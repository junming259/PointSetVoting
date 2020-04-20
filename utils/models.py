import torch
import numpy as np
import sys
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius
from torch_geometric.utils import scatter_
from .model_utils import mlp


class Model(torch.nn.Module):
    '''
    Point clouds completion model.

    Arguments:
        radius: float, radius for generating sub point clouds
        bottleneck: int, the size of bottleneck
        ratio_train: float, sampling ratio in further points sampling (FPS) during training
        ratio_test: float, sampling ratio in FPS during test
        num_contri_feats_train: int, the number of contribution features during training
        num_contri_feats_test: int, the number of contribution features during test
        is_fidReg: bool, flag for fidelity regularization during training
    '''

    def __init__(
            self,
            radius,
            bottleneck,
            ratio_train,
            ratio_test,
            num_contri_feats_train,
            num_contri_feats_test,
            is_fidReg=True,
            is_classifier=False):
        super(Model, self).__init__()
        self.num_contri_feats_train = num_contri_feats_train
        self.num_contri_feats_test = num_contri_feats_test
        self.is_fidReg = is_fidReg
        self.is_classifier = is_classifier
        self.encoder = Encoder(radius, bottleneck, ratio_train, ratio_test)
        self.latent_module = LatentModule()
        self.decoder = Decoder(bottleneck)
        if is_classifier:
            self.classifier = Classifier(bottleneck, 40)

    def forward(self, x=None, pos=None, batch=None):
        bsize = batch.max() + 1
        fidelity = 0

        # extract feature for each sub point clouds
        mean, std, x_idx, y_idx = self.encoder(x, pos, batch)

        # contribution feature selection
        contribution_mean, contribution_std, mapping = self.feature_selection(mean, std, bsize,
                                                                              self.num_contri_feats_train,
                                                                              self.num_contri_feats_test)
        self.contribution_mean, self.contribution_std = contribution_mean, contribution_std

        # compute optimal latent feature
        optimal_z = self.latent_module(contribution_mean, contribution_std)

        # generate point clouds from latent feature
        generated_pc = self.decoder(optimal_z)

        # maskout contribution points from input points
        mask = []
        for item in mapping.keys():
            mask.append(y_idx==item)
        mask = torch.stack(mask, dim=-1).any(dim=-1)
        contribution_pos = pos[x_idx[mask]]
        contribution_batch = batch[x_idx[mask]]

        if self.training and self.is_fidReg:
            # during training reduce fidelity, which is the average distance from
            # each point in the input to its nearest neighbour in the output. This
            # measures how well the input is preserved.
            masked_y_idx = y_idx[mask].detach().cpu().numpy()
            mapped_masked_y_idx = list(map(lambda x: mapping[x], masked_y_idx))

            # generate point clouds from each contribution latent feature
            latent_pcs = self.decoder(contribution_mean.view(-1, contribution_mean.size(2)))

            # compute fidelity
            diff = contribution_pos.unsqueeze(1) - latent_pcs[mapped_masked_y_idx]
            min_dist = diff.norm(dim=-1).min(dim=1)[0]
            fidelity = scatter_('mean', min_dist, y_idx[mask])
            fidelity = torch.mean(fidelity)

        if not self.training:
            # select the first contribution point clouds for visualization
            self.contribution_pc = contribution_pos[contribution_batch==0]

        if self.is_classifier:
            score = self.classifier(optimal_z)
            return generated_pc, fidelity, score

        return generated_pc, fidelity, None

    def generate_pc_from_latent(self, x):
        '''
        Generate point clouds from latent features.

        Arguments:
            x: [bsize, bottleneck]
        '''
        x = self.decoder(x)
        return x

    def feature_selection(self, mean, std, bsize, num_feats_train, num_feats_test):
        '''
        Not all features generated from sub point clouds will be considered during
        optimal latent feature computation. During training, random feature selection
        is adapted. During test, features with top probability will be contributed
        to calculate the optimal latent feature.

        Arguments:
            mean: [-1, f], computed mean for each sub point cloud
            std: [-1, f], computed std for each sub point cloud
            bsize: int, batch size
            num_feats_train: int, maximum number of candidate features comtributing to
                             final latent features during training
            num_feats_test: int, number of candidate features comtributing to final latent
                             features during test

        Returns:
            new_mean: [bsize, k, f], selected contribution means
            new_std: bsize, k, f], selected contribution std
            mapping: dict,
        '''
        mean = mean.view(bsize, -1, mean.size(1))
        std = std.view(bsize, -1, std.size(1))

        if self.training:
            # feature random selection
            num_feats = np.random.choice(np.arange(1, num_feats_train+1), 1, False)
            idx = np.random.choice(mean.size(1), num_feats, False)
            new_mean = mean[:, idx, :]
            new_std = std[:, idx, :]

            # build a mapping
            source_idx = torch.arange(mean.size(0)*mean.size(1))
            target_idx = torch.arange(new_mean.size(0)*new_mean.size(1))
            source_idx = source_idx.view(bsize, -1)[:, idx].view(-1)
            mapping = dict(zip(source_idx.numpy(), target_idx.numpy()))

        else:
            # compute probability of each sub mean feature in other else distribution
            diff = mean.unsqueeze(2) - mean.unsqueeze(1)
            expanded_std = std.unsqueeze(1).expand(-1, std.size(1), -1, -1)
            prob = -torch.log(expanded_std) - 0.5*torch.pow(diff/expanded_std, 2)      # [bsize, k, k, bottleneck]
            prob = prob.sum(dim=-1).sum(dim=-1)     # [bsize, k]

            # choose features with top probability
            sorted, indices = prob.sort(dim=1, descending=True)
            top_indices = indices[:, :num_feats_test]
            expanded_top_indices = top_indices.unsqueeze(-1).expand(-1, -1, mean.size(-1))     # [bsize, num, bottleneck]
            new_mean = torch.gather(mean, dim=1, index=expanded_top_indices)
            new_std = torch.gather(std, dim=1, index=expanded_top_indices)

            # build a mapping
            source_idx = torch.arange(indices.numel()).view(bsize, -1)
            target_idx = torch.arange(top_indices.numel())
            source_idx = torch.gather(source_idx, 1, top_indices.cpu()).view(-1)
            mapping = dict(zip(source_idx.numpy(), target_idx.numpy()))

        return new_mean, new_std, mapping


class Encoder(torch.nn.Module):
    def __init__(self, radius, bottleneck, ratio_train, ratio_test):
        super(Encoder, self).__init__()
        self.sa_module = SAModule(radius, ratio_train, ratio_test,
                                  mlp([3, 64, 128, 512], leaky=True))
        self.mlp = mlp([512+3, 512, bottleneck*2], last=True, leaky=True)

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, pos, batch):
        x, new_pos, new_batch, x_idx, y_idx = self.sa_module(x, pos, batch)
        x = self.mlp(torch.cat([x, new_pos], dim=-1))
        mean, logvar = torch.chunk(x, 2, dim=-1)
        std = torch.exp(0.5*logvar)
        return mean, std, x_idx, y_idx


class SAModule(torch.nn.Module):
    def __init__(self, r, ratio_train, ratio_test, nn):
        '''
        Set abstraction module, which is proposed by Pointnet++.
        r: ball query radius
        ratio_train: sampling ratio in further points sampling (FPS) during training.
        ratio_test: sampling ratio in FPS during test.
        nn: mlp.
        '''
        super(SAModule, self).__init__()
        self.r = r
        self.ratio_train = ratio_train
        self.ratio_test = ratio_test
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        if self.training:
            ratio = self.ratio_train
        else:
            ratio = self.ratio_test
        idx = fps(pos, batch, ratio=ratio)
        # ball query searches neighbors
        y_idx, x_idx = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=128)

        edge_index = torch.stack([x_idx, y_idx], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch, x_idx, y_idx


class LatentModule(torch.nn.Module):
    def __init__(self):
        super(LatentModule, self).__init__()

    def forward(self, mean, std):
        '''
        mean: [bsize, n, k]
        '''
        # guassian model to get optimal
        x = mean
        denorm = torch.sum(1/std, dim=1)
        nume = torch.sum(x/std, dim=1)
        optimal_x = nume / denorm       # [bsize, k]
        return optimal_x


class Decoder(torch.nn.Module):
    def __init__(self, bottleneck):
        '''
        Same decoder structure as proposed in the FoldingNet
        '''
        super(Decoder, self).__init__()
        self.fold1 = FoldingNetDecFold1(bottleneck)
        self.fold2 = FoldingNetDecFold2(bottleneck)

    def forward(self, x):  # input x = batch, 512
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, 45 ** 2, 1)  # x = batch,45^2,512
        code = x
        code = x.transpose(2, 1)  # x = batch,512,45^2

        meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        grid = GridSamplingLayer(batch_size, meshgrid)  # grid = batch,45^2,2
        grid = torch.from_numpy(grid)

        if x.is_cuda:
            grid = grid.cuda()

        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2

        x = self.fold1(x)  # x = batch,3,45^2
        x = torch.cat((code, x), 1)  # x = batch,515,45^2
        x = self.fold2(x)  # x = batch,3,45^2

        return x.transpose(2, 1)


class FoldingNetDecFold1(torch.nn.Module):
    def __init__(self, bottleneck):
        super(FoldingNetDecFold1, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck+2, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 3, 1)
        self.relu = torch.nn.ReLU()
        # self.bn1 = torch.nn.BatchNorm1d(512)
        # self.bn2 = torch.nn.BatchNorm1d(512)

    def forward(self, x):  # input x = batch,514,45^2
        x = self.relu(self.conv1(x))  # x = batch,512,45^2
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        # x = self.relu(self.bn1(self.conv1(x)))  # x = batch,512,45^2
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.conv3(x)
        return x


class FoldingNetDecFold2(torch.nn.Module):
    def __init__(self, bottleneck):
        super(FoldingNetDecFold2, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck+3, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 3, 1)
        self.relu = torch.nn.ReLU()
        # self.bn1 = torch.nn.BatchNorm1d(512)
        # self.bn2 = torch.nn.BatchNorm1d(512)

    def forward(self, x):  # input x = batch,515,45^2
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        # x = self.relu(self.bn1(self.conv1(x)))  # x = batch,512,45^2
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.conv3(x)
        return x


def GridSamplingLayer(batch_size, meshgrid):
    '''
    output Grid points as a NxD matrix
    params = {
    'batch_size': 8
    'meshgrid': [[-0.3,0.3,45],[-0.3,0.3,45]]
    }
    '''
    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
    return g


class Classifier(torch.nn.Module):
    '''
    Classifier to do classification from the latent feature vector

    Arguments:
        bottleneck: bottleneck size
        k: the number of output categories
    '''
    def __init__(self, bottleneck, k):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(bottleneck, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)
