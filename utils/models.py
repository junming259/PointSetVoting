import torch
import numpy as np
import sys
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius
from .model_utils import mlp, create_batch_one_hot_category, chamfer_loss


class Model(torch.nn.Module):
    """
    Arguments:
        radius: float, radius for generating sub point clouds
        bottleneck: int, the size of bottleneck
        num_pts: int, the number of points in a complete point cloud
        num_pts_observed: int, the number of points a partial point cloud
        num_vote_train: int, the number of votes during training
        num_contrib_vote_train: int, the maximum number of selected votes during training
        num_vote_test: int, the number of votes during test
        is_vote: bool, enable voting strategy
        task: string, [segmentation, classification, completion]
    """

    def __init__(
            self,
            radius,
            bottleneck,
            num_pts,
            num_pts_observed,
            num_vote_train,
            num_contrib_vote_train,
            num_vote_test,
            is_vote,
            task
        ):
        super(Model, self).__init__()
        self.num_vote_train = num_vote_train
        self.num_contrib_vote_train = num_contrib_vote_train
        self.num_vote_test = num_vote_test
        self.is_vote = is_vote
        self.task = task
        self.bottleneck = bottleneck

        ratio_train = num_vote_train / num_pts
        ratio_test = num_vote_test / num_pts_observed

        self.transformer = mlp([3, 64], leaky=True)

        self.encoder = Encoder(radius, bottleneck, ratio_train, ratio_test)
        self.latent_module = LatentModule(self.is_vote)
        if task == 'completion':
            self.decoder = FoldingBasedDecoder(bottleneck)
        if task == 'classification':
            self.decoder = Classifier(bottleneck, 40)
        if task == 'segmentation':
            self.decoder = Segmentator(bottleneck, 50)

    def forward(self, x=None, pos=None, batch=None, category=None, label=None):
        batch = batch - batch.min()

        x = self.transformer(pos)

        # extract feature for each sub point clouds
        mean, std, x_idx, y_idx = self.encoder(x, pos, batch)

        # select contribution features
        if self.training:
            contrib_mean, contrib_std = \
                self.feature_selection(mean, std, self.num_vote_train, self.num_contrib_vote_train)
        else:
            contrib_mean, contrib_std = \
                self.feature_selection(mean, std, self.num_vote_test)

        # compute optimal latent feature
        optimal_z = self.latent_module(contrib_mean, contrib_std)

        self.contrib_mean, self.contrib_std = contrib_mean, contrib_std
        self.optimal_z = optimal_z

        # generate prediction
        if self.task == 'segmentation':
            one_hot_category = create_batch_one_hot_category(category)
            # pred = self.decoder(optimal_z, pos, one_hot_category)
            pred = self.decoder(optimal_z, x, one_hot_category)
        else:
            pred = self.decoder(optimal_z)

        # compute loss
        loss = None
        if label is not None:
            if self.task == 'completion':
                loss = chamfer_loss(pred, label.view(pred.size(0), -1, 3))
            elif self.task == 'classification':
                loss = F.nll_loss(pred, label, reduction='none')
            elif self.task == 'segmentation':
                loss = F.nll_loss(pred, label, reduction='none')
            else:
                assert False

        return pred, loss

    def generate_pc_from_latent(self, x):
        """
        Generate point clouds from latent features.

        Arguments:
            x: [bsize, bottleneck]
        """
        x = self.decoder(x)
        return x

    def feature_selection(
            self,
            mean,
            std,
            num_vote,
            num_contrib_vote=None,
        ):
        """
        During training, random feature selection is adapted. Only a portion of
        features generated from local point clouds are considered for optimal
        latent feature computation. During testing, all generated features will be
        contributed to calculate the optimal latent feature.

        Arguments:
            mean: [-1, bottleneck], computed mean for each sub point cloud
            std: [-1, bottleneck], computed std for each sub point cloud
            num_vote: int, the total number of extracted features from encoder.
            num_contrib_vote: int, maximum number of candidate features comtributing
                                    to optimal latent features during training

        Returns:
            new_mean: [bsize, num_contrib_vote, f], selected contribution means
            new_std: [bsize, num_contrib_vote, f], selected contribution std
        """
        mean = mean.view(-1, num_vote, mean.size(1))
        std = std.view(-1, num_vote, std.size(1))

        # feature random selection
        if self.training:
            num = np.random.choice(np.arange(1, num_contrib_vote+1), 1, False)
            idx = np.random.choice(mean.size(1), num, False)
        else:
            idx = np.arange(num_vote)
        new_mean = mean[:, idx, :]
        new_std = std[:, idx, :]

        return new_mean, new_std


class Encoder(torch.nn.Module):
    def __init__(self, radius, bottleneck, ratio_train, ratio_test):
        super(Encoder, self).__init__()
        self.sa_module = SAModule(radius, ratio_train, ratio_test,
                                  mlp([64+3, 64, 128, 512], leaky=True))
        self.mlp = mlp([512+3, 512, bottleneck*2], last=True, leaky=True)

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, pos, batch):
        x, new_pos, new_batch, x_idx, y_idx = self.sa_module(x, pos, batch)

        x = self.mlp(torch.cat([x, new_pos], dim=-1))
        mean, logvar = torch.chunk(x, 2, dim=-1)
        # in case gradient explodes
        # logvar = torch.clamp(logvar, min=-100, max=100)
        std = torch.exp(0.5*logvar)
        # self.new_pos = new_pos
        return mean, std, x_idx, y_idx


class SAModule(torch.nn.Module):
    def __init__(self, r, ratio_train, ratio_test, nn):
        """
        Set abstraction module, which is proposed by Pointnet++.
        r: ball query radius
        ratio_train: sampling ratio in further points sampling (FPS) during training.
        ratio_test: sampling ratio in FPS during test.
        nn: mlp.
        """
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
    def __init__(self, is_vote):
        super(LatentModule, self).__init__()
        self.is_vote = is_vote

    def forward(self, mean, std):
        """
        mean: [bsize, n, bottleneck]
        """
        # guassian model to get optimal
        if self.is_vote:
            denorm = torch.sum(1/std, dim=1)
            nume = torch.sum(mean/std, dim=1)
            optimal_z = nume / denorm       # [bsize, bottleneck]
        # max pooling
        else:
            optimal_z = mean.max(dim=1)[0]
        return optimal_z


class FoldingBasedDecoder(torch.nn.Module):
    def __init__(self, bottleneck):
        """
        Same decoder structure as proposed in the FoldingNet
        """
        super(FoldingBasedDecoder, self).__init__()
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
    """
    output Grid points as a NxD matrix
    params = {
    'batch_size': 8
    'meshgrid': [[-0.3,0.3,45],[-0.3,0.3,45]]
    }
    """
    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
    return g


class Classifier(torch.nn.Module):
    """
    Classifier to do classification from the latent feature vector

    Arguments:
        bottleneck: bottleneck size
        k: the number of output categories
    """
    def __init__(self, bottleneck, k):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(bottleneck, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)


class Segmentator(torch.nn.Module):
    """
    Pointwise part segmentation

    Arguments:
        k: the number of output categories
    """
    def __init__(self, bottleneck, k):
        super(Segmentator, self).__init__()
        self.bottleneck = bottleneck
        # self.fc1 = torch.nn.Linear(bottleneck+3+16, 512)
        self.fc1 = torch.nn.Linear(bottleneck+64+16, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, latent, pos, one_hot_category):

        bsize = one_hot_category.size(0)
        pos = pos.view(bsize, -1, 64)
        latent = torch.unsqueeze(latent, 1)     # batch, 1, bottleneck
        latent = latent.repeat(1, pos.size(1), 1)
        one_hot_category = torch.unsqueeze(one_hot_category, 1)     # batch, 1, 16
        one_hot_category = one_hot_category.repeat(1, pos.size(1), 1)

        x = torch.cat([latent, pos, one_hot_category], dim=-1).view(-1, self.bottleneck+64+16)

        x = self.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=-1)
