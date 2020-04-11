import torch
import numpy as np
import sys
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius
from torch_geometric.utils import scatter_
from .model_utils import FCBlock, mlp


class Model(torch.nn.Module):
    def __init__(self, num_sub_feats, is_subReg=True):
        super(Model, self).__init__()
        self.num_sub_feats = num_sub_feats
        self.is_subReg = is_subReg
        self.encoder = Encoder()
        self.latent_module = LatentModule()
        self.decoder = Decoder()

    def forward(self, x=None, pos=None, batch=None):
        bsize = batch.max() + 1
        sub_pc_reg = 0

        # extract feature for each sub point clouds
        mean, std, x_idx, y_idx = self.encoder(x, pos, batch)
        self.mean, self.std = mean, std

        # random selection
        selected_mean, selected_std, mapping = self.feature_selection(mean, std, bsize, self.num_sub_feats)
        self.selected_mean, self.selected_std = selected_mean, selected_std

        # latent feature
        optimal_z = self.latent_module(selected_mean, selected_std)

        # generate point clouds from latent feature
        generated_pc = self.decoder(optimal_z)

        mask = []
        for item in mapping.keys():
            mask.append(y_idx==item)
        mask = torch.stack(mask, dim=-1).any(dim=-1)
        partial_pos = pos[x_idx[mask]]
        partial_batch = batch[x_idx[mask]]

        if self.training and self.is_subReg:
            # during training, generate point clouds from each latent feature
            masked_y_idx = y_idx[mask].detach().cpu().numpy()
            mapped_masked_y_idx = list(map(lambda x: mapping[x], masked_y_idx))

            latent_pcs = self.decoder(selected_mean.view(-1, selected_mean.size(2)))
            diff = partial_pos.unsqueeze(1) - latent_pcs[mapped_masked_y_idx]
            min_dist = diff.norm(dim=-1).min(dim=1)[0]
            partial_chamfer_dist = scatter_('mean', min_dist, y_idx[mask])
            sub_pc_reg = torch.mean(partial_chamfer_dist)

        if not self.training:
            self.partial_pc = partial_pos[partial_batch==0]     # select the first point clouds

        return generated_pc, sub_pc_reg

    def generate_pc_from_latent(self, x):
        '''
        Generate point clouds from latent features.

        Arguments:
            x: [bsize, 512]
        '''
        x = self.decoder(x)
        return x

    def feature_selection(self, mean, std, bsize, num_feats):
        '''
        mean: computed mean for each sub point cloud
        std: computed std for each sub point cloud
        bsize: batch size
        num_feats: number of candidate features comtributing to final latent features
        '''
        mean = mean.view(bsize, -1, mean.size(1))
        std = std.view(bsize, -1, std.size(1))

        # feature random selection
        if self.training:
            num_feats = np.random.choice(np.arange(1, mean.size(1)//2), 1, False)

        idx = np.random.choice(mean.size(1), num_feats, False)
        new_mean = mean[:, idx, :]
        new_std = std[:, idx, :]

        # build a mapping function
        source_idx = torch.arange(mean.size(0)*mean.size(1)).to(mean.device)
        target_idx = torch.arange(new_mean.size(0)*new_mean.size(1)).to(new_mean.device)
        source_idx = source_idx.view(bsize, -1)[:, idx].view(-1)
        mapping = dict(zip(source_idx.detach().cpu().numpy(), target_idx.detach().cpu().numpy()))

        return new_mean, new_std, mapping


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sa_module = SAModule(0.2, 64/2048, mlp([3, 64, 128, 512], leaky=True))
        self.mlp = mlp([512+3, 512, 512*2], last=True, leaky=True)

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, pos, batch):
        x, new_pos, new_batch, x_idx, y_idx = self.sa_module(x, pos, batch)
        x = self.mlp(torch.cat([x, new_pos], dim=-1))
        mean, logvar = torch.split(x, x.size(-1)//2, dim=-1)
        std = torch.exp(0.5*logvar)
        return mean, std, x_idx, y_idx


class SAModule(torch.nn.Module):
    def __init__(self, r, ratio, nn):
        super(SAModule, self).__init__()
        self.r = r
        self.ratio = ratio
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
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
    def __init__(self):
        '''
        Same decoder structure as proposed in the FoldingNet
        '''
        super(Decoder, self).__init__()
        self.fold1 = FoldingNetDecFold1()
        self.fold2 = FoldingNetDecFold2()

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
    def __init__(self):
        super(FoldingNetDecFold1, self).__init__()
        self.conv1 = torch.nn.Conv1d(514, 512, 1)
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
    def __init__(self):
        super(FoldingNetDecFold2, self).__init__()
        self.conv1 = torch.nn.Conv1d(515, 512, 1)
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
