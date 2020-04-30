import torch
import numpy as np
import os
import torch.nn.functional as F

from torch.nn import Linear as Lin, Sequential as Seq, ReLU, BatchNorm1d, LeakyReLU
from torch_geometric.nn import fps, radius, knn_interpolate, PointConv, knn_graph, knn, DynamicEdgeConv
from torch_geometric.utils import scatter_
from torch_scatter import scatter_add, scatter_min, scatter_max


def mlp(channels, last=False, leaky=False):
    if leaky:
        rectifier = LeakyReLU
    else:
        rectifier = Relu
    l = [Seq(Lin(channels[i - 1], channels[i], bias=False), BatchNorm1d(channels[i]), rectifier())
            for i in range(1, len(channels)-1)]
    if last:
        l.append(Seq(Lin(channels[-2], channels[-1], bias=True)))
    else:
        l.append(Seq(Lin(channels[-2], channels[-1], bias=False), BatchNorm1d(channels[-1]), rectifier()))
    return Seq(*l)


class SimuOcclusion(object):
    """
    Simulate occlusion. Random select half side of points
    pos: [N, 3]
    batch: [N]
    npts: the number of output sampled points
    """
    def __call__(self, pos, batch, npts):
        bsize = batch.max() + 1
        pos = pos.view(bsize, -1, 3)
        batch = batch.view(bsize, -1)

        out_pos, out_batch = [], []
        for i in range(pos.size(0)):
            while True:
                # define a plane by its normal and it goes through origin
                vec = torch.rand(3).to(pos.device) - 0.5
                # mask out half side of points
                mask = pos[i].matmul(vec) > 0
                # mask = mask & (pos[i, :, 1] < 0)
                p, b = pos[i][mask], batch[i][mask]
                if p.size(0) >= 200:
                    break
            # ensure output contains self.npts points
            idx = np.random.choice(p.size(0), npts, True)
            out_pos.append(p[idx])
            out_batch.append(b[idx])

        out_pos = torch.cat(out_pos, dim=0)
        out_batch = torch.cat(out_batch, dim=0)
        return out_pos, out_batch

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def chamfer_loss(x, y):
    """
    Compute chamfer distance for x and y. Note there are multiple version of chamfer
    distance. The implemented chamfer distance is defined in:

        https://arxiv.org/pdf/1612.00603.pdf.

    It finds the nearest neighbor in the other set and computes their squared
    distances which are summed over both target and ground-truth sets.

    Arguments:
        x: [bsize, m, 3]
        y: [bsize, n, 3]

    Returns:
        dis: [bsize]
    """
    x = x.unsqueeze(1)
    y = y.unsqueeze(2)
    # diff = (x - y).norm(dim=-1)
    diff = (x - y).pow(2).sum(dim=-1)
    dis1 = diff.min(dim=1)[0].mean(dim=1)
    dis2 = diff.min(dim=2)[0].mean(dim=1)
    dis = dis1 + dis2
    return dis


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
