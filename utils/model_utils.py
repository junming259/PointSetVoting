import torch
import numpy as np
import os
import torch.nn.functional as F

from torch.nn import Linear as Lin, Sequential as Seq, ReLU, BatchNorm1d, LeakyReLU
from torch_geometric.nn import fps, radius, knn_interpolate, PointConv, knn_graph, knn, DynamicEdgeConv
from torch_geometric.utils import scatter_
from torch_scatter import scatter_add, scatter_min, scatter_max


class FCBlock(torch.nn.Module):
    def __init__(self, chs):
        """
        Fully connected layers with batchnorm and relu every layer afterwards.

        Arguments:
            chs: list of number of fully connected layers. [x1, x2, x3]

        """
        super().__init__()
        ls = [Lin(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        bs = [BatchNorm1d(chs[i+1]) for i in range(len(chs)-2)]
        self.lins = torch.nn.ModuleList(ls)
        self.bns = torch.nn.ModuleList(bs)
        self.relu = ReLU()

    def forward(self, x):
        '''
        Arguments:
            x: [bs, k, m]

        Returns:
            pred: predicated label, [bs, num_cls]
        '''
        for i in range(len(self.lins)-1):
            x = self.relu(self.bns[i](self.lins[i](x)))
        x = self.lins[-1](x)
        return x


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


def ChamferDistance(x, y):
    x_size = x.size()
    y_size = y.size()
    assert (x_size[0] == y_size[0])
    assert (x_size[2] == y_size[2])
    x = torch.unsqueeze(x, 1)  # x = batch,1,2025,3
    y = torch.unsqueeze(y, 2)  # y = batch,2048,1,3

    x = x.repeat(1, y_size[1], 1, 1)  # x = batch,2048,2025,3
    y = y.repeat(1, 1, x_size[1], 1)  # y = batch,2048,2025,3
    # x_y = (x - y).norm(dim=-1)


    x_y = x - y
    x_y = torch.pow(x_y, 2)  # x_y = batch,2048,2025,3
    x_y = torch.sum(x_y, 3, keepdim=True)  # x_y = batch,2048,2025,1
    x_y = torch.squeeze(x_y, 3)  # x_y = batch,2048,2025
    x_y= torch.pow(x_y, 0.5)

    # x = x.unsqueeze(1)
    # y = y.unsqueeze(2)
    # x_y = (x - y).norm(dim=-1)
    x_y_row, _ = torch.min(x_y, 1, keepdim=True)  # x_y_row = batch,1,2025
    x_y_col, _ = torch.min(x_y, 2, keepdim=True)  # x_y_col = batch,2048,1

    x_y_row = torch.mean(x_y_row, 2, keepdim=True)  # x_y_row = batch,1,1
    x_y_col = torch.mean(x_y_col, 1, keepdim=True)  # batch,1,1
    x_y_row_col = torch.cat((x_y_row, x_y_col), 2)  # batch,1,2
    chamfer_distance, _ = torch.max(x_y_row_col, 2, keepdim=True)  # batch,1,1
    # chamfer_distance = torch.reshape(chamfer_distance,(x_size[0],-1))  #batch,1
    # chamfer_distance = torch.squeeze(chamfer_distance,1)    # batch
    chamfer_distance = torch.mean(chamfer_distance)
    return chamfer_distance


def chamfer_loss(x, y):
    '''
    Compute chamfer distance for x and y
    '''
    x = x.unsqueeze(1)
    y = y.unsqueeze(2)
    diff = (x - y).norm(dim=-1)
    dis1 = diff.min(dim=1)[0].mean(dim=1)
    dis2 = diff.min(dim=2)[0].mean(dim=1)
    dis = dis1 + dis2
    # dis = torch.stack([dis1, dis2], dim=-1).max(-1)[0]
    return dis.mean()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
