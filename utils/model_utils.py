import torch
import numpy as np
import os
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear as Lin, Sequential as Seq, ReLU, BatchNorm1d, LeakyReLU
from torch_geometric.nn import fps, radius, PointConv

MODELNET_TO_SCANOBJECTNN = {
    2:10,
    4:8,
    8:4,
    12:5,
    13:7,
    14:3,
    22:6,
    3:4,
    29:12,
    30:13,
    32:4,
    33:9,
    35:14,
    38:3
}

SCANOBJECTNN_TO_MODELNET = {
    10:[2],
    8:[4],
    4:[8,32,3],
    5:[12],
    7:[13],
    3:[14,38],
    6:[22],
    12:[29],
    13:[30],
    9:[33],
    14:[35]
}

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

# def mlp(channels, last=False, leaky=False):
#     if leaky:
#         rectifier = LeakyReLU
#     else:
#         rectifier = Relu
#     l = [Seq(Lin(channels[i - 1], channels[i], bias=False), rectifier())
#             for i in range(1, len(channels)-1)]
#     if last:
#         l.append(Seq(Lin(channels[-2], channels[-1], bias=True)))
#     else:
#         l.append(Seq(Lin(channels[-2], channels[-1], bias=False), rectifier()))
#     return Seq(*l)


# class SimuOcclusion(object):
#     """
#     Simulate occlusion. Random select half side of points
#     pos: [N, 3]
#     batch: [N]
#     npts: the number of output sampled points
#     """
#     def __call__(self, pos, batch, npts):
#         bsize = batch.max() + 1
#         pos = pos.view(bsize, -1, 3)
#         batch = batch.view(bsize, -1)
#
#         out_pos, out_batch = [], []
#         for i in range(pos.size(0)):
#             while True:
#                 # # define a plane by its normal and it goes through the origin
#                 # vec = torch.randn(3).to(pos.device)
#                 # # mask out half side of points
#                 # mask = pos[i].matmul(vec) > 0
#                 # # mask = mask & (pos[i, :, 1] < 0)
#                 # p, b = pos[i][mask], batch[i][mask]
#                 # if p.size(0) >= 256:
#                 #     break
#
#             mask = pos[i, :, 1]>0
#             if torch.sum(mask) == 0:
#                 mask = pos[i, :, 1]>-0.3
#             if torch.sum(mask) == 0:
#                 mask = pos[i, :, 1]>-0.5
#
#             # p, b = pos[i][mask], batch[i][mask]
#             # idx = np.random.choice(p.size(0), p.size(0)//8, False)
#             # p, b = p[idx], b[idx]
#
#             p, b = pos[i][mask], batch[i][mask]
#             # ensure output contains fixed number of points
#             idx = np.random.choice(p.size(0), npts, True)
#             out_pos.append(p[idx])
#             out_batch.append(b[idx])
#
#         out_pos = torch.cat(out_pos, dim=0)
#         out_batch = torch.cat(out_batch, dim=0)
#         return out_pos, out_batch
#
#     def __repr__(self):
#         return '{}()'.format(self.__class__.__name__)


def simulate_partial_point_clouds(data, npts, task):
    """
    Simulate partial point clouds.
    """
    pos, batch, label = data.pos, data.batch, data.y

    # noise = torch.randn_like(pos)*0.08
    # pos = pos + noise

    bsize = batch.max() + 1
    pos = pos.view(bsize, -1, 3)
    batch = batch.view(bsize, -1)
    if task == 'segmentation':
        label = label.view(bsize, -1)

    out_pos, out_batch, out_label = [], [], []
    for i in range(pos.size(0)):
        while True:
            # define a plane by its normal and it goes through the origin
            vec = torch.randn(3).to(pos.device)
            # mask out half side of points
            mask = pos[i].matmul(vec) > 0
            p = pos[i][mask]
            if p.size(0) >= 256:
                break

        # mask = pos[i, :, 2]>0
        # if torch.sum(mask) == 0:
        #     mask = pos[i, :, 1]>-0.3
        # if torch.sum(mask) == 0:
        #     mask = pos[i, :, 1]>-0.5
        # p = pos[i][mask]

        # ensure output contains fixed number of points
        idx = np.random.choice(p.size(0), npts, True)
        out_pos.append(pos[i][mask][idx])
        out_batch.append(batch[i][mask][idx])
        if task == 'segmentation':
            out_label.append(label[i][mask][idx])

    data.pos = torch.cat(out_pos, dim=0)
    data.batch = torch.cat(out_batch, dim=0)
    if task == 'segmentation':
        data.y = torch.cat(out_label, dim=0)
    return data



class NormalizeSphere(object):
    """
    Normalize point clouds into a unit sphere
    """
    def __init__(self, center):
        self.is_center = center
        if center:
            self.center = T.Center()
        else:
            self.center = None

    def __call__(self, data):
        if self.center is not None:
            data = self.center(data)

        scale = (1 / data.pos.norm(dim=-1).max()) * 0.999999
        data.pos = data.pos * scale

        return data

    def __repr__(self):
        return '{}(center={})'.format(self.__class__.__name__, self.is_center)
    # def __repr__(self):
    #     return '{}()'.format(self.__class__.__name__)


class NormalizeBox(object):
    """
    Normalize point clouds into a box
    """
    def __call__(self, data):
        pos_min = data.pos.min(dim=-2, keepdim=True)[0]
        pos_max = data.pos.max(dim=-2, keepdim=True)[0]
        center = (pos_max + pos_min) / 2
        scale = pos_max - pos_min
        scale = (1 / scale.max()) * 0.999999

        # print('size:', data.pos.size())
        # print(pos_min, pos_max)
        # print('center:', center)
        # print('scale:', scale)

        data.pos = (data.pos - center) * scale
        return data

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
    diff = (x - y).norm(dim=-1)
    # diff = (x - y).pow(2).sum(dim=-1)
    dis1 = diff.min(dim=1)[0].mean(dim=1)
    dis2 = diff.min(dim=2)[0].mean(dim=1)
    dis = dis1 + dis2
    return dis


def augment_transforms(args):
    """
    build augmentation transforms
    """
    pre_transform = None
    # if args.is_normalizeScale:
    #     pre_transform = T.NormalizeScale()
    # if args.is_normalizeSphere:
    #     pre_transform = NormalizeSphere()

    if args.norm == 'scale':
        pre_transform = T.NormalizeScale()
    elif args.norm == 'sphere':
        pre_transform = NormalizeSphere(center=True)
    elif args.norm == 'sphere_wo_center':
        pre_transform = NormalizeSphere(center=False)
    else:
        pass

    transform = []
    if args.dataset == 'shapenet':
        transform.append(T.FixedPoints(args.num_pts))
    if args.dataset == 'modelnet':
        transform.append(T.SamplePoints(args.num_pts))

    # if args.is_randRotY:
    #     transform.append(T.RandomRotate(180, axis=1))
    # if args.is_randST:
    #     transform.append(T.RandomScale((2/3, 3/2)))
    #     transform.append(T.RandomTranslate(0.1))
    transform = T.Compose(transform)
    return pre_transform, transform


def create_batch_one_hot_category(category):
    """
    Create batch one-hot vector for indicating category. ShapeNet.

    Arguments:
        category: [batch]
    """
    batch_one_hot_category = np.zeros((len(category), 16))
    for b in range(len(category)):
        batch_one_hot_category[b, int(category[b])] = 1
    batch_one_hot_category = torch.from_numpy(batch_one_hot_category).float().cuda()
    return batch_one_hot_category


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
