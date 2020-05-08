# https://github.com/shizikc/completion3d

import h5py
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def read_matching_files(path_1, path_2):
    return list(set(os.listdir(path_1)).intersection(os.listdir(path_2)))


def load_h5(path_x, path_y, size=10, verbose=True):
    """
    returns a size number of files from the intersection path_x path_y
    :param path: string : folder path - should contain models
    :param size: int:
    :param verbose: boolean:
    :return:
    """
    models = read_matching_files(path_x, path_y)
    size = min(size, len(models))

    def load_single_file(file_name_x, file_name_y, file_cnt=1):
        if verbose:
            print("Loading " + str((file_cnt / size) * 100) + "%")
        fx = h5py.File(file_name_x, 'r')
        fy = h5py.File(file_name_y, 'r')
        datax = torch.tensor(fx['data'])
        datay = torch.tensor(fy['data'])
        fx.close()
        fy.close()
        return datax, datay

    lst_x = []
    lst_y = []
    for idx, model in enumerate(models):
        temp_x = os.path.join(path_x, model)
        temp_y = os.path.join(path_y, model)
        if idx >= size:
            break
        out_x, out_y = load_single_file(temp_x, temp_y, idx + 1)
        lst_x.append(out_x)
        lst_y.append(out_y)
    return torch.stack(lst_x), torch.stack(lst_y)


def plot_pc(pc, vec=None):
    x, y, z = convert_to_np(pc)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, s=4, cmap='Greens')
    if vec is not None:
        v1, v2, v3 = convert_to_np(vec)
        ax.scatter3D(v1, v2, v3, s=4, color="r")
    plt.show()


def convert_to_np(pc):
    return pc[:, 0], pc[:, 1], pc[:, 2]


def get_data(train_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        # DataLoader(valid_ds, batch_size=bs * 2)
    )


def uniform_sampling(num_samples, r1, r2, dtype=torch.double):
    """
    samples num samples uniformly over the unit cube
    :param num_samples: int
    :return:
    """
    u = torch.rand(num_samples, dtype=dtype)
    u = (r1 - r2) * u + r2
    return u


def batch_pairwise_dist(a, b):
    """

    :param a:
    :param b:
    :return: torch.Size([bs, num_samples, num_samples]) the i,j,k element is the distance  of the point j in a
    # in batch sample i, from point k in b
    """
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points)  # .type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


def batch_min_dist(x, y, dim=2):
    """

    :param x:
    :param y:
    :param dim:
    :return: minimum dist for each point in x from y
    """
    assert dim != 0
    dist = batch_pairwise_dist(x, y)  # torch.Size([bs, num_samples, num_samples])
    values, indices = dist.min(dim=dim)
    return values.unsqueeze(2).requires_grad_(False)


def preprocess(x, y):
    """
    TODO: return a centered around zero and at most range 1 edges
    :param x: input sample in (batch_size, num_points, 3)
    :param y:  input sample in (batch_size, num_points, 3)
    :return: x, y, u in device of shape (batch_size, num_points, 3), emd_dist tensor in (batch_size, num_points, 1)
    """
    sharp = 100
    u = uniform_sampling(y.shape, x.min().item(), x.max().item())
    if y is not None:
        dsts = batch_min_dist(u, y) * sharp
        return x.to(dev), y.to(dev), u.to(dev), dsts.to(dev)
    # when inference
    return x.to(dev), torch.zeros(x.shape), u.to(dev), torch.zeros(x.shape)


class WrappedDataLoader:
    """
    Add a pre processing stage to get unique points
    """

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)
