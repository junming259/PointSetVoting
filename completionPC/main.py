import torch
import numpy as np
import os
import argparse
import torch.nn.functional as F
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pdb
import traceback
import colorama
from colorama import Fore, Back, Style
from torch import autograd


from itertools import chain
from data import data_utils as d_utils
from utils.logger import Logger
from torch.autograd import grad
from utils.models import Model
from utils.model_utils import get_lr, chamfer_loss
from torch.utils.data import DataLoader
from shutil import copyfile
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch_geometric.nn import fps, knn
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


def train_one_epoch(loader, epoch, check_dir):

    model.train()
    total_loss = []
    global i

    for j, data in enumerate(loader, 0):
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y

        # training
        model.zero_grad()
        generated_pc, sub_pc_reg = model(None, pos, batch)
        loss_chf = chamfer_loss(generated_pc, pos.view(-1, args.num_pts, 3))
        loss = loss_chf + sub_pc_reg
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        # write summary
        if i % 100 == 0:
            # print('Training, Chamfer_loss: {:.5f}, Loss_d: {:.4f}, Loss_e: {:.4f}'.format(loss_g, loss_d, loss_e))
            # print('Training, Chamfer_loss: {:.5f}'.format(loss))
            print('Train, chamfer_loss: {:.5f}, sub_Reg: {:.4f}'.format(loss_chf, sub_pc_reg))
            logger.scalar_summary('loss_chamfer', loss_chf, i)
            logger.scalar_summary('loss_sub_reg', sub_pc_reg, i)
            logger.scalar_summary('lr', get_lr(optimizer), i)

        i = i + 1


def test(loader, epoch, check_dir):
    model.eval()
    total_chamfer_loss = []

    for j, data in enumerate(loader, 0):
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y
        # pos_partial, batch_partial = create_partial_pc(pos, batch, 1, 1200)
        pos_partial, batch_partial = pos, batch

        with torch.no_grad():
            generated_pc, _ = model(None, pos_partial, batch_partial)
            pred_latent_pos = model.generate_pc_from_latent(model.selected_mean[0, 0, :].view(1, 512))
            pos_partial = model.partial_pc
        total_chamfer_loss.append(chamfer_loss(generated_pc, pos.view(-1, args.num_pts, 3)))

        # save sample results
        if j == len(loader)-1:
            pos = pos.cpu().detach().numpy()
            pos_partial = pos_partial.cpu().detach().numpy()
            pred_pos = generated_pc.cpu().detach().numpy()
            pred_latent_pos = pred_latent_pos.cpu().detach().numpy()

            np.save(os.path.join(check_dir, 'pos_{}'.format(epoch)), pos.reshape(-1, args.num_pts, 3)[0])
            np.save(os.path.join(check_dir, 'pos_partial_{}'.format(epoch)), pos_partial)
            np.save(os.path.join(check_dir, 'pred_pos_{}'.format(epoch)), pred_pos[0])
            np.save(os.path.join(check_dir, 'pred_latent_pos_{}'.format(epoch)), pred_latent_pos)

    avg_chamfer_loss = sum(total_chamfer_loss) / len(total_chamfer_loss)
    logger.scalar_summary('test_chamfer_dist', avg_chamfer_loss, epoch)
    print('Epoch: {:03d}, eval_loss: {:.5f}\n'.format(epoch, avg_chamfer_loss))


def create_partial_pc(pos, batch, num, npts):
    '''
    Create sythetic partial point clouds.
    pos: [N, 3]
    batch: [N]
    num: the number of sub point clouds
    npts: the number of points in each sub point clouds
    '''
    bsize = batch.max() + 1
    pos = pos.view(bsize, -1, 3)
    batch = batch.view(-1, args.num_pts)
    pc, bh = [], []
    for i in range(pos.size(0)):
        p, b = pos[i], batch[i]
        idx = np.random.choice(pos.size(1), num, False)
        y_idx, x_idx = knn(p, p[idx], npts)
        p_partial = p[x_idx]
        b_partial = b[:p_partial.size(0)]
        pc.append(p_partial)
        bh.append(b_partial)
    partial_pos = torch.cat(pc, dim=0)
    partial_batch = torch.cat(bh, dim=0)
    return partial_pos, partial_batch


# def create_partial_pc(pos, batch, num, npts):
#     '''
#     Create sythetic partial point clouds.
#     pos: [N, 3]
#     batch: [N]
#     num: the number of points in each point clouds
#     '''
#     bsize = batch.max() + 1
#     pos = pos.view(bsize, -1, 3)
#     batch = batch.view(-1, args.num_pts)
#     pc, bh = [], []
#     for i in range(pos.size(0)):
#         p, b = pos[i], batch[i]
#         idx = fps(p, ratio=32/pos.size(1))
#         rand_idx = np.random.choice(32, 16, False)
#         idx = idx[rand_idx]
#         y_idx, x_idx = knn(p, p[idx], 64)
#         p_partial = p[x_idx]
#         print(p_partial.size())
#         b_partial = b[:p_partial.size(0)]
#         # p_partial = p[x_idx].unique(dim=0)
#         # b_partial = b[:p_partial.size(0)]
#         pc.append(p_partial)
#         bh.append(b_partial)
#     partial_pos = torch.cat(pc, dim=0)
#     partial_batch = torch.cat(bh, dim=0)
#     return partial_pos, partial_batch


def backup(log_dir, parser):
    copyfile('main.py', os.path.join(log_dir, 'main.py'))
    copyfile('../utils/models.py', os.path.join(log_dir, 'models.py'))
    copyfile('../utils/model_utils.py', os.path.join(log_dir, 'model_utils.py'))

    file = open(os.path.join(log_dir, 'parameters.txt'), 'w')
    adict = vars(parser.parse_args())
    keys = list(adict.keys())
    keys.sort()
    for item in keys:
        file.write('{0}:{1}\n'.format(item, adict[item]))
        print('{0}:{1}'.format(item, adict[item]))
    file.close()


colorama.init()
class GuruMeditation(autograd.detect_anomaly):
    def __init__(self):
        super(GuruMeditation, self).__init__()
    def __enter__(self):
        super(GuruMeditation, self).__enter__()
        return self
    def __exit__(self, type, value, trace):
        super(GuruMeditation, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            halt(str(value))

def halt(msg):
    print (Fore.RED + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print (Fore.RED + "┃ Software Failure. Press left mouse button to continue ┃")
    print (Fore.RED + "┃        Guru Meditation 00000004, 0000AAC0             ┃")
    print (Fore.RED + "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print(Style.RESET_ALL)
    print (msg)
    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='model', help="model name")
    parser.add_argument("--category", default='Chair', help="point clouds category")
    parser.add_argument("--num_pts", type=int, help="number of sampled points")
    parser.add_argument("--bsize", type=int, default=32, help="batch size")
    parser.add_argument("--max_epoch", type=int, default=250, help="max epoch to train")
    parser.add_argument("--lr", type=float, default=0.001, help="batch size")
    parser.add_argument("--step_size", type=int, default=300, help="step size to reduce lr")
    parser.add_argument("--num_sub_feats", type=int, help="number of features during test")
    parser.add_argument("--is_subReg", action='store_true', help="flag for sub point clouds regularization")
    parser.add_argument("--randRotY", action='store_true', help="flag for random rotation along Y axis")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    category = 'Chair'
    pre_transform = T.NormalizeScale()
    transform = T.Compose([T.FixedPoints(args.num_pts), T.RandomRotate(180, axis=1)])
    train_dataset = ShapeNet('../data_root/ShapeNet_normal', category, split='trainval',
                             include_normals=False, pre_transform=pre_transform, transform=transform)
    test_dataset = ShapeNet('../data_root/ShapeNet_normal', category, split='test',
                            include_normals=False, pre_transform=pre_transform, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                  num_workers=6, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=True,
                                 num_workers=6)

    model = Model(num_sub_feats=args.num_sub_feats, is_subReg=args.is_subReg)
    # to do: using multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.2)

    model_name = args.model_name
    check_dir = 'checkpoint/{}'.format(model_name)
    if not os.path.exists('checkpoint/{}'.format(model_name)):
        os.makedirs('checkpoint/{}'.format(model_name))

    log_dir = './logs/{}'.format(model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    # backup
    backup(log_dir, parser)

    global i
    i = 1

    # with GuruMeditation():
    # with torch.autograd.detect_anomaly():
    for epoch in range(1, args.max_epoch+1):

        # do training
        train_one_epoch(train_dataloader, epoch, check_dir)

        # reduce learning rate
        scheduler.step()

        # evaluation
        test(test_dataloader, epoch, check_dir)

    # save model
    torch.save(model.state_dict(), 'checkpoint/{}/model.pth'.format(model_name))
