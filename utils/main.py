import torch
import numpy as np
import os
import shutil
import argparse
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.models import Model
from utils.model_utils import SimuOcclusion, get_lr, chamfer_loss
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch_geometric.datasets import ShapeNet, ModelNet
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from utils.class_completion3D import *

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


def train_one_epoch(args, loader, optimizer, logger, epoch):

    model.train()
    loss_summary = {}
    global i

    for j, data in enumerate(loader, 0):
        # data.pos = data.pos.double()
        # data.y = data.y.double()
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y
        # print(pos)
        # pos = pos.float()
        # label = label.double()
        # batch = batch.double()
        # print(batch)
        # print(torch.type(pos))
        # training
        model.zero_grad()
        generated_pc, fidelity, score = model(None, pos, batch)

        loss = 0
        if args.is_pCompletion:
            loss_summary['loss_chamfer'] = chamfer_loss(generated_pc, pos.view(-1, args.num_pts, 3)).mean()
            loss += args.weight_chamfer*loss_summary['loss_chamfer']
        if args.is_fidReg:
            loss_summary['loss_fidelity'] = fidelity.mean()
            loss += args.weight_fidelity*loss_summary['loss_fidelity']
        if args.is_classifier:
            loss_summary['loss_cls'] = F.nll_loss(score, label)
            loss += args.weight_cls*loss_summary['loss_cls']

        loss.backward()
        optimizer.step()

        # write summary
        if i % 100 == 0:
            for item in loss_summary:
                logger.add_scalar(item, loss_summary[item], i)
            logger.add_scalar('lr', get_lr(optimizer), i)
            print(''.join(['{}: {:.4f}, '.format(k, v) for k,v in loss_summary.items()]))
        i = i + 1


def test_one_epoch(args, loader, logger, epoch):

    model.eval()
    trans = SimuOcclusion()
    loss_summary = {}
    if args.is_pCompletion:
        loss_summary['test_chamfer_dist'] = 0
    if args.is_fidReg:
        loss_summary['test_fidelity'] = 0
    if args.is_classifier:
        loss_summary['test_acc'] = 0

    for j, data in enumerate(loader, 0):
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y
        if args.is_simuOcc:
            pos_observed, batch_observed = trans(pos, batch, args.num_pts_observed)
        else:
            pos_observed, batch_observed = pos, batch

        # inference
        with torch.no_grad():
            generated_pc, fidelity, score = model(None, pos_observed, batch_observed)

        if args.is_pCompletion:
            loss_summary['test_chamfer_dist'] += chamfer_loss(generated_pc, pos.view(-1, args.num_pts, 3)).mean()
        if args.is_fidReg:
            loss_summary['test_fidelity'] += fidelity.mean()
        if args.is_classifier:
            pred = score.max(1)[1]
            loss_summary['test_acc'] += pred.eq(label).float().mean()

    for item in loss_summary:
        loss_summary[item] /= len(loader)
        logger.add_scalar(item, loss_summary[item], epoch)
    print('Epoch: {:03d}, '.format(epoch), end='')
    print(''.join(['{}: {:.4f}, '.format(k, v) for k,v in loss_summary.items()]), '\n')


def check_overwrite(model_name):
    check_dir = 'checkpoint/{}'.format(model_name)
    log_dir = 'logs/{}'.format(model_name)
    if os.path.exists(check_dir) or os.path.exists(log_dir):
        valid = ['y', 'yes', 'no', 'n']
        inp = None
        while inp not in valid:
            inp = input('{} already exists. Do you want to overwrite it? (y/n)'.format(model_name))
            if inp.lower() in ['n', 'no']:
                raise Exception('Please create new experiment.')
        # remove the existing dir if overwriting.
        if os.path.exists(check_dir):
            shutil.rmtree(check_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

    # create directory
    os.makedirs(check_dir)
    os.makedirs(log_dir)

    return check_dir, log_dir


def train(args, train_dataloader, test_dataloader):

    check_dir, log_dir = check_overwrite(args.model_name)
    logger = SummaryWriter(log_dir=log_dir)
    backup(log_dir, parser)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.2)

    global i
    i = 1
    for epoch in range(1, args.max_epoch+1):
        # do training
        train_one_epoch(args, train_dataloader, optimizer, logger, epoch)
        # reduce learning rate
        scheduler.step()
        # validation
        test_one_epoch(args, test_dataloader, logger, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(check_dir, 'model.pth'))


def evaluate(args, dataloader, save_dir):

    model.eval()
    trans = SimuOcclusion()
    loss_summary = {}
    if args.is_pCompletion:
        loss_summary['Avg_chamfer_dist'] = 0
    if args.is_fidReg:
        loss_summary['Avg_fidelity'] = 0
    if args.is_classifier:
        loss_summary['Avg_acc'] = 0

    for j, data in enumerate(dataloader, 0):
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y
        # pos_observed, batch_observed = trans(pos, batch, args.num_pts_observed)
        if args.is_simuOcc:
            pos_observed, batch_observed = trans(pos, batch, args.num_pts_observed)
        else:
            pos_observed, batch_observed = pos, batch

        with torch.no_grad():
            generated_pc, fidelity, score = model(None, pos_observed, batch_observed)
            if args.is_pCompletion:
                random_latent = model.module.contrib_mean[0, 0, :].view(1, -1)
                generated_latent_pc = model.module.generate_pc_from_latent(random_latent)
                contribution_pc = model.module.contrib_pc

        if args.is_pCompletion:
            loss_summary['Avg_chamfer_dist'] += chamfer_loss(generated_pc, pos.view(-1, args.num_pts, 3)).mean()
        if args.is_fidReg:
            loss_summary['Avg_fidelity'] += fidelity.mean()
        if args.is_classifier:
            pred = score.max(1)[1]
            loss_summary['Avg_acc'] += pred.eq(label).float().mean()

        if args.is_pCompletion:
            # save the first sample results for visualization
            # pos = pos.cpu().detach().numpy().reshape(-1, args.num_pts, 3)[0]
            pos = label.cpu().detach().numpy().reshape(-1, args.num_pts, 3)[0]
            pos_observed = pos_observed.cpu().detach().numpy().reshape(-1, args.num_pts_observed, 3)[0]
            contribution_pc = contribution_pc.cpu().detach().numpy()
            generated_pc = generated_pc.cpu().detach().numpy()[0]
            generated_latent_pc = generated_latent_pc.cpu().detach().numpy()

            np.save(os.path.join(save_dir, 'pos_{}'.format(j)), pos)
            np.save(os.path.join(save_dir, 'pos_observed_{}'.format(j)), pos_observed)
            np.save(os.path.join(save_dir, 'contribution_pc_{}'.format(j)), contribution_pc)
            np.save(os.path.join(save_dir, 'generated_pc_{}'.format(j)), generated_pc)
            np.save(os.path.join(save_dir, 'generated_latent_pc_{}'.format(j)), generated_latent_pc)

    for item in loss_summary:
        loss_summary[item] /= len(dataloader)
        print('{}: {:.5f}'.format(item, loss_summary[item]))
    print('{} point clouds are evaluated.'.format(len(dataloader.dataset)))

    if args.is_pCompletion:
        print('Sample results are saved to: {}'.format(save_dir))


def load_dataset(args):
    # load ShapeNet dataset
    if args.dataset == 'shapenet':
        pre_transform = T.NormalizeScale()
        if args.randRotY:
            transform = T.Compose([T.FixedPoints(args.num_pts), T.RandomRotate(180, axis=1)])
        else:
            transform =T.FixedPoints(args.num_pts)

        categories = args.categories.split(',')
        train_dataset = ShapeNet('../data_root/ShapeNet_normal', categories, split='trainval',
                                 include_normals=False, pre_transform=pre_transform, transform=transform)
        test_dataset = ShapeNet('../data_root/ShapeNet_normal', categories, split='test',
                                include_normals=False, pre_transform=pre_transform, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                      num_workers=8, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=True,
                                     num_workers=8, drop_last=True)

    # load ModelNet dataset
    if args.dataset == 'modelnet':
        pre_transform = T.NormalizeScale()
        if args.randRotY:
            transform = T.Compose([T.SamplePoints(args.num_pts), T.RandomRotate(180, axis=1)])
        else:
            transform = T.SamplePoints(args.num_pts)

        train_dataset = ModelNet('../data_root/ModelNet40', name='40', train=True,
                                 pre_transform=pre_transform, transform=transform)
        test_dataset = ModelNet('../data_root/ModelNet40', name='40', train=False,
                                 pre_transform=pre_transform, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                      num_workers=8, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=True,
                                     num_workers=8, drop_last=True)

    # load completion3D dataset
    if args.dataset == 'completion3D':
        pre_transform = T.NormalizeScale()
        if args.randRotY:
            transform = T.Compose([T.FixedPoints(args.num_pts), T.RandomRotate(180, axis=1)])
        else:
            transform =T.FixedPoints(args.num_pts)
        
        categories = args.categories.split(',')
        train_dataset = completion3D_class('../data_root/shapenet', categories, split='train',
                            include_normals=False, pre_transform=pre_transform, transform=transform)
        print('Finished train_dataset')
        test_dataset = completion3D_class('../data_root/shapenet', categories, split='test',
                            include_normals=False, pre_transform=pre_transform, transform=transform)
        print('Finished test_dataset')
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                      num_workers=8, drop_last=True)
        print('Finished train_dataloader')
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=True,
                                     num_workers=8, drop_last=True)
        print('Finished test_dataloader')

    return train_dataloader, test_dataloader


def backup(log_dir, parser):
    shutil.copyfile('../utils/main.py', os.path.join(log_dir, 'main.py'))
    shutil.copyfile('../utils/models.py', os.path.join(log_dir, 'models.py'))
    shutil.copyfile('../utils/model_utils.py', os.path.join(log_dir, 'model_utils.py'))

    file = open(os.path.join(log_dir, 'parameters.txt'), 'w')
    adict = vars(parser.parse_args())
    keys = list(adict.keys())
    keys.sort()
    for item in keys:
        file.write('{0}:{1}\n'.format(item, adict[item]))
        print('{0}:{1}'.format(item, adict[item]))
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='model',
                        help="model name")
    parser.add_argument("--dataset", type=str,
                        help="shapenet or modelnet")
    parser.add_argument("--is_pCompletion", action='store_true',
                        help="flag for doing point clouds completion.")
    parser.add_argument("--is_classifier", action='store_true',
                        help="flag for computing classification score this is only valid for ModelNet.")
    parser.add_argument("--is_fidReg", action='store_true',
                        help="flag for fidelity regularization during training")
    parser.add_argument("--is_vote", action='store_true',
                        help="flag for computing latent feature by voting, otherwise max pooling")
    parser.add_argument("--categories", default='Chair',
                        help="point clouds categories in ShapeNet, string or [string]. Airplane, Bag, \
                        Cap, Car, Chair, Earphone, Guitar, Knife, Lamp, Laptop, Motorbike, Mug, Pistol, \
                        Rocket, Skateboard, Table")
    parser.add_argument("--num_pts", type=int,
                        help="the number of input points")
    parser.add_argument("--num_pts_observed", type=int,
                        help="the number of points in observed point clouds")
    parser.add_argument("--bsize", type=int, default=8,
                        help="batch size")
    parser.add_argument("--max_epoch", type=int, default=250,
                        help="max epoch to train")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="batch size")
    parser.add_argument("--step_size", type=int, default=300,
                        help="step size to reduce lr")
    parser.add_argument("--radius", type=float,
                        help="radius for generating sub point clouds")
    parser.add_argument("--bottleneck", type=int,
                        help="the size of bottleneck")
    parser.add_argument("--num_subpc_train", type=int,
                        help="the number of sub point clouds sampled during training")
    parser.add_argument("--num_subpc_test", type=int,
                        help="the number of sub point clouds sampled during test")
    parser.add_argument("--num_contrib_feats_train", type=int,
                        help="the number of contribution features during training")
    parser.add_argument("--num_contrib_feats_test", type=int,
                        help="the number of contribution features during test")
    parser.add_argument("--weight_chamfer", type=float, default=1.0,
                        help="weight for chamfer distance")
    parser.add_argument("--weight_fidelity", type=float, default=0.1,
                        help="weight for fidelity regularization")
    parser.add_argument("--weight_cls", type=float, default=1.0,
                        help="weight for classification loss")
    parser.add_argument("--is_simuOcc", action='store_true',
                        help="flag for simulating partial point clouds during test.")
    parser.add_argument("--randRotY", action='store_true',
                        help="flag for random rotation along Y axis")
    parser.add_argument("--eval", action='store_true',
                        help="flag for doing evaluation")
    parser.add_argument("--checkpoint", type=str,
                        help="directory which contains pretrained model (.pth)")

    args = parser.parse_args()
    assert args.is_pCompletion or args.is_classifier
    assert args.dataset in ['shapenet', 'modelnet', 'completion3D']

    train_dataloader, test_dataloader = load_dataset(args)
    # # only support classification on modelnet
    # args.is_classifier = args.is_classifier and args.dataset=='modelnet'

    model = Model(
        radius=args.radius,
        bottleneck=args.bottleneck,
        num_pts=args.num_pts,
        num_pts_observed=args.num_pts_observed,
        num_subpc_train=args.num_subpc_train,
        num_subpc_test=args.num_subpc_test,
        num_contrib_feats_train=args.num_contrib_feats_train,
        num_contrib_feats_test=args.num_contrib_feats_test,
        is_vote=args.is_vote,
        is_pCompletion=args.is_pCompletion,
        is_fidReg=args.is_fidReg,
        is_classifier=args.is_classifier
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

    # evaluation
    if args.eval:
        model_path = os.path.join(args.checkpoint, 'model.pth')
        if not os.path.isfile(model_path):
            raise ValueError('Please provide a valid path for pretrained model!'.format(model_path))
        model.load_state_dict(torch.load(model_path))
        print('Load model successfully from: {}'.format(args.checkpoint))

        save_dir = os.path.join(args.checkpoint, 'eval_sample_results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        evaluate(args=args, dataloader=test_dataloader, save_dir=save_dir)

    # training
    else:
        train(args=args, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
        print('Training is done: {}'.format(args.model_name))