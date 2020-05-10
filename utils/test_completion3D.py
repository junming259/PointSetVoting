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
from utils.class_shapenet_source import *

def load_dataset(args):
    # load completion3D dataset
    # TODO convert .h5 to .txt and use Completion3D_Dataset?
    # convert : https://gist.github.com/andrewfowlie/da173e2a476945a96039fb14e8b3a38a
    if args.dataset == 'completion3D':
        # pre_transform = T.NormalizeScale()
        # if args.randRotY:
        #     transform = T.Compose([T.FixedPoints(args.num_pts), T.RandomRotate(180, axis=1)])
        # else:
        #     transform =T.FixedPoints(args.num_pts)
        
        
        #set paths
        categories = args.categories.split(',')
        path_train_x = '../data_root/dataset2019/shapenet/train/partial/03001627'
        path_train_y = '../data_root/dataset2019/shapenet/train/gt/03001627'

        # create dataLoader objects
        x_train_dataset, y_train_dataset = load_h5(path_train_x, path_train_y, size=TRAIN_SIZE)
        train_dataloader = DataLoader(TensorDataset(x_train_dataset, y_train_dataset), 
                                      batch_size=bs, shuffle=True, num_workers=8, drop_last=True)

        train_dataloader = WrappedDataLoader(train_dataloader, preprocess)

        x_test_dataset, y_test_dataset
        test_dataloader
 
    # load ShapeNet dataset
    if args.dataset == 'shapenet':
        pre_transform = T.NormalizeScale()
        if args.randRotY:
            transform = T.Compose([T.FixedPoints(args.num_pts), T.RandomRotate(180, axis=1)])
        else:
            transform =T.FixedPoints(args.num_pts)

        categories = args.categories.split(',')
        train_dataset = ShapeNet_test('../data_root/ShapeNet_normal', categories, split='trainval',
                                 include_normals=False, pre_transform=pre_transform, transform=transform)
        test_dataset = ShapeNet_test('../data_root/ShapeNet_normal', categories, split='test',
                                include_normals=False, pre_transform=pre_transform, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                      num_workers=8, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=True,
                                     num_workers=8, drop_last=True)

        print(train_dataset)
        print(train_dataset.categories)
        print(train_dataset.processed_file_names)
        print(train_dataset.raw_file_names)
        print(len(train_dataset))
        print(train_dataset.__init__)
        print(train_dataset.__repr__)
        print(train_dataset.num_classes)
        print(train_dataset.num_node_features)
        print(train_dataset[0])
        # train_dataset.categories
        # train_dataset.processed_file_names
        # train_dataset.raw_file_names
    return train_dataloader, test_dataloader

def main():
    print("Hello World!")
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

    args = parser.parse_args()
    assert args.is_pCompletion or args.is_classifier
    assert args.dataset in ['shapenet', 'modelnet', 'completion3D']

    train_dataloader, test_dataloader = load_dataset(args)


if __name__ == "__main__":
    main()