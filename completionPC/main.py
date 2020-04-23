import torch
import numpy as np
import os
import argparse
import torch.nn.functional as F
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.logger import Logger
from utils.models import Model
from utils.model_utils import SimuOcclusion, get_lr, chamfer_loss
from shutil import copyfile
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch_geometric.datasets import ShapeNet, ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


def train_one_epoch(loader, optimizer, logger, epoch):

    model.train()
    global i

    for j, data in enumerate(loader, 0):
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y

        # training
        model.zero_grad()
        generated_pc, fidelity, score = model(None, pos, batch)
        loss_chf = chamfer_loss(generated_pc, pos.view(-1, args.num_pts, 3))

        fidelity = fidelity.mean()
        loss = loss_chf + fidelity
        # TO DO, classification
        # if model.is_classifier:
        #     loss_cls = F.nll_loss(score, label)
        #     loss += 0.2*loss_cls
        loss.backward()
        optimizer.step()

        # write summary
        if i % 100 == 0:
            logger.scalar_summary('loss_chamfer', loss_chf, i)
            logger.scalar_summary('loss_fidelity', fidelity, i)
            logger.scalar_summary('lr', get_lr(optimizer), i)
            print('Train, chamfer_loss: {:.4f}, fidelity: {:.4f}'.format(loss_chf, fidelity))
            # if model.is_classifier:
            #     logger.scalar_summary('loss_cls', loss_cls, i)
            #     print('Train, chamfer_loss: {:.4f}, fidelity: {:.4f}, cls_loss: {:.4f}'.format(
            #         loss_chf, fidelity, loss_cls))
            # else:
            #     print('Train, chamfer_loss: {:.4f}, fidelity: {:.4f}'.format(loss_chf, fidelity))
        i = i + 1


def test_one_epoch(loader, logger, epoch, save_dir):

    model.eval()
    total_chamfer_loss = []
    total_fidelity = []
    acc = 0
    trans = SimuOcclusion()

    for j, data in enumerate(loader, 0):
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y
        pos_observed, batch_observed = trans(pos, batch, args.num_pts_observed)

        with torch.no_grad():
            generated_pc, fidelity, score = model(None, pos_observed, batch_observed)
        total_chamfer_loss.append(chamfer_loss(generated_pc, pos.view(-1, args.num_pts, 3)))
        total_fidelity.append(fidelity.mean())
        # if model.is_classifier:
        #     pred = score.max(1)[1]
        #     acc += pred.eq(label).sum().item()

    avg_chamfer_loss = sum(total_chamfer_loss) / len(total_chamfer_loss)
    avg_fidelity = sum(total_fidelity) / len(total_fidelity)
    logger.scalar_summary('test_chamfer_dist', avg_chamfer_loss, epoch)
    logger.scalar_summary('test_fidelity', avg_fidelity, epoch)
    print('Epoch: {:03d}, eval_chamfer_loss: {:.4f}, eval_fidelity: {:.5f}\n'.format(epoch, avg_chamfer_loss, avg_fidelity))
    # if model.is_classifier:
    #     acc = acc / len(loader.dataset)
    #     logger.scalar_summary('test_acc', acc, epoch)
    #     print('Epoch: {:03d}, eval_chamfer_loss: {:.4f}, acc: {:.4f}\n'.format(epoch, avg_chamfer_loss, acc))
    # else:
    #     print('Epoch: {:03d}, eval_chamfer_loss: {:.4f}\n'.format(epoch, avg_chamfer_loss))


def train(args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.2)

    model_name = args.model_name
    check_dir = 'checkpoint/{}'.format(model_name)
    if not os.path.exists('checkpoint/{}'.format(model_name)):
        os.makedirs('checkpoint/{}'.format(model_name))
    save_dir = os.path.join(check_dir, 'train_sample_results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_dir = './logs/{}'.format(model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    backup(log_dir, parser)

    global i
    i = 1
    for epoch in range(1, args.max_epoch+1):
        # do training
        train_one_epoch(train_dataloader, optimizer, logger, epoch)
        # reduce learning rate
        scheduler.step()
        # validation
        test_one_epoch(test_dataloader, logger, epoch, save_dir)
        # save model
        torch.save(model.state_dict(), 'checkpoint/{}/model.pth'.format(model_name))


def evaluate(args, save_dir):

    model.eval()
    total_chamfer_loss = []
    total_fidelity = []
    acc = 0
    trans = SimuOcclusion()

    for j, data in enumerate(test_dataloader, 0):
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y
        pos_observed, batch_observed = trans(pos, batch, args.num_pts_observed)

        with torch.no_grad():
            generated_pc, fidelity, score = model(None, pos_observed, batch_observed)
            random_latent = model.module.contribution_mean[0, 0, :].view(1, -1)
            generated_latent_pc = model.module.generate_pc_from_latent(random_latent)
            # generated_latent_pc = model.generate_pc_from_latent(model.contribution_mean[0, 0, :].view(1, -1))
            contribution_pc = model.module.contribution_pc
        total_chamfer_loss.append(chamfer_loss(generated_pc, pos.view(-1, args.num_pts, 3)))
        total_fidelity.append(fidelity.mean())

        # if model.is_classifier:
        #     pred = score.max(1)[1]
        #     acc += pred.eq(label).sum().item()

        # save the first sample results for visualization
        pos = pos.cpu().detach().numpy().reshape(-1, args.num_pts, 3)[0]
        pos_observed = pos_observed.cpu().detach().numpy().reshape(-1, args.num_pts_observed, 3)[0]
        contribution_pc = contribution_pc.cpu().detach().numpy()
        generated_pc = generated_pc.cpu().detach().numpy()[0]
        generated_latent_pc = generated_latent_pc.cpu().detach().numpy()

        np.save(os.path.join(save_dir, 'pos_{}'.format(j)), pos)
        np.save(os.path.join(save_dir, 'pos_observed_{}'.format(j)), pos_observed)
        np.save(os.path.join(save_dir, 'contribution_pc_{}'.format(j)), contribution_pc)
        np.save(os.path.join(save_dir, 'generated_pc_{}'.format(j)), generated_pc)
        np.save(os.path.join(save_dir, 'generated_latent_pc_{}'.format(j)), generated_latent_pc)

    avg_chamfer_loss = sum(total_chamfer_loss) / len(total_chamfer_loss)
    avg_fidelity = sum(total_fidelity) / len(total_fidelity)
    print('{} point clouds are evaluated.'.format(len(test_dataloader.dataset)))
    print('Avg_chamfer_loss: {:.5f}'.format(avg_chamfer_loss))
    print('Avg_fidelity: {:.5f}'.format(avg_fidelity))
    print('Sample results are saved to: {}'.format(save_dir))
    # if model.is_classifier:
    #     acc = acc / len(test_dataloader.dataset)
    #     print('Avg_accuracy: {:.4f}'.format(acc))


def load_dataset(args):
    assert args.dataset in ['ShapeNet', 'ModelNet']

    # load ShapeNet dataset
    if args.dataset == 'ShapeNet':
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
                                      num_workers=6, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=True,
                                     num_workers=6, drop_last=True)

    # load ModelNet dataset
    else:
        pre_transform = T.NormalizeScale()
        if args.randRotY:
            transform = T.Compose([T.SamplePoints(args.num_pts), T.RandomRotate(180, axis=1)])
        else:
            transform = T.SamplePoints(args.num_pts)

        train_dataset = ModelNet('../data_root/ModelNet10', name='10', train=True,
                                 pre_transform=pre_transform, transform=transform)
        test_dataset = ModelNet('../data_root/ModelNet10', name='10', train=False,
                                 pre_transform=pre_transform, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                      num_workers=8, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=True,
                                     num_workers=8)

    return train_dataloader, test_dataloader


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='model',
                        help="model name")
    parser.add_argument("--dataset", type=str,
                        help="ShapeNet or ModelNet")
    parser.add_argument("--is_classifier", action='store_true',
                        help="flag for computing classification score this is only valid for ModelNet.")
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
    parser.add_argument("--num_contri_feats_train", type=int,
                        help="the number of contribution features during training")
    parser.add_argument("--num_contri_feats_test", type=int,
                        help="the number of contribution features during test")
    parser.add_argument("--is_fidReg", action='store_true',
                        help="flag for fidelity regularization during training")
    parser.add_argument("--randRotY", action='store_true',
                        help="flag for random rotation along Y axis")
    parser.add_argument("--eval", action='store_true',
                        help="flag for doing evaluation")
    parser.add_argument("--checkpoint", type=str,
                        help="directory which contains pretrained model (.pth)")

    args = parser.parse_args()
    train_dataloader, test_dataloader = load_dataset(args)
    # only support classification on modelnet
    is_classifier = args.is_classifier and args.dataset=='ModelNet'

    model = Model(
        radius=args.radius,
        bottleneck=args.bottleneck,
        num_pts=args.num_pts,
        num_subpc_train=args.num_subpc_train,
        num_pts_observed=args.num_pts_observed,
        num_subpc_test=args.num_subpc_test,
        num_contri_feats_train=args.num_contri_feats_train,
        num_contri_feats_test=args.num_contri_feats_test,
        is_fidReg=args.is_fidReg,
        is_classifier=is_classifier
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
        evaluate(args=args, save_dir=save_dir)

    # training
    else:
        train(args=args)
        print('Training is done: {}'.format(args.model_name))
