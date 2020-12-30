import torch
import numpy as np
import os
import time
import shutil
import argparse
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.models import Model
from utils.model_utils import augment_transforms, get_lr, simulate_partial_point_clouds
from utils.class_completion3D import completion3D_class
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch_geometric.datasets import ShapeNet, ModelNet
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


def train_one_epoch(args, loader, optimizer, logger, epoch):
    '''
    Note: only complete point clouds are loaded during training, so data.x is
    both the input and label for the point cloud completion task. While partial
    point clouds (data.y) are loaded at testing.
    '''

    model.train()
    loss_summary = {}
    global i

    for j, data in enumerate(loader, 0):
        data = data.to(device)
        pos, batch = data.pos, data.batch
        label = pos if args.task == 'completion' else data.y
        category = data.category if args.task == 'segmentation' else None

        # training
        model.zero_grad()
        pred, loss = model(None, pos, batch, category, label)
        loss = loss.mean()

        if args.task == 'completion':
            loss_summary['loss_chamfer'] = loss
        elif args.task == 'classification':
            loss_summary['loss_cls'] = loss
        elif args.task == 'segmentation':
            loss_summary['loss_seg'] = loss

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
    results = []
    intersections, unions, categories = [], [], []

    for j, data in enumerate(loader, 0):
        data = data.to(device)
        pos, batch, label = data.pos, data.batch, data.y
        category = data.category if args.task == 'segmentation' else None

        if args.is_simuOcc:
            data_observed = simulate_partial_point_clouds(data, args.num_pts_observed, args.task)
            pos_observed, batch_observed, label_observed = data_observed.pos, data_observed.batch, data_observed.y
        else:
            pos_observed, batch_observed, label_observed = pos, batch, label

        # inference
        with torch.no_grad():
            pred, loss = model(None, pos_observed, batch_observed, category, label_observed)

        if args.task == 'completion':
            results.append(loss)

        elif args.task == 'classification':
            pred = pred.max(1)[1]
            results.append(pred.eq(label).float())

        elif args.task == 'segmentation':
            pred = pred.max(1)[1]
            i, u = i_and_u(pred, label_observed, loader.dataset.num_classes, batch_observed)
            intersections.append(i.to(torch.device('cpu')))
            unions.append(u.to(torch.device('cpu')))
            categories.append(category.to(torch.device('cpu')))

    if args.task == 'completion':
        results = torch.cat(results, dim=0).mean().item()
        logger.add_scalar('test_chamfer_dist', results, epoch)
        print('Epoch: {:03d}, Test Chamfer: {:.4f}'.format(epoch, results))
        results = -results

    elif args.task == 'classification':
        results = torch.cat(results, dim=0).mean().item()
        logger.add_scalar('test_acc', results, epoch)
        print('Epoch: {:03d}, Test Acc: {:.4f}'.format(epoch, results))

    elif args.task == 'segmentation':
        category = torch.cat(categories, dim=0)
        intersection = torch.cat(intersections, dim=0)
        union = torch.cat(unions, dim=0)
        ious = [[] for _ in range(len(loader.dataset.categories))]
        for j in range(category.size(0)):
            i = intersection[j, loader.dataset.y_mask[category[j]]]
            u = union[j, loader.dataset.y_mask[category[j]]]
            iou = i.to(torch.float) / u.to(torch.float)
            iou[torch.isnan(iou)] = 1
            ious[category[j]].append(iou.mean().item())

        for cat in range(len(loader.dataset.categories)):
            ious[cat] = torch.tensor(ious[cat]).mean().item()
        results = torch.tensor(ious).mean().item()
        logger.add_scalar('test_mIoU', results, epoch)
        print('Epoch: {:03d}, Test mIoU: {:.4f}'.format(epoch, results))

    return results


def train(args, train_dataloader, test_dataloader):

    check_dir, log_dir = check_overwrite(args.model_name)
    logger = SummaryWriter(log_dir=log_dir)
    backup(log_dir, parser)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.2)

    global i
    i = 1
    acc = -10000
    for epoch in range(1, args.max_epoch+1):
        # do training
        # with torch.autograd.detect_anomaly():
        train_one_epoch(args, train_dataloader, optimizer, logger, epoch)
        # reduce learning rate
        scheduler.step()
        # validation
        acc_ = test_one_epoch(args, test_dataloader, logger, epoch)
        if acc_ > acc:
            # save model
            torch.save(model.state_dict(), os.path.join(check_dir, 'model.pth'))
            acc = acc_


def evaluate(args, loader, save_dir):

    model.eval()
    results = []
    intersections, unions, categories = [], [], []
    if args.task == 'completion':
        categories_summary = {k:[] for k in loader.dataset.idx2cat.keys()}
        idx2cat = loader.dataset.idx2cat

    for _ in range(1):
        for j, data in enumerate(loader, 0):
            data = data.to(device)
            pos, batch, label = data.pos, data.batch, data.y
            try:
                category = data.category
            except AttributeError:
                category = None
    
            if args.is_simuOcc:
                data_observed = simulate_partial_point_clouds(data, args.num_pts_observed, args.task)
                pos_observed, batch_observed, label_observed = data_observed.pos, data_observed.batch, data_observed.y
            else:
                pos_observed, batch_observed, label_observed = pos, batch, label
    
            with torch.no_grad():
                pred, loss = model(None, pos_observed, batch_observed, category, label_observed)
                if args.task == 'completion':
                    # sampling in the latent space to generate diverse prediction
                    latent = model.module.optimal_z[0, :].view(1, -1)
                    idx = np.random.choice(args.num_vote_test, 1, False)
                    random_latent = model.module.contrib_mean[0, idx, :].view(1, -1)
                    random_latent = (random_latent + latent) / 2
                    pred_diverse = model.module.generate_pc_from_latent(random_latent)
    
    
            if args.task == 'classification':
                pred = pred.max(1)[1]
                results.append(pred.eq(label).float())
    
            elif args.task == 'segmentation':
                pred = pred.max(1)[1]
                i, u = i_and_u(pred, label_observed, loader.dataset.num_classes, batch_observed)
                intersections.append(i.to(torch.device('cpu')))
                unions.append(u.to(torch.device('cpu')))
                categories.append(category.to(torch.device('cpu')))
    
                if args.save:
                    pos = pos.cpu().detach().numpy().reshape(-1, args.num_pts, 3)[0]
                    pos_observed = pos_observed.cpu().detach().numpy().reshape(-1, args.num_pts_observed, 3)[0]
                    pred = pred.cpu().detach().numpy().reshape(-1, args.num_pts_observed)[0]
                    label = label.cpu().detach().numpy().reshape(-1, args.num_pts)[0]
                    np.save(os.path.join(save_dir, 'pos_{}'.format(j)), pos)
                    np.save(os.path.join(save_dir, 'pos_observed_{}'.format(j)), pos_observed)
                    np.save(os.path.join(save_dir, 'pred_{}'.format(j)), pred)
                    np.save(os.path.join(save_dir, 'label_{}'.format(j)), label)

            elif args.task == 'completion':
                results.append(loss)
                categories.append(category.to(torch.device('cpu')))

                if args.save:
                    pos = label.cpu().detach().numpy().reshape(-1, args.num_pts, 3)[0]
                    pos_observed = pos_observed.cpu().detach().numpy().reshape(-1, args.num_pts_observed, 3)[0]
                    pred = pred.cpu().detach().numpy()[0]
                    pred_diverse = pred_diverse.cpu().detach().numpy()[0]
                    np.save(os.path.join(save_dir, 'pos_{}'.format(j)), pos)
                    np.save(os.path.join(save_dir, 'pos_observed_{}'.format(j)), pos_observed)
                    np.save(os.path.join(save_dir, 'pred_{}'.format(j)), pred)
                    np.save(os.path.join(save_dir, 'pred_diverse_{}'.format(j)), pred_diverse)
        
    if args.task == 'completion':
        results = torch.cat(results, dim=0)
        category = torch.cat(categories, dim=0)
        for i in range(category.size(0)):
            categories_summary[category[i].item()].append(results[i])
        total_chamfer_distance = 0
        for idx in categories_summary:
            chamfer_distance_cat = torch.stack(categories_summary[idx], dim=0).mean().item()
            total_chamfer_distance += chamfer_distance_cat
            print('{}: {:.7f}'.format(idx2cat[idx], chamfer_distance_cat))
        print('Mean Class Chamfer Distance: {:.6f}'.format(total_chamfer_distance/len(categories_summary)))
        # results = torch.cat(results, dim=0).mean().item()
        # print('Test Chamfer: {:.4f}'.format(results))

    elif args.task == 'classification':
        results = torch.cat(results, dim=0).mean().item()
        print('Test Acc: {:.4f}'.format(results))

    elif args.task == 'segmentation':
        category = torch.cat(categories, dim=0)
        intersection = torch.cat(intersections, dim=0)
        union = torch.cat(unions, dim=0)
        ious = [[] for _ in range(len(loader.dataset.categories))]
        for j in range(category.size(0)):
            i = intersection[j, loader.dataset.y_mask[category[j]]]
            u = union[j, loader.dataset.y_mask[category[j]]]
            iou = i.to(torch.float) / u.to(torch.float)
            iou[torch.isnan(iou)] = 1
            ious[category[j]].append(iou.mean().item())

        for cat in range(len(loader.dataset.categories)):
            ious[cat] = torch.tensor(ious[cat]).mean().item()
        miou = torch.tensor(ious).mean().item()
        print('Test mIoU: {:.4f}'.format(miou))

    print('{} point clouds are evaluated.'.format(len(loader.dataset)))

    if args.task == 'completion' or args.task == 'segmentation':
        print('Sample results are saved to: {}'.format(save_dir))


def load_dataset(args):

    # load ShapeNet dataset
    if args.task == 'segmentation':
        pre_transform, transform = augment_transforms(args)

        categories = args.categories.split(',')
        train_dataset = ShapeNet('../data_root/ShapeNet_normal', categories, split='trainval', include_normals=False,
                                 pre_transform=pre_transform, transform=transform)
        test_dataset = ShapeNet('../data_root/ShapeNet_normal', categories, split='test', include_normals=False,
                                pre_transform=pre_transform, transform=T.FixedPoints(args.num_pts))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                      num_workers=6, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=False,
                                     num_workers=6, drop_last=True)


    # load ModelNet dataset
    elif args.task == 'classification':
        pre_transform, transform = augment_transforms(args)

        train_dataset = ModelNet('../data_root/ModelNet40', name='40', train=True,
                                 pre_transform=pre_transform, transform=transform)
        test_dataset = ModelNet('../data_root/ModelNet40', name='40', train=False,
                                 pre_transform=pre_transform, transform=T.SamplePoints(args.num_pts))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                      num_workers=6, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=False,
                                     num_workers=6, drop_last=True)

    # load completion3D dataset
    elif args.task == 'completion':
        pre_transform, transform = augment_transforms(args)

        categories = args.categories.split(',')
        train_dataset = completion3D_class('../data_root/completion3D', categories, split='train',
                            include_normals=False, pre_transform=pre_transform, transform=transform)
        test_dataset = completion3D_class('../data_root/completion3D', categories, split='val',
                            include_normals=False, pre_transform=pre_transform, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                      num_workers=6, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=False,
                                     num_workers=6, drop_last=True)

    return train_dataloader, test_dataloader


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

        # waiting for updating of tensorboard
        time.sleep(5)

    # create directory
    os.makedirs(check_dir)
    os.makedirs(log_dir)
    return check_dir, log_dir


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
    print()
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='model',
                        help="model name")
    parser.add_argument("--task", type=str, choices=['completion', 'classification', 'segmentation'],
                        help=' '.join([
                            'completion: point cloud completion on Completion3D',
                            'classification: shape classification on ModelNet40',
                            'segmentation: part segmentation on ShapeNet'
                        ]))
    parser.add_argument("--is_vote", action='store_true',
                        help="flag for computing latent feature by voting, otherwise max pooling")
    parser.add_argument("--categories", default='Chair',
                        help="point clouds categories, string or [string]. For ShapeNet: Airplane, Bag, \
                        Cap, Car, Chair, Earphone, Guitar, Knife, Lamp, Laptop, Motorbike, Mug, Pistol, \
                        Rocket, Skateboard, Table; For Completion3D: plane,cabinet,car,chair,lamp,couch,table,watercraft")
    parser.add_argument("--num_pts", type=int,
                        help="the number of input points")
    parser.add_argument("--num_pts_observed", type=int,
                        help="the number of points in observed point clouds")
    parser.add_argument("--bsize", type=int, default=32,
                        help="batch size")
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="max epoch to train")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="batch size")
    parser.add_argument("--step_size", type=int, default=200,
                        help="step size to reduce lr")
    parser.add_argument("--radius", type=float,
                        help="radius for generating sub point clouds")
    parser.add_argument("--bottleneck", type=int,
                        help="the size of bottleneck")
    parser.add_argument("--num_vote_train", type=int, default=64,
                        help="the number of votes (sub point clouds) during training")
    parser.add_argument("--num_contrib_vote_train", type=int, default=10,
                        help="the maximum number of contribution votes during training")
    parser.add_argument("--num_vote_test", type=int,
                        help="the number of votes (sub point clouds) during test")
    parser.add_argument("--is_simuOcc", action='store_true',
                        help="flag for simulating partial point clouds during test.")
    parser.add_argument("--norm", type=str, choices=['scale', 'bbox', 'sphere', 'sphere_wo_center'],
                        help="flag for normalization")
    parser.add_argument("--eval", action='store_true',
                        help="flag for doing evaluation")
    parser.add_argument("--save", action='store_true',
                        help="flag for writing prediction results")
    parser.add_argument("--checkpoint", type=str,
                        help="directory which contains pretrained model (.pth)")

    args = parser.parse_args()
    assert args.task in ['completion', 'classification', 'segmentation']

    # construct data loader
    train_dataloader, test_dataloader = load_dataset(args)

    model = Model(
        radius=args.radius,
        bottleneck=args.bottleneck,
        num_pts=args.num_pts,
        num_pts_observed=args.num_pts_observed,
        num_vote_train=args.num_vote_train,
        num_contrib_vote_train=args.num_contrib_vote_train,
        num_vote_test=args.num_vote_test,
        is_vote=args.is_vote,
        task=args.task
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

    # evaluation
    if args.eval:
        model_path = os.path.join(args.checkpoint)
        if not os.path.isfile(model_path):
            raise ValueError('{} does not exist. Please provide a valid path for pretrained model!'.format(model_path))
        model.load_state_dict(torch.load(model_path))
        print('Successfully load model from: {}'.format(args.checkpoint))

        path, _ = os.path.split(args.checkpoint)
        save_dir = os.path.join(path, 'eval_sample_results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        evaluate(args=args, loader=test_dataloader, save_dir=save_dir)

    # training
    else:
        train(args=args, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
        print('Training is done: {}'.format(args.model_name))
