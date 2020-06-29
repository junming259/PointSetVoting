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
from utils.model_utils import augment_transforms, get_lr, chamfer_loss, simulate_partial_point_clouds
from utils.class_completion3D_demo import completion3D_class
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch_geometric.datasets import ShapeNet, ModelNet
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius
from torch_geometric.utils import intersection_and_union as i_and_u

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

def evaluate(args, loader, save_dir):

    model.eval()
    results = []
    intersections, unions, categories = [], [], []
    # if args.task == 'completion':
    #     categories_summary = {k:[] for k in loader.dataset.idx2cat.keys()}
    #     idx2cat = loader.dataset.idx2cat

    for _ in range(10):
        for j, data in enumerate(loader, 0):
            data = data.to(device)
            pos, batch = data.pos, data.batch
            try:
                category = data.category
            except AttributeError:
                category = None
                
            if args.is_simuOcc:
                data_observed = simulate_partial_point_clouds(data, args.num_pts_observed, args.task)
                pos_observed, batch_observed = data_observed.pos, data_observed.batch
            else:
                pos_observed, batch_observed = pos, batch
    
            with torch.no_grad():
                pred = model(None, pos_observed, batch_observed, category)
                if args.task == 'completion':
                    # sampling in the latent space to generate diverse prediction
                    latent = model.module.optimal_z[0, :].view(1, -1)
                    idx = np.random.choice(args.num_vote_test, 1, False)
                    random_latent = model.module.contrib_mean[0, idx, :].view(1, -1)
                    random_latent = (random_latent + latent) / 2
                    pred_diverse = model.module.generate_pc_from_latent(random_latent)
    
                    # for i in range(16):
                    #     # latent = model.module.optimal_z[0, :].view(1, -1)
                    #     # idx = np.random.choice(args.num_vote_test, 1, False)
                    #     # random_latent = model.module.contrib_mean[0, idx, :].view(1, -1)
                    #     # random_latent = (random_latent + latent) / 2
                    #
                    #     random_latent = model.module.contrib_mean[0, i, :].view(1, -1)
                    #     pred_diverse_tmp = model.module.generate_pc_from_latent(random_latent)
                    #     pred_diverse_tmp = pred_diverse_tmp.cpu().detach().numpy()[0]
                    #     np.save(os.path.join(save_dir, 'pred_diverse_{}_{}'.format(j, i)), pred_diverse_tmp)

    
            if args.task == 'completion':
                # # save key points for visualization
                # key_pos = model.module.encoder.new_pos.view(args.bsize, -1, 3)
                # key_pos = key_pos.cpu().detach().numpy()[0]
                # np.save(os.path.join(save_dir, 'key_pos_{}'.format(j)), key_pos)
                # pos = label.cpu().detach().numpy().reshape(-1, args.num_pts, 3)[0]
    
                categories.append(category.to(torch.device('cpu')))
                pos_observed = pos_observed.cpu().detach().numpy().reshape(-1, args.num_pts_observed, 3)[0]
                pred = pred.cpu().detach().numpy()[0]
                pred_diverse = pred_diverse.cpu().detach().numpy()[0]
                # np.save(os.path.join(save_dir, 'pos_{}'.format(j)), pos)
                np.save(os.path.join(save_dir, 'pos_observed_{}'.format(j)), pos_observed)
                np.save(os.path.join(save_dir, 'pred_{}'.format(j)), pred)
                np.save(os.path.join(save_dir, 'pred_diverse_{}'.format(j)), pred_diverse)

    print('{} point clouds are evaluated.'.format(len(loader.dataset)))

    if args.task == 'completion' or args.task == 'segmentation':
        print('Sample results are saved to: {}'.format(save_dir))

def load_dataset(args):
    # load completion3D dataset
    if args.dataset == 'completion3D':
        pre_transform, transform = augment_transforms(args)
        # pre_transform = None
        # if args.randRotY:
        #     transform = T.Compose([T.FixedPoints(args.num_pts), T.RandomRotate(180, axis=1)])
        # else:
        #     transform =T.FixedPoints(args.num_pts)
        
        categories = args.categories.split(',')

        test_dataset = completion3D_class('./partial_point_cloud_demo', categories, split='test',
                            include_normals=False, pre_transform=pre_transform, transform=transform)
        print('Finished test_dataset')
        test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=False,
                                     num_workers=1, drop_last=True)
        print('Finished test_dataloader')

    return test_dataloader

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
    parser.add_argument("--dataset", type=str, choices=['shapenet', 'modelnet', 'completion3D', 'scanobjectnn'],
                        help="shapenet or modelnet or completion3D")
    parser.add_argument("--task", type=str, choices=['completion', 'classification', 'segmentation'],
                        help=' '.join([
                            'completion: point clouds completion',
                            'classification: shape classification on ModelNet40',
                            'segmentation: part segmentation on ShapeNet'
                        ]))
    parser.add_argument("--is_vote", action='store_true',
                        help="flag for computing latent feature by voting, otherwise max pooling")
    parser.add_argument("--categories", default='Chair',
                        help="point clouds categories, string or [string]. For ShapeNet: Airplane, Bag, \
                        Cap, Car, Chair, Earphone, Guitar, Knife, Lamp, Laptop, Motorbike, Mug, Pistol, \
                        Rocket, Skateboard, Table; For Completion3D: plane;cabinet;car;chair;lamp;couch;table;watercraft")
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
    parser.add_argument("--norm", type=str, choices=['scale', 'sphere', 'sphere_wo_center'],
                        help="flag for normalization")
    parser.add_argument("--is_randRotY", action='store_true',
                        help="flag for random rotation along Y axis")
    parser.add_argument("--eval", action='store_true',
                        help="flag for doing evaluation")
    parser.add_argument("--checkpoint", type=str,
                        help="directory which contains pretrained model (.pth)")

    args = parser.parse_args()
    assert args.dataset in ['shapenet', 'modelnet', 'completion3D']
    assert args.task in ['completion', 'classification', 'segmentation']

    # construct data loader
    test_dataloader = load_dataset(args)

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
        model_path = os.path.join(args.checkpoint, 'model.pth')
        if not os.path.isfile(model_path):
            raise ValueError('{} does not exist. Please provide a valid path for pretrained model!'.format(model_path))
        model.load_state_dict(torch.load(model_path))
        print('Load model successfully from: {}'.format(args.checkpoint))

        save_dir = os.path.join(args.checkpoint, 'eval_sample_results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        evaluate(args=args, loader=test_dataloader, save_dir=save_dir)

    # training
    else:
        train(args=args, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
        print('Training is done: {}'.format(args.model_name))
