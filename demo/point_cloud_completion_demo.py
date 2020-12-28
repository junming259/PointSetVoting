import torch
import glob
import os
import argparse
import numpy as np
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.models import Model

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


def load_data(data_path):
    def resample(x):
        idx = np.random.choice(x.size(0), 2048, True)
        x = x[idx]
        return x

    def load_single_data(path):
        if not path.endswith('.npy'):
            raise NotImplementedError

        points = np.load(path).reshape(-1, 3)
        points = torch.tensor(points, dtype=torch.float32)
        points = resample(points)
        batch = torch.zeros(points.size(-2)).long()
        _, filename = os.path.split(path)
        return (points, batch, filename)

    if os.path.isdir(data_path):
        data_file_list = glob.glob(os.path.join(data_path, '*.npy'))
    else:
        data_file_list = [data_path]

    data = [load_single_data(item) for item in data_file_list]

    return data 


def main():

    args = parse_args()

    # build model
    model = Model(
        radius=0.1,
        bottleneck=1024,
        num_pts=2048,
        num_pts_observed=2048,
        num_vote_train=64,
        num_contrib_vote_train=10,
        num_vote_test=128,
        is_vote=True,
        task='completion'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

    # load model
    model_path = args.checkpoint
    if not os.path.isfile(model_path):
        raise ValueError('{} does not exist. Please provide a valid path for pretrained model!'.format(model_path))
    model.load_state_dict(torch.load(model_path))
    print('Load model successfully from: {}'.format(args.checkpoint))

    save_dir = 'demo_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = load_data(args.data_path)

    model.eval()
    with torch.no_grad():
        for data in dataset:
            pos, batch, filename = data
            pos, batch = pos.to(device), batch.to(device)

            pred, _ = model(None, pos, batch)

            pred = pred.cpu().detach().numpy()[0]
            np.save(os.path.join(save_dir, 'pred_{}'.format(filename)), pred)

    print('Done.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help="path of folder containing the input PCs")
    parser.add_argument("--checkpoint", type=str,
                        help="path of pretrained model (.pth)")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
