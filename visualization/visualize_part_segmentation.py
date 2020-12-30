import numpy as np
import argparse
import open3d as o3d


color_map = 1 / 255.0 * np.array(
    [[47, 40, 49],
    [231, 29, 54],
    [26, 154, 140],
    [243, 98, 54],
    [231, 213, 76],
    [231, 213, 76]]
)


def visualize_point_cloud(points, cid):
    '''
    points: (N, 3)
    cid: (N), color id.
    '''
    colors = np.zeros(points.shape)
    for i in range(len(cid)):
        color = color_map[cid[i]]
        colors[i, :] = color

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="model name")
parser.add_argument("--idx", help="index for samples, like pos_{idx}.npy")
args = parser.parse_args()

base_dir = '../shapenet_seg/checkpoint/{}/eval_sample_results/'.format(args.model_name)

# load pos
filename = '{}/pos_{}.npy'.format(base_dir, args.idx)
pos = np.load(filename)
pos = pos.reshape(-1, 3)

# load pos_observed
filename = '{}/pos_observed_{}.npy'.format(base_dir, args.idx)
pos_observed = np.load(filename)
pos_observed = pos_observed.reshape(-1, 3)
pos_observed[:, 2] -= 2

# load pred
filename = '{}/pred_{}.npy'.format(base_dir, args.idx)
segp = np.load(filename).astype(np.int64)
segp = segp - segp.min()

# load label
filename = '{}/label_{}.npy'.format(base_dir, args.idx)
segl = np.load(filename).astype(np.int64)
segl = segl - segl.min()

pts_pred = visualize_point_cloud(pos, segl)
pts_gt = visualize_point_cloud(pos_observed, segp)
o3d.visualization.draw_geometries([pts_pred, pts_gt])
