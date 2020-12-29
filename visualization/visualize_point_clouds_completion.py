import argparse
import numpy as np
import open3d as o3d


def visualize_point_cloud(points, color='r'):
    '''
    points: (N, 3)
    color: string, ['r', 'g']
    '''
    colors = np.zeros(points.shape, dtype=np.int32)
    if color=='r':
        colors[:, 0] = 1    # red
    elif color=='g':
        colors[:, 1] = 1    # green
    elif color=='b':
        colors[:, 2] = 1    # blue
    elif color=='p':
        colors[:, 2] = 1    # pink
        colors[:, 0] = 1    # pink

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def custom_draw_geometry(inputs, title=None):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    if title:
        vis.create_window(window_name=title)
    else:
        vis.create_window()

    for item in inputs:
        vis.add_geometry(item)
    vis.get_render_option().point_size = 10
    #vis.get_render_option().show_coordinate_frame = True

    vis.run()
    vis.destroy_window()


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="model name")
parser.add_argument("--idx", help="index for samples, like pos_{idx}.npy")
args = parser.parse_args()

base_dir = '../completion3D/checkpoint/{}/eval_sample_results'.format(args.model_name)

# load observed partial points
filename = '{}/pos_observed_{}.npy'.format(base_dir, args.idx)
points = np.load(filename).reshape(-1, 3)
pts_observed = visualize_point_cloud(points, color='p')

# load pred
filename = '{}/pred_{}.npy'.format(base_dir, args.idx)
points = np.load(filename).reshape(-1, 3)
points[:, 2] -= 2
pts_pred = visualize_point_cloud(points, color='r')

# load complete point clouds
filename = '{}/pos_{}.npy'.format(base_dir, args.idx)
points = np.load(filename).reshape(-1, 3)
points[:, 2] -= 4
pts_gt = visualize_point_cloud(points, color='g')

# load pred_diverse
filename = '{}/pred_diverse_{}.npy'.format(base_dir, args.idx)
points = np.load(filename).reshape(-1, 3)
points[:, 2] -= 6
pts_diverse = visualize_point_cloud(points, color='b')

o3d.visualization.draw_geometries([pts_observed, pts_pred, pts_gt, pts_diverse])

