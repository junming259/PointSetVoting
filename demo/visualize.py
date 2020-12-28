import os
import glob
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
parser.add_argument('--data_path', type=str, default='demo_inputs/',
                    help='Specify the path of input partial point clouds. \
                        if path is a diretory, a random .npy will be choosen to \
                        visualize.')
args = parser.parse_args()


data_path = args.data_path
if os.path.isdir(args.data_path):
    data_file_list = glob.glob(os.path.join(args.data_path, '*.npy'))
    idx = np.random.choice(len(data_file_list))
    data_path = data_file_list[idx]

_, tail = os.path.split(data_path)
pred_path = os.path.join('demo_results', 'pred_{}'.format(tail))

pts = np.load(data_path).reshape(-1, 3)
pred = np.load(pred_path).reshape(-1, 3)
input_set = visualize_point_cloud(pts, color='b')
pred_set = visualize_point_cloud(pred, color='g')
o3d.visualization.draw_geometries([pred_set, input_set])
