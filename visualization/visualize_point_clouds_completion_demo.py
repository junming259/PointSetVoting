import argparse
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def rotate(pos, deg, axis):
    degree = np.pi * deg / 180.0
    sin, cos = np.sin(degree), np.cos(degree)

    if axis == 0:
        matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
    elif axis == 1:
        matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
    else:
        matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

    return pos.dot(matrix)

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
    vis.get_render_option().point_size = 8
    #vis.get_render_option().show_coordinate_frame = True

    vis.run()
    vis.destroy_window()


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="model name")
parser.add_argument("--idx", help="index for samples, like pos_{idx}.npy")
args = parser.parse_args()

#idx = '13'
#model_name = 'cpc3d_b64e500s200lr2e-4_r010tr64-16_te32_bn1024'
base_dir = '../demo/pretrained/{}/eval_sample_results'.format(args.model_name)

# load observed partial points
filename = '{}/pos_observed_{}.npy'.format(base_dir, args.idx)
points0 = np.load(filename).reshape(-1, 3)

# load pred
filename = '{}/pred_{}.npy'.format(base_dir, args.idx)
points1 = np.load(filename).reshape(-1, 3)
points1[:, 2] -= 2

# # load complete point clouds
# filename = '{}/pos_{}.npy'.format(base_dir, args.idx)
# points2 = np.load(filename).reshape(-1, 3)
# points2[:, 2] -= 4

# load pred_diverse
filename = '{}/pred_diverse_{}.npy'.format(base_dir, args.idx)
points3 = np.load(filename).reshape(-1, 3)
points3[:, 2] -= 4


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points0)
pcd.colors = o3d.utility.Vector3dVector([[1, 0, 1] for i in range(len(points0))])
o3d.io.write_point_cloud("points0.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points1)
pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(points1))])
o3d.io.write_point_cloud("points1.ply", pcd)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points2)
# pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(points2))])
# o3d.io.write_point_cloud("points2.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3)
pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(points3))])
o3d.io.write_point_cloud("points2.ply", pcd)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points4)
# pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(points4))])
# o3d.io.write_point_cloud("points4.ply", pcd)


# Load saved point cloud and visualize it
pcd0_load = o3d.io.read_point_cloud("points0.ply")
pcd1_load = o3d.io.read_point_cloud("points1.ply")
# pcd2_load = o3d.io.read_point_cloud("points2.ply")
pcd3_load = o3d.io.read_point_cloud("points2.ply")
# o3d.visualization.draw_geometries([pcd_load])
custom_draw_geometry([pcd0_load, pcd1_load, pcd3_load], title=filename)
