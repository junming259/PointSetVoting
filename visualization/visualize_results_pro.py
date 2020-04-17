from glob import glob
import numpy as np
import open3d as o3d


idx = '599'
folder = 'cpc_b8e600s250lr2e-4_r02sub16-16_chair_randHalf'


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
    # vis.get_render_option().show_coordinate_frame = True
    vis.run()
    vis.destroy_window()


# generated_latent_pc
filename = '../completionPC/checkpoint/{}/generated_latent_pc_{}.npy'.format(folder, idx)
points0 = np.load(filename)
points0 = points0.reshape(-1, 3)


# generated_pc
filename = '../completionPC/checkpoint/{}/generated_pc_{}.npy'.format(folder, idx)
points1 = np.load(filename)
points1 = points1.reshape(-1, 3)
points1_ = points1.copy()
points1[:, 2] += 2
points1_[:, 2] += 8
points1 = np.concatenate([points1, points1_], 0)
print('Number of generated points: {}'.format(points1_.shape[0]))


# complete point clouds
filename = '../completionPC/checkpoint/{}/pos_{}.npy'.format(folder, idx)
points2 = np.load(filename)
points2 = points2.reshape(-1, 3)
points2_1 = points2.copy()
points2_2 = points2.copy()
points2[:, 2] += 4
points2_1[:, 2] += 6
points2_2[:, 2] += 8
points2 = np.concatenate([points2, points2_1, points2_2], 0)
print('Number of full points: {}'.format(points2_1.shape[0]))


# observed points
filename = '../completionPC/checkpoint/{}/pos_observed_{}.npy'.format(folder, idx)
points3 = np.load(filename)
points3 = points3.reshape(-1, 3) + 0.01
points3[:, 2] += 4
print('Number of observed points: {}'.format(np.unique(points3, axis=0).shape[0]))


# contribution points
filename = '../completionPC/checkpoint/{}/contribution_pc_{}.npy'.format(folder, idx)
points4 = np.load(filename)
points4 = points4.reshape(-1, 3) + 0.01
points4[:, 2] += 6
print('Number of contribution points: {}'.format(np.unique(points4, axis=0).shape[0]))


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points0)
pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(points0))])
o3d.io.write_point_cloud("points0.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points1)
pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(points1))])
o3d.io.write_point_cloud("points1.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points2)
pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(points2))])
o3d.io.write_point_cloud("points2.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3)
pcd.colors = o3d.utility.Vector3dVector([[1, 0, 1] for i in range(len(points3))])
o3d.io.write_point_cloud("points3.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points4)
pcd.colors = o3d.utility.Vector3dVector([[1, 0.5, 1] for i in range(len(points4))])
o3d.io.write_point_cloud("points4.ply", pcd)


# Load saved point cloud and visualize it
pcd0_load = o3d.io.read_point_cloud("points0.ply")
pcd1_load = o3d.io.read_point_cloud("points1.ply")
pcd2_load = o3d.io.read_point_cloud("points2.ply")
pcd3_load = o3d.io.read_point_cloud("points3.ply")
pcd4_load = o3d.io.read_point_cloud("points4.ply")
# o3d.visualization.draw_geometries([pcd_load])
custom_draw_geometry([pcd0_load, pcd1_load, pcd2_load, pcd3_load, pcd4_load], title=filename)
# custom_draw_geometry([pcd_load, pcd2_load], title=filename)
