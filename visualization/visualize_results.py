from glob import glob
import numpy as np
import open3d as o3d
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D


# idx = '330, 324, 341'
idx = '331'
folder = 'cpc_b8e600s250lr2e-4_r02sub8'
filename = '../completionPC/checkpoint/{}/pred_pos_{}.npy'.format(folder, idx)
points = np.load(filename)
points = points.reshape(-1, 2025, 3)
points = points[0]
points_ = points.copy()
points_[:, 2] += 4
points = np.concatenate([points, points_], 0)
print('Number of predicted points: {}'.format(points_.shape[0]))


filename = '../completionPC/checkpoint/{}/pos_partial_{}.npy'.format(folder, idx)
points1 = np.load(filename)
points1 = points1.reshape(1, -1, 3)
points1 = points1[0]+0.01
points1[:, 2] += 2
print('Number of observed points: {}'.format(np.unique(points1, axis=0).shape[0]))


filename = '../completionPC/checkpoint/{}/pos_{}.npy'.format(folder, idx)
points2 = np.load(filename)
points2 = points2.reshape(1, 2048, 3)
points2 = points2[0]
points2_ = points2.copy()
points2_[:, 2] += 2
points2 = np.concatenate([points2, points2_], 0)
print('Number of full points: {}'.format(points2_.shape[0]))


filename = '../completionPC/checkpoint/{}/pred_latent_pos_{}.npy'.format(folder, idx)
points3 = np.load(filename)
points3 = points3.reshape(1, 2025, 3)
points3 = points3[0]
points3[:, 2] += 6


######### Visualize in open3d ########
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

# total points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(points))])
# pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("points.ply", pcd)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points1)
pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(points1))])
# pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("points1.ply", pcd)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points2)
pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(points2))])
# pcd.colors = o3d.utility.Vector3dVector(colors2)
o3d.io.write_point_cloud("points2.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3)
pcd.colors = o3d.utility.Vector3dVector([[1, 0, 1] for i in range(len(points3))])
# pcd.colors = o3d.utility.Vector3dVector(colors2)
o3d.io.write_point_cloud("points3.ply", pcd)

# Load saved point cloud and visualize it
pcd_load = o3d.io.read_point_cloud("points.ply")
pcd1_load = o3d.io.read_point_cloud("points1.ply")
pcd2_load = o3d.io.read_point_cloud("points2.ply")
pcd3_load = o3d.io.read_point_cloud("points3.ply")
# o3d.visualization.draw_geometries([pcd_load])
custom_draw_geometry([pcd_load, pcd1_load, pcd2_load, pcd3_load], title=filename)
# custom_draw_geometry([pcd_load, pcd2_load], title=filename)


######### Visualize in matplotlib ########
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.set_aspect('equal')
# for i in range(x1.shape[0]):
#     x = [x1[i, 0], x2[i, 0]]
#     y = [x1[i, 1], x2[i, 1]]
#     z = [x1[i, 2], x2[i, 2]]
#     ax.plot(x, y, z, c='purple', alpha=0.15)
# # ax.scatter(x7[:, 0], x7[:, 1], x7[:, 2], c='r', edgecolors='r')
# ax.scatter(x7[:, 0], x7[:, 1], x7[:, 2], s=60, c=colors, lw=0)
#
# # make scale look equal
# max_range = np.array([x1[:, 0].max()-x1[:, 0].min(),
#                       x1[:, 1].max()-x1[:, 1].min(),
#                       x1[:, 2].max()-x1[:, 2].min()]).max() / 2.0
#
# mid_x = (x1[:, 0].max()+x1[:, 0].min()) * 0.5
# mid_y = (x1[:, 1].max()+x1[:, 1].min()) * 0.5
# mid_z = (x1[:, 2].max()+x1[:, 2].min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.title('Pairs')

# x1 = points1
# x2 = points2
# fig = plt.figure()
# ax = fig.add_subplot(122, projection='3d')
# ax.set_aspect('equal')
#
# ax.scatter(x2[:, 0], x2[:, 1], x2[:, 2])
# ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2])
#
# # make scale look equal
# max_range = np.array([x1[:, 0].max()-x1[:, 0].min(),
#                       x1[:, 1].max()-x1[:, 1].min(),
#                       x1[:, 2].max()-x1[:, 2].min()]).max() / 2.0
#
# mid_x = (x1[:, 0].max()+x1[:, 0].min()) * 0.5
# mid_y = (x1[:, 1].max()+x1[:, 1].min()) * 0.5
# mid_z = (x1[:, 2].max()+x1[:, 2].min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.title('Points')
#
# plt.show()
