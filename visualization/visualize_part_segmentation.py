import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


color_map = 1 / 255.0 * np.array(
    [[47, 40, 49],
    [231, 29, 54],
    [26, 154, 140],
    [243, 98, 54],
    [231, 213, 76],
    [231, 213, 76]]
)

def rotate(pos, degree, axis):
    """
    Rotate points along specified axis
    """
    degree = np.pi * degree / 180.0
    sin, cos = np.sin(degree), np.cos(degree)

    if axis == 0:
        matrix = np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
    elif axis == 1:
        matrix = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
    else:
        matrix = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

    pos = np.matmul(pos, matrix)
    return pos


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="model name")
parser.add_argument("--idx", help="index for samples, like pos_{idx}.npy")
args = parser.parse_args()

#idx = '0'
#model_name = 'seg_b128e500s200lr1e-3_r020tr64-16_te64_bn1024_vote_simocc_xtransformer'
base_dir = '../shapenet_seg/checkpoint/{}/eval_sample_results/'.format(args.model_name)

# load pos
filename = '{}/pos_{}.npy'.format(base_dir, args.idx)
pos = np.load(filename)
pos = pos.reshape(-1, 3)
pos = rotate(pos, 90, axis=0)

# load pred
filename = '{}/pred_{}.npy'.format(base_dir, args.idx)
segp = np.load(filename).astype(np.int64)
segp = segp - segp.min()

# load label
filename = '{}/label_{}.npy'.format(base_dir, args.idx)
segl = np.load(filename).astype(np.int64)
segl = segl - segl.min()

# assign colors
colorp = np.zeros(pos.shape)
for i in range(len(segp)):
    color = color_map[segp[i]]
    colorp[i, :] = color
colorl = np.zeros(pos.shape)
for i in range(len(segl)):
    color = color_map[segl[i]]
    colorl[i, :] = color


# ######### Visualize in matplotlib ########
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, c=colorp, lw=0, alpha=1)
ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, c=colorl, lw=0, alpha=1)

ax1.title.set_text("prediction")
ax2.title.set_text("ground truth")

# compute the scale for visualization
max_range = np.array([pos[:, 0].max()-pos[:, 0].min(),
                      pos[:, 1].max()-pos[:, 1].min(),
                      pos[:, 2].max()-pos[:, 2].min()]).max() / 2.0

mid_x = (pos[:, 0].max()+pos[:, 0].min()) * 0.5
mid_y = (pos[:, 1].max()+pos[:, 1].min()) * 0.5
mid_z = (pos[:, 2].max()+pos[:, 2].min()) * 0.5

# make scale look equal
ax1.set_xlim(mid_x - max_range, mid_x + max_range)
ax1.set_ylim(mid_y - max_range, mid_y + max_range)
ax1.set_zlim(mid_z - max_range, mid_z + max_range)

ax2.set_xlim(mid_x - max_range, mid_x + max_range)
ax2.set_ylim(mid_y - max_range, mid_y + max_range)
ax2.set_zlim(mid_z - max_range, mid_z + max_range)

#plt.axis('off')
plt.show()
