import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def visual_right_scale(pos, ax):
    max_range = np.array([pos[:, 0].max()-pos[:, 0].min(),
                          pos[:, 1].max()-pos[:, 1].min(),
                          pos[:, 2].max()-pos[:, 2].min()]).max() / 2.0

    mid_x = (pos[:, 0].max()+pos[:, 0].min()) * 0.5
    mid_y = (pos[:, 1].max()+pos[:, 1].min()) * 0.5
    mid_z = (pos[:, 2].max()+pos[:, 2].min()) * 0.5

    # make scale look equal
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def sample(pos, n):
    num = pos.shape[0]
    idx = np.random.choice(num, n, False)
    return pos[idx]


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='demo_inputs/',
                    help='Specify the path of input partial point clouds. \
                        if path is a diretory, a random .npy will be choosen to
                        visualize.')
args = parser.parse_args()


######### Visualize in matplotlib ########
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data_path = args.data_path
if os.path.isdir(args.data_path):
    data_file_list = glob.glob(os.path.join(args.data_path, '*.npy'))
    idx = np.random.choice(len(data_file_list))
    data_path = data_file_list[idx]

_, tail = os.path.split(data_path)
pred_path = os.path.join('demo_results', 'pred_{}'.format(tail))

pts = np.load(data_path).reshape(-1, 3)
pred = np.load(pred_path).reshape(-1, 3)
# pts = sample(pts, 5000)

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, c='g', lw=0, alpha=1)
ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=5, c='r', lw=0, alpha=1)

visual_right_scale(pred, ax)
ax.title.set_text(data_path)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()



