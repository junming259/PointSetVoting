#!/bin/bash

# The script is for evaluating part segmentation on partial point clouds. If you
# want to evaluate on complete point clouds, comment `--is_simuOcc` and set
# `--num_pts_observed` to 2048 instead of 1024.

python ../utils/main.py \
--eval \
--checkpoint checkpoint/seg_b128e500s200lr1e-3_r020tr64-10_te32_bn1024/model.pth \
--categories Airplane,Bag,Cap,Car,Chair,Earphone,Guitar,Knife,Lamp,Laptop,Motorbike,Mug,Pistol,Rocket,Skateboard,Table \
--task segmentation \
--num_pts 2048 \
--num_pts_observed 1024 \
--bsize 64 \
--radius 0.20 \
--bottleneck 1024 \
--num_vote_test 256 \
--is_vote \
--is_simuOcc \
--save \
