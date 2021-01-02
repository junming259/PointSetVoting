#!/bin/bash

# The script is for evaluating shape classification on partial point clouds. If
# you want to evaluate on complete point clouds, comment `--is_simuOcc` and set
# `--num_pts_observed` to 1024 instead of 512.

python ../utils/main.py \
--eval \
--checkpoint checkpoint/cls_b64e500s200lr1e-3_r020tr64-10_te32_bn1024/model.pth \
--task classification \
--num_pts 1024 \
--num_pts_observed 512 \
--bsize 64 \
--radius 0.20 \
--bottleneck 1024 \
--num_vote_test 256 \
--is_vote \
--norm scale \
--is_simuOcc \
