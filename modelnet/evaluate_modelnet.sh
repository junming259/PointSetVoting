#!/bin/bash
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
--is_simuOcc \
--norm scale \
