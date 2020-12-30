#!/bin/bash
python ../utils/main.py \
--model_name cls_b64e500s200lr1e-3_r020tr64-10_te32_bn1024 \
--task classification \
--num_pts 1024 \
--num_pts_observed 512 \
--lr 0.001 \
--step_size 200 \
--max_epoch 500 \
--bsize 64 \
--radius 0.20 \
--bottleneck 1024 \
--num_vote_train 64 \
--num_contrib_vote_train 10 \
--num_vote_test 32 \
--is_vote \
--is_simuOcc \
--norm scale \

