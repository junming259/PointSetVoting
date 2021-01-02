#!/bin/bash

python ../utils/main.py \
--eval \
--checkpoint checkpoint/cpc3d_b64e500s200lr2e-4_r010tr64-10_te32_bn1024/model.pth \
--task completion \
--categories plane,cabinet,car,chair,lamp,couch,table,watercraft \
--num_pts 2048 \
--num_pts_observed 2048 \
--bsize 64 \
--radius 0.10 \
--bottleneck 1024 \
--num_vote_test 256 \
--is_vote \
--save \

