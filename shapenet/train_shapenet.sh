#!/bin/bash
python ../utils/main.py \
--model_name seg_b128e500s200lr1e-3_r020tr64-10_te32_bn1024 \
--categories Airplane,Bag,Cap,Car,Chair,Earphone,Guitar,Knife,Lamp,Laptop,Motorbike,Mug,Pistol,Rocket,Skateboard,Table \
--task segmentation \
--num_pts 2048 \
--num_pts_observed 1024 \
--lr 0.001 \
--step_size 200 \
--max_epoch 500 \
--bsize 128 \
--radius 0.20 \
--bottleneck 1024 \
--num_vote_train 64 \
--num_contrib_vote_train 10 \
--num_vote_test 32 \
--is_vote \
--is_simuOcc \
