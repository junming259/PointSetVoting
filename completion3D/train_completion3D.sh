#!/bin/bash
docker run -it --rm \
  --gpus '"device='0'"' \
  -v $(pwd):/cpc/training \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/training \
  completion-pc \
  python ../utils/main.py \
  --model_name cpc3d_b64e500s200lr2e-4_r010tr64-10_te32_bn1024 \
  --dataset completion3D \
  --categories plane,cabinet,car,chair,lamp,couch,table,watercraft \
  --task completion \
  --num_pts 2048 \
  --num_pts_observed 2048 \
  --lr 0.0002 \
  --step_size 200 \
  --max_epoch 500 \
  --bsize 64 \
  --radius 0.10 \
  --bottleneck 1024 \
  --num_vote_train 64 \
  --num_contrib_vote_train 10 \
  --num_vote_test 32 \
  --is_vote \
