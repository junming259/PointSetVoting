#!/bin/bash
docker run -it --rm \
  --gpus '"device='0'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/completionPC/cpc \
  -v $(pwd)/../data:/completionPC/data \
  -v $(pwd)/../utils:/completionPC/utils \
  -v $(pwd)/../data_root:/completionPC/data_root \
  -w /completionPC/cpc \
  completion-pc \
  python main.py \
  --model_name cpc \
  --category Chair \
  --num_pts 2048 \
  --lr 0.0002 \
  --step_size 250 \
  --max_epoch 600 \
  --bsize 8 \
  --num_sub_feats 16 \
  --is_subReg \
  --randRotY
