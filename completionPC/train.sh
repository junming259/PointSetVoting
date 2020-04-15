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
  --categories Chair \
  --num_pts 2048 \
  --lr 0.0001 \
  --step_size 200 \
  --max_epoch 500 \
  --bsize 8 \
  --num_subpc_train 64 \
  --num_subpc_test 16 \
  --num_contri_feats 16 \
  --is_fidReg
  # --randRotY
