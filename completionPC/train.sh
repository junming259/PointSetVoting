#!/bin/bash
docker run -it --rm \
  --gpus '"device='6'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/completionPC/cpc \
  -v $(pwd)/../data:/completionPC/data \
  -v $(pwd)/../utils:/completionPC/utils \
  -v $(pwd)/../data_root:/completionPC/data_root \
  -w /completionPC/cpc \
  completion-pc \
  python main.py \
  --model_name cpc_b7e600s250lr2e-4_r02sub16_bndec \
  --num_pts 2048 \
  --lr 0.0002 \
  --step_size 250 \
  --max_epoch 600 \
  --bsize 7 \
  --num_sub_feats 16 \
  --is_subReg
