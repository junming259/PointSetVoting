#!/bin/bash
docker run -it --rm \
  --gpus '"device='3'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/completionPC/cpc \
  -v $(pwd)/../utils:/completionPC/utils \
  -v $(pwd)/../data_root:/completionPC/data_root \
  -w /completionPC/cpc \
  completion-pc \
  python main.py \
  --model_name cpc_b12e600s250lr1e-3_r025tr64-16_te16-16_bn512_MN40_cls \
  --dataset ModelNet40 \
  --is_classifier \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --lr 0.001 \
  --step_size 250 \
  --max_epoch 600 \
  --bsize 12 \
  --radius 0.25 \
  --bottleneck 512 \
  --num_subpc_train 64 \
  --num_contri_feats_train 16 \
  --num_subpc_test 16 \
  --num_contri_feats_test 16 \
  --is_fidReg
  # --randRotY
