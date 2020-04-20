#!/bin/bash
docker run -it --rm \
  --gpus '"device='1'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/completionPC/cpc \
  -v $(pwd)/../utils:/completionPC/utils \
  -v $(pwd)/../data_root:/completionPC/data_root \
  -w /completionPC/cpc \
  completion-pc \
  python main.py \
  --eval \
  --checkpoint checkpoint/cpc_b12e600s250lr2e-4_r025tr64-16_te16-16_bn512_MN40_cls \
  --dataset ModelNet40 \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --radius 0.25 \
  --bottleneck 512 \
  --num_subpc_train 64 \
  --num_contri_feats_train 16 \
  --num_subpc_test 4 \
  --num_contri_feats_test 4 \
  --is_classifier
  # --randRotY
