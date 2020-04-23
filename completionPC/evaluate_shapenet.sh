#!/bin/bash
docker run -it --rm \
  --gpus '"device='0'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/completionPC/cpc \
  -v $(pwd)/../utils:/completionPC/utils \
  -v $(pwd)/../data_root:/completionPC/data_root \
  -w /completionPC/cpc \
  completion-pc \
  python main.py \
  --eval \
  --checkpoint checkpoint/cpc_model \
  --dataset ShapeNet \
  --categories Chair,Airplane,Car \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --bsize 32 \
  --radius 0.25 \
  --bottleneck 512 \
  --num_subpc_train 64 \
  --num_contri_feats_train 16 \
  --num_subpc_test 16 \
  --num_contri_feats_test 12 \
  --is_fidReg
  # --randRotY
