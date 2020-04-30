#!/bin/bash
docker run -it --rm \
  --gpus '"device='0'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/cpc/shapenet \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/shapenet \
  completion-pc \
  python ../utils/main.py \
  --eval \
  --checkpoint checkpoint/completion_shapenet \
  --dataset shapenet \
  --categories Chair,Airplane,Car \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --bsize 32 \
  --radius 0.25 \
  --bottleneck 512 \
  --num_subpc_train 64 \
  --num_contrib_feats_train 16 \
  --num_subpc_test 16 \
  --num_contrib_feats_test 16 \
  --is_vote \
  --is_pCompletion \
  --is_fidReg \
  --is_simuOcc \
