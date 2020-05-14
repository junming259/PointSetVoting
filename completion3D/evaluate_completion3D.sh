#!/bin/bash
docker run -it --rm \
  --gpus '"device='0'"' \
  -u $(id -u):$(id -g) \
  -e SKIMAGE_DATADIR=/tmp \
  -v $(pwd):/cpc/completion3D \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/completion3D \
  completion-pc \
  python ../utils/main.py \
  --eval \
  --checkpoint checkpoint/completion3D_benchmark_model \
  --dataset completion3D \
  --categories chair,plane,car \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --bsize 16 \
  --radius 0.25 \
  --bottleneck 512 \
  --num_subpc_train 64 \
  --num_contrib_feats_train 16 \
  --num_subpc_test 16 \
  --num_contrib_feats_test 16 \
  --is_vote \
  --is_pCompletion \
  --is_simuOcc \