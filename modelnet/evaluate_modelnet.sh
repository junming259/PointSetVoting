#!/bin/bash
docker run -it --rm \
  --gpus '"device='0'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/cpc/modelnet \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/modelnet \
  completion-pc \
  python ../utils/main.py \
  --eval \
  --checkpoint checkpoint/cls_modelnet \
  --dataset modelnet \
  --num_pts 1024 \
  --num_pts_observed 512 \
  --radius 0.25 \
  --bottleneck 1024 \
  --num_subpc_train 64 \
  --num_contrib_feats_train 10 \
  --num_subpc_test 128 \
  --num_contrib_feats_test 128 \
  --is_classifier \
  --is_vote \
  --is_simuOcc \
  # --randRotY
