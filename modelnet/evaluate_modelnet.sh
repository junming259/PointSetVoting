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
  --checkpoint checkpoint/test_cls_modelnet_b64e500s200lr1e-3_r020tr64-10_te16_bn1024_sphere \
  --dataset modelnet \
  --task classification \
  --num_pts 1024 \
  --num_pts_observed 512 \
  --bsize 64 \
  --radius 0.20 \
  --bottleneck 1024 \
  --num_vote_test 128 \
  --is_rand \
  --is_vote \
  --is_simuOcc \
  --norm sphere \
