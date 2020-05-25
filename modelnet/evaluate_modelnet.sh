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
  --task classification \
  --num_pts 1024 \
  --num_pts_observed 512 \
  --bsize 64 \
  --radius 0.20 \
  --bottleneck 1024 \
  --num_vote_train 64 \
  --num_contrib_vote_train 10 \
  --num_vote_test 128 \
  --is_rand \
  --is_vote \
  --is_normalizeSphere \
  --is_simuOcc \
  # --is_normalizeScale \
  # --is_randRotY \
  # --is_randST \
