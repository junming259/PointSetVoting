#!/bin/bash
docker run -it --rm \
  --gpus '"device='0'"' \
  -u $(id -u):$(id -g) \
  -e SKIMAGE_DATADIR=/tmp \
  -v $(pwd):/cpc/shapenet \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/shapenet \
  completion-pc \
  python ../utils/main.py \
  --eval \
  --checkpoint checkpoint/cpc_b32e500s200lr2e-4_r020tr64-10_te16_bn1024_ChairAirplaneCarTable_vote_nofid_CD_normSphere \
  --task completion \
  --dataset shapenet \
  --categories Airplane \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --bsize 4 \
  --radius 0.20 \
  --bottleneck 1024 \
  --num_vote_train 64 \
  --num_contrib_vote_train 10 \
  --num_vote_test 64 \
  --is_rand \
  --is_vote \
  --is_normalizeSphere \
  --is_simuOcc \
  # --is_normalizeScale \
  # --is_randRotY \
  # --is_randST \
