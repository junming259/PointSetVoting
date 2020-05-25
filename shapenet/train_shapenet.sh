#!/bin/bash
docker run -it --rm \
  --gpus '"device='0,1,2,3'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/cpc/shapenet \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/shapenet \
  completion-pc \
  python ../utils/main.py \
  --model_name completion_shapenet \
  --dataset shapenet \
  --categories Chair,Airplane \
  --task completion \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --lr 0.0002 \
  --step_size 200 \
  --max_epoch 500 \
  --bsize 32 \
  --radius 0.20 \
  --bottleneck 1024 \
  --num_vote_train 64 \
  --num_contrib_vote_train 10 \
  --num_vote_test 64 \
  --is_vote \
  --is_rand \
  --is_simuOcc \
  --is_normalizeSphere \
  # --is_normalizeScale \
  # --is_randRotY \
  # --is_randST \
