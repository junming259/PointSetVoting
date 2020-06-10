#!/bin/bash
docker run -it --rm \
  --gpus '"device='2'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/cpc/modelnet \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/modelnet \
  completion-pc \
  python ../utils/main.py \
  --eval \
  --checkpoint checkpoint/cls_modelnet_b64e500s200lr1e-3_r020tr64-10_te16_bn1024_scalebox \
  --dataset scanobjectnn \
  --categories bag,bin,box,cabinet,chair,desk,display,door,shelf,table,bed,pillow,sink,sofa,toilet \
  --task classification \
  --num_pts 1024 \
  --num_pts_observed 2048 \
  --bsize 64 \
  --radius 0.20 \
  --bottleneck 1024 \
  --num_vote_train 64 \
  --num_contrib_vote_train 10 \
  --num_vote_test 128 \
  --is_rand \
  --is_vote \
  # --norm scalebox \
  # --is_simuOcc \
  # --is_randRotY \
