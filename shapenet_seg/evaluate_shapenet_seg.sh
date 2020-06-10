#!/bin/bash
docker run -it --rm \
  --gpus '"device='3'"' \
  -u $(id -u):$(id -g) \
  -e SKIMAGE_DATADIR=/tmp \
  -v $(pwd):/cpc/shapenet \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/shapenet \
  completion-pc \
  python ../utils/main.py \
  --eval \
  --checkpoint checkpoint/seg_b128e500s200lr1e-3_r020tr64-16_te64_bn1024_vote_simocc_xtransformer \
  --dataset shapenet \
  --categories Airplane \
  --task segmentation \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --bsize 2 \
  --radius 0.20 \
  --bottleneck 1024 \
  --num_vote_train 64 \
  --num_contrib_vote_train 10 \
  --num_vote_test 64 \
  --is_rand \
  --is_vote \
  --is_simuOcc \
  # --is_normalizeSphere \
  # --is_normalizeScale \
  # --is_randRotY \
  # --is_randST \
  # --categories Airplane,Bag,Cap,Car,Chair,Earphone,Guitar,Knife,Lamp,Laptop,Motorbike,Mug,Pistol,Rocket,Skateboard,Table \
