#!/bin/bash
docker run -it --rm \
  --gpus '"device='0'"' \
  -u $(id -u):$(id -g) \
  -e SKIMAGE_DATADIR=/tmp \
  -v $(pwd):/cpc/test \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/test \
  completion-pc \
  python ../utils/main.py \
  --eval \
  --checkpoint checkpoint/cpc3d_b64e500s200lr2e-4_r010tr64-10_te32_bn1024_trainval_test \
  --task completion \
  --dataset completion3D \
  --categories plane,cabinet,car,chair,lamp,couch,table,watercraft \
  --num_pts 2048 \
  --num_pts_observed 2048 \
  --bsize 32 \
  --radius 0.10 \
  --bottleneck 1024 \
  --num_vote_test 32 \
  --is_vote \



#!/bin/bash
# docker run -it --rm \
#   --gpus '"device='0'"' \
#   -u $(id -u):$(id -g) \
#   -e SKIMAGE_DATADIR=/tmp \
#   -v $(pwd):/cpc/test \
#   -v $(pwd)/../utils:/cpc/utils \
#   -v $(pwd)/../data_root:/cpc/data_root \
#   -w /cpc/test \
#   completion-pc \
#   python ../utils/main.py \
#   --eval \
#   --checkpoint checkpoint/cpc3d_b64e500s200lr2e-4_r010tr64-10_te32_bn1024 \
#   --task completion \
#   --dataset scanobjectnn \
#   --categories chair,table \
#   --num_pts 2048 \
#   --num_pts_observed 2048 \
#   --bsize 4 \
#   --radius 0.10 \
#   --bottleneck 1024 \
#   --num_vote_train 64 \
#   --num_contrib_vote_train 10 \
#   --num_vote_test 128 \
#   --is_rand \
#   --is_vote \
#   # --is_normalizeSphere \
#   # --is_simuOcc \
#   # --is_normalizeScale \
#   # --is_randRotY \
#   # --is_randST \
