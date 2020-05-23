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
  python ../utils/main_test_Benchmark.py \
  --eval \
  --checkpoint checkpoint/completion3D_benchmark_model_8ctg500epoch_r25 \
  --dataset completion3D \
  --categories plane,cabinet,car,chair,lamp,couch,table,watercraft \
  --num_pts 2048 \
  --num_pts_observed 2048 \
  --bsize 1 \
  --radius 0.25 \
  --bottleneck 1024 \
  --num_subpc_train 64 \
  --num_contrib_feats_train 16 \
  --num_subpc_test 64 \
  --num_contrib_feats_test 64 \
  --is_vote \
  --is_pCompletion \
