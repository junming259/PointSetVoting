#!/bin/bash
docker run -it --rm \
  --gpus '"device='0,1,2,3'"' \
  -u $(id -u):$(id -g) \
  -e SKIMAGE_DATADIR=/tmp \
  -v $(pwd):/cpc/completion3D \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/completion3D \
  completion-pc \
  python ../utils/main.py \
  --model_name completion3D_benchmark_model_8ctg500epoch \
  --dataset completion3D \
  --categories plane,cabinet,car,chair,lamp,couch,table,watercraft \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --lr 0.0002 \
  --step_size 200 \
  --max_epoch 500 \
  --bsize 32 \
  --radius 0.2 \
  --bottleneck 1024 \
  --num_subpc_train 64 \
  --num_contrib_feats_train 16 \
  --num_subpc_test 16 \
  --num_contrib_feats_test 16 \
  --weight_chamfer 1.0 \
  --weight_fidelity 0.1 \
  --is_vote \
  --is_pCompletion \
  --is_simuOcc \
