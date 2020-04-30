#!/bin/bash
docker run -it --rm \
  --gpus '"device='0,1,2,3'"' \
  -u $(id -u):$(id -g) \
  -v $(pwd):/cpc/modelnet \
  -v $(pwd)/../utils:/cpc/utils \
  -v $(pwd)/../data_root:/cpc/data_root \
  -w /cpc/modelnet \
  completion-pc \
  python ../utils/main.py \
  --model_name cls_modelnet \
  --dataset modelnet \
  --num_pts 1024 \
  --num_pts_observed 1024 \
  --lr 0.001 \
  --step_size 200 \
  --max_epoch 500 \
  --bsize 64 \
  --radius 0.25 \
  --bottleneck 1024 \
  --num_subpc_train 64 \
  --num_contrib_feats_train 64 \
  --num_subpc_test 64 \
  --num_contrib_feats_test 64 \
  --weight_cls 1.0 \
  --is_classifier \
  # --is_vote \
  # --is_simuOcc \
  # --is_pCompletion \
  # --is_fidReg \
