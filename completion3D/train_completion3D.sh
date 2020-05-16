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
  --model_name completion3D_benchmark_model_200epoch \
  --dataset completion3D \
  --categories chair,plane,car \
  --num_pts 2048 \
  --num_pts_observed 1024 \
  --lr 0.0002 \
  --step_size 250 \
  --max_epoch 200 \
  --bsize 32 \
  --radius 0.25 \
  --bottleneck 512 \
  --num_subpc_train 64 \
  --num_contrib_feats_train 16 \
  --num_subpc_test 16 \
  --num_contrib_feats_test 16 \
  --weight_chamfer 1.0 \
  --weight_fidelity 0.1 \
  --is_vote \
  --is_pCompletion \
  --is_simuOcc \
