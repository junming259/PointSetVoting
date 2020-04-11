#! /bin/bash
docker run -it --rm \
  -v $PWD/logs:/logs:ro \
  -p 6006:6006 \
  tensorflow/tensorflow:1.11.0-devel-gpu-py3 \
  tensorboard --logdir=/logs --port=6006
