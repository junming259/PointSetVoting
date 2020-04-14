#! /bin/bash
docker run -it --rm \
  -v $PWD/logs:/logs:ro \
  -p 6005:6006 \
  tensorflow/tensorflow:1.14.0-gpu-py3 \
  tensorboard --logdir=/logs --port=6006
