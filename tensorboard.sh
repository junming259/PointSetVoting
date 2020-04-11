#! /bin/bash
docker run -it --rm \
  -v $PWD/cls/logs:/logs:ro \
  -p 6004:6005 \
  tensorflow/tensorflow:1.11.0-devel-gpu-py3 \
  tensorboard --logdir=/logs --port=6005
