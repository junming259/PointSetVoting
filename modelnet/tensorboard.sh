#! /bin/bash
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v $PWD/logs:/logs:ro \
  -p 6005:6006 \
  completion-pc \
  tensorboard --logdir=/logs --port=6006 --bind_all
