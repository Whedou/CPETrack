#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3

python tracking/train.py \
  --script cpetrack \
  --config deep_rgbt_256 \
  --save_dir ./output \
  --mode single \
  --nproc_per_node 1


