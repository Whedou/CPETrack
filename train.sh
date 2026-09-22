#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3

python tracking/train.py \
  --script cpetrack \
  --config deep_rgbt_256 \
  --save_dir ./output \
  --mode single \
  --nproc_per_node 1

for epoch in 45
do
  python ./RGBT_workspace/test_rgbt_mgpus.py \
    --script_name cpetrack \
    --yaml_name deep_rgbt_256 \
    --dataset_name LasHeR \
    --epoch "$epoch" \
    --threads 4 \
    --num_gpus 1 \
    --mode sequential
done


