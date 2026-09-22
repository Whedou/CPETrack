#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2
for epoch in 60 55 50
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
