#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2
python ./RGBT_workspace/test_rgbt_mgpus.py \
    --script_name cpetrack \
    --yaml_name deep_rgbt_256 \
    --dataset_name LasHeR \
    --epoch 45 \
    --threads 4 \
    --num_gpus 1 
