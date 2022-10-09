#!/usr/bin/env bash
dataset_name=$1
model=$2

echo Which PYTHON: `which python`
python eyelash_test.py \
--config=config/${model}.toml \
--checkpoint=$(pwd)/checkpoints/${model}/best_model.pth \
--image-dir=$(pwd)/${dataset_name} \
--output=$(pwd)/${dataset_name}.${model}_predict

