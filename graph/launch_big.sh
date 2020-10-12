#!/bin/bash

DATASET_NAMES=("REDDIT-BINARY")

for DATASET_NAME in "${DATASET_NAMES[@]}"; do  
	CUDA_VISIBLE_DEVICES=1, python deepinfomax.py --DS "$DATASET_NAME" --lr 0.001 --num-gc-layers 3 --repeats 5 --batch_size 256
done