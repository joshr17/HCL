#!/bin/bash

DATASET_NAMES=("MUTAG" "PTC_MR" "ENZYMES" "IMDB-BINARY" "IMDB-MULTI" "DD" "REDDIT-BINARY" "PROTEINS")

for DATASET_NAME in "${DATASET_NAMES[@]}"; do  
	CUDA_VISIBLE_DEVICES=1, python deepinfomax.py --DS "$DATASET_NAME" --lr 0.001 --num-gc-layers 3 --repeats 10
done