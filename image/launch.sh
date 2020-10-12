#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1, python main.py --dataset_name stl10 --batch_size 256 --estimator hard --tau_plus 0.1 --beta 1.0
CUDA_VISIBLE_DEVICES=0,1, python linear.py --dataset_name stl10 --model_path ../results/stl10/stl10_hard_model_256_0.1_1.0_400.pth