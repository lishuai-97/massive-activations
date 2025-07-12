#!/bin/bash

ROOT_DIR=./
export EXP_NAME="gpt2-345M-default-run-lr-3e-4-beta2-0p999-50k"
TENSORBOARD_PATH=$ROOT_DIR/results/logs/$EXP_NAME

mkdir -p $TENSORBOARD_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_default.py | tee -a ${TENSORBOARD_PATH}/output.log