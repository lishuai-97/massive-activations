#!/bin/bash

export CKPT_DIR="/data3/lishuai_datasets/llm/checkpoints/nanogpt_ckpt"
export EXP_NAME="gpt2-345M-default-run-lr-1e-3-beta2-0p999"

# Define checkpoint iterations to evaluate
CKPT_ITERATIONS=(3000 5000 6000 7000 8000 9000 11000)

# Loop through each checkpoint for evaluation
for ckpt_iter in "${CKPT_ITERATIONS[@]}"; do
    echo "========================================="
    echo "Analyzing checkpoint: $ckpt_iter"
    echo "========================================="
    
    export CKPT_ITER=$ckpt_iter
    CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_default.py
    
    # Check the exit status of the previous command
    if [ $? -ne 0 ]; then
        echo "Error: Analysis failed for checkpoint $ckpt_iter"
        # You can choose to continue or exit
        # exit 1  # Exit on failure
        continue  # Continue to next checkpoint
    fi
    
    echo "Completed analysis for checkpoint: $ckpt_iter"
    echo ""
done

echo "All checkpoint analyses completed!"