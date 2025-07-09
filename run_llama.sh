#!/bin/bash

# Configuration
# NOTE: llama2_13b model requires 2 GPUs, so set GPU_ID accordingly.
GPU_ID=0
# MODELS=(llama2_7b llama2_7b_chat llama2_13b)
MODELS=(llama2_7b_chat)
BASE_SAVEDIR="results/llm"

# Create base directory
mkdir -p "$BASE_SAVEDIR"

# Function to run experiment
run_exp() {
    local model=$1
    local args=$2
    local desc=$3
    
    echo "Running: $desc for $model"
    if CUDA_VISIBLE_DEVICES=$GPU_ID python main_llm.py --model $model $args; then
        echo "✓ Completed: $desc for $model"
    else
        echo "✗ Failed: $desc for $model"
        echo "Stopping execution due to error."
        exit 1
    fi
}

# Main execution
for MODEL_NAME in "${MODELS[@]}"; do
    echo "========================================="
    echo "Processing model: $MODEL_NAME"
    echo "========================================="
    
    # Experiment 1: 3D feature visualization
    run_exp "$MODEL_NAME" "--exp1 --layer_id 2 --savedir $BASE_SAVEDIR/3d_feat_vis/" \
        "3D Feature Visualization"
    
    # Experiment 2: Layerwise analysis
    run_exp "$MODEL_NAME" "--exp2 --savedir $BASE_SAVEDIR/layerwise/" \
        "Layerwise Analysis"
    
    # Experiment 3: Intervention analysis
    run_exp "$MODEL_NAME" "--exp3 --reset_type set_zero --layer_id 2 --savedir $BASE_SAVEDIR/intervention_analysis/" \
        "Intervention Analysis (set_zero)"
    
    run_exp "$MODEL_NAME" "--exp3 --reset_type set_mean --layer_id 2 --savedir $BASE_SAVEDIR/intervention_analysis/" \
        "Intervention Analysis (set_mean)"
    
    # Experiment 4: Attention visualization
    run_exp "$MODEL_NAME" "--exp4 --layer_id 3 --savedir $BASE_SAVEDIR/attn_vis/" \
        "Attention Visualization"
    
    echo "Completed all experiments for $MODEL_NAME"
    echo ""
done

echo "All experiments completed!"