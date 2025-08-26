#!/bin/bash

# DeepSpeed training script for GraspGPT
# Usage: ./run_deepspeed_training.sh [num_gpus] [additional_args...]

set -e

# Get the number of GPUs (default to 2)
NUM_GPUS=${1:-2}

# Shift to get additional arguments
shift 2>/dev/null || true




# Set CUDA visible devices if needed
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Change to the graspGPT directory
cd graspGPT

echo "Starting DeepSpeed training with $NUM_GPUS GPUs..."
echo "Additional arguments: $@"

# Run DeepSpeed training
deepspeed --num_gpus=$NUM_GPUS train_deepspeed.py \
    --deepspeed_config ../deepspeed_config.json \
    --batch_size 30 \
    --micro_batch_size 9 \
    --learning_rate 3e-4 \
    --max_iters 200000 \
    --wandb_project "graspgpt-deepspeed" \
    "$@"

echo "Training completed!"