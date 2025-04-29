#!/bin/bash
# Simple training script for Linux systems with GPU

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Print system info
echo "Starting training on Linux system..."
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "Using GPU for training"

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints

# Run the training script with GPU optimizations
python src/train.py \
  --config src/config/config.yaml \
  --accelerator gpu \
  --devices 1 \
  --precision 16 \
  --batch_size 4 \
  --accumulate_grad_batches 2 \
  --max_epochs 10

echo "Training completed!" 