#!/bin/bash
# Simple training script for Linux systems with GPU

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print system info
echo "Starting training on Linux system..."
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "Using GPU for training"

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints

# Run the training script with GPU optimizations
python src/main.py \
  --config src/config/config.yaml \
  --mode train \
  --cluster

echo "Training completed!" 