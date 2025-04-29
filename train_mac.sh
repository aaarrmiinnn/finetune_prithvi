#!/bin/bash
# Simple training script for Mac systems

# Set environment variables for better performance on Mac
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Print system info
echo "Starting training on Mac OS system..."
echo "Python version: $(python --version)"
echo "Using CPU for training"

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints

# Run the training script with Mac-specific optimizations
python src/train.py \
  --config src/config/config.yaml \
  --accelerator cpu \
  --precision 32 \
  --batch_size 1 \
  --accumulate_grad_batches 4 \
  --max_epochs 5

echo "Training completed!" 