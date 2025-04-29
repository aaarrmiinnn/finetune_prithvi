#!/bin/bash
# Simple training script for Mac systems

# Set environment variables for better performance on Mac
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print system info
echo "Starting training on Mac OS system..."
echo "Python version: $(python --version)"
echo "Using CPU for training"

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints

# Run the training script with Mac-specific optimizations
python src/main.py \
  --config src/config/config.yaml \
  --mode train

echo "Training completed!" 