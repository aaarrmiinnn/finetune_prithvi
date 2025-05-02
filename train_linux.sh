#!/bin/bash
# Standard training script for Linux systems

# Set environment variables for better GPU performance
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print system info
echo "Starting training on Linux system..."
echo "Python version: $(python --version)"
echo "Using GPU for training if available..."

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints
mkdir -p cache

# Check if CUDA is available using PyTorch
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo "PyTorch CUDA is available"
    python -c "import torch; print('CUDA Devices:', torch.cuda.device_count()); print('Current Device:', torch.cuda.current_device()); print('Device Name:', torch.cuda.get_device_name(0))"
else
    echo "PyTorch CUDA not available, will use CPU"
fi

# Empty GPU cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('No CUDA available to empty cache')"

# Start training
echo "Starting training..."
python src/main.py \
  --config src/config/config.yaml \
  --mode train

echo "Training completed!" 