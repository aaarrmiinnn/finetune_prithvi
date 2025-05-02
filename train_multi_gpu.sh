#!/bin/bash
# Multi-GPU training script optimized for distributed training

# Set environment variables for better GPU and CPU performance
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export OMP_NUM_THREADS=$(nproc)  # Use all CPU cores for OpenMP
export MKL_NUM_THREADS=$(nproc)  # Use all CPU cores for MKL

# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=$(nvidia-smi -L | wc -l)  # Number of GPUs
export NODE_RANK=0  # Single node training

# Print system info
echo "Starting multi-GPU training setup..."
echo "Python version: $(python --version)"
echo "CPU Cores: $(nproc)"
echo "Number of GPUs: $WORLD_SIZE"

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints
mkdir -p cache

# Check available GPUs and print info
python -c "
import torch
print(f'Available GPUs: {torch.cuda.device_count()}')
if torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'Available CPU cores: {os.cpu_count()}')"

# Empty GPU cache before starting
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
"

# Function to run training on a specific GPU
run_training() {
    local GPU_RANK=$1
    export RANK=$GPU_RANK
    export LOCAL_RANK=$GPU_RANK
    export CUDA_VISIBLE_DEVICES=$GPU_RANK
    
    echo "Starting training process on GPU $GPU_RANK"
    python src/main.py \
        --config src/config/config_multi_gpu.yaml \
        --mode train \
        --cluster
}

# Launch training processes for each GPU
echo "Launching distributed training on $WORLD_SIZE GPUs..."
for ((GPU_RANK=0; GPU_RANK<$WORLD_SIZE; GPU_RANK++)); do
    run_training $GPU_RANK &
done

# Wait for all processes to complete
wait

echo "Multi-GPU training completed!" 