#!/bin/bash
# Multi-GPU training script optimized for distributed training
# This script automatically detects and uses all available GPUs and CPU cores

# Set environment variables for better GPU and CPU performance
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export OMP_NUM_THREADS=$(nproc)  # Use all CPU cores for OpenMP
export MKL_NUM_THREADS=$(nproc)  # Use all CPU cores for MKL
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print system info
echo "Starting multi-GPU training setup..."
echo "Python version: $(python --version)"
echo "CPU Cores: $(nproc)"

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints
mkdir -p cache

# Create a temporary modified config file with multi-GPU settings
CONFIG_FILE="src/config/config.yaml"
TEMP_CONFIG_FILE="src/config/config_multi_gpu.yaml"

# Check available GPUs and CPU cores
python -c "
import torch
import os
import yaml
import math

# Load the base config
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Get number of available GPUs
num_gpus = torch.cuda.device_count()
print(f'Available GPUs: {num_gpus}')
if num_gpus > 0:
    for i in range(num_gpus):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

# Get number of CPU cores
num_cores = os.cpu_count()
print(f'Available CPU cores: {num_cores}')

# Configure for multi-GPU training
if num_gpus > 0:
    # Hardware settings
    config['hardware']['accelerator'] = 'gpu'
    config['hardware']['devices'] = num_gpus
    config['hardware']['num_workers'] = max(4, num_cores // num_gpus)  # Distribute cores among GPUs
    config['hardware']['pin_memory'] = True
    
    # Distributed training settings
    config['distributed']['enabled'] = True
    config['distributed']['strategy'] = 'ddp'
    config['distributed']['sync_batchnorm'] = True
    config['distributed']['find_unused_parameters'] = False
    
    # Adjust batch size and learning rate for multi-GPU
    base_batch_size = config['training']['batch_size']
    config['training']['batch_size'] = base_batch_size * num_gpus
    config['training']['optimizer']['lr'] *= math.sqrt(num_gpus)  # Scale learning rate with sqrt of number of GPUs
    
    # Memory optimizations
    config['training']['precision'] = 16  # Use mixed precision
    config['model']['gradient_checkpointing'] = True  # Enable gradient checkpointing
    
    print(f'Configured for {num_gpus} GPUs:')
    print(f'- Total batch size: {config['training']['batch_size']} ({base_batch_size} per GPU)')
    print(f'- Workers per GPU: {config['hardware']['num_workers']}')
    print(f'- Learning rate: {config['training']['optimizer']['lr']:.2e}')
else:
    print('No GPUs available, falling back to CPU training')
    config['hardware']['accelerator'] = 'cpu'
    config['hardware']['devices'] = 1
    config['hardware']['num_workers'] = num_cores
    config['hardware']['pin_memory'] = False
    config['distributed']['enabled'] = False

# Save the multi-GPU config
with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"

# Empty GPU cache before starting
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
"

# Start distributed training
echo "Starting multi-GPU training..."
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train \
  --cluster

# Clean up temporary config
rm $TEMP_CONFIG_FILE

echo "Training completed!" 