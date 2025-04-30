#!/bin/bash
# Multi-GPU training script for Linux systems
# This script uses all available GPUs for distributed training

# Set environment variables for better GPU performance
# Removed CUDA_VISIBLE_DEVICES to allow using all GPUs
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print system info
echo "Starting multi-GPU distributed training..."
echo "Python version: $(python --version)"
echo "This script will use all available GPUs for distributed training"

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints
mkdir -p cache

# Create a temporary modified config file with multi-GPU settings
CONFIG_FILE="src/config/config.yaml"
TEMP_CONFIG_FILE="src/config/config_multi_gpu.yaml"

# Check if CUDA is available and count GPUs
CUDA_INFO=$(python -c "
import torch
if torch.cuda.is_available():
    print('available', torch.cuda.device_count())
else:
    print('unavailable', 0)
")

CUDA_AVAILABLE=$(echo $CUDA_INFO | cut -d' ' -f1)
NUM_GPUS=$(echo $CUDA_INFO | cut -d' ' -f2)

if [ "$CUDA_AVAILABLE" = "available" ] && [ "$NUM_GPUS" -gt 1 ]; then
    echo "PyTorch CUDA is available with $NUM_GPUS GPUs, configuring for multi-GPU training"
    
    # Use Python to update the config file for multi-GPU training
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Multi-GPU hardware settings
config['hardware']['accelerator'] = 'gpu'
config['hardware']['devices'] = -1  # Use all available GPUs
config['model']['device'] = 'cuda'

# Distributed training settings
config['cluster']['enabled'] = True
config['cluster']['strategy'] = 'ddp'  # Use DistributedDataParallel
config['cluster']['sync_batchnorm'] = True  # Synchronize batch normalization

# Increase batch size for multi-GPU efficiency
# Each GPU will use this batch size, so effective batch size = batch_size * num_gpus
config['training']['batch_size'] = 4  # Increase batch size for efficiency
config['training']['accumulate_grad_batches'] = 4

# Use mixed precision for speed and memory efficiency
config['training']['precision'] = 16

# Other optimizations
config['hardware']['num_workers'] = 4  # Increase workers for faster data loading
config['hardware']['pin_memory'] = True
config['model']['freeze_encoder'] = True  # Still keep backbone frozen for stability

# Learning rate needs to be scaled with batch size
# LR = base_lr * (effective_batch_size / 256)
effective_batch_size = config['training']['batch_size'] * $NUM_GPUS * config['training']['accumulate_grad_batches']
base_lr = 1e-6
scaled_lr = base_lr * (effective_batch_size / 256)
config['training']['optimizer']['lr'] = max(scaled_lr, base_lr)  # Don't go below base LR

# Set up data and dates - use a subset for faster iterations
config['data']['dates'] = [
    '20250315', '20250316', '20250317', '20250318', '20250319',
    '20250320', '20250321', '20250322', '20250323', '20250324',
    '20250325', '20250326', '20250327', '20250328', '20250329'
]

# Enable Weights & Biases logging
config['logging']['wandb'] = True
config['logging']['wandb_project'] = 'prithvi-downscaling'
config['logging']['wandb_tags'] = ['multi-gpu', 'distributed', 'frozen-backbone']
config['logging']['wandb_notes'] = f'Multi-GPU training with {$NUM_GPUS} GPUs'

with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
    
    # Print GPU information
    python -c "
import torch
print(f'CUDA Devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'Total GPU Memory: {sum([torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]) / 1024**3:.1f} GB')
"
    
else
    if [ "$CUDA_AVAILABLE" = "available" ] && [ "$NUM_GPUS" -eq 1 ]; then
        echo "Only 1 GPU available, configuring for single-GPU training"
        # Create a single-GPU config
        python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['hardware']['accelerator'] = 'gpu'
config['hardware']['devices'] = 1
config['model']['device'] = 'cuda'
config['training']['batch_size'] = 4
config['training']['precision'] = 16
config['model']['freeze_encoder'] = True  # Make sure encoder is frozen
config['training']['optimizer']['lr'] = 1e-6
config['logging']['wandb'] = True
config['logging']['wandb_project'] = 'prithvi-downscaling'
config['logging']['wandb_tags'] = ['single-gpu', 'frozen-backbone']
with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
    else
        echo "CUDA is not available, multi-GPU training is not possible"
        exit 1
    fi
fi

# Validate cache files and remove corrupted ones
echo "Validating cache files to prevent EOFError..."
python -c "
import os
import numpy as np
import glob

cache_dir = 'cache'
cache_files = glob.glob(os.path.join(cache_dir, 'merra2_prism_*.npz'))
print(f'Checking {len(cache_files)} cache files...')

for cache_file in cache_files:
    try:
        # Try to load the file to see if it's valid
        data = np.load(cache_file, allow_pickle=True)
        # Verify key content exists
        if 'patches' not in data:
            print(f'Warning: Missing data in {cache_file}, removing...')
            os.remove(cache_file)
            continue
        
        # Try to access the data to ensure it's readable
        _ = data['patches'].tolist()
        print(f'✓ Valid cache file: {os.path.basename(cache_file)}')
    except (EOFError, KeyError, ValueError, Exception) as e:
        # If any error occurs, remove the file
        print(f'Found corrupted cache file {cache_file}: {str(e)}')
        print(f'Removing {cache_file}...')
        os.remove(cache_file)
"

# Empty GPU cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('No CUDA available to empty cache')"

# Run with multi-GPU settings
echo "Running with multi-GPU configuration (using $NUM_GPUS GPUs)"
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train \
  --cluster

# Clean up the temporary config file
rm $TEMP_CONFIG_FILE

echo "Multi-GPU training completed!" 