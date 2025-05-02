#!/bin/bash
# Memory-efficient training script for Linux systems
# This script freezes the Prithvi backbone and uses other memory optimizations

# Set environment variables for better GPU performance and memory management
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print system info
echo "Starting memory-efficient training on Linux system..."
echo "Python version: $(python --version)"
echo "Using GPU for training with memory optimizations..."

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints
mkdir -p cache

# Create a temporary modified config file with memory-efficient settings
CONFIG_FILE="src/config/config.yaml"
TEMP_CONFIG_FILE="src/config/config_memory_efficient.yaml"

# Check if CUDA is available using PyTorch
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo "PyTorch CUDA is available, configuring for memory-efficient GPU training"
    
    # Create memory-efficient config
    python -c "
import yaml
import torch

# Load the base config
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Memory optimizations
config['model']['freeze_encoder'] = True
config['model']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['hardware']['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'
config['hardware']['pin_memory'] = torch.cuda.is_available()
config['training']['batch_size'] = 8
config['training']['accumulate_grad_batches'] = 4
config['training']['precision'] = 16

# Optimizer adjustments for stability
config['training']['optimizer']['lr'] = 1e-5
config['training']['gradient_clip_val'] = 0.5

# Save the memory-efficient config
with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
    
    # Print CUDA memory info
    python -c "import torch; print('CUDA Devices:', torch.cuda.device_count()); print('Current Device:', torch.cuda.current_device()); print('Device Name:', torch.cuda.get_device_name(0)); print('Memory Allocated:', torch.cuda.memory_allocated(0)/1024**3, 'GB')"
else
    echo "PyTorch CUDA not available, configuring for CPU training"
    cp $CONFIG_FILE $TEMP_CONFIG_FILE
fi

# Empty GPU cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('No CUDA available to empty cache')"

# Validate cache files
echo "Validating cache files..."
python -c "
import os
import numpy as np
import glob

cache_dir = 'cache'
cache_files = glob.glob(os.path.join(cache_dir, 'merra2_prism_*.npz'))
print(f'Checking {len(cache_files)} cache files...')

for cache_file in cache_files:
    try:
        data = np.load(cache_file, allow_pickle=True)
        if 'patches' not in data:
            print(f'Warning: Missing data in {cache_file}, removing...')
            os.remove(cache_file)
            continue
        _ = data['patches'].tolist()
        print(f'✓ Valid cache file: {os.path.basename(cache_file)}')
    except Exception as e:
        print(f'Found corrupted cache file {cache_file}: {str(e)}')
        print(f'Removing {cache_file}...')
        os.remove(cache_file)
"

# Start training with memory optimizations
echo "Starting memory-efficient training..."
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train \
  --memory_efficient

# Clean up the temporary config file
rm $TEMP_CONFIG_FILE

echo "Training completed!" 