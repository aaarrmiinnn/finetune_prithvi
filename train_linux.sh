#!/bin/bash
# Simple training script for Linux systems with GPU

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
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

# Create a temporary modified config file that removes the problematic parameter
CONFIG_FILE="src/config/config.yaml"
TEMP_CONFIG_FILE="src/config/config_linux_temp.yaml"

# Create a copy of the config file with modified cluster settings
cp $CONFIG_FILE $TEMP_CONFIG_FILE

# Check if CUDA is available using PyTorch
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo "PyTorch CUDA is available, configuring for GPU training"
    # Update the hardware and cluster settings for GPU
    python -c "
import yaml
with open('$TEMP_CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# GPU hardware settings
config['hardware']['accelerator'] = 'gpu'
config['hardware']['devices'] = 1
config['hardware']['pin_memory'] = True
config['hardware']['num_workers'] = 2
config['model']['device'] = 'cuda'

# Training settings
config['training']['precision'] = 16
config['training']['batch_size'] = 4
config['training']['accumulate_grad_batches'] = 4
config['model']['freeze_encoder'] = True  # Keep backbone frozen for stability

# Cluster settings
config['cluster']['enabled'] = True
config['cluster']['strategy'] = 'auto'
config['cluster']['find_unused_parameters'] = False

# Enable Weights & Biases logging
config['logging']['wandb'] = True
config['logging']['wandb_project'] = 'prithvi-downscaling'
config['logging']['wandb_tags'] = ['single-gpu', 'frozen-backbone']
config['logging']['wandb_notes'] = 'Single GPU training run with Weights & Biases tracking'

with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
    
    # Print CUDA info for debugging
    python -c "
import torch
print(f'CUDA Devices: {torch.cuda.device_count()}')
print(f'Current Device: {torch.cuda.current_device()}')
print(f'Device Name: {torch.cuda.get_device_name(0)}')
print(f'Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
else
    echo "PyTorch CUDA not available, configuring for CPU training"
    # Keep accelerator as CPU and make sure model device also uses CPU
    python -c "
import yaml
with open('$TEMP_CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# CPU hardware settings
config['hardware']['accelerator'] = 'cpu'
config['hardware']['devices'] = 1
config['hardware']['pin_memory'] = False
config['hardware']['num_workers'] = 0
config['model']['device'] = 'cpu'

# Training settings (reduced for CPU)
config['training']['precision'] = 32
config['training']['batch_size'] = 1
config['training']['accumulate_grad_batches'] = 4
config['model']['freeze_encoder'] = True

# Cluster settings
config['cluster']['enabled'] = False
config['cluster']['strategy'] = 'auto'
config['cluster']['find_unused_parameters'] = False

# Enable Weights & Biases logging
config['logging']['wandb'] = True
config['logging']['wandb_project'] = 'prithvi-downscaling'
config['logging']['wandb_tags'] = ['cpu', 'debugging']
config['logging']['wandb_notes'] = 'CPU training run with Weights & Biases tracking'

with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
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

# Run with updated config
echo "Running with --cluster flag to ensure GPU usage"
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train \
  --cluster

# Clean up the temporary config file
rm $TEMP_CONFIG_FILE

echo "Training completed!" 