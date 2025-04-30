#!/bin/bash
# Memory-efficient training script for Linux systems with GPU
# This script freezes the Prithvi backbone and only trains adapter layers

# Set environment variables for better GPU performance and memory management
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print system info
echo "Starting memory-efficient training on Linux system..."
echo "Python version: $(python --version)"
echo "Using GPU for training with memory optimizations..."
echo "NOTE: This script freezes the Prithvi backbone and only trains adapter layers"

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
    
    # Use Python to update the config file safely (instead of multiple sed commands)
    python update_config.py --input $CONFIG_FILE --output $TEMP_CONFIG_FILE
    
    # Add additional stability measures for NaN prevention
    python -c "
import yaml
with open('$TEMP_CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
# Set an even lower learning rate with longer warmup
config['training']['optimizer']['lr'] = 1e-6  # Lower learning rate
config['training']['scheduler']['warmup_epochs'] = 5  # Longer warmup
config['training']['gradient_clip_val'] = 0.5  # Lower gradient clipping
# Add data validation checks
config['data']['validate_data'] = True  # Enable data validation
config['data']['clip_extreme_values'] = True  # Clip extreme values
config['data']['replace_nan_with_mean'] = True  # Replace NaNs with mean values
# Add NaN detection during training
config['training']['detect_anomaly'] = True  # Enable PyTorch anomaly detection
with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
    
    # Print CUDA info for debugging
    python -c "import torch; print('CUDA Devices:', torch.cuda.device_count()); print('Current Device:', torch.cuda.current_device()); print('Device Name:', torch.cuda.get_device_name(0)); print('Memory Allocated:', torch.cuda.memory_allocated(0)/1024**3, 'GB'); print('Memory Reserved:', torch.cuda.memory_reserved(0)/1024**3, 'GB'); print('Max Memory Allocated:', torch.cuda.max_memory_allocated(0)/1024**3, 'GB')"
else
    echo "PyTorch CUDA not available, configuring for CPU training"
    # Create a CPU-focused config
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['hardware']['accelerator'] = 'cpu'
config['model']['device'] = 'cpu'
config['training']['epochs'] = 100
config['model']['freeze_encoder'] = True  # Make sure encoder is frozen
with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
fi

# Pre-check for NaNs in the dataset
echo "Validating dataset for NaN values before training..."
python -c "
import os
import sys
import numpy as np
import yaml
import glob

# Load config
with open('$TEMP_CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Check MERRA2 files
merra2_dir = config['data']['merra2_dir']
merra2_files = glob.glob(os.path.join(merra2_dir, '*.nc4'))
print(f'Checking {len(merra2_files)} MERRA2 files...')

has_nan = False
try:
    import xarray as xr
    for f in merra2_files:
        try:
            data = xr.open_dataset(f)
            for var in config['data']['input_vars']:
                if var in data:
                    if np.isnan(data[var].values).any():
                        print(f'WARNING: NaN values found in {var} in file {f}')
                        has_nan = True
        except Exception as e:
            print(f'Error checking file {f}: {str(e)}')
except ImportError:
    print('xarray not available, skipping NaN check')

if has_nan:
    print('WARNING: NaN values found in input data. This may cause training instability.')
    print('Consider preprocessing the data to replace or filter NaN values.')
else:
    print('No NaN values found in input data.')
"

# Empty GPU cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('No CUDA available to empty cache')"

# Run with memory efficient settings and anomaly detection
echo "Running with memory-efficient configuration for 100 epochs (with frozen backbone)"
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train \
  --cluster \
  --memory_efficient \
  --detect_anomaly  # Add anomaly detection flag

# Clean up the temporary config file
rm $TEMP_CONFIG_FILE

echo "Training completed!" 