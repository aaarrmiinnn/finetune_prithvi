#!/bin/bash
# Training script for separate temperature and precipitation models
# This script allows training one or both models

# Set environment variables for better GPU performance
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints/temperature
mkdir -p models/checkpoints/precipitation
mkdir -p cache

# Function to train a specific variable model
train_model() {
    local variable=$1
    local config_file="src/config/${variable}_config.yaml"
    
    echo "============================================================"
    echo "Starting training for ${variable} model using ${config_file}"
    echo "============================================================"
    
    # Empty GPU cache before starting
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('No CUDA available to empty cache')"
    
    # Run validation on cache files
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
    
    # Start training
    echo "Running training with config: ${config_file}"
    python src/main.py \
        --config ${config_file} \
        --mode train \
        --cluster \
        --detect_anomaly
    
    # Copy the best model to a dedicated directory
    echo "Training completed for ${variable} model"
}

# Parse command line arguments
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --temperature    Train only the temperature model"
    echo "  --precipitation  Train only the precipitation model"
    echo "  --all            Train both models (default)"
    echo "  --help           Display this help message"
}

# By default, train both models
TRAIN_TEMPERATURE=false
TRAIN_PRECIPITATION=false

# Parse arguments
if [ $# -eq 0 ]; then
    # Default: train both models
    TRAIN_TEMPERATURE=true
    TRAIN_PRECIPITATION=true
else
    for arg in "$@"; do
        case $arg in
            --temperature)
                TRAIN_TEMPERATURE=true
                ;;
            --precipitation)
                TRAIN_PRECIPITATION=true
                ;;
            --all)
                TRAIN_TEMPERATURE=true
                TRAIN_PRECIPITATION=true
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done
fi

# Print training plan
echo "Training plan:"
if [ "$TRAIN_TEMPERATURE" = true ]; then
    echo "- Temperature model: YES"
else
    echo "- Temperature model: NO"
fi

if [ "$TRAIN_PRECIPITATION" = true ]; then
    echo "- Precipitation model: YES"
else
    echo "- Precipitation model: NO"
fi

echo ""

# Train the models
if [ "$TRAIN_TEMPERATURE" = true ]; then
    train_model "temperature"
fi

if [ "$TRAIN_PRECIPITATION" = true ]; then
    train_model "precipitation"
fi

echo "All requested training completed!" 