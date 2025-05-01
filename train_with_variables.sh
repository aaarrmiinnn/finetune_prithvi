#!/bin/bash
# Renamed from train_separate_variables.sh to train_with_variables.sh
# This script allows training with one or more climate variables

# Set environment variables for better GPU performance
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints
mkdir -p cache

# Function to validate cache files
validate_cache() {
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
}

# Generate modified config with selected variables
generate_config() {
    local variables=$1
    local temp_config="src/config/temp_runtime_config.yaml"
    
    # Start with the base config
    cp src/config/config.yaml $temp_config
    
    # Update target_vars based on selection
    python -c "
import yaml

# Load the config
with open('src/config/temp_runtime_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update target variables based on input
variables = '$variables'.split(',')
config['data']['target_vars'] = variables

# Update experiment name based on variables
var_names = '_'.join(variables)
config['logging']['name'] = f'merra2-prism-{var_names}-downscaling'
config['logging']['wandb_tags'].extend(variables)
config['logging']['wandb_notes'] = f'Downscaling model for variables: {variables}'

# If we're training only one variable, apply variable-specific settings
if len(variables) == 1:
    var_map = {'tdmean': 'temperature', 'ppt': 'precipitation'}
    var_type = var_map.get(variables[0], 'common')
    
    # Apply model settings
    if var_type in config['model']['variable_config']:
        for key, value in config['model']['variable_config'][var_type].items():
            config['model'][key] = value
    
    # Apply training settings
    if var_type in config['training']['variable_config']:
        for key, value in config['training']['variable_config'][var_type].items():
            if isinstance(value, dict):
                # Handle nested dictionaries like optimizer settings
                if key not in config['training']:
                    config['training'][key] = {}
                for k, v in value.items():
                    config['training'][key][k] = v
            else:
                config['training'][key] = value
    
    # Apply loss settings
    if var_type in config['loss']:
        for key, value in config['loss'][var_type].items():
            config['loss'][key] = value
else:
    # Apply common settings for multiple variables
    var_type = 'common'
    
    # Apply model settings
    if var_type in config['model']['variable_config']:
        for key, value in config['model']['variable_config'][var_type].items():
            config['model'][key] = value
    
    # Apply training settings
    if var_type in config['training']['variable_config']:
        for key, value in config['training']['variable_config'][var_type].items():
            if isinstance(value, dict):
                # Handle nested dictionaries like optimizer settings
                if key not in config['training']:
                    config['training'][key] = {}
                for k, v in value.items():
                    config['training'][key][k] = v
            else:
                config['training'][key] = value
    
    # Apply loss settings
    if var_type in config['loss']:
        for key, value in config['loss'][var_type].items():
            config['loss'][key] = value

# Save the modified config
with open('src/config/temp_runtime_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f'Created config for variables: {variables}')
"
    echo $temp_config
}

# Function to train with specific variables
train_with_variables() {
    local variables=$1
    local config_file=$(generate_config "$variables")
    
    echo "============================================================"
    echo "Starting training for variables: $variables"
    echo "Using config: $config_file"
    echo "============================================================"
    
    # Empty GPU cache before starting
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('No CUDA available to empty cache')"
    
    # Validate cache files
    validate_cache
    
    # Start training
    echo "Running training..."
    python src/main.py \
        --config $config_file \
        --mode train \
        --cluster \
        --detect_anomaly
    
    # Clean up temp config
    rm $config_file
    
    echo "Training completed for variables: $variables"
}

# Parse command line arguments
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --variables=VAR1[,VAR2,...]  Train specific variables (tdmean, ppt)"
    echo "  --temperature               Train only temperature (tdmean)"
    echo "  --precipitation             Train only precipitation (ppt)"
    echo "  --all                       Train all variables (default)"
    echo "  --help                      Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --temperature            Train temperature model"
    echo "  $0 --precipitation          Train precipitation model"
    echo "  $0 --variables=tdmean,ppt   Train with specified variables"
    echo "  $0 --all                    Train with all variables"
}

# Default variables to train
VARIABLES="tdmean,ppt"

# Parse arguments
if [ $# -eq 0 ]; then
    # Default: train all variables
    VARIABLES="tdmean,ppt"
else
    for arg in "$@"; do
        case $arg in
            --variables=*)
                VARIABLES="${arg#*=}"
                ;;
            --temperature)
                VARIABLES="tdmean"
                ;;
            --precipitation)
                VARIABLES="ppt"
                ;;
            --all)
                VARIABLES="tdmean,ppt"
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
echo "Variables: $VARIABLES"
echo ""

# Start training
train_with_variables "$VARIABLES"

echo "Training completed!" 