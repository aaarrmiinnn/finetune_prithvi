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
mkdir -p cache

# Create a temporary modified config file with Mac-specific settings
CONFIG_FILE="src/config/config.yaml"
TEMP_CONFIG_FILE="src/config/config_mac_temp.yaml"

# Create a copy of the config file
cp $CONFIG_FILE $TEMP_CONFIG_FILE

# Update hardware settings if needed (CPU is already the default in our config)
# No need to modify the existing settings as they're already optimized for Mac

# Run the training script with Mac-specific optimizations
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train

# Clean up the temporary config file
rm $TEMP_CONFIG_FILE

echo "Training completed!" 