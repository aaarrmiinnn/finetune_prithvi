#!/bin/bash
# Simple training script for Linux systems with GPU

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# Add the current directory to Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print system info
echo "Starting training on Linux system..."
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "Using GPU for training"

# Create directories if they don't exist
mkdir -p logs
mkdir -p models/checkpoints
mkdir -p cache

# Create a temporary modified config file that removes the problematic parameter
CONFIG_FILE="src/config/config.yaml"
TEMP_CONFIG_FILE="src/config/config_linux_temp.yaml"

# Create a copy of the config file with modified cluster settings
cp $CONFIG_FILE $TEMP_CONFIG_FILE

# Update the hardware and cluster settings
sed -i 's/accelerator: "cpu"/accelerator: "gpu"/' $TEMP_CONFIG_FILE
sed -i 's/devices: 1/devices: 1/' $TEMP_CONFIG_FILE
sed -i 's/precision: 32/precision: 16/' $TEMP_CONFIG_FILE
sed -i 's/find_unused_parameters: false/# find_unused_parameters disabled for compatibility/' $TEMP_CONFIG_FILE
sed -i 's/strategy: "ddp"/strategy: "auto"/' $TEMP_CONFIG_FILE

# Run the training script with GPU optimizations
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train

# Clean up the temporary config file
rm $TEMP_CONFIG_FILE

echo "Training completed!" 