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

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    echo "CUDA found, configuring for GPU training"
    # Update the hardware and cluster settings for GPU
    sed -i 's/accelerator: "cpu"/accelerator: "gpu"/' $TEMP_CONFIG_FILE
    sed -i 's/devices: 1/devices: 1/' $TEMP_CONFIG_FILE
    sed -i 's/precision: 32/precision: 16/' $TEMP_CONFIG_FILE
    sed -i 's/find_unused_parameters: false/# find_unused_parameters disabled for compatibility/' $TEMP_CONFIG_FILE
    sed -i 's/strategy: "ddp"/strategy: "auto"/' $TEMP_CONFIG_FILE
    
    # Most importantly: Update the device setting for model to match accelerator
    sed -i 's/device: "cpu"/device: "cuda"/' $TEMP_CONFIG_FILE
    
    # Enable pin_memory for faster data transfer to GPU
    sed -i 's/pin_memory: false/pin_memory: true/' $TEMP_CONFIG_FILE
    
    # Increase number of workers for better GPU utilization
    sed -i 's/num_workers: 0/num_workers: 2/' $TEMP_CONFIG_FILE
else
    echo "CUDA not found, configuring for CPU training"
    # Keep accelerator as CPU and make sure model device also uses CPU
    sed -i 's/accelerator: "gpu"/accelerator: "cpu"/' $TEMP_CONFIG_FILE
    sed -i 's/device: "cuda"/device: "cpu"/' $TEMP_CONFIG_FILE
    sed -i 's/find_unused_parameters: false/# find_unused_parameters disabled for compatibility/' $TEMP_CONFIG_FILE
    sed -i 's/strategy: "ddp"/strategy: "auto"/' $TEMP_CONFIG_FILE
    
    # Disable pin_memory for CPU training
    sed -i 's/pin_memory: true/pin_memory: false/' $TEMP_CONFIG_FILE
fi

# Run the training script with optimizations
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train

# Clean up the temporary config file
rm $TEMP_CONFIG_FILE

echo "Training completed!" 