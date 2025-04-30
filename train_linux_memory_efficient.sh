#!/bin/bash
# Memory-efficient training script for Linux systems with GPU

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
with open('$TEMP_CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
fi

# Empty GPU cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('No CUDA available to empty cache')"

# Run with memory efficient settings
echo "Running with memory-efficient configuration for 100 epochs"
python src/main.py \
  --config $TEMP_CONFIG_FILE \
  --mode train \
  --cluster \
  --memory_efficient

# Clean up the temporary config file
rm $TEMP_CONFIG_FILE

echo "Training completed!" 