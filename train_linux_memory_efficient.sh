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

# Create a copy of the config file
cp $CONFIG_FILE $TEMP_CONFIG_FILE

# Check if CUDA is available using PyTorch
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo "PyTorch CUDA is available, configuring for memory-efficient GPU training"
    
    # Hardware and precision settings
    sed -i 's/accelerator: "cpu"/accelerator: "gpu"/' $TEMP_CONFIG_FILE
    sed -i 's/devices: 1/devices: 1/' $TEMP_CONFIG_FILE
    sed -i 's/precision: 32/precision: 16/' $TEMP_CONFIG_FILE
    sed -i 's/device: "cpu"/device: "cuda"/' $TEMP_CONFIG_FILE
    
    # Memory efficiency settings
    sed -i 's/batch_size: [0-9]*/batch_size: 1/' $TEMP_CONFIG_FILE
    sed -i 's/accumulate_grad_batches: [0-9]*/accumulate_grad_batches: 16/' $TEMP_CONFIG_FILE
    sed -i 's/num_workers: [0-9]*/num_workers: 2/' $TEMP_CONFIG_FILE
    sed -i 's/pin_memory: false/pin_memory: true/' $TEMP_CONFIG_FILE
    
    # Model size reduction settings
    sed -i 's/hidden_dim: [0-9]*/hidden_dim: 16/' $TEMP_CONFIG_FILE
    sed -i 's/patch_size: [0-9]*/patch_size: 8/' $TEMP_CONFIG_FILE
    
    # Training settings - increase to 100 epochs
    sed -i 's/epochs: [0-9]*/epochs: 100/' $TEMP_CONFIG_FILE
    
    # Gradient checkpointing to save memory
    echo "gradient_checkpointing: true" >> $TEMP_CONFIG_FILE
    
    # Cluster settings
    sed -i 's/enabled: false/enabled: true/' $TEMP_CONFIG_FILE
    sed -i 's/strategy: "ddp"/strategy: "auto"/' $TEMP_CONFIG_FILE
    
    # Print CUDA info for debugging
    python -c "import torch; print('CUDA Devices:', torch.cuda.device_count()); print('Current Device:', torch.cuda.current_device()); print('Device Name:', torch.cuda.get_device_name(0)); print('Memory Allocated:', torch.cuda.memory_allocated(0)/1024**3, 'GB'); print('Memory Reserved:', torch.cuda.memory_reserved(0)/1024**3, 'GB'); print('Max Memory Allocated:', torch.cuda.max_memory_allocated(0)/1024**3, 'GB')"
else
    echo "PyTorch CUDA not available, configuring for CPU training"
    sed -i 's/accelerator: "gpu"/accelerator: "cpu"/' $TEMP_CONFIG_FILE
    sed -i 's/device: "cuda"/device: "cpu"/' $TEMP_CONFIG_FILE
    sed -i 's/epochs: [0-9]*/epochs: 100/' $TEMP_CONFIG_FILE
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