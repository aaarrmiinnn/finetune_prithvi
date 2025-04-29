#!/bin/bash

# Set environment variables for Mac MPS memory optimization
if [[ "$(uname)" == "Darwin"* ]]; then
  # Increase MPS memory allocation on Mac
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment if it exists
if command -v conda &> /dev/null && conda info --envs | grep -q "prithvi"; then
    echo "Activating conda environment 'prithvi'..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate prithvi
fi

# Set Python path to include the project
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training with default configuration
echo "Starting training..."
python src/main.py "$@" 