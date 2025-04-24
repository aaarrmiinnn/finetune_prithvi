#!/bin/bash

# Create cache and logs directories
mkdir -p cache logs

# Check if conda environment exists
if ! conda info --envs | grep -q "prithvi"; then
    echo "Creating conda environment 'prithvi'..."
    conda create -n prithvi python=3.8 -y
    conda activate prithvi
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Using existing conda environment 'prithvi'..."
    conda activate prithvi
fi

# Run the main script
echo "Running the training pipeline..."
python src/main.py "$@" 