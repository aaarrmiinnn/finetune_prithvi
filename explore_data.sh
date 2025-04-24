#!/bin/bash

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

# Run the explore data script
echo "Running the data exploration script..."
python src/scripts/explore_data.py "$@" 