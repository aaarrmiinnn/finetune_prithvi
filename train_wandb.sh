#!/bin/bash

# Script to run training with Weights & Biases logging

# Display help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --cluster     Run in cluster mode (configures for GPU)"
    echo "  --resume PATH Resume from checkpoint PATH"
    echo "  --entity NAME Set wandb entity (username or team name)"
    echo "  --project NAME Set wandb project name (default: merra2-prism-downscaling)"
    echo "  --name NAME   Set run name in wandb"
    echo "  --epochs N    Set number of epochs"
    echo "  --batch N     Set batch size"
    echo "  --help        Show this help message"
    exit 0
fi

# Set environment variables for Mac MPS memory optimization
if [[ "$(uname)" == "Darwin"* ]]; then
    # Increase MPS memory allocation on Mac
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
fi

# Create logs directory if it doesn't exist
mkdir -p logs cache models/cache

# Activate conda environment if it exists
if command -v conda &> /dev/null && conda info --envs | grep -q "prithvi"; then
    echo "Activating conda environment 'prithvi'..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate prithvi
fi

# Default values
CLUSTER_MODE=""
RESUME_ARG=""
RUN_NAME=""
EPOCHS=""
BATCH_SIZE=""
WANDB_ENTITY=""
WANDB_PROJECT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER_MODE="--cluster"
            shift
            ;;
        --resume)
            RESUME_ARG="--checkpoint $2"
            shift 2
            ;;
        --entity)
            WANDB_ENTITY=$2
            shift 2
            ;;
        --project)
            WANDB_PROJECT=$2
            shift 2
            ;;
        --name)
            RUN_NAME=$2
            shift 2
            ;;
        --epochs)
            EPOCHS=$2
            shift 2
            ;;
        --batch)
            BATCH_SIZE=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set Python path to include the project
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configure wandb (login if needed)
if ! python -c "import wandb; wandb.login(anonymous='never')" > /dev/null 2>&1; then
    echo "Please log in to Weights & Biases:"
    python -c "import wandb; wandb.login()"
fi

# Create a temporary config file with any custom settings
CONFIG_FILE="src/config/config_temp.yaml"
cp src/config/config.yaml $CONFIG_FILE

# Update config if entity is provided
if [[ -n "$WANDB_ENTITY" ]]; then
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['logging']['wandb_entity'] = '$WANDB_ENTITY'
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f)
"
fi

# Update config if project is provided
if [[ -n "$WANDB_PROJECT" ]]; then
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['logging']['wandb_project'] = '$WANDB_PROJECT'
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f)
"
fi

# Update config if run name is provided
if [[ -n "$RUN_NAME" ]]; then
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['logging']['name'] = '$RUN_NAME'
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f)
"
fi

# Update config if epochs is provided
if [[ -n "$EPOCHS" ]]; then
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['training']['epochs'] = $EPOCHS
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f)
"
fi

# Update config if batch size is provided
if [[ -n "$BATCH_SIZE" ]]; then
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['training']['batch_size'] = $BATCH_SIZE
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f)
"
fi

# Run the training script
echo "Starting training with Weights & Biases logging"
echo "Config: $CONFIG_FILE"
echo "Cluster mode: ${CLUSTER_MODE:-disabled}"
echo "Resume from: ${RESUME_ARG:-N/A}"

python src/main.py --config $CONFIG_FILE --mode train $CLUSTER_MODE $RESUME_ARG

# Clean up temp config
rm -f $CONFIG_FILE 