#!/bin/bash

# Consolidated script for Prithvi Downscaler project
# Combines functionality of train.sh, run.sh, train_wandb.sh, explore_data.sh, and cluster_train.sh

# Display help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  train              Run basic training"
    echo "  explore            Run data exploration and visualization"
    echo "  cluster            Run training on a cluster with SLURM"
    echo ""
    echo "Options for 'train' command:"
    echo "  --wandb            Enable Weights & Biases tracking"
    echo "  --entity NAME      Set wandb entity (username or team name)"
    echo "  --project NAME     Set wandb project name"
    echo "  --name NAME        Set run name in wandb"
    echo "  --epochs N         Set number of epochs"
    echo "  --batch N          Set batch size"
    echo "  --resume PATH      Resume from checkpoint PATH"
    echo "  --install          Create conda environment if it doesn't exist"
    echo ""
    echo "Examples:"
    echo "  $0 train                           # Basic training"
    echo "  $0 train --wandb                   # Training with W&B"
    echo "  $0 train --wandb --entity user     # W&B with custom entity"
    echo "  $0 explore                         # Data exploration"
    echo "  $0 cluster                         # Train on cluster"
    exit 0
fi

# Set environment variables for Mac MPS memory optimization
if [[ "$(uname)" == "Darwin"* ]]; then
    # Increase MPS memory allocation on Mac
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
fi

# Create necessary directories
mkdir -p logs cache models/cache

# Parse the main command
COMMAND=${1:-train}  # Default to train if no command provided
shift || true

# Handle conda environment setup
SETUP_ENV=false
for arg in "$@"; do
    if [[ "$arg" == "--install" ]]; then
        SETUP_ENV=true
        break
    fi
done

# Set up conda environment if requested or needed
if [[ "$SETUP_ENV" == "true" ]]; then
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
else
    # Activate conda environment if it exists
    if command -v conda &> /dev/null && conda info --envs | grep -q "prithvi"; then
        echo "Activating conda environment 'prithvi'..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate prithvi
    fi
fi

# Set Python path to include the project
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Process command
case "$COMMAND" in
    train)
        # Check if W&B mode is enabled
        USE_WANDB=false
        for arg in "$@"; do
            if [[ "$arg" == "--wandb" ]]; then
                USE_WANDB=true
                break
            fi
        done
        
        if [[ "$USE_WANDB" == "true" ]]; then
            # Remove --wandb from arguments
            ARGS=()
            for arg in "$@"; do
                if [[ "$arg" != "--wandb" && "$arg" != "--install" ]]; then
                    ARGS+=("$arg")
                fi
            done
            
            # Configure wandb (login if needed)
            if ! python -c "import wandb; wandb.login(anonymous='never')" > /dev/null 2>&1; then
                echo "Please log in to Weights & Biases:"
                python -c "import wandb; wandb.login()"
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
            i=0
            while [[ $i -lt ${#ARGS[@]} ]]; do
                case "${ARGS[$i]}" in
                    --cluster)
                        CLUSTER_MODE="--cluster"
                        ((i++))
                        ;;
                    --resume)
                        RESUME_ARG="--checkpoint ${ARGS[$((i+1))]}"
                        ((i+=2))
                        ;;
                    --entity)
                        WANDB_ENTITY="${ARGS[$((i+1))]}"
                        ((i+=2))
                        ;;
                    --project)
                        WANDB_PROJECT="${ARGS[$((i+1))]}"
                        ((i+=2))
                        ;;
                    --name)
                        RUN_NAME="${ARGS[$((i+1))]}"
                        ((i+=2))
                        ;;
                    --epochs)
                        EPOCHS="${ARGS[$((i+1))]}"
                        ((i+=2))
                        ;;
                    --batch)
                        BATCH_SIZE="${ARGS[$((i+1))]}"
                        ((i+=2))
                        ;;
                    *)
                        echo "Unknown option: ${ARGS[$i]}"
                        exit 1
                        ;;
                esac
            done
            
            # Create a temporary config file with any custom settings
            CONFIG_FILE="src/config/config_temp.yaml"
            cp src/config/config.yaml $CONFIG_FILE
            
            # Update config based on parameters
            python - <<EOF
import yaml

# Load the original config
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Enable W&B
config['logging']['wandb'] = True

# Update entity if provided
if '$WANDB_ENTITY':
    config['logging']['wandb_entity'] = '$WANDB_ENTITY'

# Update project if provided
if '$WANDB_PROJECT':
    config['logging']['wandb_project'] = '$WANDB_PROJECT'

# Update run name if provided
if '$RUN_NAME':
    config['logging']['name'] = '$RUN_NAME'

# Update epochs if provided
if '$EPOCHS':
    config['training']['epochs'] = int('$EPOCHS')

# Update batch size if provided
if '$BATCH_SIZE':
    config['training']['batch_size'] = int('$BATCH_SIZE')

# Save the updated config
with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f)
EOF
            
            # Run the training script with W&B
            echo "Starting training with Weights & Biases logging"
            echo "Config: $CONFIG_FILE"
            echo "Cluster mode: ${CLUSTER_MODE:-disabled}"
            echo "Resume from: ${RESUME_ARG:-N/A}"
            
            python src/main.py --config $CONFIG_FILE $CLUSTER_MODE $RESUME_ARG
            
            # Clean up temp config
            rm -f $CONFIG_FILE
            
        else
            # Basic training mode
            echo "Starting training..."
            python src/main.py "$@"
        fi
        ;;
        
    explore)
        # Run data exploration
        echo "Running data exploration..."
        python src/scripts/explore_data.py "$@"
        ;;
        
    cluster)
        # Check if this is a SLURM environment
        if command -v sbatch &> /dev/null; then
            # Create a SLURM script
            SLURM_SCRIPT="tmp_slurm_script.sh"
            cat > $SLURM_SCRIPT << 'EOF'
#!/bin/bash
#SBATCH --job-name=prithvi_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on your cluster setup)
echo "Loading modules..."
module purge
module load anaconda3 cuda/11.7 cudnn

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prithvi

# Set Python path to include the project
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training with cluster mode enabled
echo "Starting training in cluster mode..."
python src/main.py --cluster
EOF
            
            # Submit the job
            echo "Submitting SLURM job..."
            sbatch $SLURM_SCRIPT
            rm -f $SLURM_SCRIPT
        else
            # If not in a SLURM environment, run with --cluster flag
            echo "Running in cluster mode (not using SLURM)..."
            python src/main.py --cluster "$@"
        fi
        ;;
        
    *)
        echo "Unknown command: $COMMAND"
        echo "Run '$0 --help' for usage information"
        exit 1
        ;;
esac 