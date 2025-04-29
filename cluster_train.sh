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