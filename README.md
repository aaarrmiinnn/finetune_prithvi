# Prithvi Downscaler

A deep learning model for downscaling MERRA2 climate data to PRISM resolution using IBM's Prithvi-WxC-1.0-2300M model.

## Overview

This project implements a transformer-based downscaling approach to increase the spatial resolution of MERRA2 global climate data to match PRISM's regional 4km resolution. The model leverages the pretrained Prithvi-WxC model from IBM and NASA, incorporating topographical information through Digital Elevation Model (DEM) data.

## Features

- Deep learning-based downscaling of climate variables (temperature and precipitation)
- Integration of terrain information through DEM data
- Configurable model architecture and training parameters
- Mixed precision training for efficiency
- Optimized for Mac M1/M2 using MPS backend
- Comprehensive logging with TensorBoard and Weights & Biases

## Requirements

- Python 3.8
- PyTorch 2.0+
- PyTorch Lightning
- Hugging Face Transformers
- NetCDF4
- GDAL
- NumPy
- Pandas
- Weights & Biases (optional)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd finetune_prithvi
   ```

2. Create a conda environment:
   ```bash
   conda create -n prithvi python=3.8
   conda activate prithvi
   ```

3. Install GDAL using conda (recommended) before other dependencies:
   ```bash
   conda install -c conda-forge gdal
   ```

4. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### Data Directory Structure

The project expects the following data directory structure:
```
data/
├── merra2/           # MERRA2 NetCDF4 files
├── prism/            # PRISM data in zip or extracted format
└── dem/              # Digital Elevation Model data (BIL format)
```

### MERRA2 Data

1. Download MERRA2 data from NASA GES DISC in NetCDF4 format
2. Place the files in the `data/merra2` directory
3. Files should contain variables: T2MMAX, T2MMEAN, T2MMIN, TPRECMAX

### PRISM Data

1. Download PRISM data from PRISM Climate Group
2. Place the zip or extracted files in the `data/prism` directory
3. Files should contain variables: tdmean (temperature), ppt (precipitation)

### DEM Data

1. Download the Digital Elevation Model data in BIL format
2. Place the file in the `data/dem` directory
3. The default expected filename is `PRISM_us_dem_4km_bil.bil`

## Configuration

The model and training parameters are configured through YAML files in `src/config/`:

```bash
# Edit the configuration file as needed
nano src/config/config.yaml
```

Key configuration sections include:
- Data configuration (paths, variables, patch size)
- Model configuration (hidden dimension, pretrained weights)
- Training configuration (batch size, learning rate, optimizer)
- Hardware configuration (accelerator, devices)

### Memory Optimization

For Mac M1/M2 users, the configuration is already optimized with:
- Reduced hidden dimension (128)
- Smaller batch size (1)
- MPS backend for GPU acceleration
- Smaller patch size (32)

For systems with more memory, you can increase these values for better performance.

## Running Scripts

### Consolidated Script (Recommended)

The project provides a single consolidated script `prithvi.sh` that combines all functionality:

```bash
# Show available commands and options
./prithvi.sh --help

# Basic training
./prithvi.sh train

# Data exploration
./prithvi.sh explore

# Training with Weights & Biases
./prithvi.sh train --wandb

# Resume training from a checkpoint with W&B
./prithvi.sh train --wandb --resume logs/path/to/checkpoint.ckpt

# Training on a cluster
./prithvi.sh cluster
```

This consolidated script also handles environment setup:
```bash
# Setup conda environment if needed and run training
./prithvi.sh train --install
```

### Individual Scripts

The project also includes several specialized shell scripts if needed:

```bash
# Explore and visualize the input data
./explore_data.sh

# Basic training script with memory optimization for Mac
./train.sh

# Full setup and training (creates conda env if needed)
./run.sh

# Training with Weights & Biases tracking and custom parameters
./train_wandb.sh [options]

# Cluster training with SLURM integration
./cluster_train.sh
```

The `train_wandb.sh` script provides additional options:
```
Options:
  --cluster           Run in cluster mode (configures for GPU)
  --resume PATH       Resume from checkpoint PATH
  --entity NAME       Set wandb entity (username or team name)
  --project NAME      Set wandb project name
  --name NAME         Set run name in wandb
  --epochs N          Set number of epochs
  --batch N           Set batch size
  --help              Show this help message
```

You can also run Python directly:

```bash
python src/main.py
```

This will:
1. Load and preprocess the data
2. Create and initialize the model
3. Train the model with the configured parameters
4. Save checkpoints and logs

## Monitoring

Training progress can be monitored using:

- TensorBoard:
  ```bash
  tensorboard --logdir logs
  ```

- Weights & Biases (if enabled):
  Monitor at https://wandb.ai/YOUR_USERNAME/merra2-prism-downscaling

## Model Outputs

The trained model will produce:
- Checkpoints in the `logs/` directory
- Performance metrics (MAE, MSE, RMSE)
- Visualization plots (if enabled)

## Troubleshooting

### Memory Issues

If you encounter memory errors:
1. Reduce batch size in config.yaml
2. Decrease hidden_dim in config.yaml
3. Reduce patch_size in config.yaml
4. Use mixed precision training (set precision to 16)

For MPS backend memory issues on Mac:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
```

### GDAL Installation Issues

If GDAL installation fails:
1. Install via conda first: `conda install -c conda-forge gdal`
2. Then install other requirements: `pip install -r requirements.txt`

## License

[Specify your license here]

## Acknowledgments

- IBM and NASA for the Prithvi-WxC model
- MERRA2 and PRISM for climate data 