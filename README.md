# Prithvi Downscaler

A deep learning model for downscaling MERRA2 climate data to PRISM resolution using IBM's Prithvi-WxC-1.0-2300M model.

## Overview

This project implements a transformer-based downscaling approach to increase the spatial resolution of MERRA2 global climate data to match PRISM's regional 4km resolution. The model leverages the pretrained Prithvi-WxC model from IBM and NASA, incorporating topographical information through Digital Elevation Model (DEM) data.

## Features

- Deep learning-based downscaling of climate variables (temperature and precipitation)
- Integration of terrain information through DEM data
- Configurable model architecture and training parameters
- Mixed precision training for efficiency
- Platform-specific optimizations for Mac (CPU) and Linux (GPU)
- Memory-efficient training with frozen encoder backbone
- Robust NaN detection and handling during training
- Comprehensive logging with TensorBoard and Weights & Biases

## Requirements

- Python 3.8+
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

The project includes memory optimization strategies:

- **Memory-efficient training** with frozen Prithvi backbone
- Reduced hidden dimension (16-32)
- Smaller batch size with gradient accumulation
- Mixed precision training (16-bit)
- Gradient checkpointing for memory efficiency
- NaN detection and handling for training stability

## Running Scripts

### Platform-Specific Training Scripts

The project provides platform-specific training scripts:

```bash
# For Linux systems with NVIDIA GPUs - standard training
./train_linux.sh

# For Linux systems with NVIDIA GPUs - memory-efficient training
./train_linux_memory_efficient.sh

# For Linux systems with NVIDIA GPUs - multi-GPU distributed training (NEW)
./train_linux_multi_gpu.sh

# For Mac systems (optimized for CPU)
./train_mac.sh
```

### Memory-Efficient Training

The memory-efficient training script is recommended for most use cases:

```bash
# Run memory-efficient training on Linux
./train_linux_memory_efficient.sh
```

This script:
- Freezes the Prithvi backbone to reduce memory usage
- Applies aggressive memory optimizations
- Uses gradient checkpointing and accumulation
- Includes NaN detection and handling
- Automatically enables Weights & Biases logging
- Pre-validates data for NaN values before training
- Configures optimal training parameters

### Multi-GPU Distributed Training

For systems with multiple GPUs, the multi-GPU training script enables distributed training:

```bash
# Run distributed training across all available GPUs
./train_linux_multi_gpu.sh
```

This script:
- Automatically detects and uses all available GPUs
- Configures distributed data parallelism (DDP)
- Scales batch size and learning rate appropriately
- Synchronizes batch normalization across devices
- Distributes data efficiently across GPUs
- Provides significant training speed improvements
- Enables training with larger effective batch sizes

For fine-grained control, you can also use the Python command directly:

```bash
# Enable multi-GPU training with the --multi_gpu flag
python src/main.py --config src/config/config.yaml --mode train --multi_gpu
```

### Direct Python Execution

You can also run Python directly with custom arguments:

```bash
# Basic training
python src/main.py --config src/config/config.yaml --mode train

# Memory-efficient training
python src/main.py --config src/config/config.yaml --mode train --memory_efficient

# GPU cluster mode
python src/main.py --config src/config/config.yaml --mode train --cluster

# Enable NaN detection
python src/main.py --config src/config/config.yaml --mode train --detect_anomaly
```

## Monitoring

Training progress can be monitored using:

- TensorBoard:
  ```bash
  tensorboard --logdir logs
  ```

- Weights & Biases (if enabled):
  Monitor at https://wandb.ai/YOUR_USERNAME/prithvi-downscaling

## Model Outputs

The trained model will produce:
- Checkpoints in the `logs/` directory
- Performance metrics (MAE, MSE, RMSE)
- Per-variable metrics for temperature and precipitation
- NaN occurrence tracking
- Visualization plots (if enabled)

## NaN Handling and Debugging

The model includes robust NaN detection and handling:

- Automatic NaN detection in inputs, outputs, and losses
- NaN replacement to prevent training failures
- Detailed logging of tensor statistics when NaNs are detected
- Reduced frequency NaN warnings to avoid console flooding
- PyTorch anomaly detection for better debugging
- W&B logging of NaN occurrence counts

## Troubleshooting

### Memory Issues

If you encounter memory errors:
1. Use the memory-efficient training script (`train_linux_memory_efficient.sh`)
2. Reduce batch size further in config.yaml
3. Decrease hidden_dim in config.yaml
4. Reduce patch_size in config.yaml

### NaN Issues

If you encounter NaN values during training:
1. Check your input data for NaN values
2. Reduce learning rate in config.yaml
3. Enable robust NaN detection with `--detect_anomaly`
4. Try setting `replace_nan_with_mean: True` in your config

### GDAL Installation Issues

If GDAL installation fails:
1. Install via conda first: `conda install -c conda-forge gdal`
2. Then install other requirements: `pip install -r requirements.txt`

## Version History

### v1.1.0
- Added memory-efficient training with frozen backbone
- Implemented robust NaN detection and handling
- Added Weights & Biases integration for experiment tracking
- Optimized training scripts for platform-specific execution
- Streamlined codebase by removing unnecessary scripts
- Fixed data loading and preprocessing issues

### v1.0.0
- Initial implementation of downscaling model
- Basic data processing functionality
- PyTorch Lightning integration
- Multi-variable prediction

## License
[Your License Here] 