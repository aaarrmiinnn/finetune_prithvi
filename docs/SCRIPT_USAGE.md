# Script Usage Guide

This guide explains the usage of the scripts provided in the Prithvi Downscaler project, with a focus on the recommended consolidated script `prithvi.sh`.

## Consolidated Script: `prithvi.sh`

The `prithvi.sh` script is a unified command-line interface that combines all the functionality of the individual scripts into one convenient tool. It's recommended to use this script for all operations as it provides a consistent interface and eliminates the need to remember multiple script names.

### Basic Usage

```bash
./prithvi.sh [command] [options]
```

### Available Commands

- `train`: Run model training
- `explore`: Run data exploration and visualization
- `cluster`: Submit a training job to a cluster (with SLURM)

### Common Options

- `--help`: Display help information
- `--install`: Create and set up the conda environment if it doesn't exist
- `--device DEVICE`: Force a specific device type: 'cuda', 'mps', or 'cpu'

### Training Options

When using the `train` command, the following options are available:

- `--wandb`: Enable Weights & Biases tracking
- `--entity NAME`: Set W&B entity (username or team name)
- `--project NAME`: Set W&B project name
- `--name NAME`: Set run name in W&B
- `--epochs N`: Set number of epochs
- `--batch N`: Set batch size
- `--resume PATH`: Resume from checkpoint PATH

### Platform Detection

The script automatically detects your hardware platform and configures the appropriate settings:

- **macOS with M1/M2**: Uses MPS (Metal Performance Shaders) acceleration
- **Linux with NVIDIA GPU**: Uses CUDA GPU acceleration
- **Any platform without GPU**: Falls back to CPU

You can override the automatic detection using the `--device` option.

### Examples

1. **Show help information**:
   ```bash
   ./prithvi.sh --help
   ```

2. **Basic training with automatic platform detection**:
   ```bash
   ./prithvi.sh train
   ```

3. **Force using CUDA GPU (useful for multi-GPU systems)**:
   ```bash
   ./prithvi.sh train --device cuda
   ```

4. **Force using CPU (even if GPU is available)**:
   ```bash
   ./prithvi.sh train --device cpu
   ```

5. **Data exploration**:
   ```bash
   ./prithvi.sh explore
   ```

6. **Training with W&B tracking**:
   ```bash
   ./prithvi.sh train --wandb --entity "your-username" --project "your-project"
   ```

7. **Resuming training from a checkpoint on a specific device**:
   ```bash
   ./prithvi.sh train --resume logs/merra2-prism-downscaling/version_1/checkpoints/epoch=10.ckpt --device cuda
   ```

8. **Training on a cluster**:
   ```bash
   ./prithvi.sh cluster
   ```

9. **Training with environment setup**:
   ```bash
   ./prithvi.sh train --install
   ```

## Hardware-Specific Optimizations

The script applies different optimizations based on the detected hardware:

### CUDA (NVIDIA GPU)
- Uses mixed precision (16-bit) for faster training
- Automatically increases batch size for better GPU utilization
- Sets appropriate CUDA environment variables
- Uses the GPU accelerator in PyTorch Lightning

### MPS (Mac M1/M2)
- Sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` for better memory management
- Keeps smaller batch sizes to avoid memory issues
- Uses the MPS accelerator in PyTorch Lightning

### CPU
- Configures for CPU-only operation
- Adjusts parameters for CPU efficiency

## Individual Scripts

While the consolidated script is recommended, the project also provides individual scripts for specific purposes:

- **train.sh**: Basic training script optimized for Mac M1/M2
- **run.sh**: Training with environment setup (creates conda env if needed)
- **explore_data.sh**: Data exploration and visualization
- **train_wandb.sh**: Training with Weights & Biases integration
- **cluster_train.sh**: Training on a cluster with SLURM

## Why Use the Consolidated Script?

1. **Simplicity**: One script to remember instead of five
2. **Consistency**: Uniform command structure and options
3. **Completeness**: All functionality available in one place
4. **Maintainability**: Easier to update a single script than multiple ones
5. **Flexibility**: Combines options from all scripts
6. **Advanced Features**: Automatic detection of hardware and environment
7. **Cross-Platform**: Works across macOS, Linux, and different hardware configurations

## Script Dependencies

The consolidated script handles all necessary dependencies:

- Python 3.8
- Conda environment management
- PYTHONPATH setting
- Hardware detection and configuration
- Directory creation

## Troubleshooting

### Common Issues

1. **MPS-related errors on Linux**:
   - The script now automatically detects the platform and avoids MPS-specific settings on Linux
   - You can force a specific device with `--device cuda` or `--device cpu`

2. **CUDA not detected**:
   - Ensure NVIDIA drivers are installed and working
   - Try running `nvidia-smi` to verify GPU accessibility
   - Use `--device cuda` to force CUDA if detection fails

3. **Environment Issues**:
   - Use the `--install` option to create and set up the conda environment

## Migration from Individual Scripts

If you've been using the individual scripts, here's how to migrate to the consolidated script:

| Old Script | New Command |
|------------|-------------|
| `./train.sh` | `./prithvi.sh train` |
| `./run.sh` | `./prithvi.sh train --install` |
| `./explore_data.sh` | `./prithvi.sh explore` |
| `./train_wandb.sh --entity user` | `./prithvi.sh train --wandb --entity user` |
| `./cluster_train.sh` | `./prithvi.sh cluster` | 