# Model Configuration and Training Guide

This guide provides details on configuring and training the Prithvi Downscaler model.

## Configuration Overview

The model configuration is managed through a YAML file located at `src/config/config.yaml`. This file controls all aspects of data preprocessing, model architecture, training parameters, and hardware utilization.

## Configuration Sections

### Data Configuration

```yaml
data:
  merra2_dir: "data/merra2"        # Directory containing MERRA2 NetCDF4 files
  prism_dir: "data/prism"          # Directory containing PRISM data
  dem_dir: "data/dem"              # Directory containing DEM data
  dem_file: "PRISM_us_dem_4km_bil.bil"  # DEM filename
  dates: ["20250302"]              # List of dates to process
  spatial_extent: null             # Geographical bounds [min_lon, min_lat, max_lon, max_lat] or null for full domain
  cache_dir: "cache"               # Directory for caching preprocessed data
  patch_size: 32                   # Size of image patches for training
  input_vars: ["T2MMAX", "T2MMEAN", "T2MMIN", "TPRECMAX"]  # MERRA2 input variables
  target_vars: ["tdmean", "ppt"]   # PRISM target variables
  mask_ratio: 0.3                  # Ratio of pixels to mask during training
  train_test_split: [0.7, 0.15, 0.15]  # Train/validation/test split ratios
```

### Model Configuration

```yaml
model:
  # Prithvi model configuration
  prithvi_checkpoint: "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M"  # Model name or path
  cache_dir: "models/cache"        # Directory to cache downloaded models
  use_pretrained: true             # Whether to use pretrained weights
  device: "mps"                    # Device to use: "mps" for Mac M1/M2, "cuda" for NVIDIA GPU, "cpu" for CPU
  freeze_encoder: false            # Whether to freeze encoder layers during training
  
  # Architecture configuration
  hidden_dim: 128                  # Hidden dimension size (reduced for memory efficiency)
```

### Training Configuration

```yaml
training:
  epochs: 30                       # Number of training epochs
  batch_size: 1                    # Batch size (reduced for memory efficiency)
  optimizer:
    name: "AdamW"                  # Optimizer name
    lr: 1.0e-4                     # Learning rate
    weight_decay: 0.01             # Weight decay factor
  scheduler:
    name: "cosine"                 # Learning rate scheduler type
    warmup_epochs: 3               # Warmup epochs for scheduler
  precision: 32                    # Precision (16 for mixed precision, 32 for full precision)
  gradient_clip_val: 1.0           # Gradient clipping value
  log_every_n_steps: 1             # Logging frequency
  val_check_interval: 1.0          # Validation check interval
  save_top_k: 3                    # Number of best checkpoints to save
  resume_from_checkpoint: null     # Path to checkpoint for resuming training
```

### Loss Configuration

```yaml
loss:
  mae_weight: 1.0                  # Weight for Mean Absolute Error loss
  mse_weight: 0.5                  # Weight for Mean Squared Error loss
  ssim_weight: 0.2                 # Weight for Structural Similarity Index Measure loss
```

### Logging Configuration

```yaml
logging:
  save_dir: "logs"                 # Directory to save logs
  name: "merra2-prism-downscaling" # Experiment name
  version: null                    # Version (null for auto-increment)
  log_graph: true                  # Whether to log model graph
  tensorboard: true                # Enable TensorBoard logging
  wandb: true                      # Enable Weights & Biases logging
  wandb_project: "merra2-prism-downscaling"  # W&B project name
  wandb_entity: null               # W&B username or team name
  wandb_tags: ["prithvi", "downscaling", "merra2", "prism"]  # W&B tags
  wandb_notes: "Initial training run on local machine"  # W&B run notes
```

### Hardware Configuration

```yaml
hardware:
  accelerator: "mps"               # Accelerator type: "mps", "gpu", "cpu"
  devices: 1                       # Number of devices to use
  num_workers: 1                   # Number of data loading workers
  pin_memory: true                 # Whether to pin memory for GPU transfers
```

### Cluster Configuration

```yaml
cluster:
  enabled: false                   # Whether to use cluster settings
  distributed_backend: "ddp"       # Distributed data parallel backend
  sync_batchnorm: true             # Synchronize batch normalization
  strategy: "ddp"                  # Training strategy
  find_unused_parameters: false    # Whether to find unused parameters in DDP
```

## Memory Optimization

For systems with limited memory (like Mac M1/M2), the following settings are recommended:

1. **Reduce model size**:
   ```yaml
   model:
     hidden_dim: 128  # Reduced from 256
   ```

2. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 1    # Reduced from 2
   ```

3. **Reduce patch size**:
   ```yaml
   data:
     patch_size: 32   # Reduced from 64
   ```

4. **Use mixed precision** (on supported hardware):
   ```yaml
   training:
     precision: 16    # Use 16-bit precision instead of 32
   ```

## Training the Model

### Local Training

1. **Basic training command**:
   ```bash
   python src/main.py
   ```

2. **Custom configuration**:
   ```bash
   python src/main.py --config path/to/custom/config.yaml
   ```

3. **Using a training script**:
   ```bash
   ./train.sh
   ```

### Cluster Training

1. **Enable cluster mode**:
   ```yaml
   cluster:
     enabled: true
   ```

2. **Run with cluster script**:
   ```bash
   ./cluster_train.sh
   ```
   
3. **Manual SLURM submission**:
   ```bash
   sbatch -J prithvi_train -p gpu --gres=gpu:1 -t 24:00:00 -o logs/slurm-%j.out -e logs/slurm-%j.err ./train.sh
   ```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs
```

### Weights & Biases

1. **Login to W&B**:
   ```bash
   wandb login
   ```

2. **View runs**:
   - Visit: https://wandb.ai/YOUR_USERNAME/merra2-prism-downscaling

## Resuming Training

To resume from a checkpoint:

1. **Update configuration**:
   ```yaml
   training:
     resume_from_checkpoint: "logs/merra2-prism-downscaling/version_X/checkpoints/epoch=N.ckpt"
   ```

2. **Run training command**:
   ```bash
   python src/main.py
   ```

## Debugging Memory Issues

If you encounter memory errors:

1. **Enable memory logging** (NVIDIA GPUs):
   ```bash
   CUDA_LAUNCH_BLOCKING=1 python src/main.py
   ```

2. **Increase MPS memory limit** (Mac M1/M2):
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 
   python src/main.py
   ```

3. **Monitor memory usage**:
   ```bash
   # For NVIDIA GPUs
   watch -n 0.5 nvidia-smi
   
   # For Mac M1/M2
   top -o mem
   ```

## Troubleshooting Training

### Common Issues and Solutions

1. **Out of Memory Errors**:
   - Reduce batch size
   - Reduce model hidden dimension
   - Reduce patch size
   - Enable mixed precision training

2. **Slow Training**:
   - Increase number of workers
   - Enable pin_memory
   - Use smaller patch size
   - Use mixed precision training

3. **Convergence Issues**:
   - Adjust learning rate
   - Modify loss weights
   - Increase training epochs
   - Check data preprocessing

4. **NaN Loss Values**:
   - Enable gradient clipping
   - Reduce learning rate
   - Check for data normalization issues 