# Weights & Biases Integration

This guide explains how to use Weights & Biases (W&B) for experiment tracking with the Prithvi Downscaler model.

## Overview

Weights & Biases is an experiment tracking tool that helps you:
- Track model training metrics in real-time
- Compare different runs with different parameters
- Visualize model performance
- Share results with collaborators

## Setup

1. Install the wandb package (included in requirements.txt):
   ```bash
   pip install wandb
   ```

2. Sign up for a free account at [wandb.ai](https://wandb.ai) if you don't have one

3. Log in to your account:
   ```bash
   wandb login
   ```

## Configuration

The W&B integration is configured in the `src/config/config.yaml` file:

```yaml
logging:
  # ...other logging settings...
  wandb: true                      # Enable Weights & Biases logging
  wandb_project: "merra2-prism-downscaling"  # W&B project name
  wandb_entity: null               # Your W&B username or team name
  wandb_tags: ["prithvi", "downscaling", "merra2", "prism"]  # Tags for the run
  wandb_notes: "Initial training run on local machine"  # Notes for the run
```

## Using train_wandb.sh

The `train_wandb.sh` script provides a convenient way to run training with W&B integration and custom parameters:

```bash
./train_wandb.sh [options]
```

### Options

- `--cluster`: Run in cluster mode (configures for GPU)
- `--resume PATH`: Resume from checkpoint PATH
- `--entity NAME`: Set W&B entity (username or team name)
- `--project NAME`: Set W&B project name
- `--name NAME`: Set run name in W&B
- `--epochs N`: Set number of epochs
- `--batch N`: Set batch size
- `--help`: Show help message

### Examples

1. Basic training with W&B:
   ```bash
   ./train_wandb.sh
   ```

2. Training with custom entity and project:
   ```bash
   ./train_wandb.sh --entity "your-username" --project "prithvi-experiments"
   ```

3. Resuming from a checkpoint:
   ```bash
   ./train_wandb.sh --resume "logs/merra2-prism-downscaling/version_1/checkpoints/epoch=10.ckpt"
   ```

4. Custom training parameters:
   ```bash
   ./train_wandb.sh --name "experiment-1" --epochs 50 --batch 2
   ```

5. Cluster training:
   ```bash
   ./train_wandb.sh --cluster --project "cluster-runs"
   ```

## Tracked Metrics

The following metrics are automatically tracked in W&B:

1. **Training Metrics**:
   - train_loss: Combined training loss
   - train_mae: Mean Absolute Error on training data
   - train_mse: Mean Squared Error on training data

2. **Validation Metrics**:
   - val_mae: Mean Absolute Error on validation data
   - val_mse: Mean Squared Error on validation data
   - val_rmse: Root Mean Squared Error on validation data

3. **Testing Metrics** (when applicable):
   - test_mae: Mean Absolute Error on test data
   - test_mse: Mean Squared Error on test data
   - test_rmse: Root Mean Squared Error on test data

## Viewing Results

1. During or after training, navigate to [wandb.ai](https://wandb.ai)
2. Select your project (default: "merra2-prism-downscaling")
3. View your runs, metrics, and visualizations

## Advanced Usage

### Offline Mode

If you don't have internet access during training, you can use W&B in offline mode:

```bash
export WANDB_MODE="offline"
./train_wandb.sh
```

Later, you can sync the results:

```bash
wandb sync ./wandb/offline-run-*
```

### Custom Logging

You can log additional custom metrics in your code:

```python
import wandb

# Log a metric
wandb.log({"custom_metric": value})

# Log an image
wandb.log({"prediction": wandb.Image(image_array)})
```

### Hyperparameter Sweeps

For advanced hyperparameter optimization, you can use W&B Sweeps:

1. Create a sweep configuration file `sweep.yaml`:
   ```yaml
   program: src/main.py
   method: bayes
   metric:
     name: val_rmse
     goal: minimize
   parameters:
     learning_rate:
       min: 0.0001
       max: 0.01
     hidden_dim:
       values: [128, 256, 512]
     batch_size:
       values: [1, 2, 4]
   ```

2. Initialize the sweep:
   ```bash
   wandb sweep sweep.yaml
   ```

3. Run the sweep agent:
   ```bash
   wandb agent SWEEP_ID
   ``` 