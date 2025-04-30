"""Main script for running the MERRA-2 to PRISM downscaling pipeline."""
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch

from src.utils import load_config, setup_paths, get_experiment_name
from src.data import create_dataloaders
from src.trainers import DownscalingModule
from src.visualizations import visualize_predictions


def get_gpu_info():
    """Get GPU information."""
    gpu_info = {
        'available': torch.cuda.is_available(),
        'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'names': [],
        'total_memory_gb': 0.0
    }
    
    if gpu_info['available']:
        for i in range(gpu_info['count']):
            gpu_info['names'].append(torch.cuda.get_device_name(i))
            gpu_info['total_memory_gb'] += torch.cuda.get_device_properties(i).total_memory / (1024**3)
    
    return gpu_info


def configure_multi_gpu(config, num_gpus):
    """Configure training for multi-GPU setup.
    
    Args:
        config: Configuration dictionary.
        num_gpus: Number of GPUs available.
        
    Returns:
        Updated configuration dictionary.
    """
    # Configure for multi-GPU training
    config['hardware']['accelerator'] = 'gpu'
    config['hardware']['devices'] = num_gpus
    config['model']['device'] = 'cuda'
    
    # Set distributed training strategy
    config['cluster']['enabled'] = True
    config['cluster']['strategy'] = 'ddp'  # Use DistributedDataParallel
    config['cluster']['sync_batchnorm'] = True
    
    # Scale batch size and learning rate for multi-GPU training
    # Keep per-GPU batch size small for memory efficiency
    per_gpu_batch_size = max(1, min(4, config['training']['batch_size']))
    
    # Calculate effective batch size
    effective_batch_size = per_gpu_batch_size * num_gpus * config['training'].get('accumulate_grad_batches', 1)
    
    # Scale learning rate using square root scaling rule: lr = base_lr * sqrt(batch_size / base_batch_size)
    base_lr = config['training']['optimizer']['lr']
    base_batch_size = 1
    scaled_lr = base_lr * (effective_batch_size / base_batch_size) ** 0.5
    
    # Update config with the new values
    config['training']['batch_size'] = per_gpu_batch_size
    config['training']['optimizer']['lr'] = scaled_lr
    config['training']['precision'] = 16  # Use mixed precision for efficiency
    
    # Set data loader settings for efficiency
    config['hardware']['num_workers'] = min(4, os.cpu_count() // num_gpus)
    config['hardware']['pin_memory'] = True
    
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MERRA-2 to PRISM Downscaling Pipeline")
    
    parser.add_argument(
        "--config", type=str, default="src/config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "predict"], default="train",
        help="Mode to run: train, test, or predict"
    )
    
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for testing or prediction"
    )
    
    parser.add_argument(
        "--cluster", action="store_true",
        help="Enable cluster mode configuration"
    )
    
    parser.add_argument(
        "--memory_efficient", action="store_true",
        help="Enable memory-efficient settings to prevent CUDA OOM errors"
    )
    
    parser.add_argument(
        "--detect_anomaly", action="store_true",
        help="Enable PyTorch anomaly detection to help debug NaN values"
    )
    
    parser.add_argument(
        "--multi_gpu", action="store_true",
        help="Enable multi-GPU training with automatic configuration"
    )
    
    return parser.parse_args()


def train(config, cluster_mode=False, memory_efficient=False, detect_anomaly=False, multi_gpu=False):
    """Train the model.
    
    Args:
        config: Configuration dictionary.
        cluster_mode: Whether to use cluster-specific configurations.
        memory_efficient: Whether to use memory-efficient settings.
        detect_anomaly: Whether to enable PyTorch anomaly detection for debugging NaNs.
        multi_gpu: Whether to automatically configure for multi-GPU training.
    """
    # Empty CUDA cache before starting if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Emptied CUDA cache before training")
    
    # Enable anomaly detection if requested
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("PyTorch anomaly detection enabled - this will help debug NaN values but slow down training")
        
        # Also set deterministic mode for better debugging
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Set PyTorch to deterministic mode for better debugging")
    
    # Configure for multi-GPU if requested
    if multi_gpu:
        gpu_info = get_gpu_info()
        if gpu_info['available'] and gpu_info['count'] > 1:
            print(f"Multi-GPU mode enabled with {gpu_info['count']} GPUs")
            print(f"GPUs available: {', '.join(gpu_info['names'])}")
            print(f"Total GPU memory: {gpu_info['total_memory_gb']:.1f} GB")
            
            # Configure for multi-GPU training
            config = configure_multi_gpu(config, gpu_info['count'])
            
            # Set cluster mode to True since we're using multiple GPUs
            cluster_mode = True
            
            print(f"Configured for {gpu_info['count']} GPUs with batch size {config['training']['batch_size']} per GPU")
            print(f"Effective batch size: {config['training']['batch_size'] * gpu_info['count'] * config['training'].get('accumulate_grad_batches', 1)}")
            print(f"Scaled learning rate: {config['training']['optimizer']['lr']}")
        else:
            print(f"Multi-GPU mode requested but only {gpu_info['count']} GPUs available. Using single GPU.")
    
    # Update configuration for cluster if needed
    if cluster_mode:
        config['hardware']['accelerator'] = 'gpu'
        config['hardware']['num_workers'] = min(8, os.cpu_count())
        config['model']['device'] = 'cuda'
        config['training']['precision'] = 16
        config['training']['batch_size'] = 8  # Increase batch size on GPU
        config['cluster']['enabled'] = True
        config['logging']['wandb_notes'] = "Training run on GPU cluster"
        config['cluster'].setdefault('find_unused_parameters', False)
        print(f"Cluster mode enabled with device: {config['model']['device']}, accelerator: {config['hardware']['accelerator']}")
    
    # Apply memory-efficient settings if requested
    if memory_efficient:
        # Reduce model complexity
        config['model']['hidden_dim'] = min(16, config['model']['hidden_dim'])
        
        # Reduce batch size and increase gradient accumulation
        config['training']['batch_size'] = 1
        config['training']['accumulate_grad_batches'] = max(16, config['training'].get('accumulate_grad_batches', 1))
        
        # Enable mixed precision
        config['training']['precision'] = 16
        
        # Enable gradient checkpointing (saves memory at cost of computation)
        config.setdefault('model', {}).setdefault('gradient_checkpointing', True)
        
        # Reduce patch size for less memory usage
        config['data']['patch_size'] = min(8, config['data']['patch_size'])
        
        # Set optimized dataloader settings
        config['hardware']['num_workers'] = 2
        config['hardware']['pin_memory'] = True
        
        print("Memory-efficient settings applied:")
        print(f"- Model hidden dim: {config['model']['hidden_dim']}")
        print(f"- Batch size: {config['training']['batch_size']}")
        print(f"- Gradient accumulation: {config['training']['accumulate_grad_batches']}")
        print(f"- Precision: {config['training']['precision']}")
        print(f"- Patch size: {config['data']['patch_size']}")
        print(f"- Gradient checkpointing: {config['model'].get('gradient_checkpointing', False)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        train_split=config['data']['train_test_split'][0],
        val_split=config['data']['train_test_split'][1],
        test_split=config['data']['train_test_split'][2],
        num_workers=config['hardware']['num_workers']
    )
    
    # Create Lightning module
    model = DownscalingModule(config)
    
    # Create experiment name
    experiment_name = get_experiment_name(config)
    
    # Set up logging
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config['logging']['save_dir'],
        name=config['logging']['name'],
        version=config['logging']['version'],
        log_graph=config['logging']['log_graph']
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger
    if config['logging']['wandb']:
        wandb_logger = WandbLogger(
            project=config['logging']['wandb_project'],
            entity=config['logging']['wandb_entity'],
            name=experiment_name,
            tags=config['logging']['wandb_tags'],
            notes=config['logging']['wandb_notes'],
            log_model=True,
            save_dir=config['logging']['save_dir']
        )
        loggers.append(wandb_logger)
        
        # Log hyperparameters explicitly
        wandb_logger.log_hyperparams(config)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/loss',
            dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
            filename=f'{experiment_name}-{{epoch:02d}}-{{val/loss:.4f}}',
            save_top_k=config['training']['save_top_k'],
            mode='min'
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=10,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Create trainer with appropriate configuration
    trainer_kwargs = {
        'max_epochs': config['training']['epochs'],
        'accelerator': config['hardware']['accelerator'],
        'devices': config['hardware']['devices'],
        'precision': config['training']['precision'],
        'callbacks': callbacks,
        'logger': loggers,
        'log_every_n_steps': config['training']['log_every_n_steps'],
        'val_check_interval': config['training']['val_check_interval'],
        'gradient_clip_val': config['training']['gradient_clip_val'],
        'accumulate_grad_batches': config['training'].get('accumulate_grad_batches', 1),
    }
    
    # Add cluster-specific configurations if enabled
    if cluster_mode or config['cluster']['enabled']:
        # Make sure all required keys exist with safe defaults
        config['cluster'].setdefault('strategy', 'auto')
        config['cluster'].setdefault('sync_batchnorm', True)
        config['cluster'].setdefault('find_unused_parameters', False)
        
        trainer_kwargs.update({
            'strategy': config['cluster']['strategy'],
            'sync_batchnorm': config['cluster']['sync_batchnorm'],
        })
        
        # Only add find_unused_parameters if strategy is ddp (newer PyTorch Lightning versions don't support this)
        if config['cluster']['strategy'] == 'ddp':
            trainer_kwargs.update({
                'find_unused_parameters': config['cluster']['find_unused_parameters'],
            })
        
        print(f"Using cluster settings: {config['cluster']}")
    
    # Add checkpoint for resuming if specified
    if config['training']['resume_from_checkpoint']:
        trainer_kwargs['resume_from_checkpoint'] = config['training']['resume_from_checkpoint']
    
    # Create trainer
    print(f"Using trainer parameters: {trainer_kwargs}")
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, test_loader)
    
    return model, trainer


def test(config, checkpoint_path, cluster_mode=False):
    """Test the model.
    
    Args:
        config: Configuration dictionary.
        checkpoint_path: Path to model checkpoint.
        cluster_mode: Whether to use cluster-specific configurations.
    """
    # Update configuration for cluster if needed
    if cluster_mode:
        config['hardware']['accelerator'] = 'gpu'
        config['hardware']['num_workers'] = min(8, os.cpu_count())
        config['model']['device'] = 'cuda'
        config['training']['precision'] = 16
        config['cluster']['enabled'] = True
        config['cluster'].setdefault('find_unused_parameters', False)
    
    # Create data loaders
    _, _, test_loader = create_dataloaders(
        config,
        train_split=config['data']['train_test_split'][0],
        val_split=config['data']['train_test_split'][1],
        test_split=config['data']['train_test_split'][2],
        num_workers=config['hardware']['num_workers']
    )
    
    # Load model from checkpoint
    model = DownscalingModule.load_from_checkpoint(checkpoint_path)
    
    # Create logger for testing
    loggers = []
    if config['logging']['wandb']:
        experiment_name = get_experiment_name(config) + "_test"
        wandb_logger = WandbLogger(
            project=config['logging']['wandb_project'],
            entity=config['logging']['wandb_entity'],
            name=experiment_name,
            tags=config['logging']['wandb_tags'] + ["testing"],
            log_model=False,
            save_dir=config['logging']['save_dir']
        )
        loggers.append(wandb_logger)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['training']['precision'],
        logger=loggers if loggers else None
    )
    
    # Test the model
    results = trainer.test(model, test_loader)
    
    # Visualize predictions
    for batch in test_loader:
        # Move batch to appropriate device
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Make predictions
        with torch.no_grad():
            predictions = model(batch)
        
        # Visualize
        save_dir = os.path.join(config['logging']['save_dir'], 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        visualize_predictions(batch, predictions, config, save_dir=save_dir)
        
        # Log images to wandb if enabled
        if config['logging']['wandb'] and loggers:
            try:
                import wandb
                # Implement wandb image logging here if needed
                # This would depend on the specific format of your visualizations
            except ImportError:
                print("wandb not available for image logging")
        
        # Just process one batch for visualization
        break
    
    return results


def predict(config, checkpoint_path, cluster_mode=False):
    """Make predictions with the model.
    
    Args:
        config: Configuration dictionary.
        checkpoint_path: Path to model checkpoint.
        cluster_mode: Whether to use cluster-specific configurations.
    """
    # This is similar to test, but would typically be used for new data
    # For the purpose of this example, we'll just use the test data
    return test(config, checkpoint_path, cluster_mode)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up paths and validate configuration
    config = setup_paths(config)
    
    # Ensure cluster section exists
    if 'cluster' not in config:
        config['cluster'] = {'enabled': False, 'strategy': 'auto', 'sync_batchnorm': True, 'find_unused_parameters': False}
    
    # Run selected mode
    if args.mode == "train":
        model, trainer = train(config, cluster_mode=args.cluster, memory_efficient=args.memory_efficient, detect_anomaly=args.detect_anomaly, multi_gpu=args.multi_gpu)
    elif args.mode == "test":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for test mode")
        results = test(config, args.checkpoint, cluster_mode=args.cluster)
    elif args.mode == "predict":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for predict mode")
        results = predict(config, args.checkpoint, cluster_mode=args.cluster)


if __name__ == "__main__":
    main() 