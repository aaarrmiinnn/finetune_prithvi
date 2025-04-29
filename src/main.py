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
    
    return parser.parse_args()


def train(config, cluster_mode=False):
    """Train the model.
    
    Args:
        config: Configuration dictionary.
        cluster_mode: Whether to use cluster-specific configurations.
    """
    # Update configuration for cluster if needed
    if cluster_mode:
        config['hardware']['accelerator'] = 'gpu'
        config['hardware']['num_workers'] = min(8, os.cpu_count())
        config['model']['device'] = 'cuda'
        config['training']['precision'] = 16
        config['training']['batch_size'] = 8  # Increase batch size on GPU
        config['cluster']['enabled'] = True
        config['logging']['wandb_notes'] = "Training run on GPU cluster"
    
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
    }
    
    # Add cluster-specific configurations if enabled
    if cluster_mode or config['cluster']['enabled']:
        trainer_kwargs.update({
            'strategy': config['cluster']['strategy'],
            'sync_batchnorm': config['cluster']['sync_batchnorm'],
            'find_unused_parameters': config['cluster']['find_unused_parameters'],
        })
    
    # Add checkpoint for resuming if specified
    if config['training']['resume_from_checkpoint']:
        trainer_kwargs['resume_from_checkpoint'] = config['training']['resume_from_checkpoint']
    
    # Create trainer
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
    
    # Run selected mode
    if args.mode == "train":
        model, trainer = train(config, cluster_mode=args.cluster)
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