"""Main script for running the MERRA-2 to PRISM downscaling pipeline."""
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from utils import load_config, setup_paths, get_experiment_name
from data import create_dataloaders
from trainers import DownscalingModule
from visualizations import visualize_predictions


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
    
    return parser.parse_args()


def train(config):
    """Train the model.
    
    Args:
        config: Configuration dictionary.
    """
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
    logger = TensorBoardLogger(
        save_dir=config['logging']['save_dir'],
        name=config['logging']['name'],
        version=config['logging']['version'],
        log_graph=config['logging']['log_graph']
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/loss',
            dirpath=os.path.join(logger.log_dir, 'checkpoints'),
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
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['training']['precision'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['training']['log_every_n_steps'],
        val_check_interval=config['training']['val_check_interval'],
        gradient_clip_val=config['training']['gradient_clip_val']
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, test_loader)
    
    return model, trainer


def test(config, checkpoint_path):
    """Test the model.
    
    Args:
        config: Configuration dictionary.
        checkpoint_path: Path to model checkpoint.
    """
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
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['training']['precision']
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
        visualize_predictions(batch, predictions, config, save_dir=save_dir)
        
        # Just process one batch for visualization
        break
    
    return results


def predict(config, checkpoint_path):
    """Make predictions with the model.
    
    Args:
        config: Configuration dictionary.
        checkpoint_path: Path to model checkpoint.
    """
    # This is similar to test, but would typically be used for new data
    # For the purpose of this example, we'll just use the test data
    return test(config, checkpoint_path)


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
        model, trainer = train(config)
    elif args.mode == "test":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for test mode")
        results = test(config, args.checkpoint)
    elif args.mode == "predict":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for predict mode")
        results = predict(config, args.checkpoint)


if __name__ == "__main__":
    main() 