"""Training script for PrithviDownscaler model."""
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from argparse import ArgumentParser
from pathlib import Path

from src.models.prithvi_downscaler import PrithviDownscalerModule
from src.data.datamodule import MERRA2PRISMDataModule

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                      help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint for resuming training')
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create data module
    datamodule = MERRA2PRISMDataModule(config)
    
    # Create model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = PrithviDownscalerModule.load_from_checkpoint(args.checkpoint)
    else:
        print("Creating new model")
        model = PrithviDownscalerModule(config)
    
    # Configure logging
    log_dir = Path(config['logging']['save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    loggers = []
    if config['logging']['tensorboard']:
        tensorboard_logger = TensorBoardLogger(
            save_dir=str(log_dir),
            name=config['logging']['name'],
            version=config['logging']['version'],
            log_graph=config['logging']['log_graph']
        )
        loggers.append(tensorboard_logger)
    
    if config['logging']['wandb']:
        wandb_logger = WandbLogger(
            project=config['logging']['wandb_project'],
            name=config['logging']['name'],
            save_dir=str(log_dir)
        )
        loggers.append(wandb_logger)
    
    # Configure callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(log_dir / 'checkpoints'),
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=config['training']['save_top_k'],
        monitor='val_loss',
        mode='min'
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        val_check_interval=config['training']['val_check_interval'],
        callbacks=callbacks,
        logger=loggers if loggers else True
    )
    
    # Train model
    trainer.fit(model, datamodule=datamodule)
    
    # Test model
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    main() 