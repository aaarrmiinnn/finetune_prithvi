"""PyTorch Lightning module for training the downscaling model."""
import torch
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union
import os
from pathlib import Path

from ..models import PrithviDownscaler, create_model
from ..utils.losses import DownscalingLoss
from ..utils.metrics import calculate_metrics


class DownscalingModule(pl.LightningModule):
    """PyTorch Lightning module for MERRA-2 to PRISM downscaling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the lightning module.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create model
        self.model = create_model(config)
        
        # Create loss function
        loss_config = config['loss']
        self.loss_fn = DownscalingLoss(
            mae_weight=loss_config['mae_weight'],
            mse_weight=loss_config['mse_weight'],
            ssim_weight=loss_config['ssim_weight']
        )
        
        # Store variable names for metrics
        self.input_vars = config['data']['input_vars']
        self.target_vars = config['data']['target_vars']
        
        # Initialize variables to track best metrics
        self.best_val_loss = float('inf')
        self.best_val_metrics = None
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Dictionary containing input tensors.
            
        Returns:
            Model output tensor.
        """
        merra2_input = x['merra2_input']
        dem = x.get('dem')  # Optional DEM input
        
        return self.model(merra2_input, dem)
    
    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary with loss.
        """
        # Extract data
        merra2_input = batch['merra2_input']
        prism_target = batch['prism_target']
        dem = batch.get('dem')  # Optional DEM input
        mask = batch.get('mask')  # Optional mask
        
        # Forward pass
        predictions = self.model(merra2_input, dem)
        
        # Calculate loss
        loss_dict = self.loss_fn(predictions, prism_target, mask)
        loss = loss_dict['loss']
        
        # Log losses
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/mae_loss', loss_dict['mae_loss'], on_step=False, on_epoch=True)
        self.log('train/mse_loss', loss_dict['mse_loss'], on_step=False, on_epoch=True)
        self.log('train/ssim_loss', loss_dict['ssim_loss'], on_step=False, on_epoch=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary with loss and predictions.
        """
        # Extract data
        merra2_input = batch['merra2_input']
        prism_target = batch['prism_target']
        dem = batch.get('dem')  # Optional DEM input
        mask = batch.get('mask')  # Optional mask
        
        # Forward pass
        predictions = self.model(merra2_input, dem)
        
        # Calculate loss
        loss_dict = self.loss_fn(predictions, prism_target, mask)
        loss = loss_dict['loss']
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, prism_target, self.target_vars)
        
        # Log losses
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mae_loss', loss_dict['mae_loss'], on_step=False, on_epoch=True)
        self.log('val/mse_loss', loss_dict['mse_loss'], on_step=False, on_epoch=True)
        self.log('val/ssim_loss', loss_dict['ssim_loss'], on_step=False, on_epoch=True)
        
        # Log overall metrics
        for metric_name, metric_value in metrics['overall'].items():
            self.log(f'val/metrics/{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        # Log per-variable metrics
        for var_name, var_metrics in metrics['variables'].items():
            for metric_name, metric_value in var_metrics.items():
                self.log(f'val/metrics/{var_name}/{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        return {
            'val_loss': loss,
            'predictions': predictions,
            'targets': prism_target,
            'metrics': metrics
        }
    
    def validation_epoch_end(self, outputs):
        """Validation epoch end hook.
        
        Args:
            outputs: List of outputs from validation_step.
        """
        # Calculate average loss
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        # Track best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            
            # Store best metrics
            # Since metrics are already logged per step, we don't need to average them again here
            # Just store the metrics from the last batch as a representative
            self.best_val_metrics = outputs[-1]['metrics']
            
            # Log best metrics
            self.log('val/best_loss', self.best_val_loss, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        """Test step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary with predictions and metrics.
        """
        # Extract data
        merra2_input = batch['merra2_input']
        prism_target = batch['prism_target']
        dem = batch.get('dem')  # Optional DEM input
        
        # Forward pass
        predictions = self.model(merra2_input, dem)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, prism_target, self.target_vars)
        
        # Log overall metrics
        for metric_name, metric_value in metrics['overall'].items():
            self.log(f'test/metrics/{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        # Log per-variable metrics
        for var_name, var_metrics in metrics['variables'].items():
            for metric_name, metric_value in var_metrics.items():
                self.log(f'test/metrics/{var_name}/{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        return {
            'predictions': predictions,
            'targets': prism_target,
            'metrics': metrics
        }
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Optimizer configuration.
        """
        # Get optimizer configuration
        training_config = self.config['training']
        optimizer_config = training_config['optimizer']
        scheduler_config = training_config['scheduler']
        
        # Create optimizer
        if optimizer_config['name'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
        
        # Create scheduler
        if scheduler_config['name'].lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config['epochs'] - scheduler_config['warmup_epochs']
            )
            
            # Wrap with warmup
            if scheduler_config['warmup_epochs'] > 0:
                from pytorch_lightning.callbacks import LearningRateMonitor
                
                # Create dummy scheduler for warmup
                # This is a simple linear warmup
                class WarmupScheduler:
                    def __init__(self, optimizer, warmup_epochs, base_lr):
                        self.optimizer = optimizer
                        self.warmup_epochs = warmup_epochs
                        self.base_lr = base_lr
                        self.current_epoch = 0
                    
                    def step(self, epoch=None):
                        if epoch is not None:
                            self.current_epoch = epoch
                        
                        # Linear warmup
                        lr_scale = min(1.0, (self.current_epoch + 1) / self.warmup_epochs)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.base_lr * lr_scale
                        
                        self.current_epoch += 1
                
                warmup_scheduler = WarmupScheduler(
                    optimizer,
                    scheduler_config['warmup_epochs'],
                    optimizer_config['lr']
                )
                
                # Return as a list of dictionaries
                return [
                    {
                        'optimizer': optimizer,
                        'lr_scheduler': {
                            'scheduler': warmup_scheduler if self.current_epoch < scheduler_config['warmup_epochs'] else scheduler,
                            'interval': 'epoch',
                            'frequency': 1
                        }
                    }
                ]
            
            # Return without warmup
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")
    
    def on_save_checkpoint(self, checkpoint):
        """Add extra information to checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary.
        """
        checkpoint['best_val_loss'] = self.best_val_loss
        checkpoint['best_val_metrics'] = self.best_val_metrics
        checkpoint['config'] = self.config 