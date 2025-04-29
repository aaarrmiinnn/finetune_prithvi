"""PyTorch Lightning module for training the downscaling model."""
import torch
import pytorch_lightning as pl
from typing import Dict, List, Any, Optional, Union
import os
from pathlib import Path
import logging
import numpy as np

from src.models import PrithviDownscaler, create_model
from src.utils.losses import DownscalingLoss
from src.utils.metrics import calculate_metrics

# Set up logger
logger = logging.getLogger(__name__)

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
        
        # Initialize counter for NaN losses
        self.nan_loss_count = 0
        self.max_nan_losses = config.get('training', {}).get('max_nan_losses', 10)
        
        # Save last good values for debug
        self.last_good_input = None
        self.last_good_output = None
        self.last_good_target = None
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Dictionary containing input tensors.
            
        Returns:
            Model output tensor.
        """
        merra2_input = x['merra2_input']
        dem = x.get('dem')  # Optional DEM input
        
        # Check for NaN in input - new
        if torch.isnan(merra2_input).any() or (dem is not None and torch.isnan(dem).any()):
            logger.warning(f"NaN values in input tensors! merra2_input: {torch.isnan(merra2_input).sum().item()}, dem: {torch.isnan(dem).sum().item() if dem is not None else 'N/A'}")
            # Replace NaNs with zeros
            merra2_input = torch.nan_to_num(merra2_input, nan=0.0)
            if dem is not None:
                dem = torch.nan_to_num(dem, nan=0.0)
        
        return self.model(merra2_input, dem)
    
    def preprocess_batch(self, batch):
        """Preprocess batch to handle potential NaNs and infinites.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Preprocessed batch with NaNs and infinites replaced
        """
        # Extract data
        merra2_input = batch['merra2_input']
        prism_target = batch['prism_target']
        dem = batch.get('dem')  # Optional DEM input
        mask = batch.get('mask')  # Optional mask
        
        # Handle NaNs and infinites in input
        if torch.isnan(merra2_input).any() or torch.isinf(merra2_input).any():
            logger.warning(f"NaN/Inf values in merra2_input: {torch.isnan(merra2_input).sum().item()} NaNs, {torch.isinf(merra2_input).sum().item()} Infs")
            merra2_input = torch.nan_to_num(merra2_input, nan=0.0, posinf=1.0, neginf=-1.0)
            batch['merra2_input'] = merra2_input
            
        # Handle NaNs and infinites in target
        if torch.isnan(prism_target).any() or torch.isinf(prism_target).any():
            logger.warning(f"NaN/Inf values in prism_target: {torch.isnan(prism_target).sum().item()} NaNs, {torch.isinf(prism_target).sum().item()} Infs")
            prism_target = torch.nan_to_num(prism_target, nan=0.0, posinf=1.0, neginf=-1.0)
            batch['prism_target'] = prism_target
            
        # Handle NaNs and infinites in DEM
        if dem is not None and (torch.isnan(dem).any() or torch.isinf(dem).any()):
            logger.warning(f"NaN/Inf values in dem: {torch.isnan(dem).sum().item()} NaNs, {torch.isinf(dem).sum().item()} Infs")
            dem = torch.nan_to_num(dem, nan=0.0, posinf=1.0, neginf=-1.0)
            batch['dem'] = dem
            
        # Handle NaNs and infinites in mask
        if mask is not None and (torch.isnan(mask).any() or torch.isinf(mask).any()):
            logger.warning(f"NaN/Inf values in mask: {torch.isnan(mask).sum().item()} NaNs, {torch.isinf(mask).sum().item()} Infs")
            mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
            batch['mask'] = mask
            
        return batch
    
    def log_tensor_stats(self, tensor, name):
        """Log tensor statistics for debugging.
        
        Args:
            tensor: Tensor to log
            name: Name of tensor for logging
        """
        if tensor is None:
            logger.info(f"{name} is None")
            return
            
        try:
            logger.info(f"{name} - shape: {tensor.shape}, "
                      f"min: {tensor.min().item():.6f}, "
                      f"max: {tensor.max().item():.6f}, "
                      f"mean: {tensor.mean().item():.6f}, "
                      f"std: {tensor.std().item():.6f}, "
                      f"NaNs: {torch.isnan(tensor).sum().item()}, "
                      f"Infs: {torch.isinf(tensor).sum().item()}")
        except Exception as e:
            logger.warning(f"Error logging tensor stats for {name}: {str(e)}")
    
    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary with loss.
        """
        # Preprocess batch to handle NaNs
        batch = self.preprocess_batch(batch)
        
        # Extract data
        merra2_input = batch['merra2_input']
        prism_target = batch['prism_target']
        dem = batch.get('dem')  # Optional DEM input
        mask = batch.get('mask')  # Optional mask
        
        # Forward pass
        try:
            predictions = self.model(merra2_input, dem)
            
            # Save last good values if no NaNs
            if not torch.isnan(predictions).any() and not torch.isinf(predictions).any():
                self.last_good_input = merra2_input.detach().clone()
                self.last_good_output = predictions.detach().clone()
                self.last_good_target = prism_target.detach().clone()
            
            # Check for NaN or Inf values in predictions
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                nan_count = torch.isnan(predictions).sum().item()
                inf_count = torch.isinf(predictions).sum().item()
                logger.warning(f"[Batch {batch_idx}] NaN/Inf detected in model predictions: {nan_count} NaNs, {inf_count} Infs")
                
                # Log detailed tensor stats
                self.log_tensor_stats(merra2_input, "merra2_input")
                self.log_tensor_stats(dem, "dem")
                self.log_tensor_stats(predictions, "predictions")
                
                # Replace NaN/Inf with zeros to continue training
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # If we have good values from previous batch, log a comparison
                if self.last_good_input is not None:
                    # Compare input stats between good and bad batches
                    input_diff = torch.abs(merra2_input - self.last_good_input).mean().item()
                    logger.info(f"Mean difference from last good input: {input_diff:.6f}")
        except Exception as e:
            logger.error(f"[Batch {batch_idx}] Error in forward pass: {str(e)}")
            if self.last_good_output is not None:
                logger.info("Using last good output as fallback")
                predictions = self.last_good_output
            else:
                logger.info("No good previous output, creating zeros tensor")
                predictions = torch.zeros_like(prism_target)
        
        # Calculate loss
        try:
            loss_dict = self.loss_fn(predictions, prism_target, mask)
            loss = loss_dict['loss']
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"[Batch {batch_idx}] NaN or Inf loss detected: {loss.item()}")
                self.nan_loss_count += 1
                
                # Log detailed tensor stats
                self.log_tensor_stats(predictions, "predictions")
                self.log_tensor_stats(prism_target, "prism_target")
                self.log_tensor_stats(mask, "mask")
                
                # If too many NaN losses, raise an exception to stop training
                if self.nan_loss_count >= self.max_nan_losses:
                    logger.error(f"Too many NaN/Inf losses ({self.nan_loss_count}). Stopping training.")
                    raise ValueError("Training stopped due to excessive NaN/Inf losses")
                
                # Use a dummy loss to continue training
                loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                loss_dict = {
                    'loss': loss,
                    'mae_loss': torch.tensor(0.0, device=self.device),
                    'mse_loss': torch.tensor(0.0, device=self.device),
                    'ssim_loss': torch.tensor(0.0, device=self.device)
                }
            else:
                # Reset counter if loss is valid
                self.nan_loss_count = 0
                
        except Exception as e:
            logger.error(f"[Batch {batch_idx}] Error in loss calculation: {str(e)}")
            # Use a dummy loss to continue training
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            loss_dict = {
                'loss': loss,
                'mae_loss': torch.tensor(0.0, device=self.device),
                'mse_loss': torch.tensor(0.0, device=self.device),
                'ssim_loss': torch.tensor(0.0, device=self.device)
            }
        
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
        # Preprocess batch to handle NaNs
        batch = self.preprocess_batch(batch)
        
        # Extract data
        merra2_input = batch['merra2_input']
        prism_target = batch['prism_target']
        dem = batch.get('dem')  # Optional DEM input
        mask = batch.get('mask')  # Optional mask
        
        # Forward pass
        try:
            predictions = self.model(merra2_input, dem)
            
            # Check for NaN or Inf values in predictions
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                logger.warning(f"[Validation Batch {batch_idx}] NaN/Inf detected in predictions")
                # Log detailed tensor stats
                self.log_tensor_stats(merra2_input, "val_merra2_input")
                self.log_tensor_stats(dem, "val_dem")
                self.log_tensor_stats(predictions, "val_predictions")
                
                # Replace NaN/Inf with zeros for evaluation
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            logger.error(f"[Validation Batch {batch_idx}] Error in forward pass: {str(e)}")
            predictions = torch.zeros_like(prism_target)
        
        # Calculate loss
        try:
            loss_dict = self.loss_fn(predictions, prism_target, mask)
            loss = loss_dict['loss']
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"[Validation Batch {batch_idx}] NaN/Inf loss detected: {loss.item()}")
                # Use a dummy loss for validation
                loss = torch.tensor(1.0, device=self.device)
                loss_dict = {
                    'loss': loss,
                    'mae_loss': torch.tensor(0.0, device=self.device),
                    'mse_loss': torch.tensor(0.0, device=self.device),
                    'ssim_loss': torch.tensor(0.0, device=self.device)
                }
        except Exception as e:
            logger.error(f"[Validation Batch {batch_idx}] Error in loss calculation: {str(e)}")
            # Use a dummy loss for validation
            loss = torch.tensor(1.0, device=self.device)
            loss_dict = {
                'loss': loss,
                'mae_loss': torch.tensor(0.0, device=self.device),
                'mse_loss': torch.tensor(0.0, device=self.device),
                'ssim_loss': torch.tensor(0.0, device=self.device)
            }
        
        # Calculate metrics (with error handling)
        try:
            metrics = calculate_metrics(predictions, prism_target, self.target_vars, mask)
            
            # Check if metrics contain NaN values
            has_nan_metrics = False
            for metric_name, metric_value in metrics['overall'].items():
                if isinstance(metric_value, (torch.Tensor, np.ndarray)) and (
                    np.isnan(metric_value).any() if isinstance(metric_value, np.ndarray) else torch.isnan(metric_value).any()
                ):
                    has_nan_metrics = True
                    logger.warning(f"[Validation Batch {batch_idx}] NaN in metric '{metric_name}'")
            
            if has_nan_metrics:
                # Log tensor stats for debugging
                self.log_tensor_stats(predictions, "val_pred_for_metrics")
                self.log_tensor_stats(prism_target, "val_target_for_metrics")
        except Exception as e:
            logger.error(f"[Validation Batch {batch_idx}] Error in metrics calculation: {str(e)}")
            # Create dummy metrics
            metrics = {
                'overall': {'rmse': torch.tensor(1.0), 'mae': torch.tensor(1.0), 'r2': torch.tensor(0.0)},
                'variables': {}
            }
            for var in self.target_vars:
                metrics['variables'][var] = {'rmse': torch.tensor(1.0), 'mae': torch.tensor(1.0), 'r2': torch.tensor(0.0)}
        
        # Log losses
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mae_loss', loss_dict['mae_loss'], on_step=False, on_epoch=True)
        self.log('val/mse_loss', loss_dict['mse_loss'], on_step=False, on_epoch=True)
        self.log('val/ssim_loss', loss_dict['ssim_loss'], on_step=False, on_epoch=True)
        
        # Log overall metrics
        for metric_name, metric_value in metrics['overall'].items():
            self.log(f'val/metrics/{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        # Log per-variable metrics
        if 'variables' in metrics:
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
        try:
            # Calculate average loss (with NaN check)
            val_losses = [x['val_loss'] for x in outputs]
            # Remove any NaN losses before averaging
            valid_losses = [loss for loss in val_losses if not (torch.isnan(loss) or torch.isinf(loss))]
            
            if not valid_losses:
                logger.warning("All validation losses were NaN or Inf")
                avg_loss = torch.tensor(float('inf'), device=self.device)
            else:
                avg_loss = torch.stack(valid_losses).mean()
            
            # Track best model
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                
                # Store best metrics
                # Since metrics are already logged per step, we don't need to average them again here
                # Just store the metrics from the last batch as a representative
                self.best_val_metrics = outputs[-1]['metrics']
                
                # Log best metrics
                self.log('val/best_loss', self.best_val_loss, on_step=False, on_epoch=True)
        
        except Exception as e:
            logger.error(f"Error in validation_epoch_end: {str(e)}")
    
    def test_step(self, batch, batch_idx):
        """Test step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary with predictions and metrics.
        """
        # Preprocess batch to handle NaNs
        batch = self.preprocess_batch(batch)
        
        # Extract data
        merra2_input = batch['merra2_input']
        prism_target = batch['prism_target']
        dem = batch.get('dem')  # Optional DEM input
        mask = batch.get('mask')  # Optional mask
        
        # Forward pass
        try:
            predictions = self.model(merra2_input, dem)
            
            # Check for NaN or Inf values in predictions
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                logger.warning(f"[Test Batch {batch_idx}] NaN or Inf values detected in model predictions")
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            logger.error(f"[Test Batch {batch_idx}] Error in forward pass: {str(e)}")
            predictions = torch.zeros_like(prism_target)
        
        # Calculate metrics
        try:
            metrics = calculate_metrics(predictions, prism_target, self.target_vars, mask)
        except Exception as e:
            logger.error(f"[Test Batch {batch_idx}] Error in metrics calculation: {str(e)}")
            metrics = {
                'overall': {'rmse': torch.tensor(1.0), 'mae': torch.tensor(1.0), 'r2': torch.tensor(0.0)},
                'variables': {}
            }
            for var in self.target_vars:
                metrics['variables'][var] = {'rmse': torch.tensor(1.0), 'mae': torch.tensor(1.0), 'r2': torch.tensor(0.0)}
        
        # Log overall metrics
        for metric_name, metric_value in metrics['overall'].items():
            self.log(f'test/metrics/{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        # Log per-variable metrics
        if 'variables' in metrics:
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