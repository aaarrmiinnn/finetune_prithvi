"""PyTorch Lightning module for MERRA-2 to PRISM downscaling."""
import torch
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Optional, List

from src.models.prithvi_downscaler import create_model


class DownscalingModule(pl.LightningModule):
    """Lightning module for downscaling MERRA-2 to PRISM."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the module.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create model
        self.model = create_model(config)
        
        # Store loss weights
        self.loss_weights = config['loss']
        
        # Track best metrics
        self.best_val_loss = float('inf')
        
        # Track target variables for detailed logging
        self.target_vars = config['data']['target_vars']
        
        # Store validation outputs
        self.validation_step_outputs = []
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            batch: Dictionary containing:
                - merra2_input: MERRA-2 input tensor
                - prism_target: PRISM target tensor
                - dem: Optional DEM tensor
                - mask: Optional mask tensor
        
        Returns:
            Model predictions
        """
        x = batch['merra2_input']
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # Get predictions
        y_pred = self(batch)
        y_true = batch['prism_target']
        
        # Calculate losses
        mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)
        mae_loss = torch.nn.functional.l1_loss(y_pred, y_true)
        
        # Combine losses
        loss = (
            self.loss_weights['mse_weight'] * mse_loss +
            self.loss_weights['mae_weight'] * mae_loss
        )
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/mse', mse_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/mae', mae_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log learning rate
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            self.log('train/lr', scheduler.get_last_lr()[0], on_step=False, on_epoch=True, sync_dist=True)
        
        # Calculate and log per-variable metrics for more detailed tracking
        if batch_idx % 5 == 0 and self.config['logging']['wandb']:  # Only log detailed metrics occasionally to save memory
            self._log_per_variable_metrics(y_pred, y_true, prefix='train')
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """Validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Dictionary with validation results
        """
        # Get predictions
        y_pred = self(batch)
        y_true = batch['prism_target']
        
        # Calculate losses
        mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)
        mae_loss = torch.nn.functional.l1_loss(y_pred, y_true)
        
        # Combine losses
        loss = (
            self.loss_weights['mse_weight'] * mse_loss +
            self.loss_weights['mae_weight'] * mae_loss
        )
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/mse', mse_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/mae', mae_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Calculate and log per-variable metrics
        self._log_per_variable_metrics(y_pred, y_true, prefix='val')
        
        # Return dictionary and save to list for on_validation_epoch_end
        output = {
            'val_loss': loss,
            'val_mse': mse_loss,
            'val_mae': mae_loss,
            'predictions': y_pred,
            'targets': y_true
        }
        
        self.validation_step_outputs.append(output)
        return output
    
    def on_validation_epoch_start(self) -> None:
        """Reset validation step outputs list at the start of each validation epoch."""
        self.validation_step_outputs = []
    
    def on_validation_epoch_end(self) -> None:
        """Process validation epoch end.
        
        This is the updated version of validation_epoch_end to be compatible with PyTorch Lightning v2.0.0+
        """
        if not self.validation_step_outputs:
            return
            
        # Average the loss across all validation steps
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        
        # Track best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.log('val/best_loss', self.best_val_loss, sync_dist=True)
            
            # Sample visualization for best model (if using wandb)
            if self.config['logging']['wandb'] and len(self.validation_step_outputs) > 0:
                try:
                    import wandb
                    # Log example images if using wandb - can be implemented
                    # based on your specific visualization needs
                    pass
                except ImportError:
                    pass
        
        # Clear the outputs to free memory
        self.validation_step_outputs = []
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """Test step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Dictionary with test results
        """
        # Get predictions
        y_pred = self(batch)
        y_true = batch['prism_target']
        
        # Calculate losses
        mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)
        mae_loss = torch.nn.functional.l1_loss(y_pred, y_true)
        
        # Combine losses
        loss = (
            self.loss_weights['mse_weight'] * mse_loss +
            self.loss_weights['mae_weight'] * mae_loss
        )
        
        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mae', mae_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Calculate and log per-variable metrics
        self._log_per_variable_metrics(y_pred, y_true, prefix='test')
        
        return {
            'test_loss': loss,
            'test_mse': mse_loss,
            'test_mae': mae_loss,
            'predictions': y_pred,
            'targets': y_true
        }
    
    def _log_per_variable_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, prefix: str) -> None:
        """Log metrics for each target variable separately.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            prefix: Prefix for metric names (train/val/test)
        """
        # Skip if no variable names are defined
        if not hasattr(self, 'target_vars') or not self.target_vars:
            return
            
        # Calculate per-variable metrics
        for i, var_name in enumerate(self.target_vars):
            # Extract predictions and targets for this variable (assume channel dimension is 1)
            pred_var = predictions[:, i:i+1]
            target_var = targets[:, i:i+1]
            
            # Calculate metrics
            var_mse = torch.nn.functional.mse_loss(pred_var, target_var)
            var_mae = torch.nn.functional.l1_loss(pred_var, target_var)
            
            # Log metrics
            self.log(f'{prefix}/{var_name}/mse', var_mse, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'{prefix}/{var_name}/mae', var_mae, on_step=False, on_epoch=True, sync_dist=True)
            
            # Calculate and log additional metrics for each variable
            if var_name == 'tdmean':
                # Temperature-specific metrics
                temp_bias = torch.mean(pred_var - target_var)
                self.log(f'{prefix}/{var_name}/bias', temp_bias, on_step=False, on_epoch=True, sync_dist=True)
            elif var_name == 'ppt':
                # Precipitation-specific metrics
                # Relative bias is more relevant for precipitation
                # Avoid division by zero
                valid_mask = target_var > 0.01  # Only consider values above 0.01 mm
                if torch.sum(valid_mask) > 0:
                    rel_bias = torch.mean((pred_var[valid_mask] - target_var[valid_mask]) / target_var[valid_mask])
                    self.log(f'{prefix}/{var_name}/rel_bias', rel_bias, on_step=False, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer parameters
        optimizer_config = self.config['training']['optimizer']
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
        
        # Create scheduler
        scheduler_config = self.config['training']['scheduler']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val/loss'
            }
        } 