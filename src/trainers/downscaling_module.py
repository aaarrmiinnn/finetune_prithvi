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
        
        # NaN tracking for less verbose logging
        self._nan_detected = {
            'input': False,
            'output': False,
            'target': False,
            'loss': False
        }
        self._nan_count = {
            'input': 0,
            'output': 0,
            'target': 0,
            'loss': 0
        }
        self._nan_warning_frequency = 10  # Only warn every N occurrences
    
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
        # Check for NaNs in input
        if torch.isnan(batch['merra2_input']).any():
            self._nan_count['input'] += 1
            # Only log first occurrence and then at frequency intervals
            if not self._nan_detected['input'] or self._nan_count['input'] % self._nan_warning_frequency == 0:
                self.print(f"WARNING: NaN detected in input data at batch {batch_idx}. Total occurrences: {self._nan_count['input']}")
                self._nan_detected['input'] = True
            # Replace NaNs with zeros to prevent propagation
            batch['merra2_input'] = torch.nan_to_num(batch['merra2_input'], nan=0.0)
        
        # Get predictions
        y_pred = self(batch)
        y_true = batch['prism_target']
        
        # Check for NaNs in output
        if torch.isnan(y_pred).any():
            self._nan_count['output'] += 1
            # Only log first occurrence and then at frequency intervals
            if not self._nan_detected['output'] or self._nan_count['output'] % self._nan_warning_frequency == 0:
                self.print(f"WARNING: NaN detected in model output at batch {batch_idx}. Total occurrences: {self._nan_count['output']}")
                self._nan_detected['output'] = True
            # Replace NaNs with zeros for loss calculation
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
        
        # Check for NaNs in target
        if torch.isnan(y_true).any():
            self._nan_count['target'] += 1
            # Only log first occurrence and then at frequency intervals
            if not self._nan_detected['target'] or self._nan_count['target'] % self._nan_warning_frequency == 0:
                self.print(f"WARNING: NaN detected in target data at batch {batch_idx}. Total occurrences: {self._nan_count['target']}")
                self._nan_detected['target'] = True
            # Replace NaNs with zeros for loss calculation
            y_true = torch.nan_to_num(y_true, nan=0.0)
        
        # Calculate losses with safeguards
        try:
            mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)
            mae_loss = torch.nn.functional.l1_loss(y_pred, y_true)
            
            # Check for NaN losses
            if torch.isnan(mse_loss) or torch.isnan(mae_loss):
                self._nan_count['loss'] += 1
                # Always report loss NaNs with detailed stats
                self.print(f"WARNING: NaN loss detected at batch {batch_idx}. Total occurrences: {self._nan_count['loss']}")
                self.print(f"MSE: {mse_loss}, MAE: {mae_loss}")
                self.print(f"Prediction stats - min: {y_pred.min()}, max: {y_pred.max()}, mean: {y_pred.mean()}")
                self.print(f"Target stats - min: {y_true.min()}, max: {y_true.max()}, mean: {y_true.mean()}")
                self._nan_detected['loss'] = True
                
                # Use a small constant loss to prevent training failure
                mse_loss = torch.tensor(1.0, device=self.device)
                mae_loss = torch.tensor(1.0, device=self.device)
        except Exception as e:
            self.print(f"ERROR during loss calculation: {str(e)}")
            # Use a small constant loss to prevent training failure
            mse_loss = torch.tensor(1.0, device=self.device)
            mae_loss = torch.tensor(1.0, device=self.device)
        
        # Combine losses
        loss = (
            self.loss_weights['mse_weight'] * mse_loss +
            self.loss_weights['mae_weight'] * mae_loss
        )
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/mse', mse_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/mae', mae_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log NaN counts to track progress
        if any(self._nan_count.values()):
            for key, count in self._nan_count.items():
                if count > 0:
                    self.log(f'train/nan_{key}_count', count, on_step=False, on_epoch=True, sync_dist=True)
        
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
        # Check for NaNs in input (less verbose for validation)
        if torch.isnan(batch['merra2_input']).any():
            if not self._nan_detected['input'] or batch_idx == 0:  # Only log first batch of each validation run
                self.print(f"WARNING: NaN detected in validation input data at batch {batch_idx}")
            # Replace NaNs with zeros to prevent propagation
            batch['merra2_input'] = torch.nan_to_num(batch['merra2_input'], nan=0.0)
        
        # Get predictions
        y_pred = self(batch)
        y_true = batch['prism_target']
        
        # Check for NaNs in output (less verbose for validation)
        if torch.isnan(y_pred).any():
            if not self._nan_detected['output'] or batch_idx == 0:
                self.print(f"WARNING: NaN detected in validation model output at batch {batch_idx}")
            # Replace NaNs with zeros for loss calculation
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
        
        # Check for NaNs in target (less verbose for validation)
        if torch.isnan(y_true).any():
            if not self._nan_detected['target'] or batch_idx == 0:
                self.print(f"WARNING: NaN detected in validation target data at batch {batch_idx}")
            # Replace NaNs with zeros for loss calculation
            y_true = torch.nan_to_num(y_true, nan=0.0)
        
        # Calculate losses with safeguards
        try:
            mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)
            mae_loss = torch.nn.functional.l1_loss(y_pred, y_true)
            
            # Check for NaN losses
            if torch.isnan(mse_loss) or torch.isnan(mae_loss):
                # Always report validation loss NaNs with detailed stats
                self.print(f"WARNING: NaN validation loss detected at batch {batch_idx}")
                self.print(f"MSE: {mse_loss}, MAE: {mae_loss}")
                self.print(f"Prediction stats - min: {y_pred.min()}, max: {y_pred.max()}, mean: {y_pred.mean()}")
                self.print(f"Target stats - min: {y_true.min()}, max: {y_true.max()}, mean: {y_true.mean()}")
                
                # Use a small constant loss to prevent validation failure
                mse_loss = torch.tensor(1.0, device=self.device)
                mae_loss = torch.tensor(1.0, device=self.device)
        except Exception as e:
            self.print(f"ERROR during validation loss calculation: {str(e)}")
            # Use a small constant loss to prevent validation failure
            mse_loss = torch.tensor(1.0, device=self.device)
            mae_loss = torch.tensor(1.0, device=self.device)
        
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
        # Check for NaNs in input (least verbose for testing)
        has_input_nan = torch.isnan(batch['merra2_input']).any()
        if has_input_nan:
            if batch_idx == 0:  # Only log once during testing
                self.print(f"WARNING: NaN values detected in test input data (quietly replacing)")
            # Replace NaNs with zeros to prevent propagation
            batch['merra2_input'] = torch.nan_to_num(batch['merra2_input'], nan=0.0)
        
        # Get predictions
        y_pred = self(batch)
        y_true = batch['prism_target']
        
        # Check for NaNs in output (least verbose for testing)
        has_output_nan = torch.isnan(y_pred).any()
        if has_output_nan:
            if batch_idx == 0:  # Only log once during testing
                self.print(f"WARNING: NaN values detected in test model output (quietly replacing)")
            # Replace NaNs with zeros for loss calculation
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
        
        # Check for NaNs in target (least verbose for testing)
        has_target_nan = torch.isnan(y_true).any()
        if has_target_nan:
            if batch_idx == 0:  # Only log once during testing
                self.print(f"WARNING: NaN values detected in test target data (quietly replacing)")
            # Replace NaNs with zeros for loss calculation
            y_true = torch.nan_to_num(y_true, nan=0.0)
        
        # Calculate losses with safeguards
        try:
            mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)
            mae_loss = torch.nn.functional.l1_loss(y_pred, y_true)
            
            # Check for NaN losses
            if torch.isnan(mse_loss) or torch.isnan(mae_loss):
                if batch_idx == 0:  # Only log once during testing
                    self.print(f"WARNING: NaN test loss detected (replacing with constant loss)")
                
                # Use a small constant loss to prevent test failure
                mse_loss = torch.tensor(1.0, device=self.device)
                mae_loss = torch.tensor(1.0, device=self.device)
        except Exception as e:
            if batch_idx == 0:  # Only log once during testing
                self.print(f"ERROR during test loss calculation: {str(e)}")
            # Use a small constant loss to prevent test failure
            mse_loss = torch.tensor(1.0, device=self.device)
            mae_loss = torch.tensor(1.0, device=self.device)
        
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