"""Metrics for evaluating downscaling models."""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
from typing import Dict, Union, List
from skimage.metrics import structural_similarity as ssim


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    variable_names: List[str] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Calculate evaluation metrics.
    
    Args:
        predictions: Predicted tensor of shape (B, C, H, W).
        targets: Target tensor of shape (B, C, H, W).
        variable_names: List of variable names for detailed metrics.
        
    Returns:
        Dictionary of metrics.
    """
    # Move to CPU and convert to numpy
    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    batch_size, n_channels = preds_np.shape[0], preds_np.shape[1]
    
    # Overall metrics
    overall_metrics = {}
    
    # Calculate RMSE
    mse = np.mean((preds_np - targets_np) ** 2)
    rmse = np.sqrt(mse)
    overall_metrics['rmse'] = float(rmse)
    
    # Calculate MAE
    mae = np.mean(np.abs(preds_np - targets_np))
    overall_metrics['mae'] = float(mae)
    
    # Calculate bias
    bias = np.mean(preds_np - targets_np)
    overall_metrics['bias'] = float(bias)
    
    # Calculate R^2
    # Flatten all dimensions except channel dimension
    preds_flat = preds_np.reshape(batch_size * n_channels, -1)
    targets_flat = targets_np.reshape(batch_size * n_channels, -1)
    
    # Calculate R^2 for each sample and average
    r2_values = []
    for i in range(preds_flat.shape[0]):
        r2 = r2_score(targets_flat[i], preds_flat[i])
        r2_values.append(r2)
    
    overall_metrics['r2'] = float(np.mean(r2_values))
    
    # Calculate SSIM
    ssim_values = []
    for b in range(batch_size):
        for c in range(n_channels):
            ssim_val = ssim(
                preds_np[b, c],
                targets_np[b, c],
                data_range=targets_np[b, c].max() - targets_np[b, c].min()
            )
            ssim_values.append(ssim_val)
    
    overall_metrics['ssim'] = float(np.mean(ssim_values))
    
    # Per-variable metrics if variable names are provided
    variable_metrics = {}
    if variable_names and len(variable_names) == n_channels:
        for i, var_name in enumerate(variable_names):
            # Extract data for this variable
            var_preds = preds_np[:, i]
            var_targets = targets_np[:, i]
            
            # Calculate metrics
            var_mse = np.mean((var_preds - var_targets) ** 2)
            var_rmse = np.sqrt(var_mse)
            var_mae = np.mean(np.abs(var_preds - var_targets))
            var_bias = np.mean(var_preds - var_targets)
            
            # Calculate R^2
            var_preds_flat = var_preds.reshape(batch_size, -1)
            var_targets_flat = var_targets.reshape(batch_size, -1)
            
            var_r2_values = []
            for b in range(batch_size):
                var_r2 = r2_score(var_targets_flat[b], var_preds_flat[b])
                var_r2_values.append(var_r2)
            
            var_r2 = float(np.mean(var_r2_values))
            
            # Calculate SSIM
            var_ssim_values = []
            for b in range(batch_size):
                var_ssim_val = ssim(
                    var_preds[b],
                    var_targets[b],
                    data_range=var_targets[b].max() - var_targets[b].min()
                )
                var_ssim_values.append(var_ssim_val)
            
            var_ssim = float(np.mean(var_ssim_values))
            
            # Store metrics for this variable
            variable_metrics[var_name] = {
                'rmse': float(var_rmse),
                'mae': float(var_mae),
                'bias': float(var_bias),
                'r2': var_r2,
                'ssim': var_ssim
            }
    
    # Return combined metrics
    return {
        'overall': overall_metrics,
        'variables': variable_metrics
    } 