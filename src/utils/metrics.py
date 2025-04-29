"""Metrics for evaluating downscaling models."""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
from typing import Dict, Union, List, Optional
from skimage.metrics import structural_similarity as ssim
import logging


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    variable_names: Optional[List[str]] = None,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Calculate evaluation metrics.
    
    Args:
        predictions: Predicted tensor of shape (B, C, H, W).
        targets: Target tensor of shape (B, C, H, W).
        variable_names: List of variable names for detailed metrics.
        mask: Mask tensor of shape (B, C, H, W) with 1 for valid pixels and 0 for invalid pixels.
        
    Returns:
        Dictionary of metrics.
    """
    # Initialize logger
    logger = logging.getLogger(__name__)
    
    # Preprocess inputs to handle NaNs and Infs
    predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)
    targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=-1.0)
    if mask is not None:
        mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Print debug info
    print(f"DEBUG: Metrics calculation input shapes - preds: {predictions.shape}, targets: {targets.shape}")
    print(f"DEBUG: Metrics calculation input ranges - preds: [{predictions.min().item()}, {predictions.max().item()}], targets: [{targets.min().item()}, {targets.max().item()}]")
    print(f"DEBUG: Any NaN in preds: {torch.isnan(predictions).any().item()}, targets: {torch.isnan(targets).any().item()}")
    print(f"DEBUG: Any Inf in preds: {torch.isinf(predictions).any().item()}, targets: {torch.isinf(targets).any().item()}")
    
    if mask is not None:
        print(f"DEBUG: Mask shape: {mask.shape}, Non-zero elements: {mask.sum().item()}")
    
    # If mask is provided, apply it
    if mask is not None:
        predictions = predictions * mask
        targets = targets * mask
    
    # Ensure predictions and targets have the same shape
    if predictions.shape != targets.shape:
        logger.warning(f"Shape mismatch: preds {predictions.shape}, targets {targets.shape}")
        # Try to reshape predictions to match targets
        try:
            predictions = F.interpolate(
                predictions, 
                size=targets.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        except Exception as e:
            logger.error(f"Failed to reshape predictions: {str(e)}")
            # Return NaN metrics if reshaping fails
            return create_dummy_metrics(variable_names, logger)
    
    # Get batch size and number of channels
    batch_size, num_channels = predictions.shape[0], predictions.shape[1]
    
    # Initialize metrics dictionary
    metrics = {'overall': {}, 'variables': {}}
    
    try:
        # Calculate RMSE
        mse = F.mse_loss(predictions, targets, reduction='none')
        print(f"DEBUG: MSE shape: {mse.shape}, Any NaN: {torch.isnan(mse).any().item()}")
        if mask is not None:
            # Average over non-masked pixels
            mask_sum = torch.sum(mask) * num_channels
            if mask_sum > 0:
                rmse = torch.sqrt(torch.sum(mse * mask) / mask_sum)
            else:
                logger.warning("Mask sum is zero, using unmasked RMSE")
                rmse = torch.sqrt(torch.mean(mse))
        else:
            rmse = torch.sqrt(torch.mean(mse))
        
        # Handle NaN RMSE
        if torch.isnan(rmse) or torch.isinf(rmse):
            logger.warning("RMSE calculation resulted in NaN or Inf, using default value")
            rmse = torch.tensor(1.0, device=predictions.device)
            
        metrics['overall']['rmse'] = rmse.item()
        print(f"DEBUG: Overall RMSE: {metrics['overall']['rmse']}")
        
        # Calculate MAE
        mae = F.l1_loss(predictions, targets, reduction='none')
        print(f"DEBUG: MAE shape: {mae.shape}, Any NaN: {torch.isnan(mae).any().item()}")
        if mask is not None:
            # Average over non-masked pixels
            mask_sum = torch.sum(mask) * num_channels
            if mask_sum > 0:
                mae = torch.sum(mae * mask) / mask_sum
            else:
                logger.warning("Mask sum is zero, using unmasked MAE")
                mae = torch.mean(mae)
        else:
            mae = torch.mean(mae)
            
        # Handle NaN MAE
        if torch.isnan(mae) or torch.isinf(mae):
            logger.warning("MAE calculation resulted in NaN or Inf, using default value")
            mae = torch.tensor(1.0, device=predictions.device)
            
        metrics['overall']['mae'] = mae.item()
        print(f"DEBUG: Overall MAE: {metrics['overall']['mae']}")
        
        # Calculate bias
        if mask is not None:
            # Average over non-masked pixels
            mask_sum = torch.sum(mask) * num_channels
            if mask_sum > 0:
                bias = torch.sum((predictions - targets) * mask) / mask_sum
            else:
                logger.warning("Mask sum is zero, using unmasked bias")
                bias = torch.mean(predictions - targets)
        else:
            bias = torch.mean(predictions - targets)
            
        # Handle NaN bias
        if torch.isnan(bias) or torch.isinf(bias):
            logger.warning("Bias calculation resulted in NaN or Inf, using default value")
            bias = torch.tensor(0.0, device=predictions.device)
            
        metrics['overall']['bias'] = bias.item()
        print(f"DEBUG: Overall bias: {metrics['overall']['bias']}")
        
        # Calculate R²
        try:
            if mask is not None:
                # Only consider non-masked pixels
                mask_expanded = mask.expand_as(predictions)
                valid_mask = mask_expanded > 0
                if valid_mask.sum() > 0:
                    preds_flat = predictions[valid_mask]
                    targets_flat = targets[valid_mask]
                else:
                    logger.warning("No valid masked pixels, using all pixels for R²")
                    preds_flat = predictions.reshape(-1)
                    targets_flat = targets.reshape(-1)
            else:
                preds_flat = predictions.reshape(-1)
                targets_flat = targets.reshape(-1)
            
            print(f"DEBUG: Flattened shapes - preds: {preds_flat.shape}, targets: {targets_flat.shape}")
            
            # Check if flattened tensors have valid values
            if preds_flat.numel() > 0 and not (torch.isnan(preds_flat).any() or torch.isnan(targets_flat).any()):
                targets_mean = torch.mean(targets_flat)
                ss_tot = torch.sum((targets_flat - targets_mean) ** 2)
                ss_res = torch.sum((targets_flat - preds_flat) ** 2)
                
                print(f"DEBUG: R² calculation - ss_tot: {ss_tot.item()}, ss_res: {ss_res.item()}")
                
                if ss_tot > 0:
                    r2 = 1 - ss_res / ss_tot
                    
                    # Handle NaN R²
                    if torch.isnan(r2) or torch.isinf(r2) or r2 < -1:
                        logger.warning("R² calculation resulted in NaN, Inf, or < -1, using default value")
                        r2 = torch.tensor(0.0, device=predictions.device)
                        
                    metrics['overall']['r2'] = r2.item()
                    print(f"DEBUG: Overall R²: {metrics['overall']['r2']}")
                else:
                    logger.warning("ss_tot is zero or negative, using default R²")
                    metrics['overall']['r2'] = 0.0
            else:
                logger.warning("No valid elements for R², using default value")
                metrics['overall']['r2'] = 0.0
        except Exception as e:
            logger.error(f"Error calculating R²: {str(e)}")
            metrics['overall']['r2'] = 0.0
        
        # Calculate SSIM (with safe handling)
        try:
            # Convert to numpy for SSIM calculation
            preds_np = predictions.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            # Check for NaN/Inf values
            if np.isnan(preds_np).any() or np.isnan(targets_np).any() or np.isinf(preds_np).any() or np.isinf(targets_np).any():
                logger.warning("NaN/Inf values in inputs to SSIM, replacing with safe values")
                preds_np = np.nan_to_num(preds_np, nan=0.0, posinf=1.0, neginf=-1.0)
                targets_np = np.nan_to_num(targets_np, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Calculate data range carefully to avoid division by zero
            data_range = np.maximum(
                np.ptp(targets_np),  # peak-to-peak (max - min)
                np.ptp(preds_np),
                1e-6  # minimum non-zero range
            )
            
            # Calculate SSIM per channel and average
            ssim_values = []
            for c in range(num_channels):
                for b in range(batch_size):
                    try:
                        channel_ssim = ssim(
                            preds_np[b, c], 
                            targets_np[b, c], 
                            data_range=data_range
                        )
                        if not (np.isnan(channel_ssim) or np.isinf(channel_ssim)):
                            ssim_values.append(channel_ssim)
                    except Exception as e:
                        logger.warning(f"Error calculating SSIM for batch {b}, channel {c}: {str(e)}")
            
            # Calculate mean SSIM if there are valid values
            if ssim_values:
                metrics['overall']['ssim'] = np.mean(ssim_values)
            else:
                logger.warning("No valid SSIM values calculated, using default")
                metrics['overall']['ssim'] = 0.0
                
            print(f"DEBUG: Overall SSIM: {metrics['overall']['ssim']}")
            
        except Exception as e:
            logger.error(f"Error calculating SSIM: {str(e)}")
            metrics['overall']['ssim'] = 0.0
        
        # Calculate per-variable metrics if variable names are provided
        if variable_names is not None and len(variable_names) == num_channels:
            for i, var_name in enumerate(variable_names):
                print(f"DEBUG: Calculating metrics for variable: {var_name}")
                metrics['variables'][var_name] = {}
                
                # Extract single channel
                pred_var = predictions[:, i:i+1]
                target_var = targets[:, i:i+1]
                
                try:
                    # Calculate RMSE
                    mse_var = F.mse_loss(pred_var, target_var, reduction='none')
                    if mask is not None:
                        # Average over non-masked pixels
                        mask_sum = torch.sum(mask)
                        if mask_sum > 0:
                            rmse_var = torch.sqrt(torch.sum(mse_var * mask) / mask_sum)
                        else:
                            rmse_var = torch.sqrt(torch.mean(mse_var))
                    else:
                        rmse_var = torch.sqrt(torch.mean(mse_var))
                        
                    # Handle NaN RMSE
                    if torch.isnan(rmse_var) or torch.isinf(rmse_var):
                        logger.warning(f"RMSE calculation for {var_name} resulted in NaN or Inf, using default value")
                        rmse_var = torch.tensor(1.0, device=predictions.device)
                        
                    metrics['variables'][var_name]['rmse'] = rmse_var.item()
                    
                    # Calculate MAE
                    mae_var = F.l1_loss(pred_var, target_var, reduction='none')
                    if mask is not None:
                        # Average over non-masked pixels
                        mask_sum = torch.sum(mask)
                        if mask_sum > 0:
                            mae_var = torch.sum(mae_var * mask) / mask_sum
                        else:
                            mae_var = torch.mean(mae_var)
                    else:
                        mae_var = torch.mean(mae_var)
                        
                    # Handle NaN MAE
                    if torch.isnan(mae_var) or torch.isinf(mae_var):
                        logger.warning(f"MAE calculation for {var_name} resulted in NaN or Inf, using default value")
                        mae_var = torch.tensor(1.0, device=predictions.device)
                        
                    metrics['variables'][var_name]['mae'] = mae_var.item()
                    
                    # Calculate bias
                    if mask is not None:
                        # Average over non-masked pixels
                        mask_sum = torch.sum(mask)
                        if mask_sum > 0:
                            bias_var = torch.sum((pred_var - target_var) * mask) / mask_sum
                        else:
                            bias_var = torch.mean(pred_var - target_var)
                    else:
                        bias_var = torch.mean(pred_var - target_var)
                        
                    # Handle NaN bias
                    if torch.isnan(bias_var) or torch.isinf(bias_var):
                        logger.warning(f"Bias calculation for {var_name} resulted in NaN or Inf, using default value")
                        bias_var = torch.tensor(0.0, device=predictions.device)
                        
                    metrics['variables'][var_name]['bias'] = bias_var.item()
                    
                    # Calculate R²
                    try:
                        if mask is not None:
                            # Only consider non-masked pixels
                            mask_expanded = mask.expand_as(pred_var)
                            valid_mask = mask_expanded > 0
                            if valid_mask.sum() > 0:
                                pred_var_flat = pred_var[valid_mask]
                                target_var_flat = target_var[valid_mask]
                            else:
                                logger.warning(f"No valid masked pixels for {var_name}, using all pixels for R²")
                                pred_var_flat = pred_var.reshape(-1)
                                target_var_flat = target_var.reshape(-1)
                        else:
                            pred_var_flat = pred_var.reshape(-1)
                            target_var_flat = target_var.reshape(-1)
                        
                        # Check if flattened tensors have valid values
                        if pred_var_flat.numel() > 0 and not (torch.isnan(pred_var_flat).any() or torch.isnan(target_var_flat).any()):
                            target_var_mean = torch.mean(target_var_flat)
                            ss_tot_var = torch.sum((target_var_flat - target_var_mean) ** 2)
                            ss_res_var = torch.sum((target_var_flat - pred_var_flat) ** 2)
                            
                            if ss_tot_var > 0:
                                r2_var = 1 - ss_res_var / ss_tot_var
                                
                                # Handle NaN R²
                                if torch.isnan(r2_var) or torch.isinf(r2_var) or r2_var < -1:
                                    logger.warning(f"R² calculation for {var_name} resulted in NaN, Inf, or < -1, using default value")
                                    r2_var = torch.tensor(0.0, device=predictions.device)
                                    
                                metrics['variables'][var_name]['r2'] = r2_var.item()
                            else:
                                logger.warning(f"ss_tot is zero or negative for {var_name}, using default R²")
                                metrics['variables'][var_name]['r2'] = 0.0
                        else:
                            logger.warning(f"No valid elements for R² for {var_name}, using default value")
                            metrics['variables'][var_name]['r2'] = 0.0
                            
                    except Exception as e:
                        logger.error(f"Error calculating R² for {var_name}: {str(e)}")
                        metrics['variables'][var_name]['r2'] = 0.0
                    
                    # Calculate SSIM
                    try:
                        # Convert to numpy for SSIM calculation
                        pred_var_np = pred_var.detach().cpu().numpy()
                        target_var_np = target_var.detach().cpu().numpy()
                        
                        # Check for NaN/Inf values
                        if np.isnan(pred_var_np).any() or np.isnan(target_var_np).any() or np.isinf(pred_var_np).any() or np.isinf(target_var_np).any():
                            logger.warning(f"NaN/Inf values in inputs to SSIM for {var_name}, replacing with safe values")
                            pred_var_np = np.nan_to_num(pred_var_np, nan=0.0, posinf=1.0, neginf=-1.0)
                            target_var_np = np.nan_to_num(target_var_np, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # Calculate data range carefully to avoid division by zero
                        data_range = np.maximum(
                            np.ptp(target_var_np),  # peak-to-peak (max - min)
                            np.ptp(pred_var_np),
                            1e-6  # minimum non-zero range
                        )
                        
                        # Calculate SSIM per batch and average
                        ssim_values = []
                        for b in range(batch_size):
                            try:
                                var_ssim = ssim(
                                    pred_var_np[b, 0], 
                                    target_var_np[b, 0], 
                                    data_range=data_range
                                )
                                if not (np.isnan(var_ssim) or np.isinf(var_ssim)):
                                    ssim_values.append(var_ssim)
                            except Exception as e:
                                logger.warning(f"Error calculating SSIM for batch {b}, variable {var_name}: {str(e)}")
                        
                        # Calculate mean SSIM if there are valid values
                        if ssim_values:
                            metrics['variables'][var_name]['ssim'] = np.mean(ssim_values)
                        else:
                            logger.warning(f"No valid SSIM values calculated for {var_name}, using default")
                            metrics['variables'][var_name]['ssim'] = 0.0
                        
                    except Exception as e:
                        logger.error(f"Error calculating SSIM for {var_name}: {str(e)}")
                        metrics['variables'][var_name]['ssim'] = 0.0
                        
                except Exception as e:
                    logger.error(f"Error calculating metrics for {var_name}: {str(e)}")
                    metrics['variables'][var_name] = {
                        'rmse': 1.0,
                        'mae': 1.0,
                        'bias': 0.0,
                        'r2': 0.0,
                        'ssim': 0.0
                    }
                
                print(f"DEBUG: Metrics for {var_name} - "
                      f"RMSE: {metrics['variables'][var_name].get('rmse', 'N/A')}, "
                      f"MAE: {metrics['variables'][var_name].get('mae', 'N/A')}, "
                      f"bias: {metrics['variables'][var_name].get('bias', 'N/A')}, "
                      f"R²: {metrics['variables'][var_name].get('r2', 'N/A')}, "
                      f"SSIM: {metrics['variables'][var_name].get('ssim', 'N/A')}")
        
        # Return combined metrics
        result = {
            'overall': metrics['overall'],
            'variables': metrics['variables']
        }
        print(f"[METRICS DEBUG] Final metrics: {result}")
        return result
    
    except Exception as e:
        print(f"DEBUG: Exception in metrics calculation: {str(e)}")
        logger.error(f"Error calculating metrics: {str(e)}")
        # Return default metrics
        return create_dummy_metrics(variable_names, logger)


def create_dummy_metrics(variable_names=None, logger=None):
    """Create a dummy metrics dictionary with default values.
    
    Args:
        variable_names: Optional list of variable names.
        logger: Optional logger for warning messages.
        
    Returns:
        Dictionary with default metric values.
    """
    if logger:
        logger.warning("Creating dummy metrics due to calculation failure")
        
    result = {
        'overall': {
            'rmse': 1.0,
            'mae': 1.0,
            'bias': 0.0,
            'r2': 0.0,
            'ssim': 0.0
        },
        'variables': {}
    }
    
    if variable_names:
        for var_name in variable_names:
            result['variables'][var_name] = {
                'rmse': 1.0,
                'mae': 1.0,
                'bias': 0.0,
                'r2': 0.0,
                'ssim': 0.0
            }
    
    return result 