"""Loss functions for the downscaling task."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
import logging


# Add a PyTorch-native SSIM implementation to avoid numpy conversion issues
class SSIM(nn.Module):
    """PyTorch implementation of SSIM."""
    
    def __init__(self, window_size=11, size_average=True):
        """Initialize SSIM module.
        
        Args:
            window_size: Size of gaussian kernel
            size_average: Whether to average the loss over the batch
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer('window', self._create_window(window_size))

    def _gaussian(self, window_size, sigma=1.5):
        """Create a 1D gaussian kernel.
        
        Args:
            window_size: Size of window
            sigma: Standard deviation
            
        Returns:
            1D gaussian kernel
        """
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
        
    def _create_window(self, window_size):
        """Create a 2D gaussian kernel.
        
        Args:
            window_size: Size of window
            
        Returns:
            2D gaussian kernel
        """
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, self.channel, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2, data_range=None):
        """Calculate SSIM.
        
        Args:
            img1: First image
            img2: Second image
            data_range: Data range for normalization
            
        Returns:
            SSIM value
        """
        # Ensure inputs are at least 2D (B, C, H, W)
        if len(img1.shape) < 4:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) < 4:
            img2 = img2.unsqueeze(0)
            
        # Check for NaN values
        if torch.isnan(img1).any() or torch.isnan(img2).any():
            img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=-1.0)
            img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Auto-calculate data range if not provided
        if data_range is None:
            data_range = torch.max(torch.stack([
                img1.max() - img1.min(),
                img2.max() - img2.min(),
                torch.tensor(1e-8, device=img1.device)  # Prevent division by zero
            ]))
            
        # Ensure data_range is at least a small positive value
        data_range = torch.max(data_range, torch.tensor(1e-8, device=img1.device))
            
        # Number of channels
        _, channels, height, width = img1.size()
        
        # If window is not the right size, recreate it
        if self.window.size(2) != window_size:
            window = self._create_window(window_size).to(img1.device)
            self.window = window.expand(channels, 1, window_size, window_size)
            
        # Compute SSIM
        window_size = self.window_size
        
        # Pad the images if they're smaller than window size
        pad = window_size // 2
        if height < window_size or width < window_size:
            img1 = F.pad(img1, (pad, pad, pad, pad), mode='replicate')
            img2 = F.pad(img2, (pad, pad, pad, pad), mode='replicate')
            
        # Normalize inputs
        img1 = img1 / data_range
        img2 = img2 / data_range
        
        # Apply gaussian window (convolution)
        mu1 = F.conv2d(img1, self.window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 ** 2, self.window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, self.window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=channels) - mu1_mu2
        
        # Add small constants for stability
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Replace any NaN values with 1.0 (perfect similarity)
        ssim_map = torch.nan_to_num(ssim_map, nan=1.0)
        
        # Average over spatial dimensions
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class DownscalingLoss(nn.Module):
    """Combined loss for downscaling task."""
    
    def __init__(
        self,
        mae_weight: float = 1.0,
        mse_weight: float = 0.5,
        ssim_weight: float = 0.2
    ):
        """Initialize the downscaling loss.
        
        Args:
            mae_weight: Weight for Mean Absolute Error loss.
            mse_weight: Weight for Mean Squared Error loss.
            ssim_weight: Weight for Structural Similarity Index loss.
        """
        super().__init__()
        
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM(window_size=7, size_average=True)
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Calculate losses.
        
        Args:
            y_pred: Predicted tensor of shape (B, C, H, W).
            y_true: Target tensor of shape (B, C, H, W).
            mask: Optional mask tensor of shape (B, C, H, W).
            
        Returns:
            Dictionary with individual losses and total loss.
        """
        # Debug information
        print(f"DEBUG LOSS: Input shapes - preds: {y_pred.shape}, targets: {y_true.shape}")
        print(f"DEBUG LOSS: Value ranges - preds: [{y_pred.min().item():.6f}, {y_pred.max().item():.6f}], targets: [{y_true.min().item():.6f}, {y_true.max().item():.6f}]")
        print(f"DEBUG LOSS: Any NaN - preds: {torch.isnan(y_pred).any().item()}, targets: {torch.isnan(y_true).any().item()}")
        print(f"DEBUG LOSS: Any Inf - preds: {torch.isinf(y_pred).any().item()}, targets: {torch.isinf(y_true).any().item()}")
        
        if mask is not None:
            print(f"DEBUG LOSS: Mask shape: {mask.shape}, Non-zero elements: {mask.sum().item()}")
            print(f"DEBUG LOSS: Mask min: {mask.min().item()}, max: {mask.max().item()}")
        
        # Check for NaN or Inf values in inputs
        has_nan_pred = torch.isnan(y_pred).any().item()
        has_inf_pred = torch.isinf(y_pred).any().item()
        has_nan_target = torch.isnan(y_true).any().item()
        has_inf_target = torch.isinf(y_true).any().item()
        
        print(f"[LOSS DEBUG] Predictions contain NaN: {has_nan_pred}, Inf: {has_inf_pred}")
        print(f"[LOSS DEBUG] Targets contain NaN: {has_nan_target}, Inf: {has_inf_target}")
        
        # Always replace NaN/Inf values to ensure stability
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=-1.0)
        y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply mask if provided
        if mask is not None:
            print(f"[LOSS DEBUG] Applying mask with shape: {mask.shape}")
            try:
                # Ensure mask has same shape as y_pred and y_true
                mask = F.interpolate(mask, size=y_pred.shape[2:], mode='nearest')
                mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
                y_pred = y_pred * mask
                y_true = y_true * mask
            except Exception as e:
                print(f"[LOSS DEBUG] Error applying mask: {str(e)}")
        
        # Calculate losses with error handling
        losses = {}
        
        # Calculate MAE loss
        try:
            mae = self.mae_loss(y_pred, y_true)
            mae = torch.nan_to_num(mae, nan=1.0)
            print(f"[LOSS DEBUG] MAE: {mae.item()}")
            losses['mae_loss'] = mae
        except Exception as e:
            print(f"[LOSS DEBUG] Error calculating MAE: {str(e)}")
            losses['mae_loss'] = torch.tensor(1.0, device=y_pred.device, requires_grad=True)
        
        # Calculate MSE loss
        try:
            mse = self.mse_loss(y_pred, y_true)
            mse = torch.nan_to_num(mse, nan=1.0)
            print(f"[LOSS DEBUG] MSE: {mse.item()}")
            losses['mse_loss'] = mse
        except Exception as e:
            print(f"[LOSS DEBUG] Error calculating MSE: {str(e)}")
            losses['mse_loss'] = torch.tensor(1.0, device=y_pred.device, requires_grad=True)
        
        # Calculate SSIM loss (only if weight > 0)
        if self.ssim_weight > 0:
            try:
                # Calculate SSIM per channel and average
                ssim_vals = []
                
                # Process each channel separately
                for c in range(y_pred.shape[1]):
                    # Get slices for current channel
                    pred_c = y_pred[:, c:c+1]
                    true_c = y_true[:, c:c+1]
                    
                    # Calculate data range for normalization
                    data_range = max(
                        true_c.max().item() - true_c.min().item(),
                        pred_c.max().item() - pred_c.min().item(),
                        1e-6  # Minimum range to prevent division by zero
                    )
                    
                    # Calculate SSIM using PyTorch implementation
                    ssim_val = self.ssim_loss(pred_c, true_c, data_range)
                    ssim_vals.append(ssim_val)
                
                # Average SSIM values across channels
                if ssim_vals:
                    ssim_mean = torch.stack(ssim_vals).mean()
                    # SSIM is similarity, so we use 1 - SSIM as a loss
                    ssim_loss = 1.0 - ssim_mean
                else:
                    ssim_loss = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
                
                print(f"[LOSS DEBUG] SSIM loss: {ssim_loss.item()}")
                losses['ssim_loss'] = ssim_loss
                
            except Exception as e:
                print(f"[LOSS DEBUG] Error calculating SSIM loss: {str(e)}")
                losses['ssim_loss'] = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        else:
            losses['ssim_loss'] = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # Combine losses
        try:
            total_loss = (
                self.mae_weight * losses['mae_loss'] + 
                self.mse_weight * losses['mse_loss'] + 
                self.ssim_weight * losses['ssim_loss']
            )
            
            # Final safety check
            total_loss = torch.nan_to_num(total_loss, nan=1.0, posinf=1.0, neginf=0.0)
            print(f"[LOSS DEBUG] Total loss: {total_loss.item()}")
            losses['loss'] = total_loss
            
        except Exception as e:
            print(f"[LOSS DEBUG] Error computing total loss: {str(e)}")
            # Fall back to MAE loss only
            losses['loss'] = losses['mae_loss']
        
        return losses 