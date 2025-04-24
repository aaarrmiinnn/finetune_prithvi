"""Loss functions for the downscaling task."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from skimage.metrics import structural_similarity as ssim
import numpy as np


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
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has same shape as y_pred and y_true
            mask = F.interpolate(mask, size=y_pred.shape[2:], mode='nearest')
            y_pred = y_pred * mask
            y_true = y_true * mask
        
        # Calculate MAE loss
        mae = self.mae_loss(y_pred, y_true)
        
        # Calculate MSE loss
        mse = self.mse_loss(y_pred, y_true)
        
        # Calculate SSIM loss (only if weight > 0)
        ssim_loss = torch.tensor(0.0, device=y_pred.device)
        if self.ssim_weight > 0:
            # SSIM is traditionally for image comparison
            # Here, we calculate it per-channel and average
            for c in range(y_pred.shape[1]):
                for b in range(y_pred.shape[0]):
                    pred_np = y_pred[b, c].detach().cpu().numpy()
                    true_np = y_true[b, c].detach().cpu().numpy()
                    
                    # Calculate SSIM
                    ssim_val = ssim(
                        pred_np, 
                        true_np, 
                        data_range=true_np.max() - true_np.min()
                    )
                    
                    # SSIM is similarity, so we use 1 - SSIM as a loss
                    ssim_loss = ssim_loss + (1.0 - ssim_val)
            
            # Average over batch and channels
            ssim_loss = ssim_loss / (y_pred.shape[0] * y_pred.shape[1])
            ssim_loss = torch.tensor(ssim_loss, device=y_pred.device, requires_grad=True)
        
        # Combine losses
        total_loss = (
            self.mae_weight * mae + 
            self.mse_weight * mse + 
            self.ssim_weight * ssim_loss
        )
        
        return {
            'loss': total_loss,
            'mae_loss': mae,
            'mse_loss': mse,
            'ssim_loss': ssim_loss
        } 