"""Prithvi-based downscaling model for MERRA-2 to PRISM."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from .prithvi_wxc import PrithviWxC, PrithviWxCConfig


class UpsampleBlock(nn.Module):
    """Upsampling block for the decoder."""
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_residual: bool = True
    ):
        """Initialize the upsampling block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            scale_factor: Upsampling scale factor.
            use_residual: Whether to use residual connection.
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.scale_factor = scale_factor
        
        # Conv before upsampling
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Upsampling layer
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Conv after upsampling
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection if dimensions match
        if use_residual and in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Upsampled tensor of shape (B, C_out, H*scale, W*scale).
        """
        identity = x
        
        # First convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Upsample
        x = self.upsample(x)
        
        # Second convolution
        x = self.bn2(self.conv2(x))
        
        # Residual connection
        if self.use_residual:
            identity = self.residual(identity)
            identity = self.upsample(identity)
            x = x + identity
        
        return F.relu(x)


class DemEncoder(nn.Module):
    """Encoder for Digital Elevation Model data."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        hidden_channels: int = 32
    ):
        """Initialize the DEM encoder.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (embedding size).
            hidden_channels: Number of hidden channels.
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsampling to match the spatial dimensions of MERRA-2
        self.pool = nn.AdaptiveAvgPool2d(output_size=(64, 64))  # Adjust to match MERRA-2 size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, H, W).
            
        Returns:
            Encoded tensor of shape (B, out_channels, H_merra, W_merra).
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class PrithviDownscaler(nn.Module):
    """A downscaling model that uses PrithviWxC as a backbone.
    
    This model takes low-resolution climate data and produces high-resolution predictions
    using a transformer-based architecture. It consists of three main components:
    1. Input projection: Adapts input channels to the transformer's hidden dimension
    2. Feature extraction: Uses PrithviWxC to extract features
    3. Decoder: Converts features to the desired output resolution and channels
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_dim: int = 768,
        prithvi_checkpoint: str = "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        cache_dir: str = "models/cache",
        device: str = "cuda",
        use_pretrained: bool = True,
        gradient_checkpointing: bool = False,
        freeze_encoder: bool = False,
    ):
        """Initialize the PrithviDownscaler model.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            hidden_dim: Hidden dimension size for transformer and conv layers
            prithvi_checkpoint: Model name or path for PrithviWxC
            cache_dir: Directory to cache downloaded models
            device: Device to use (cuda or cpu)
            use_pretrained: Whether to use pretrained weights
            gradient_checkpointing: Whether to use gradient checkpointing to save memory
            freeze_encoder: Whether to freeze the backbone transformer encoder
        """
        super().__init__()
        
        # Ensure num_attention_heads divides hidden_dim evenly
        # For small hidden_dim values, use a small number of heads (e.g., 4 or 8)
        num_attention_heads = 8 if hidden_dim >= 64 else 4
        
        # Initialize PrithviWxC backbone with the correct number of input channels
        config = PrithviWxCConfig(
            num_channels=input_channels,
            hidden_size=hidden_dim,
            patch_size=16,  # Fixed patch size for Prithvi
            num_attention_heads=num_attention_heads,  # Ensure divisibility
            num_hidden_layers=6,  # Reduce number of layers for smaller models
            intermediate_size=hidden_dim * 4  # Standard multiplier for FF size
        )
        self.backbone = PrithviWxC(
            config=config,
            model_name=prithvi_checkpoint,
            cache_dir=cache_dir,
            device=device,
            use_pretrained=use_pretrained,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        # Store whether to freeze encoder
        self.freeze_encoder = freeze_encoder
        
        # If freeze_encoder is True, freeze backbone parameters
        if freeze_encoder:
            print("Freezing Prithvi backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Input projection layers
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, input_channels, kernel_size=3, padding=1),  # Project back to input channels
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
        )
        
        # Feature extraction layers
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        
        # Decoder layers with progressive upsampling
        # Input: 64x64 -> Output: 256x256 (2 upsampling blocks)
        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            UpsampleBlock(hidden_dim, hidden_dim // 2, scale_factor=2),
            # 128x128 -> 256x256
            UpsampleBlock(hidden_dim // 2, hidden_dim // 4, scale_factor=2),
            # Final convolution to get desired number of channels
            nn.Conv2d(hidden_dim // 4, output_channels, kernel_size=1),
        )
        
        # Flag for gradient checkpointing
        self.gradient_checkpointing = gradient_checkpointing
    
    def _feature_extraction_checkpoint(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature extraction with gradient checkpointing."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0])
            return custom_forward
        
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.feature_extraction), 
            features
        )
    
    def _decoder_checkpoint(self, features: torch.Tensor) -> torch.Tensor:
        """Apply decoder with gradient checkpointing."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0])
            return custom_forward
        
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.decoder), 
            features
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, output_channels, height*4, width*4)
        """
        # Project input
        x = self.input_projection(x)
        
        # Extract features using PrithviWxC backbone
        features = self.backbone(x)
        
        # Extract additional features with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            features = self._feature_extraction_checkpoint(features)
        else:
            features = self.feature_extraction(features)
        
        # Decode to final output (4x upsampling) with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            output = self._decoder_checkpoint(features)
        else:
            output = self.decoder(features)
        
        return output
    
    def prepare_for_training(self, training_config: Dict[str, Any]) -> None:
        """Prepare the model for training.
        
        Args:
            training_config: Dictionary containing training configuration
        """
        self.train()
        
        # Enable gradient checkpointing if specified in config
        self.gradient_checkpointing = training_config.get('gradient_checkpointing', self.gradient_checkpointing)
        
        # Check if we should freeze the encoder
        freeze_encoder = training_config.get('freeze_encoder', self.freeze_encoder)
        if freeze_encoder:
            print("Freezing backbone parameters for training...")
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Even though we've frozen parameters, we still prepare the backbone
            # but with a modified config that doesn't change the frozen state
            modified_config = {**training_config, 'freeze_encoder': True}
            self.backbone.prepare_for_training(modified_config)
        else:
            # Apply to backbone normally if not freezing
            self.backbone.prepare_for_training({
                'gradient_checkpointing': self.gradient_checkpointing,
                **training_config
            })
        
        if self.gradient_checkpointing:
            print("Gradient checkpointing enabled for PrithviDownscaler")
    
    def prepare_for_inference(self) -> None:
        """Prepare the model for inference."""
        self.eval()
        # Disable gradient checkpointing for inference
        self.gradient_checkpointing = False
        self.backbone.prepare_for_inference()


def create_model(config: Dict[str, Any]) -> PrithviDownscaler:
    """Create a PrithviDownscaler model from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Initialized PrithviDownscaler model.
    """
    model_config = config['model']
    data_config = config['data']
    
    return PrithviDownscaler(
        input_channels=len(data_config['input_vars']),
        output_channels=len(data_config['target_vars']),
        hidden_dim=model_config['hidden_dim'],
        prithvi_checkpoint=model_config['prithvi_checkpoint'],
        # Use model_cache_dir if available, fall back to cache_dir for backward compatibility
        cache_dir=model_config.get('model_cache_dir', model_config.get('cache_dir', 'models/cache')),
        # Use hardware.device if available, fall back to model.device for backward compatibility
        device=config.get('hardware', {}).get('device', model_config.get('device', 'cpu')),
        use_pretrained=model_config['use_pretrained'],
        gradient_checkpointing=model_config.get('gradient_checkpointing', False),
        freeze_encoder=model_config.get('freeze_encoder', False)
    )


class PrithviDownscalerModule(pl.LightningModule):
    """Lightning module for training PrithviDownscaler."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[PrithviDownscaler] = None
    ):
        """Initialize the lightning module.
        
        Args:
            config: Configuration dictionary containing model and training parameters
            model: Optional pre-initialized PrithviDownscaler model
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize model if not provided
        if model is None:
            model = create_model(config)
        self.model = model
        
        # Loss weights
        self.mae_weight = config['loss']['mae_weight']
        self.mse_weight = config['loss']['mse_weight']
        self.ssim_weight = config['loss'].get('ssim_weight', 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Dictionary containing input and target tensors
            batch_idx: Index of the current batch
            
        Returns:
            Loss value
        """
        x, y = batch['merra2_input'], batch['prism_target']
        y_hat = self(x)
        
        # Calculate losses
        mae_loss = F.l1_loss(y_hat, y)
        mse_loss = F.mse_loss(y_hat, y)
        
        # Combine losses
        loss = self.mae_weight * mae_loss + self.mse_weight * mse_loss
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae_loss, on_step=True, on_epoch=True)
        self.log('train_mse', mse_loss, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step.
        
        Args:
            batch: Dictionary containing input and target tensors
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing validation metrics
        """
        x, y = batch['merra2_input'], batch['prism_target']
        y_hat = self(x)
        
        # Calculate metrics
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(mse)
        
        # Log metrics
        metrics = {
            'val_mae': mae,
            'val_mse': mse,
            'val_rmse': rmse
        }
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        
        return metrics
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step.
        
        Args:
            batch: Dictionary containing input and target tensors
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing test metrics
        """
        x, y = batch['merra2_input'], batch['prism_target']
        y_hat = self(x)
        
        # Calculate metrics
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(mse)
        
        # Log metrics
        metrics = {
            'test_mae': mae,
            'test_mse': mse,
            'test_rmse': rmse
        }
        self.log_dict(metrics)
        
        return metrics
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        # Get optimizer parameters
        optimizer_config = self.config['training']['optimizer']
        lr = optimizer_config['lr']
        weight_decay = optimizer_config['weight_decay']
        
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Get scheduler parameters
        scheduler_config = self.config['training']['scheduler']
        scheduler_name = scheduler_config['name']
        
        # Create scheduler
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=lr * 0.01
            )
        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.config['model']['scheduler_patience'],
                factor=self.config['model']['scheduler_factor']
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        } 