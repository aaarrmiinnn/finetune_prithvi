"""Prithvi-based downscaling model for MERRA-2 to PRISM."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Tuple, Optional, Union, Any


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
    """Downscaling model based on Prithvi WxC."""
    
    def __init__(
        self,
        prithvi_checkpoint: str = "ibm/prithvi-100m-wxc",
        use_pretrained: bool = True,
        freeze_encoder: bool = False,
        unfreeze_layers: List[str] = None,
        input_channels: int = 4,  # Number of MERRA-2 variables
        output_channels: int = 2,  # Number of PRISM variables
        embedding_dim: int = 768,
        upsampling_scales: List[int] = [2, 2],  # Total upsampling factor = product of scales
        dropout: float = 0.1,
        use_dem: bool = True
    ):
        """Initialize the Prithvi downscaler.
        
        Args:
            prithvi_checkpoint: Pretrained Prithvi checkpoint from HuggingFace.
            use_pretrained: Whether to use pretrained weights.
            freeze_encoder: Whether to freeze the Prithvi encoder.
            unfreeze_layers: List of layer names to unfreeze if encoder is frozen.
            input_channels: Number of input channels (MERRA-2 variables).
            output_channels: Number of output channels (PRISM variables).
            embedding_dim: Embedding dimension.
            upsampling_scales: List of upsampling scales for each level.
            dropout: Dropout rate.
            use_dem: Whether to use DEM as auxiliary input.
        """
        super().__init__()
        
        self.use_pretrained = use_pretrained
        self.freeze_encoder = freeze_encoder
        self.use_dem = use_dem
        
        # Load Prithvi model
        if use_pretrained:
            self.config = AutoConfig.from_pretrained(prithvi_checkpoint)
            self.prithvi = AutoModel.from_pretrained(prithvi_checkpoint)
        else:
            self.config = AutoConfig.from_pretrained(prithvi_checkpoint)
            self.prithvi = AutoModel.from_config(self.config)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.prithvi.parameters():
                param.requires_grad = False
            
            # Unfreeze specific layers if requested
            if unfreeze_layers:
                for name, param in self.prithvi.named_parameters():
                    if any(layer in name for layer in unfreeze_layers):
                        param.requires_grad = True
        
        # Input adaptation layer to convert from input_channels to Prithvi's expected channels
        # Prithvi typically expects 160 channels (80 variables x 2 timestamps)
        self.input_adapter = nn.Conv2d(input_channels, 160, kernel_size=1)
        
        # DEM encoder
        if use_dem:
            self.dem_encoder = DemEncoder(in_channels=1, out_channels=embedding_dim // 2)
            self.dem_projection = nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=1)
        
        # Feature extraction from Prithvi's output
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder with upsampling
        decoder_layers = []
        in_channels = embedding_dim
        
        for i, scale in enumerate(upsampling_scales):
            out_channels = embedding_dim // (2 ** (i + 1)) if i < len(upsampling_scales) - 1 else output_channels
            decoder_layers.append(
                UpsampleBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    scale_factor=scale,
                    use_residual=True
                )
            )
            in_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        dem: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            dem: DEM tensor of shape (B, 1, H*4, W*4).
            
        Returns:
            Output tensor of shape (B, output_channels, H*4, W*4).
        """
        batch_size, _, height, width = x.shape
        
        # Adapt input channels to Prithvi's expected format
        x = self.input_adapter(x)
        
        # Reshape to Prithvi's expected shape (B, H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)
        
        # Forward through Prithvi
        outputs = self.prithvi(inputs_embeds=x)
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state  # (B, H*W, embedding_dim)
        
        # Reshape back to spatial feature map (B, embedding_dim, H, W)
        features = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
        
        # Process DEM if available
        if self.use_dem and dem is not None:
            dem_features = self.dem_encoder(dem)
            dem_features = self.dem_projection(dem_features)
            
            # Add DEM features to Prithvi features
            features = features + dem_features
        
        # Extract features
        features = self.feature_extractor(features)
        
        # Decode and upsample
        output = self.decoder(features)
        
        return output


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
        prithvi_checkpoint=model_config['prithvi_checkpoint'],
        use_pretrained=model_config['use_pretrained'],
        freeze_encoder=model_config['freeze_encoder'],
        unfreeze_layers=model_config['unfreeze_layers'],
        input_channels=len(data_config['input_vars']),
        output_channels=len(data_config['target_vars']),
        embedding_dim=model_config['embedding_dim'],
        upsampling_scales=model_config['upsampling_scales'],
        dropout=model_config['dropout'],
        use_dem=model_config['auxiliary_input']
    ) 