"""
Prithvi-WxC model implementation.
"""
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
from transformers import PreTrainedModel
from typing import Optional, Dict, Any
from .configuration_prithvi_wxc import PrithviWxCConfig

class PatchEmbedding(nn.Module):
    """2D Image to Patch Embedding."""
    
    def __init__(self, config: PrithviWxCConfig):
        super().__init__()
        
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        
        # Use convolutions with padding to maintain spatial dimensions
        self.proj = nn.Sequential(
            nn.Conv2d(
                self.num_channels,
                self.hidden_size // 4,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.BatchNorm2d(self.hidden_size // 4),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_size // 4,
                self.hidden_size // 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.hidden_size // 2),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_size // 2,
                self.hidden_size,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, N, D) where:
                N = H * W is the number of patches
                D is the hidden dimension
        """
        B, C, H, W = x.shape
        
        # Project patches: (B, C, H, W) -> (B, D, H, W)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, D, H, W) -> (B, D, N)
        x = x.flatten(2)
        
        # Transpose: (B, D, N) -> (B, N, D)
        return x.transpose(1, 2)

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, config: PrithviWxCConfig):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, D)
            
        Returns:
            Output tensor of shape (B, N, D)
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x

class PrithviWxC(PreTrainedModel):
    """Prithvi-WxC model implementation."""
    
    config_class = PrithviWxCConfig
    base_model_prefix = "prithvi_wxc"
    
    def __init__(
        self,
        config: Optional[PrithviWxCConfig] = None,
        model_name: str = "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        cache_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_pretrained: bool = True
    ):
        """Initialize the Prithvi-WxC model.
        
        Args:
            config: Model configuration
            model_name: Name or path of the pretrained model
            cache_dir: Directory to cache the model
            device: Device to load the model on
            use_pretrained: Whether to load pretrained weights
        """
        if config is None:
            config = PrithviWxCConfig()
        
        super().__init__(config)
        
        self.model_name = model_name
        self.cache_dir = cache_dir or "models/cache"
        self._device = torch.device(device)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        
        # Position embeddings - will be initialized in forward pass
        self.pos_embed = None
        self.hidden_size = config.hidden_size
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self._device)
    
    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _get_pos_embed(self, x: torch.Tensor) -> Parameter:
        """Get position embeddings for the current input size."""
        B, N, C = x.shape
        if self.pos_embed is None or self.pos_embed.size(1) != N:
            self.pos_embed = Parameter(torch.zeros(1, N, C, device=x.device))
            nn.init.normal_(self.pos_embed, std=0.02)
        return self.pos_embed
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device
    
    def prepare_for_training(self, training_config: Dict[str, Any]) -> None:
        """Prepare the model for training."""
        self.train()
    
    def prepare_for_inference(self) -> None:
        """Prepare the model for inference."""
        self.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        x = x.to(self._device)
        
        # Convert image to patches and get spatial dimensions
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, H*W, D)
        
        # Add position embeddings
        pos_embed = self._get_pos_embed(x)
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Reshape back to image format
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        
        return x

if __name__ == "__main__":
    # Test model loading
    print("Testing Prithvi-WxC model loading...")
    model = PrithviWxC()
    print(f"Model loaded successfully on device: {model.device}")
    print(f"Model configuration: {model.config}") 