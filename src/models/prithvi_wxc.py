"""
Prithvi-WxC model wrapper for downscaling task.
"""
import os
import torch
from transformers import AutoConfig, AutoModel
from typing import Optional, Dict, Any

class PrithviWxC:
    """Wrapper for Prithvi-WxC model."""
    
    def __init__(
        self,
        model_name: str = "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        cache_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models", "cache")
        self.device = device
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load model configuration
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Load pre-trained model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            config=self.config,
            cache_dir=self.cache_dir
        )
        
        # Move model to appropriate device
        self.model.to(self.device)
        
    def prepare_for_training(self, training_config: Dict[Any, Any]) -> None:
        """
        Prepare model for fine-tuning on downscaling task.
        
        Args:
            training_config: Dictionary containing training configuration
        """
        # Set model to training mode
        self.model.train()
        
        # Configure for mixed precision if enabled
        if training_config.get("precision") == 16:
            self.model = self.model.half()
    
    def prepare_for_inference(self) -> None:
        """Prepare model for inference."""
        self.model.eval()
    
    @property
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time, height, width)
            
        Returns:
            Model output tensor
        """
        return self.model(x)

if __name__ == "__main__":
    # Test model loading
    print("Testing Prithvi-WxC model loading...")
    model = PrithviWxC()
    print(f"Model loaded successfully on device: {model.device}")
    print(f"Model configuration: {model.config}") 