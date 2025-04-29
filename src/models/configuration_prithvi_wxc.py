from transformers import PretrainedConfig
from typing import Optional

class PrithviWxCConfig(PretrainedConfig):
    """Configuration class for Prithvi-WxC model."""
    
    model_type = "prithvi_wxc"
    
    def __init__(
        self,
        num_attention_heads: int = 32,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 30,
        num_channels: int = 160,
        patch_size: int = 16,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        vocab_size: int = 1,
        encoder_layers: int = 25,
        decoder_layers: int = 5,
        max_position_embeddings: int = 4096,
        **kwargs
    ):
        """Initialize the Prithvi-WxC configuration.
        
        Args:
            num_attention_heads (int): Number of attention heads for each attention layer.
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
            intermediate_size (int): Dimensionality of the "intermediate" (often named feed-forward) layer.
            num_hidden_layers (int): Number of hidden layers in the model.
            num_channels (int): Number of channels in the input.
            patch_size (int): Size of patches for patch embedding.
            attention_probs_dropout_prob (float): Dropout ratio for attention probabilities.
            hidden_dropout_prob (float): Dropout ratio for hidden states.
            initializer_range (float): Standard deviation for initializing parameters.
            layer_norm_eps (float): Epsilon used by layer normalization layers.
            use_cache (bool): Whether to use the past key/values attentions.
            vocab_size (int): Size of the vocabulary (always 1 for vision models).
            encoder_layers (int): Number of encoder layers.
            decoder_layers (int): Number of decoder layers.
            max_position_embeddings (int): Maximum sequence length supported.
        """
        super().__init__(**kwargs)
        
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.max_position_embeddings = max_position_embeddings 