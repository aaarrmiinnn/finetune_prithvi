"""Configuration utilities for the project."""
import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set up and validate all paths in the configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Updated configuration with validated paths.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    
    # Create log directory if it doesn't exist
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Validate data paths
    for path_key in ['merra2_dir', 'prism_dir', 'dem_dir']:
        path = config['data'][path_key]
        if not os.path.exists(path):
            raise ValueError(f"Data path {path} does not exist.")
    
    # Validate DEM file
    dem_file = os.path.join(config['data']['dem_dir'], config['data']['dem_file'])
    if not os.path.exists(dem_file):
        raise ValueError(f"DEM file {dem_file} does not exist.")
    
    # Add the full DEM path to the config
    config['data']['dem_path'] = dem_file
    
    return config


def get_experiment_name(config: Dict[str, Any]) -> str:
    """Create a unique experiment name based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        A unique experiment name.
    """
    model_name = config['model']['prithvi_checkpoint'].split('/')[-1]
    freeze_str = "frozen" if config['model']['freeze_encoder'] else "unfrozen"
    batch_size = config['training']['batch_size']
    lr = float(config['training']['optimizer']['lr'])  # Ensure lr is float
    
    return f"{model_name}_{freeze_str}_bs{batch_size}_lr{lr:.1e}" 