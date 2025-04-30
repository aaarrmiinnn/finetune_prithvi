#!/usr/bin/env python3
"""
Update YAML configuration file with memory-efficient settings.
This script properly handles YAML structure and indentation.
"""
import yaml
import argparse
import os

def update_config(config_path, output_path):
    """Update the config file with memory-efficient settings."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model settings
    if 'model' in config:
        # Set hidden_dim to 64 to work with 4 or 8 attention heads
        # 64 is a good balance between memory efficiency and model capacity
        config['model']['hidden_dim'] = 64  
        config['model']['device'] = 'cuda'
        config['model']['gradient_checkpointing'] = True
        
        # Explicitly freeze the Prithvi backbone - this is important!
        config['model']['freeze_encoder'] = True
        config['model']['use_pretrained'] = True  # Ensure we're using pretrained weights
    
    # Update data settings
    if 'data' in config:
        config['data']['patch_size'] = 8
    
    # Update training settings
    if 'training' in config:
        config['training']['epochs'] = 100
        config['training']['batch_size'] = 1
        config['training']['precision'] = 16
        config['training']['accumulate_grad_batches'] = 16
        # Add gradient clipping for stability
        config['training']['gradient_clip_val'] = 1.0
    
    # Update loss settings to favor MAE over MSE (more stable)
    if 'loss' in config:
        config['loss']['mae_weight'] = 1.0
        config['loss']['mse_weight'] = 0.1
        config['loss']['ssim_weight'] = 0.0  # Disable SSIM to save memory
    
    # Update hardware settings
    if 'hardware' in config:
        config['hardware']['accelerator'] = 'gpu'
        config['hardware']['num_workers'] = 2
        config['hardware']['pin_memory'] = True
    
    # Update cluster settings
    if 'cluster' in config:
        config['cluster']['enabled'] = True
        config['cluster']['strategy'] = 'auto'
    
    # Write the updated config to the output file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config updated and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update config for memory efficiency")
    parser.add_argument("--input", type=str, default="src/config/config.yaml", 
                        help="Input config file path")
    parser.add_argument("--output", type=str, default="src/config/config_memory_efficient.yaml",
                        help="Output config file path")
    
    args = parser.parse_args()
    update_config(args.input, args.output) 