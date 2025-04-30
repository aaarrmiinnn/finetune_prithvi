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
        config['model']['hidden_dim'] = 16
        config['model']['device'] = 'cuda'
        config['model']['gradient_checkpointing'] = True
    
    # Update data settings
    if 'data' in config:
        config['data']['patch_size'] = 8
    
    # Update training settings
    if 'training' in config:
        config['training']['epochs'] = 100
        config['training']['batch_size'] = 1
        config['training']['precision'] = 16
        config['training']['accumulate_grad_batches'] = 16
    
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