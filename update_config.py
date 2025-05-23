#!/usr/bin/env python3
"""
Update YAML configuration file with memory-efficient settings.
This script properly handles YAML structure and indentation.
"""
import yaml
import argparse
import os
import glob
import re

def find_available_prism_vars(prism_dir):
    """Find available PRISM variables based on files in directory."""
    # Look for files matching pattern prism_<var>_us_25m_YYYYMMDD.zip
    prism_files = glob.glob(os.path.join(prism_dir, "prism_*_us_25m_*.zip"))
    
    # Extract variable names using regex
    var_pattern = re.compile(r'prism_([^_]+)_us_25m_\d{8}\.zip')
    available_vars = set()
    
    for f in prism_files:
        match = var_pattern.search(os.path.basename(f))
        if match:
            available_vars.add(match.group(1))
    
    return list(available_vars)

def find_available_dates(config):
    """Find dates that have both MERRA-2 and PRISM data."""
    merra2_dir = config['data']['prism_dir']
    prism_dir = config['data']['prism_dir']
    target_vars = config['data']['target_vars']
    
    # Find MERRA-2 dates
    merra2_files = glob.glob(os.path.join(merra2_dir, "MERRA2_400.statD_2d_slv_Nx.*.nc4"))
    merra2_dates = set()
    for f in merra2_files:
        match = re.search(r'MERRA2_400\.statD_2d_slv_Nx\.(\d{8})\.nc4', os.path.basename(f))
        if match:
            merra2_dates.add(match.group(1))
    
    # Find dates with all required PRISM variables
    valid_dates = []
    for date in merra2_dates:
        all_vars_exist = True
        for var in target_vars:
            prism_pattern = os.path.join(prism_dir, f"prism_{var}_us_25m_{date}.zip")
            if not glob.glob(prism_pattern):
                all_vars_exist = False
                break
        
        if all_vars_exist:
            valid_dates.append(date)
    
    return valid_dates

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
        # Find available PRISM variables
        prism_dir = config['data']['prism_dir']
        available_vars = find_available_prism_vars(prism_dir)
        
        if available_vars:
            print(f"Found these available PRISM variables: {available_vars}")
            config['data']['target_vars'] = available_vars
        
        config['data']['patch_size'] = 8
        # Set dates to an empty list to auto-detect available dates
        config['data']['dates'] = []
    
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