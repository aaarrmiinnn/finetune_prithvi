#!/usr/bin/env python
# Renamed from inference_separate_variables.py to inference_with_variables.py
"""
Script to run inference with variable selection.
This script supports inference with models trained on different variable combinations.
"""

import os
import sys
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainers import DownscalingModule
from src.data import create_dataloaders
from src.utils import load_config, setup_paths, get_experiment_name
from src.visualizations import visualize_predictions


def modify_config_for_variables(config_path: str, variables: List[str]) -> Dict[str, Any]:
    """Modify the config to use specific variables.
    
    Args:
        config_path: Path to the base config file
        variables: List of variables to use
        
    Returns:
        Modified config dict
    """
    # Load the base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the target variables
    config['data']['target_vars'] = variables
    
    # If we have a single variable, apply the variable-specific settings
    if len(variables) == 1:
        var_map = {'tdmean': 'temperature', 'ppt': 'precipitation'}
        var_type = var_map.get(variables[0], 'common')
        
        # Apply model settings
        if 'variable_config' in config['model'] and var_type in config['model']['variable_config']:
            for key, value in config['model']['variable_config'][var_type].items():
                config['model'][key] = value
        
        # Apply loss settings
        if var_type in config['loss']:
            for key, value in config['loss'][var_type].items():
                config['loss'][key] = value
    
    return config


def load_model(config_path: str, checkpoint_path: str, variables: List[str]) -> Tuple[Dict[str, Any], DownscalingModule]:
    """Load a model from a checkpoint using a config modified for the specified variables.
    
    Args:
        config_path: Path to the model's config file
        checkpoint_path: Path to the model checkpoint
        variables: List of variables the model was trained on
        
    Returns:
        Tuple of (config, model)
    """
    # Modify config for the variables
    config = modify_config_for_variables(config_path, variables)
    config = setup_paths(config)
    
    # Load the model
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = DownscalingModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Ensure model is on the right device
    device = config['hardware']['device']
    model.to(device)
    
    return config, model


def run_inference(input_data: torch.Tensor, model: DownscalingModule) -> torch.Tensor:
    """Run inference on input data.
    
    Args:
        input_data: Input tensor of shape [batch_size, channels, height, width]
        model: DownscalingModule model
        
    Returns:
        Predictions tensor
    """
    with torch.no_grad():
        return model(input_data)


def save_visualizations(
    input_data: torch.Tensor, 
    prediction: torch.Tensor,
    target: Optional[torch.Tensor],
    input_vars: List[str],
    target_vars: List[str],
    output_dir: str
) -> None:
    """Save visualization of predictions.
    
    Args:
        input_data: Input tensor
        prediction: Prediction tensor
        target: Optional target tensor for comparison
        input_vars: List of input variable names
        target_vars: List of target variable names
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=len(target_vars) + 1,  # +1 for input
        ncols=3 if target is not None else 2,  # Input, Prediction, (Target)
        figsize=(12, 4 * len(target_vars)),
        constrained_layout=True
    )
    
    # Plot input (use the first channel as example)
    ax = axes[0, 0] if len(target_vars) > 1 else axes[0]
    im = ax.imshow(input_data[0, 0].cpu().numpy())
    ax.set_title(f"Input: {input_vars[0]}")
    plt.colorbar(im, ax=ax)
    
    # Plot other columns as blank for input row
    if target is not None:
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
    else:
        axes[0, 1].axis('off')
    
    # Plot predictions and targets for each variable
    for i, var in enumerate(target_vars):
        row = i + 1  # +1 because first row is input
        
        # Plot prediction
        pred_ax = axes[row, 1] if target is not None else axes[row, 1]
        pred_data = prediction[0, i].cpu().numpy()
        im = pred_ax.imshow(pred_data)
        pred_ax.set_title(f"Prediction: {var}")
        plt.colorbar(im, ax=pred_ax)
        
        # Plot target if available
        if target is not None:
            target_ax = axes[row, 2]
            target_data = target[0, i].cpu().numpy()
            im = target_ax.imshow(target_data)
            target_ax.set_title(f"Target: {var}")
            plt.colorbar(im, ax=target_ax)
            
            # Plot difference
            diff_ax = axes[row, 0]
            diff_data = pred_data - target_data
            im = diff_ax.imshow(diff_data, cmap='RdBu_r')
            diff_ax.set_title(f"Difference: {var}")
            plt.colorbar(im, ax=diff_ax)
        else:
            # If no target, plot enhanced version of prediction in first column
            axes[row, 0].axis('off')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=300)
    plt.close(fig)
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    
    # Config and checkpoint paths
    parser.add_argument("--config", type=str, default="src/config/config.yaml",
                        help="Path to base config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Variable selection
    parser.add_argument("--variables", type=str, default="tdmean,ppt",
                        help="Comma-separated list of variables the model was trained on")
    
    # Input data options
    parser.add_argument("--input-file", type=str, default=None,
                        help="Path to input file (if not using test dataloader)")
    parser.add_argument("--use-test-data", action="store_true",
                        help="Use test data from the dataloader instead of custom input")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output/predictions",
                        help="Directory to save output predictions")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of predictions")
    
    args = parser.parse_args()
    
    # Parse variables
    variables = args.variables.split(',')
    print(f"Running inference for variables: {variables}")
    
    # Load model
    config, model = load_model(args.config, args.checkpoint, variables)
    
    # Get input data
    if args.use_test_data:
        # Create dataloaders
        _, _, test_loader = create_dataloaders(
            config,
            train_split=config['data']['train_test_split'][0],
            val_split=config['data']['train_test_split'][1],
            test_split=config['data']['train_test_split'][2],
            num_workers=config['hardware']['num_workers']
        )
        
        # Get a batch from the test dataloader
        batch = next(iter(test_loader))
        input_data = batch['merra2_input'].to(config['hardware']['device'])
        target = batch['prism_target'].to(config['hardware']['device'])
    else:
        # Load custom input file
        if args.input_file is None:
            raise ValueError("When not using test data, --input-file must be specified")
        
        # Load and preprocess input data here
        # This would depend on your input format
        raise NotImplementedError("Custom input file loading not implemented yet")
    
    # Run inference
    prediction = run_inference(input_data, model)
    
    # Save visualizations if requested
    if args.visualize:
        save_visualizations(
            input_data,
            prediction,
            target if args.use_test_data else None,
            config['data']['input_vars'],
            variables,
            args.output_dir
        )
    
    # Create metadata for saving
    metadata = {
        'model': {
            'config': args.config,
            'checkpoint': args.checkpoint,
            'variables': variables
        },
        'input_vars': config['data']['input_vars'],
        'target_vars': variables,
        'timestamp': torch.datetime.datetime.now().isoformat()
    }
    
    # Save predictions and metadata
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{'_'.join(variables)}_predictions.npz")
    
    # Convert tensors to numpy for saving
    np_preds = {
        'prediction': prediction.cpu().numpy(),
        'input_data': input_data.cpu().numpy(),
    }
    
    if args.use_test_data:
        np_preds['target'] = target.cpu().numpy()
    
    # Save with metadata
    np.savez_compressed(
        output_file,
        **np_preds,
        metadata=metadata
    )
    
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main() 