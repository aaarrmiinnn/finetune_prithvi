#!/usr/bin/env python
"""
Script to run inference with separate temperature and precipitation models
and combine their outputs into a unified prediction.
"""

import os
import sys
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainers import DownscalingModule
from src.data import create_dataloaders
from src.utils import load_config, setup_paths, get_experiment_name
from src.visualizations import visualize_predictions


def load_model(config_path: str, checkpoint_path: str) -> Tuple[Dict[str, Any], DownscalingModule]:
    """Load a model from a checkpoint using the specified config.
    
    Args:
        config_path: Path to the model's config file
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        Tuple of (config, model)
    """
    # Load and set up config
    config = load_config(config_path)
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


def combine_predictions(
    temp_pred: torch.Tensor,
    precip_pred: torch.Tensor,
    temp_config: Dict[str, Any],
    precip_config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Combine predictions from temperature and precipitation models.
    
    Args:
        temp_pred: Temperature prediction tensor
        precip_pred: Precipitation prediction tensor
        temp_config: Temperature model config
        precip_config: Precipitation model config
        
    Returns:
        Dictionary with combined predictions
    """
    # Create combined tensor
    combined_pred = torch.cat([temp_pred, precip_pred], dim=1)
    
    # Create mapping for the combined variables
    combined_vars = temp_config['data']['target_vars'] + precip_config['data']['target_vars']
    
    return {
        'combined_prediction': combined_pred,
        'target_vars': combined_vars,
        'temp_prediction': temp_pred,
        'precip_prediction': precip_pred
    }


def save_visualizations(
    input_data: torch.Tensor, 
    predictions: Dict[str, torch.Tensor],
    target: Optional[torch.Tensor],
    input_vars: List[str],
    target_vars: List[str],
    output_dir: str
) -> None:
    """Save visualization of predictions.
    
    Args:
        input_data: Input tensor
        predictions: Dictionary with prediction tensors
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
        
        # Get index of this variable in the combined prediction
        var_idx = target_vars.index(var)
        
        # Plot prediction
        pred_ax = axes[row, 1] if target is not None else axes[row, 1]
        pred_data = predictions['combined_prediction'][0, var_idx].cpu().numpy()
        im = pred_ax.imshow(pred_data)
        pred_ax.set_title(f"Prediction: {var}")
        plt.colorbar(im, ax=pred_ax)
        
        # Plot target if available
        if target is not None:
            target_ax = axes[row, 2]
            target_data = target[0, var_idx].cpu().numpy()
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
    plt.savefig(os.path.join(output_dir, 'combined_predictions.png'), dpi=300)
    plt.close(fig)
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with separate temperature and precipitation models")
    
    # Config and checkpoint paths
    parser.add_argument("--temp-config", type=str, default="src/config/temperature_config.yaml",
                        help="Path to temperature model config")
    parser.add_argument("--precip-config", type=str, default="src/config/precipitation_config.yaml",
                        help="Path to precipitation model config")
    parser.add_argument("--temp-checkpoint", type=str, required=True,
                        help="Path to temperature model checkpoint")
    parser.add_argument("--precip-checkpoint", type=str, required=True,
                        help="Path to precipitation model checkpoint")
    
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
    
    # Load models
    temp_config, temp_model = load_model(args.temp_config, args.temp_checkpoint)
    precip_config, precip_model = load_model(args.precip_config, args.precip_checkpoint)
    
    # Ensure we're using the same device for both models
    device = temp_config['hardware']['device']
    temp_model.to(device)
    precip_model.to(device)
    
    # Get input data
    if args.use_test_data:
        # Create dataloaders using the temperature config (input data is the same for both)
        _, _, test_loader = create_dataloaders(
            temp_config,
            train_split=temp_config['data']['train_test_split'][0],
            val_split=temp_config['data']['train_test_split'][1],
            test_split=temp_config['data']['train_test_split'][2],
            num_workers=temp_config['hardware']['num_workers']
        )
        
        # Get a batch from the test dataloader
        batch = next(iter(test_loader))
        input_data = batch['merra2_input'].to(device)
        target = batch['prism_target'].to(device)
        
        # Note: target contains both temperature and precipitation targets
        # We need to split it according to the config
        temp_target_idx = 0  # tdmean is first
        precip_target_idx = 1  # ppt is second
        
        # Create separate targets
        temp_target = target[:, temp_target_idx:temp_target_idx+1]
        precip_target = target[:, precip_target_idx:precip_target_idx+1]
    else:
        # Load custom input file
        if args.input_file is None:
            raise ValueError("When not using test data, --input-file must be specified")
        
        # Load and preprocess input data here
        # This would depend on your input format
        raise NotImplementedError("Custom input file loading not implemented yet")
    
    # Run inference
    temp_pred = run_inference(input_data, temp_model)
    precip_pred = run_inference(input_data, precip_model)
    
    # Combine predictions
    combined_preds = combine_predictions(temp_pred, precip_pred, temp_config, precip_config)
    
    # Save visualizations if requested
    if args.visualize:
        save_visualizations(
            input_data,
            combined_preds,
            target if args.use_test_data else None,
            temp_config['data']['input_vars'],
            combined_preds['target_vars'],
            args.output_dir
        )
    
    # Create metadata for saving
    metadata = {
        'temp_model': {
            'config': args.temp_config,
            'checkpoint': args.temp_checkpoint
        },
        'precip_model': {
            'config': args.precip_config,
            'checkpoint': args.precip_checkpoint
        },
        'target_vars': combined_preds['target_vars'],
        'input_vars': temp_config['data']['input_vars'],
        'timestamp': torch.datetime.datetime.now().isoformat()
    }
    
    # Save predictions and metadata
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'combined_predictions.npz')
    
    # Convert tensors to numpy for saving
    np_preds = {
        'combined_prediction': combined_preds['combined_prediction'].cpu().numpy(),
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