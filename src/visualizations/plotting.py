"""Visualization utilities for downscaling results."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path


def plot_comparison(
    merra2: Union[torch.Tensor, np.ndarray],
    prism: Union[torch.Tensor, np.ndarray],
    prediction: Union[torch.Tensor, np.ndarray],
    variable_names: List[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
):
    """Plot comparison between MERRA-2, PRISM, and prediction.
    
    Args:
        merra2: MERRA-2 data of shape (C, H, W).
        prism: PRISM data of shape (C, H*4, W*4).
        prediction: Predicted data of shape (C, H*4, W*4).
        variable_names: Names of variables for subplot titles.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to show the figure.
        title: Title for the figure.
    """
    # Convert tensors to numpy arrays
    if isinstance(merra2, torch.Tensor):
        merra2 = merra2.detach().cpu().numpy()
    if isinstance(prism, torch.Tensor):
        prism = prism.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    # Get number of variables (channels)
    n_vars = merra2.shape[0]
    
    # Set up variable names if not provided
    if variable_names is None:
        variable_names = [f"Variable {i+1}" for i in range(n_vars)]
    
    # Create figure
    fig, axes = plt.subplots(n_vars, 3, figsize=(15, 5 * n_vars))
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # For a single variable, make sure axes is 2D
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each variable
    for i in range(n_vars):
        # Extract data for this variable
        merra2_var = merra2[i]
        prism_var = prism[i]
        pred_var = prediction[i]
        
        # Get data ranges for consistent colormaps
        vmin = min(prism_var.min(), pred_var.min())
        vmax = max(prism_var.max(), pred_var.max())
        
        # Plot MERRA-2 (resized to match PRISM resolution)
        from scipy.ndimage import zoom
        zoom_factor = prism_var.shape[0] / merra2_var.shape[0]
        merra2_resized = zoom(merra2_var, zoom_factor, order=1)
        
        im0 = axes[i, 0].imshow(merra2_resized, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"MERRA-2: {variable_names[i]}")
        plt.colorbar(im0, ax=axes[i, 0])
        
        # Plot PRISM
        im1 = axes[i, 1].imshow(prism_var, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"PRISM: {variable_names[i]}")
        plt.colorbar(im1, ax=axes[i, 1])
        
        # Plot prediction
        im2 = axes[i, 2].imshow(pred_var, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f"Prediction: {variable_names[i]}")
        plt.colorbar(im2, ax=axes[i, 2])
        
        # Remove axis ticks
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add a row for error map
    fig, axes = plt.subplots(n_vars, 2, figsize=(10, 5 * n_vars))
    
    # For a single variable, make sure axes is 2D
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    # Plot error maps
    for i in range(n_vars):
        # Extract data for this variable
        prism_var = prism[i]
        pred_var = prediction[i]
        
        # Calculate absolute error
        abs_error = np.abs(pred_var - prism_var)
        
        # Plot absolute error
        im0 = axes[i, 0].imshow(abs_error, cmap='hot')
        axes[i, 0].set_title(f"Absolute Error: {variable_names[i]}")
        plt.colorbar(im0, ax=axes[i, 0])
        
        # Plot error histogram
        axes[i, 1].hist(abs_error.flatten(), bins=50)
        axes[i, 1].set_title(f"Error Distribution: {variable_names[i]}")
        axes[i, 1].set_xlabel("Absolute Error")
        axes[i, 1].set_ylabel("Frequency")
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_history(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
):
    """Plot metrics history.
    
    Args:
        metrics_history: Dictionary mapping metric names to lists of values.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to show the figure.
        title: Title for the figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=16)
    
    # Plot each metric
    for metric_name, values in metrics_history.items():
        ax.plot(values, label=metric_name)
    
    # Add labels and legend
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_predictions(
    test_batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    config: Dict[str, Any],
    save_dir: Optional[str] = None,
    show: bool = True
):
    """Visualize predictions for a test batch.
    
    Args:
        test_batch: Dictionary containing test data.
        predictions: Predicted tensor.
        config: Configuration dictionary.
        save_dir: Directory to save figures. If None, figures are not saved.
        show: Whether to show figures.
    """
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get variable names
    target_vars = config['data']['target_vars']
    
    # Extract data
    merra2_input = test_batch['merra2_input']
    prism_target = test_batch['prism_target']
    
    # Loop over batch
    for i in range(merra2_input.shape[0]):
        # Plot comparison for this sample
        plot_comparison(
            merra2=merra2_input[i],
            prism=prism_target[i],
            prediction=predictions[i],
            variable_names=target_vars,
            save_path=os.path.join(save_dir, f"comparison_{i}.png") if save_dir else None,
            show=show,
            title=f"Sample {i}"
        ) 