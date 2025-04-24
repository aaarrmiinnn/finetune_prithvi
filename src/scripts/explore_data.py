"""Script to explore the MERRA-2 and PRISM data."""
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import (
    load_merra2_data,
    load_prism_data,
    load_dem_data
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore MERRA-2 and PRISM data")
    
    parser.add_argument(
        "--merra2-dir", type=str, default="data/merra2",
        help="Directory containing MERRA-2 files"
    )
    
    parser.add_argument(
        "--prism-dir", type=str, default="data/prism",
        help="Directory containing PRISM files"
    )
    
    parser.add_argument(
        "--dem-path", type=str, default="data/dem/PRISM_us_dem_4km_bil.bil",
        help="Path to DEM file"
    )
    
    parser.add_argument(
        "--date", type=str, default="20250301",
        help="Date to explore in format YYYYMMDD"
    )
    
    parser.add_argument(
        "--merra2-vars", type=str, nargs="+", default=["T2MMAX", "T2MMEAN", "T2MMIN", "TPRECMAX"],
        help="MERRA-2 variables to explore"
    )
    
    parser.add_argument(
        "--prism-vars", type=str, nargs="+", default=["tdmean", "ppt"],
        help="PRISM variables to explore"
    )
    
    return parser.parse_args()


def plot_data(merra2_data, prism_data, dem_data=None, 
             merra2_vars=None, prism_vars=None, date=None):
    """Plot MERRA-2 and PRISM data.
    
    Args:
        merra2_data: MERRA-2 dataset.
        prism_data: Dictionary of PRISM data arrays.
        dem_data: DEM data array.
        merra2_vars: List of MERRA-2 variable names.
        prism_vars: List of PRISM variable names.
        date: Date string.
    """
    n_merra2_vars = len(merra2_vars) if merra2_vars else len(merra2_data.data_vars)
    n_prism_vars = len(prism_vars) if prism_vars else len(prism_data)
    
    # Create figure
    n_rows = max(n_merra2_vars, n_prism_vars) + (1 if dem_data is not None else 0)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    
    # Add title
    if date:
        fig.suptitle(f"Data for {date}", fontsize=16)
    
    # If there's only one variable, make sure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot MERRA-2 data
    for i, var in enumerate(merra2_vars):
        if var in merra2_data:
            im = axes[i, 0].imshow(merra2_data[var].values, cmap='viridis')
            axes[i, 0].set_title(f"MERRA-2: {var}")
            plt.colorbar(im, ax=axes[i, 0])
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
    
    # Plot PRISM data
    for i, var in enumerate(prism_vars):
        if var in prism_data:
            im = axes[i, 1].imshow(prism_data[var].values, cmap='viridis')
            axes[i, 1].set_title(f"PRISM: {var}")
            plt.colorbar(im, ax=axes[i, 1])
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
    
    # Plot DEM if available
    if dem_data is not None:
        i = n_rows - 1
        im = axes[i, 0].imshow(dem_data.values, cmap='terrain')
        axes[i, 0].set_title("DEM")
        plt.colorbar(im, ax=axes[i, 0])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Make the DEM plot span both columns
        axes[i, 1].remove()
        axes[i, 0].set_position([0.125, 0.1, 0.775, 0.2])
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Extract date components for MERRA-2 file matching
    year = args.date[:4]
    month = args.date[4:6]
    day = args.date[6:8]
    
    # Find MERRA-2 file
    merra2_pattern = f"MERRA2_400.statD_2d_slv_Nx.{year}{month}{day}.nc4.nc4"
    merra2_file = os.path.join(args.merra2_dir, merra2_pattern)
    
    if not os.path.exists(merra2_file):
        print(f"Error: MERRA-2 file not found: {merra2_file}")
        return
    
    # Load data
    try:
        # Load MERRA-2 data
        print(f"Loading MERRA-2 data from {merra2_file}...")
        merra2_data = load_merra2_data(merra2_file, args.merra2_vars)
        print(f"MERRA-2 data loaded with variables: {list(merra2_data.data_vars)}")
        print(f"MERRA-2 data shape: {merra2_data[args.merra2_vars[0]].shape}")
        
        # Load PRISM data
        print(f"Loading PRISM data from {args.prism_dir} for date {args.date}...")
        prism_data = load_prism_data(args.prism_dir, args.date, args.prism_vars)
        print(f"PRISM data loaded with variables: {list(prism_data.keys())}")
        print(f"PRISM data shape: {prism_data[args.prism_vars[0]].shape}")
        
        # Load DEM data
        dem_data = None
        if os.path.exists(args.dem_path):
            print(f"Loading DEM data from {args.dem_path}...")
            dem_data = load_dem_data(args.dem_path)
            print(f"DEM data shape: {dem_data.shape}")
        
        # Plot data
        plot_data(
            merra2_data=merra2_data,
            prism_data=prism_data,
            dem_data=dem_data,
            merra2_vars=args.merra2_vars,
            prism_vars=args.prism_vars,
            date=args.date
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 