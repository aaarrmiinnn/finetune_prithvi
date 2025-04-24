"""Data preprocessing for MERRA-2 and PRISM datasets."""
import os
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from typing import Dict, List, Tuple, Optional, Union
import glob
import zipfile
import tempfile
from pathlib import Path


def load_merra2_data(file_path: str, variables: List[str]) -> xr.Dataset:
    """Load MERRA-2 NetCDF4 data.
    
    Args:
        file_path: Path to MERRA-2 NetCDF4 file.
        variables: List of variables to extract.
        
    Returns:
        xarray Dataset containing the requested variables.
    """
    try:
        ds = xr.open_dataset(file_path)
        
        # Check if all variables exist in the dataset
        missing_vars = [var for var in variables if var not in ds.data_vars]
        if missing_vars:
            raise ValueError(f"Variables {missing_vars} not found in MERRA-2 file.")
        
        # Select only the requested variables
        ds = ds[variables]
        
        return ds
    except Exception as e:
        raise RuntimeError(f"Error loading MERRA-2 data: {str(e)}")


def load_prism_data(prism_dir: str, date: str, variables: List[str]) -> Dict[str, xr.DataArray]:
    """Load PRISM data from zip files.
    
    Args:
        prism_dir: Directory containing PRISM zip files.
        date: Date string in format YYYYMMDD.
        variables: List of PRISM variables to load (e.g., ['tdmean', 'ppt']).
        
    Returns:
        Dictionary mapping variable names to xarray DataArrays.
    """
    result = {}
    
    for var in variables:
        # Find the zip file for this variable and date
        zip_pattern = os.path.join(prism_dir, f"prism_{var}_us_25m_{date}.zip")
        zip_files = glob.glob(zip_pattern)
        
        if not zip_files:
            raise FileNotFoundError(f"No PRISM file found for variable {var} and date {date}")
        
        zip_file = zip_files[0]
        
        # Extract and load the GeoTIFF file
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as z:
                # Find the .tif file
                tif_files = [f for f in z.namelist() if f.endswith('.tif')]
                if not tif_files:
                    raise FileNotFoundError(f"No .tif file found in {zip_file}")
                
                tif_file = tif_files[0]
                z.extract(tif_file, temp_dir)
                
                # Load with rioxarray
                tif_path = os.path.join(temp_dir, tif_file)
                da = rioxarray.open_rasterio(tif_path)
                
                # Standardize the dimensions
                da = da.squeeze(drop=True)  # Remove single-dimension axes
                
                result[var] = da
    
    return result


def load_dem_data(dem_path: str) -> xr.DataArray:
    """Load Digital Elevation Model data.
    
    Args:
        dem_path: Path to DEM file.
        
    Returns:
        xarray DataArray containing the DEM data.
    """
    try:
        # Check file extension
        if dem_path.endswith('.bil'):
            # GDAL/rasterio can read .bil files
            dem = rioxarray.open_rasterio(dem_path)
            dem = dem.squeeze(drop=True)  # Remove single-dimension axes
            return dem
        else:
            raise ValueError(f"Unsupported DEM file format: {dem_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading DEM data: {str(e)}")


def reproject_merra2_to_prism(merra2_data: xr.Dataset, prism_sample: xr.DataArray) -> xr.Dataset:
    """Reproject MERRA-2 data to match PRISM grid.
    
    This is a preprocessing step to align the data for visualization and comparison.
    The actual downscaling will be handled by the model.
    
    Args:
        merra2_data: MERRA-2 dataset.
        prism_sample: PRISM data array to use as reference grid.
        
    Returns:
        Reprojected MERRA-2 dataset.
    """
    # This is a placeholder for a more complex reprojection
    # In practice, this would use rioxarray and pyproj to properly reproject
    # with correct handling of coordinate reference systems
    
    # For now, just return the original dataset
    # To be implemented based on the actual data format and projection
    return merra2_data


def normalize_data(data: Union[xr.Dataset, xr.DataArray], 
                  stats: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[Union[xr.Dataset, xr.DataArray], Dict[str, Dict[str, float]]]:
    """Normalize data to zero mean and unit variance.
    
    If stats are provided, use them for normalization.
    Otherwise, compute stats from the data.
    
    Args:
        data: Data to normalize.
        stats: Dictionary mapping variable names to dictionaries with 'mean' and 'std' keys.
            If None, stats are computed from the data.
            
    Returns:
        Tuple of (normalized_data, stats_dict).
    """
    if stats is None:
        # Compute stats
        stats = {}
        
        if isinstance(data, xr.Dataset):
            for var in data.data_vars:
                mean = float(data[var].mean().values)
                std = float(data[var].std().values)
                if std == 0:
                    std = 1.0  # Prevent division by zero
                stats[var] = {'mean': mean, 'std': std}
                data[var] = (data[var] - mean) / std
        else:  # DataArray
            mean = float(data.mean().values)
            std = float(data.std().values)
            if std == 0:
                std = 1.0
            stats['data'] = {'mean': mean, 'std': std}
            data = (data - mean) / std
    else:
        # Use provided stats
        if isinstance(data, xr.Dataset):
            for var in data.data_vars:
                if var in stats:
                    mean = stats[var]['mean']
                    std = stats[var]['std']
                    data[var] = (data[var] - mean) / std
        else:  # DataArray
            if 'data' in stats:
                mean = stats['data']['mean']
                std = stats['data']['std']
                data = (data - mean) / std
    
    return data, stats


def extract_patches(data: Union[xr.Dataset, xr.DataArray], 
                   patch_size: int, 
                   stride: Optional[int] = None) -> List[Union[xr.Dataset, xr.DataArray]]:
    """Extract spatial patches from the data.
    
    Args:
        data: Input data.
        patch_size: Size of the patches (assumed square).
        stride: Stride between patches. If None, use patch_size (non-overlapping).
        
    Returns:
        List of patches.
    """
    if stride is None:
        stride = patch_size
    
    patches = []
    
    # Identify spatial dimensions based on data type
    if isinstance(data, xr.Dataset):
        # Get the first variable to identify dimensions
        first_var = list(data.data_vars)[0]
        dims = data[first_var].dims
    else:  # DataArray
        dims = data.dims
    
    # Find spatial dimensions (assuming they are 'y' and 'x' or similar)
    spatial_dims = [dim for dim in dims if dim.lower() in ['y', 'x', 'lat', 'lon', 'latitude', 'longitude']]
    if len(spatial_dims) != 2:
        raise ValueError(f"Could not identify spatial dimensions in {dims}")
    
    y_dim, x_dim = spatial_dims
    
    # Get dimension sizes
    y_size = data.sizes[y_dim]
    x_size = data.sizes[x_dim]
    
    # Extract patches
    for y in range(0, y_size - patch_size + 1, stride):
        for x in range(0, x_size - patch_size + 1, stride):
            # Select the patch
            patch = data.isel({y_dim: slice(y, y + patch_size), x_dim: slice(x, x + patch_size)})
            patches.append(patch)
    
    return patches


def prepare_merra2_prism_pair(merra2_file: str, 
                             prism_dir: str, 
                             date: str,
                             merra2_vars: List[str],
                             prism_vars: List[str],
                             dem_path: Optional[str] = None,
                             normalize: bool = True) -> Dict[str, Union[xr.Dataset, xr.DataArray, Dict]]:
    """Prepare a matched pair of MERRA-2 and PRISM data for the same date.
    
    Args:
        merra2_file: Path to MERRA-2 file.
        prism_dir: Directory containing PRISM files.
        date: Date string in format YYYYMMDD.
        merra2_vars: List of MERRA-2 variables to use.
        prism_vars: List of PRISM variables to use.
        dem_path: Path to DEM file. If provided, include DEM data.
        normalize: Whether to normalize the data.
        
    Returns:
        Dictionary with keys 'merra2', 'prism', and optionally 'dem' and 'stats'.
    """
    # Load MERRA-2 data
    merra2_data = load_merra2_data(merra2_file, merra2_vars)
    
    # Load PRISM data
    prism_data = load_prism_data(prism_dir, date, prism_vars)
    
    # Load DEM if requested
    dem_data = None
    if dem_path:
        dem_data = load_dem_data(dem_path)
    
    result = {
        'merra2': merra2_data,
        'prism': prism_data,
    }
    
    if dem_data is not None:
        result['dem'] = dem_data
    
    # Normalize if requested
    if normalize:
        stats = {}
        
        # Normalize MERRA-2
        merra2_norm, merra2_stats = normalize_data(merra2_data)
        result['merra2'] = merra2_norm
        stats['merra2'] = merra2_stats
        
        # Normalize PRISM
        prism_norm = {}
        prism_stats = {}
        for var, data in prism_data.items():
            norm_data, var_stats = normalize_data(data)
            prism_norm[var] = norm_data
            prism_stats[var] = var_stats
        
        result['prism'] = prism_norm
        stats['prism'] = prism_stats
        
        # Normalize DEM
        if dem_data is not None:
            dem_norm, dem_stats = normalize_data(dem_data)
            result['dem'] = dem_norm
            stats['dem'] = dem_stats
        
        result['stats'] = stats
    
    return result 