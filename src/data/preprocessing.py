"""Data preprocessing for MERRA-2 and PRISM datasets."""
import os
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio import enums
from typing import Dict, List, Tuple, Optional, Union
import glob
import zipfile
import tempfile
from pathlib import Path


def load_merra2_data(
    file_path: Union[str, Path],
    variables: List[str],
    decode_times: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MERRA-2 data from NetCDF file.
    
    Args:
        file_path: Path to NetCDF file
        variables: List of variables to load
        decode_times: Whether to decode times in xarray
        
    Returns:
        Tuple of (data_array, latitudes, longitudes)
        - data_array: Array of shape (num_vars, height, width)
        - latitudes: Array of latitude values
        - longitudes: Array of longitude values
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"MERRA-2 file not found: {file_path}")
        
        # Load dataset
        ds = xr.open_dataset(file_path, decode_times=decode_times)
        
        # Check variables
        missing_vars = [var for var in variables if var not in ds.variables]
        if missing_vars:
            raise ValueError(f"Variables not found in dataset: {missing_vars}")
        
        # Get coordinates
        lats = ds['lat'].values
        lons = ds['lon'].values
        
        # Ensure longitudes are in -180 to 180 range
        lons = np.where(lons > 180, lons - 360, lons)
        
        # Load and stack variables
        data = []
        for var in variables:
            var_data = ds[var].values
            
            # Handle temperature variables (convert from K to C)
            if var in ['T2MMAX', 'T2MMEAN', 'T2MMIN']:
                var_data = var_data - 273.15
            
            # Ensure 3D shape (time, lat, lon)
            if var_data.ndim == 2:
                var_data = var_data[np.newaxis, ...]
            
            data.append(var_data[0])  # Take first time step
        
        return np.stack(data), lats, lons
    
    except Exception as e:
        raise RuntimeError(f"Error loading MERRA-2 data: {str(e)}")
    
    finally:
        if 'ds' in locals():
            ds.close()


def load_prism_data(
    prism_dir: Union[str, Path],
    date: str,
    variables: List[str]
) -> Tuple[Dict[str, np.ndarray], rasterio.Affine]:
    """Load PRISM data from zip files.
    
    Args:
        prism_dir: Directory containing PRISM data
        date: Date in YYYYMMDD format
        variables: List of variables to load
        
    Returns:
        Tuple of (data_dict, transform)
        - data_dict: Dictionary mapping variable names to arrays
        - transform: Affine transform for the PRISM grid
    """
    try:
        prism_dir = Path(prism_dir)
        if not prism_dir.exists():
            raise FileNotFoundError(f"PRISM directory not found: {prism_dir}")
        
        data = {}
        transform = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for var in variables:
                # Find matching file
                pattern = f"prism_{var}_us_25m_{date}.zip"
                var_files = list(prism_dir.glob(pattern))
                
                if not var_files:
                    raise FileNotFoundError(f"PRISM file not found: {pattern}")
                
                var_file = var_files[0]
                
                # Extract zip file
                with zipfile.ZipFile(var_file, 'r') as zip_ref:
                    # Get the GeoTIFF file
                    tif_files = [f for f in zip_ref.namelist() if f.endswith('.tif') and not f.endswith('.aux.xml')]
                    if not tif_files:
                        raise FileNotFoundError(f"No GeoTIFF file found in {var_file}")
                    
                    # Extract the GeoTIFF file
                    tif_file = tif_files[0]
                    zip_ref.extract(tif_file, temp_dir)
                    
                    # Load data using rasterio
                    tif_path = os.path.join(temp_dir, tif_file)
                    with rasterio.open(tif_path) as src:
                        var_data = src.read(1)  # Read first band
                        
                        # Store transform from first variable
                        if transform is None:
                            transform = src.transform
                        
                        # Apply variable-specific scaling
                        if var == 'tdmean':  # Temperature
                            var_data = var_data * 0.1  # Scale factor for temperature
                        elif var == 'ppt':  # Precipitation
                            var_data = var_data * 1.0  # Scale factor for precipitation
                        
                        # Mask no-data values
                        if src.nodata is not None:
                            var_data = np.ma.masked_array(var_data, var_data == src.nodata)
                        
                        # Convert to regular numpy array, filling masked values with NaN
                        if isinstance(var_data, np.ma.MaskedArray):
                            var_data = var_data.filled(np.nan)
                        
                        data[var] = var_data
        
        if transform is None:
            raise RuntimeError("Failed to get transform from PRISM data")
        
        return data, transform
    
    except Exception as e:
        raise RuntimeError(f"Error loading PRISM data: {str(e)}")


def load_dem_data(dem_path: Union[str, Path]) -> np.ndarray:
    """Load Digital Elevation Model data.
    
    Args:
        dem_path: Path to DEM file
        
    Returns:
        Array of shape (height, width)
    """
    try:
        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        # Load data using rasterio
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)  # Read first band
            
            # Mask no-data values
            if src.nodata is not None:
                dem_data = np.ma.masked_equal(dem_data, src.nodata)
            
            return dem_data
    
    except Exception as e:
        raise RuntimeError(f"Error loading DEM data: {str(e)}")


def reproject_merra2_to_prism(
    merra2_data: np.ndarray,
    merra2_lats: np.ndarray,
    merra2_lons: np.ndarray,
    prism_data: np.ndarray,
    prism_transform: np.ndarray,
    method: str = 'bilinear'
) -> np.ndarray:
    """Reproject MERRA-2 data to PRISM grid.
    
    Args:
        merra2_data: MERRA-2 data array of shape (vars, lat, lon)
        merra2_lats: MERRA-2 latitude array
        merra2_lons: MERRA-2 longitude array
        prism_data: PRISM data array of shape (height, width)
        prism_transform: PRISM affine transform
        method: Interpolation method ('bilinear' or 'nearest')
        
    Returns:
        Reprojected MERRA-2 data on PRISM grid
    """
    try:
        # Check input shapes
        if merra2_data.ndim != 3:
            raise ValueError("MERRA-2 data must be 3D (vars, lat, lon)")
        if prism_data.ndim != 2:
            raise ValueError("PRISM data must be 2D (height, width)")
        
        # Create source dataset
        ds = xr.Dataset(
            data_vars={
                'data': (['var', 'y', 'x'], merra2_data)
            },
            coords={
                'y': merra2_lats,
                'x': merra2_lons
            }
        )
        
        # Set spatial dimensions and CRS
        ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
        ds = ds.rio.write_crs("EPSG:4326")  # WGS84
        
        # Create target grid
        target_height, target_width = prism_data.shape
        
        # Reproject
        ds_proj = ds.rio.reproject(
            dst_crs="EPSG:4269",  # NAD83
            shape=(target_height, target_width),
            transform=prism_transform,
            resampling=getattr(rasterio.enums.Resampling, method)
        )
        
        return ds_proj['data'].values
    
    except Exception as e:
        raise RuntimeError(f"Error reprojecting data: {str(e)}")


def normalize_data(
    data: Union[xr.Dataset, xr.DataArray, np.ndarray], 
    stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[Union[xr.Dataset, xr.DataArray, np.ndarray], Dict[str, Dict[str, float]]]:
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
        elif isinstance(data, xr.DataArray):
            mean = float(data.mean().values)
            std = float(data.std().values)
            if std == 0:
                std = 1.0
            stats['data'] = {'mean': mean, 'std': std}
            data = (data - mean) / std
        else:  # numpy array
            mean = float(np.nanmean(data))
            std = float(np.nanstd(data))
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
        elif isinstance(data, xr.DataArray):
            if 'data' in stats:
                mean = stats['data']['mean']
                std = stats['data']['std']
                data = (data - mean) / std
        else:  # numpy array
            if 'data' in stats:
                mean = stats['data']['mean']
                std = stats['data']['std']
                data = (data - mean) / std
    
    return data, stats


def extract_patches(
    data: Union[xr.Dataset, xr.DataArray, np.ndarray], 
    patch_size: int, 
    stride: Optional[int] = None
) -> List[Union[xr.Dataset, xr.DataArray, np.ndarray]]:
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
    
    if isinstance(data, (xr.Dataset, xr.DataArray)):
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
    else:  # numpy array
        # For numpy arrays, assume last two dimensions are spatial
        if data.ndim < 2:
            raise ValueError("Data must have at least 2 dimensions")
        
        # Get spatial dimensions
        y_size = data.shape[-2]
        x_size = data.shape[-1]
        
        # Extract patches
        for y in range(0, y_size - patch_size + 1, stride):
            for x in range(0, x_size - patch_size + 1, stride):
                # Select the patch
                if data.ndim == 2:
                    patch = data[y:y + patch_size, x:x + patch_size]
                else:
                    patch = data[..., y:y + patch_size, x:x + patch_size]
                patches.append(patch)
    
    return patches


def prepare_merra2_prism_pair(
    merra2_file: str, 
    prism_dir: str, 
    date: str,
    merra2_vars: List[str],
    prism_vars: List[str],
    dem_path: Optional[str] = None,
    normalize: bool = False
) -> Dict[str, Union[xr.Dataset, xr.DataArray, Dict]]:
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
    merra2_data, merra2_lats, merra2_lons = load_merra2_data(merra2_file, merra2_vars)
    
    # Load PRISM data
    prism_data, prism_transform = load_prism_data(prism_dir, date, prism_vars)
    
    # Convert PRISM data to xarray DataArrays
    prism_arrays = {}
    for var, data in prism_data.items():
        prism_arrays[var] = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={
                'y': np.arange(data.shape[0]),
                'x': np.arange(data.shape[1])
            },
            name=var
        )
    
    # Load DEM if requested
    dem_data = None
    if dem_path:
        dem_data = load_dem_data(dem_path)
    
    # Reproject MERRA-2 data to match PRISM grid
    # Use the first PRISM variable as reference
    first_var = next(iter(prism_data))
    first_prism_data = prism_data[first_var]
    merra2_data = reproject_merra2_to_prism(
        merra2_data,
        merra2_lats,
        merra2_lons,
        first_prism_data,
        prism_transform
    )
    
    result = {
        'merra2': merra2_data,
        'prism': prism_arrays,
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
        for var, data in prism_arrays.items():
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