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
        
        # Convert temperature variables from Kelvin to Celsius
        temp_vars = ['T2MMAX', 'T2MMEAN', 'T2MMIN']
        for var in temp_vars:
            if var in ds:
                ds[var] = ds[var] - 273.15
                ds[var].attrs['units'] = 'degC'
        
        return ds
    except Exception as e:
        raise RuntimeError(f"Error loading MERRA-2 data: {str(e)}")


def load_prism_data(prism_dir: str, date: str, variables: List[str]) -> Dict[str, xr.DataArray]:
    """Load PRISM data from zip files.
    
    Args:
        prism_dir: Directory containing PRISM files.
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
        
        # Extract and load the data files
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as z:
                # Extract all files
                z.extractall(temp_dir)
                
                # Base name for all files
                base_name = f"prism_{var}_us_25m_{date}"
                
                # Load the GeoTIFF file with proper CRS
                tif_file = os.path.join(temp_dir, f"{base_name}.tif")
                if not os.path.exists(tif_file):
                    raise FileNotFoundError(f"No .tif file found in {zip_file}")
                
                # Read with rioxarray, explicitly setting CRS to NAD83
                da = rioxarray.open_rasterio(tif_file)
                da.rio.write_crs("EPSG:4269", inplace=True)  # NAD83
                
                # Standardize the dimensions
                da = da.squeeze(drop=True)  # Remove single-dimension axes
                
                # Set variable name
                da.name = var
                
                # Read metadata from XML for proper attributes
                xml_file = os.path.join(temp_dir, f"{base_name}.tif.aux.xml")
                if os.path.exists(xml_file):
                    try:
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(xml_file)
                        root = tree.getroot()
                        stats = {
                            elem.attrib['key']: float(elem.text)
                            for elem in root.findall(".//MDI")
                            if elem.attrib['key'].startswith('STATISTICS_')
                        }
                        
                        # Set attributes based on variable type
                        if var == 'tdmean':
                            da.attrs.update({
                                'units': 'degC',
                                'long_name': 'Mean Temperature',
                                'grid_mapping': 'NAD83',
                                'valid_range': [stats.get('STATISTICS_MINIMUM', -100.0),
                                              stats.get('STATISTICS_MAXIMUM', 100.0)],
                                '_FillValue': -9999.0
                            })
                        elif var == 'ppt':
                            da.attrs.update({
                                'units': 'mm',
                                'long_name': 'Precipitation',
                                'grid_mapping': 'NAD83',
                                'valid_range': [0.0, stats.get('STATISTICS_MAXIMUM', 1000.0)],
                                '_FillValue': -9999.0
                            })
                        
                        # Add statistics to attributes
                        da.attrs.update({
                            'min_value': stats.get('STATISTICS_MINIMUM'),
                            'max_value': stats.get('STATISTICS_MAXIMUM'),
                            'mean_value': stats.get('STATISTICS_MEAN'),
                            'std_value': stats.get('STATISTICS_STDDEV')
                        })
                    except Exception as e:
                        print(f"Warning: Could not parse XML stats: {e}")
                
                # Mask no-data values
                da = da.where(da > da.attrs['_FillValue'])
                
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


def reproject_merra2_to_prism(merra2_ds: xr.Dataset, prism_da: xr.DataArray) -> xr.Dataset:
    """Reproject MERRA-2 data to match PRISM grid.
    
    Args:
        merra2_ds: MERRA-2 dataset with variables to reproject.
        prism_da: PRISM data array to use as reference for target grid.
        
    Returns:
        MERRA-2 dataset reprojected to PRISM grid.
    """
    # Create a copy to avoid modifying the input
    merra2_ds = merra2_ds.copy()
    
    # Check and fix latitude orientation
    # MERRA-2 should have latitudes from south to north
    for var in merra2_ds.data_vars:
        if 'lat' in merra2_ds[var].coords:
            lats = merra2_ds[var].lat.values
            if lats[0] > lats[-1]:  # If latitudes are north to south
                print("Flipping MERRA-2 data to ensure south-to-north orientation")
                merra2_ds[var] = merra2_ds[var].reindex(lat=list(reversed(lats)))
    
    # Ensure MERRA-2 has proper CRS (WGS84)
    for var in merra2_ds.data_vars:
        if not hasattr(merra2_ds[var], 'rio'):
            merra2_ds[var] = merra2_ds[var].rio.set_spatial_dims(x_dim='lon', y_dim='lat')
        merra2_ds[var] = merra2_ds[var].rio.write_crs("EPSG:4326")  # WGS84
    
    # Create output dataset
    reprojected_ds = xr.Dataset()
    
    # Reproject each variable
    for var in merra2_ds.data_vars:
        # Get the MERRA-2 data array
        da = merra2_ds[var]
        
        # Convert longitude from 0-360 to -180-180 if needed
        if (da.lon > 180).any():
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
            da = da.roll(lon=(da.dims['lon'] // 2), roll_coords=True)
        
        # Reproject to match PRISM grid using bilinear interpolation
        reprojected = da.rio.reproject_match(
            prism_da,
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Fill NaN values with nearest valid data
        if np.isnan(reprojected).any():
            reprojected = reprojected.rio.reproject_match(
                prism_da,
                resampling=rasterio.enums.Resampling.nearest
            ).where(~np.isnan(reprojected), reprojected)
        
        # Add to output dataset
        reprojected_ds[var] = reprojected
        
        # Copy attributes
        reprojected_ds[var].attrs = da.attrs.copy()
        reprojected_ds[var].attrs.update({
            'grid_mapping': 'NAD83',  # Target CRS
            'resampling_method': 'bilinear',
            'original_grid': 'MERRA-2'
        })
    
    return reprojected_ds


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
                             normalize: bool = False) -> Dict[str, Union[xr.Dataset, xr.DataArray, Dict]]:
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
    
    # Reproject MERRA-2 data to match PRISM grid
    # Use the first PRISM variable as reference for the target grid
    first_prism_var = next(iter(prism_data.values()))
    merra2_data = reproject_merra2_to_prism(merra2_data, first_prism_var)
    
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