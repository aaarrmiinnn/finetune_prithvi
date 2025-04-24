"""Dataset classes for MERRA-2 to PRISM downscaling."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Callable
import random
import glob
from pathlib import Path

from .preprocessing import (
    prepare_merra2_prism_pair,
    extract_patches,
    normalize_data
)


class MERRA2PRISMDataset(Dataset):
    """Dataset for MERRA-2 to PRISM downscaling."""
    
    def __init__(
        self,
        merra2_dir: str,
        prism_dir: str,
        dates: List[str],
        merra2_vars: List[str],
        prism_vars: List[str],
        dem_path: Optional[str] = None,
        patch_size: int = 64,
        patch_stride: Optional[int] = None,
        normalize: bool = True,
        mask_ratio: float = 0.0,
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize the dataset.
        
        Args:
            merra2_dir: Directory containing MERRA-2 files.
            prism_dir: Directory containing PRISM files.
            dates: List of dates in format YYYYMMDD.
            merra2_vars: List of MERRA-2 variables to use.
            prism_vars: List of PRISM variables to use.
            dem_path: Path to DEM file. If provided, include DEM data.
            patch_size: Size of spatial patches to extract.
            patch_stride: Stride between patches. If None, use patch_size.
            normalize: Whether to normalize the data.
            mask_ratio: Ratio of input pixels to mask for training.
            transform: Optional transform to apply to the data.
            cache_dir: Directory to cache processed data. If None, no caching is used.
        """
        self.merra2_dir = merra2_dir
        self.prism_dir = prism_dir
        self.dates = dates
        self.merra2_vars = merra2_vars
        self.prism_vars = prism_vars
        self.dem_path = dem_path
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.normalize = normalize
        self.mask_ratio = mask_ratio
        self.transform = transform
        self.cache_dir = cache_dir
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Find MERRA-2 files for each date
        self.merra2_files = {}
        for date in self.dates:
            # Extract date components for file pattern matching
            year = date[:4]
            month = date[4:6]
            day = date[6:8]
            
            # Match pattern based on MERRA-2 file naming convention
            pattern = os.path.join(self.merra2_dir, f"MERRA2_400.statD_2d_slv_Nx.{year}{month}{day}.nc4.nc4")
            matching_files = glob.glob(pattern)
            
            if not matching_files:
                raise FileNotFoundError(f"No MERRA-2 file found for date {date}")
            
            self.merra2_files[date] = matching_files[0]
        
        # Prepare all data pairs and extract patches
        self.patches = []
        self.stats = {}
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare all data pairs and extract patches."""
        print(f"Preparing data for {len(self.dates)} dates...")
        
        for date_idx, date in enumerate(self.dates):
            print(f"Processing date {date} ({date_idx+1}/{len(self.dates)})")
            
            # Check if cached data exists
            cache_file = None
            if self.cache_dir:
                cache_file = os.path.join(self.cache_dir, f"merra2_prism_{date}.npz")
                if os.path.exists(cache_file):
                    print(f"Loading cached data from {cache_file}")
                    data = np.load(cache_file, allow_pickle=True)
                    date_patches = data['patches'].tolist()
                    self.patches.extend(date_patches)
                    
                    if self.normalize and 'stats' in data:
                        self.stats.update(data['stats'].tolist())
                    
                    continue
            
            # Load and prepare data for this date
            try:
                data_pair = prepare_merra2_prism_pair(
                    self.merra2_files[date],
                    self.prism_dir,
                    date,
                    self.merra2_vars,
                    self.prism_vars,
                    self.dem_path,
                    normalize=self.normalize
                )
                
                # Extract patches from MERRA-2 data
                merra2_patches = extract_patches(
                    data_pair['merra2'],
                    self.patch_size,
                    self.patch_stride
                )
                
                # Extract patches from PRISM data
                prism_patches = {}
                for var, data in data_pair['prism'].items():
                    prism_patches[var] = extract_patches(
                        data,
                        self.patch_size * 4,  # Assuming PRISM is 4x resolution of MERRA-2
                        self.patch_stride * 4
                    )
                
                # Extract DEM patches if available
                dem_patches = None
                if 'dem' in data_pair:
                    dem_patches = extract_patches(
                        data_pair['dem'],
                        self.patch_size * 4,  # Matching PRISM resolution
                        self.patch_stride * 4
                    )
                
                # Store stats if normalizing
                if self.normalize and 'stats' in data_pair:
                    self.stats.update({date: data_pair['stats']})
                
                # Create patch pairs
                date_patches = []
                for i in range(min(len(merra2_patches), min([len(prism_patches[var]) for var in prism_vars]))):
                    patch_data = {
                        'merra2': merra2_patches[i],
                        'prism': {var: prism_patches[var][i] for var in prism_vars},
                        'date': date
                    }
                    
                    if dem_patches:
                        patch_data['dem'] = dem_patches[i]
                    
                    date_patches.append(patch_data)
                
                self.patches.extend(date_patches)
                
                # Cache the processed data
                if self.cache_dir:
                    np.savez(
                        cache_file,
                        patches=date_patches,
                        stats=self.stats
                    )
            
            except Exception as e:
                print(f"Error processing date {date}: {str(e)}")
                continue
        
        print(f"Total patches: {len(self.patches)}")
    
    def _convert_to_tensor(self, data):
        """Convert different data types to PyTorch tensors."""
        if isinstance(data, xr.Dataset):
            # Convert each variable to a tensor and concatenate along channel dimension
            tensors = []
            for var in self.merra2_vars:
                if var in data:
                    tensor = torch.tensor(data[var].values, dtype=torch.float32)
                    if tensor.ndim == 2:
                        tensor = tensor.unsqueeze(0)  # Add channel dimension
                    tensors.append(tensor)
            
            return torch.cat(tensors, dim=0)
        
        elif isinstance(data, xr.DataArray):
            tensor = torch.tensor(data.values, dtype=torch.float32)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)  # Add channel dimension
            return tensor
        
        elif isinstance(data, dict):
            # For PRISM data (dictionary of variables)
            tensors = []
            for var in self.prism_vars:
                if var in data:
                    tensor = self._convert_to_tensor(data[var])
                    tensors.append(tensor)
            
            return torch.cat(tensors, dim=0)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _apply_mask(self, tensor, mask_ratio):
        """Apply random masking to the input tensor."""
        if mask_ratio <= 0:
            return tensor, None
        
        # Create mask (1 = keep, 0 = mask)
        mask = torch.ones_like(tensor)
        
        # Randomly select pixels to mask
        num_pixels = tensor.shape[1] * tensor.shape[2]
        num_masked = int(num_pixels * mask_ratio)
        
        # Randomly select indices to mask
        indices = torch.randperm(num_pixels)[:num_masked]
        rows = indices // tensor.shape[2]
        cols = indices % tensor.shape[2]
        
        # Apply mask
        mask[:, rows, cols] = 0
        
        # Return masked tensor and mask
        return tensor * mask, mask
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Returns:
            Dictionary with keys:
                - 'merra2_input': Tensor of shape (C, H, W)
                - 'prism_target': Tensor of shape (C, H*4, W*4)
                - 'dem': Tensor of shape (1, H*4, W*4) (if DEM provided)
                - 'mask': Tensor of shape (C, H, W) (if masking applied)
                - 'date': Date string
        """
        patch = self.patches[idx]
        
        # Convert data to tensors
        merra2_tensor = self._convert_to_tensor(patch['merra2'])
        prism_tensor = self._convert_to_tensor(patch['prism'])
        
        # Apply mask if needed
        mask = None
        if self.mask_ratio > 0:
            merra2_tensor, mask = self._apply_mask(merra2_tensor, self.mask_ratio)
        
        # Prepare output dictionary
        output = {
            'merra2_input': merra2_tensor,
            'prism_target': prism_tensor,
            'date': patch['date']
        }
        
        if mask is not None:
            output['mask'] = mask
        
        # Add DEM if available
        if 'dem' in patch:
            dem_tensor = self._convert_to_tensor(patch['dem'])
            output['dem'] = dem_tensor
        
        # Apply transform if provided
        if self.transform:
            output = self.transform(output)
        
        return output


def create_dataloaders(
    config: Dict,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary.
        train_split: Fraction of data to use for training.
        val_split: Fraction of data to use for validation.
        test_split: Fraction of data to use for testing.
        num_workers: Number of workers for data loading.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Validate split ratios
    assert abs(train_split + val_split + test_split - 1.0) < 1e-5, "Split ratios must sum to 1"
    
    # Get configuration parameters
    merra2_dir = config['data']['merra2_dir']
    prism_dir = config['data']['prism_dir']
    dem_dir = config['data']['dem_dir']
    dem_file = config['data']['dem_file']
    dem_path = os.path.join(dem_dir, dem_file)
    dates = config['data']['dates']
    patch_size = config['data']['patch_size']
    input_vars = config['data']['input_vars']
    target_vars = config['data']['target_vars']
    mask_ratio = config['data']['mask_ratio']
    cache_dir = config['data']['cache_dir']
    batch_size = config['training']['batch_size']
    
    # Create dataset with all data
    full_dataset = MERRA2PRISMDataset(
        merra2_dir=merra2_dir,
        prism_dir=prism_dir,
        dates=dates,
        merra2_vars=input_vars,
        prism_vars=target_vars,
        dem_path=dem_path,
        patch_size=patch_size,
        normalize=True,
        mask_ratio=mask_ratio,
        cache_dir=cache_dir
    )
    
    # Determine the split sizes
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split the dataset
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 