# Data Preparation Guide

This guide provides detailed instructions for preparing the data required for the Prithvi Downscaling model.

## Data Overview

The model requires three types of data:

1. **MERRA2**: Low-resolution climate data (input)
2. **PRISM**: High-resolution climate data (target)
3. **DEM**: Digital Elevation Model data (auxiliary)

## Directory Structure

Create the following directory structure:

```
data/
├── merra2/           # MERRA2 NetCDF4 files
├── prism/            # PRISM data in zip or extracted format
└── dem/              # Digital Elevation Model data
```

## MERRA2 Data

### Data Source

MERRA2 (Modern-Era Retrospective Analysis for Research and Applications, Version 2) data can be downloaded from the NASA GES DISC:
- Website: https://disc.gsfc.nasa.gov/datasets?project=MERRA-2
- Access requires a NASA Earthdata account

### Required Variables

The model uses the following MERRA2 variables:
- `T2MMAX`: Maximum 2-meter temperature
- `T2MMEAN`: Mean 2-meter temperature 
- `T2MMIN`: Minimum 2-meter temperature
- `TPRECMAX`: Maximum precipitation rate

### Data Format

- Format: NetCDF4 (.nc4)
- Spatial Resolution: ~50km (51x94 grid for continental US)
- Coordinate System: WGS84 (EPSG:4326)
- Temperature units: Kelvin (automatically converted to Celsius during preprocessing)

### Download Instructions

1. Navigate to the NASA GES DISC website
2. Search for "MERRA2"
3. Select the "M2SDNXSLV: MERRA-2 Daily Aggregated Surface Level Diagnostics" dataset
4. Use the Earthdata Search tool to filter by:
   - Date range
   - Spatial extent (US region)
   - Variables (T2MMAX, T2MMEAN, T2MMIN, TPRECMAX)
5. Download files in NetCDF4 format
6. Place downloaded files in the `data/merra2/` directory

## PRISM Data

### Data Source

PRISM (Parameter-elevation Regressions on Independent Slopes Model) data can be downloaded from the PRISM Climate Group:
- Website: https://prism.oregonstate.edu/
- Registration may be required

### Required Variables

The model uses the following PRISM variables:
- `tdmean`: Mean temperature
- `ppt`: Precipitation

### Data Format

- Format: BIL files in ZIP archives
- Spatial Resolution: 4km (621x1405 grid for continental US)
- Coordinate System: NAD83 (EPSG:4269)
- Temperature units: Celsius
- Precipitation units: mm

### Download Instructions

1. Navigate to the PRISM Climate Group website
2. Select "Recent Years" data
3. Choose the desired date range
4. Download daily data for:
   - Mean temperature (tdmean)
   - Precipitation (ppt)
5. Place downloaded ZIP files in the `data/prism/` directory

## DEM Data

### Data Source

Digital Elevation Model (DEM) data matching PRISM's resolution can be downloaded from:
- PRISM Climate Group: https://prism.oregonstate.edu/
- USGS: https://www.usgs.gov/3d-elevation-program/

### Data Format

- Format: BIL (Binary Interleaved by Line)
- Spatial Resolution: 4km (matching PRISM)
- Units: meters above sea level

### Download Instructions

1. Download the DEM file matching PRISM's 4km resolution grid
2. The default expected filename is `PRISM_us_dem_4km_bil.bil`
3. Place the file in the `data/dem/` directory

## Data Preprocessing

The model automatically handles data preprocessing, including:

1. **Unit Conversion**:
   - MERRA2 temperatures are converted from Kelvin to Celsius

2. **Coordinate System Alignment**:
   - MERRA2: WGS84 (EPSG:4326)
   - PRISM: NAD83 (EPSG:4269)
   - Proper reprojection is handled automatically

3. **Resolution Matching**:
   - Target: PRISM 4km grid
   - Bilinear interpolation for initial upscaling

4. **Quality Control**:
   - Data range validation
   - Missing value handling
   - Physical consistency checks

## Testing Your Data Setup

To verify your data setup:

```bash
# Run the data exploration script
python src/scripts/explore_data.py

# This will generate visualizations comparing MERRA2 and PRISM data
# Output will be saved in docs/images/data_comparison.png
```

The visualization should show:
- MERRA2 data (left) at coarse resolution
- PRISM data (right) at high resolution
- Both should cover the same geographical area
- Temperature values should be in Celsius
- DEM data should align with the same grid as PRISM 