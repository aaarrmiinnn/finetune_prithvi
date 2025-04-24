# Technical Context

## Development Environment
- Python 3.8
- Conda environment: "prithvi"
- GDAL installed

## Core Technologies
1. Deep Learning:
   - PyTorch Lightning
   - HuggingFace Transformers
   - IBM Prithvi-WxC-1.0-2300M model (2.3B parameters)
   - Pre-trained on 40 years of MERRA-2 data
   - Supports multiple tasks:
     * Downscaling
     * Gravity wave parameterization
     * Autoregressive rollout forecasting
     * Extreme events estimation

2. Data Processing:
   - NetCDF4 (for MERRA2 data)
   - GDAL (for DEM and PRISM data)
   - NumPy/Pandas

3. Training Infrastructure:
   - Mixed precision training (16-bit)
   - GPU support
   - Tensorboard logging
   - Optional Weights & Biases integration

## Data Sources
1. MERRA2:
   - Format: NetCDF4
   - Variables: T2MMAX, T2MMEAN, T2MMIN, TPRECMAX
   - Files: Daily statistics

2. PRISM:
   - Format: ZIP archives with BIL files
   - Variables: tdmean (temperature), ppt (precipitation)
   - Resolution: 4km

3. DEM:
   - Format: BIL (Binary Interleaved by Line)
   - Resolution: 4km
   - Coverage: US domain

## Dependencies
Key packages from requirements.txt:
- torch
- pytorch-lightning
- transformers
- netCDF4
- GDAL
- numpy
- pandas 

## Data Processing Pipeline

### MERRA-2 Data
- Format: NetCDF4 files
- Variables: T2MMAX, T2MMEAN, T2MMIN, TPRECMAX
- Resolution: Coarse grid (51x94)
- Temperature units: Originally Kelvin, converted to Celsius
- Coordinate system: WGS84 (EPSG:4326)

### PRISM Data
- Format: GeoTIFF files in zip archives
- Variables: tdmean (temperature), ppt (precipitation)
- Resolution: 4km grid (621x1405)
- Temperature units: Celsius
- Precipitation units: mm
- Coordinate system: NAD83 (EPSG:4269)
- Includes detailed metadata in XML format

### DEM Data
- Format: BIL (Band Interleaved by Line)
- Resolution: Matches PRISM (4km)
- Units: meters
- Used as auxiliary data for downscaling

### Data Loading Implementation
1. MERRA-2 Loading (`load_merra2_data`):
   - Uses xarray for NetCDF4 handling
   - Converts temperature from K to °C
   - Adds proper units and metadata

2. PRISM Loading (`load_prism_data`):
   - Extracts data from zip archives
   - Uses rioxarray for GeoTIFF handling
   - Reads and applies metadata from XML
   - Handles missing values and coordinates

3. DEM Loading (`load_dem_data`):
   - Uses rioxarray for BIL file handling
   - Ensures consistent grid with PRISM

### Data Preprocessing Steps
1. Coordinate System Handling:
   - MERRA-2: WGS84 (EPSG:4326)
   - PRISM: NAD83 (EPSG:4269)
   - Proper reprojection between systems

2. Resolution Matching:
   - Target: PRISM 4km grid
   - Uses bilinear interpolation for upscaling
   - Handles coordinate transformations

3. Quality Control:
   - Validates data ranges
   - Masks missing values
   - Ensures physical consistency

### Visualization Tools
- Matplotlib-based plotting
- Consistent color scales
- Unit-aware visualization
- Side-by-side comparison capability
- DEM overlay options 