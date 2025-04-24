# MERRA-2 to PRISM Downscaling Pipeline

This project implements a PyTorch Lightning pipeline to downscale MERRA-2 climate data to PRISM's higher resolution using the Prithvi WxC foundation model. The pipeline transforms coarse atmospheric variables into high-resolution outputs matching observed PRISM climatology.

## Project Structure

- `data/`: Contains input and auxiliary data
  - `merra2/`: MERRA-2 NetCDF4 files
  - `prism/`: PRISM climate data files
  - `dem/`: Digital Elevation Model data

- `src/`: Source code
  - `data/`: Data preprocessing and dataset implementations
  - `models/`: Model architecture definitions
  - `trainers/`: PyTorch Lightning trainer modules
  - `utils/`: Utility functions and helper scripts
  - `config/`: Configuration files
  - `visualizations/`: Visualization utilities

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare data files:
   - MERRA-2 NetCDF4 files in `data/merra2/`
   - PRISM climate data in `data/prism/`
   - DEM file in `data/dem/`

3. Run training:
   ```
   python src/main.py
   ```

## Input and Output

- **Input (MERRA-2):** T2MMAX, T2MMEAN, T2MMIN, TPRECMAX
- **Target (PRISM):** Corresponding temperature and precipitation variables
- **Auxiliary Data:** Digital Elevation Model (DEM)

## Model

The model uses Prithvi WxC as a foundation model for downscaling, with additional components to handle the specific task of climate downscaling. 