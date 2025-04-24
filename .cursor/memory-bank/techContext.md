# Technical Context

## Technologies Used

### Core Technologies
- **Python 3.8+** - Primary programming language
- **PyTorch 1.8+** - Deep learning framework
- **PyTorch Lightning 1.5+** - Training structure and organization
- **Transformers 4.11+** - Access to Prithvi WxC model
- **NumPy/Pandas** - Data manipulation
- **Xarray/Rioxarray** - Geospatial data handling
- **NetCDF4** - Climate data format
- **Rasterio/GDAL** - Geospatial processing

### Data Management
- **NetCDF4** - MERRA-2 data format
- **GeoTIFF** - PRISM data format
- **ZIP archives** - PRISM data distribution format
- **BIL format** - DEM elevation data
- **YAML** - Configuration management

### Development Tools
- **Conda/Pip** - Environment management
- **TensorBoard** - Experiment tracking
- **Matplotlib/Seaborn** - Visualization

## Data Structures

### MERRA-2 Data
- NetCDF4 format
- Variables: T2MMAX, T2MMEAN, T2MMIN, TPRECMAX
- Daily temporal resolution
- ~50km spatial resolution

### PRISM Data
- GeoTIFF format within ZIP archives
- Variables: tdmean (temperature), ppt (precipitation)
- Daily temporal resolution
- 4km spatial resolution

### DEM Data
- BIL format (Binary Interleaved by Line)
- Static elevation data
- 4km spatial resolution

## Development Setup

### Environment
- Created requirements.txt with all dependencies
- Conda environment setup script (run.sh)
- Data exploration script (explore_data.sh)

### Configuration
- YAML-based configuration (src/config/config.yaml)
- Command-line argument parsing
- Experiment naming based on configuration

## Project Structure

### Data Pipeline
- Loaders for MERRA-2, PRISM, and DEM data
- Preprocessing with normalization and spatial alignment
- Dataset and DataLoader creation with caching
- Patch extraction for efficient training

### Model Architecture
- Prithvi WxC foundation model
- Input adapter for variable mapping
- DEM encoder for auxiliary data
- Upsampling decoder for 4x resolution increase

### Training System
- Lightning Module with training, validation, and test steps
- Multi-component loss function
- Metrics calculation and visualization
- Checkpointing and early stopping

## Technical Constraints

### Hardware Requirements
- GPU with 16GB+ VRAM recommended
- 32GB+ system RAM
- Storage for cached preprocessed data

### Performance Considerations
- Batch size optimized for GPU memory
- Mixed precision training for efficiency
- Caching preprocessed data to avoid repeated computation

## Foundation Model: Prithvi WxC
- **Architecture:** Hierarchical 2D vision transformer with local and global attention
- **Parameters:** 2.3 billion parameters
- **Pre-training:** Masked reconstruction and forecasting on 160 MERRA-2 variables
- **Input Structure:** Dual-timestamp format with encoded time deltas
- **Pre-trained On:** 40 years of MERRA-2 reanalysis data
- **Source:** Developed by NASA and IBM, available on Hugging Face

## Development Environment
- **Python Version:** 3.8+
- **Deep Learning Framework:** PyTorch (1.10+) with PyTorch Lightning for training structure
- **CUDA Version:** 11.0+ required for GPU acceleration
- **Key Libraries:**
  - xarray and netCDF4 for climate data handling
  - rioxarray for geospatial operations
  - Weights & Biases or TensorBoard for experiment tracking
  - cartopy for visualization

## Hardware Requirements
- **GPU:** NVIDIA with 16GB+ VRAM (e.g., V100, A100, RTX 3090)
- **RAM:** 32GB+ recommended for data processing
- **Storage:** 200GB+ for datasets and model checkpoints

## Data Formats
- **Input (MERRA-2):** NetCDF4 format, ~50km resolution
- **Target (PRISM):** NetCDF4/GeoTIFF format, 4km resolution
- **Auxiliary Data:** GeoTIFF for DEM, land cover, and other static features
- **Model Outputs:** PyTorch checkpoints, ONNX, TorchScript

## Key Technical Challenges
1. **Variable Mapping:** Adapting 4 input variables to Prithvi's 160-variable structure
2. **Memory Management:** Efficient processing of high-resolution geospatial data
3. **Up-sampling:** Learning appropriate spatial patterns from coarse to fine resolution
4. **Distribution Shifts:** Handling differences between model pre-training and fine-tuning data

## Evaluation Infrastructure
- **Metrics Pipeline:** Automated calculation of pointwise, spatial, and distribution metrics
- **Visualization Tools:** Matplotlib, Cartopy for spatial visualization
- **Baselines:** Implementation of bilinear, BCSD, and other comparison methods

## Deployment Considerations
- **Inference Speed:** Processing 1 year of daily data in under 1 hour
- **Memory Usage:** Stay within 16GB VRAM during inference
- **Portability:** Export formats supporting different deployment scenarios
- **Versioning:** Git for code, DVC for model and data versioning 