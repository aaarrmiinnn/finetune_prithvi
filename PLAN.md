# MERRA-2 to PRISM Downscaling Pipeline

> Using Prithvi WxC and PyTorch Lightning

## Project Summary

Create a PyTorch Lightning pipeline to downscale MERRA-2 climate data to PRISM's higher resolution using the Prithvi WxC foundation model. The pipeline will transform coarse atmospheric variables into high-resolution outputs matching observed PRISM climatology.

**Input (MERRA-2):** T2MMAX, T2MMEAN, T2MMIN, TPRECMAX  
**Target (PRISM):** Corresponding temperature and precipitation variables  
**Scope:** Initial proof of concept with limited spatial and temporal extent

## Technical Stack

- **Framework:** PyTorch Lightning
- **Foundation Model:** Prithvi WxC
- **Data Formats:** NetCDF4 (input), GeoTIFF/NetCDF4 (output)
- **Environment:** Python 3.8+, CUDA 11.0+
- **Tracking:** TensorBoard or Weights & Biases

## Data Pipeline

### Data Sources
- **MERRA-2:** NASA GES DISC, subset of variables
- **PRISM:** Oregon State, daily data (AN81d), 4km resolution
- **Auxiliary:** DEM, land cover data, land-sea mask

### Preprocessing Requirements
- Temporal alignment between datasets
- Consistent spatial projections
- Quality control and missing value handling
- Normalization with stored parameters
- Structured inputs for two timestamps with encoded time deltas

### Dataset Implementation
- PyTorch Dataset with efficient chunked loading
- Data augmentation via spatial patches and masking
- 70/15/15 train/validation/test split

## Model Architecture

### Core Design
- Pre-trained Prithvi WxC encoder-decoder as foundation
- Input structure following Prithvi's dual-timestamp format
- Variable mapping from 4 variables to Prithvi's 160
- Masking strategy mimicking Prithvi's pre-training process
- Upsampling modules to match PRISM resolution

### Components
- **Encoder:** Prithvi WxC + auxiliary data embeddings
- **Decoder:** Multi-scale upsampling to 4km resolution
- **Loss Functions:** MAE primary + distribution-aware secondary losses

## Training Implementation

### Configuration
- **Hardware:** GPU with 16GB+ VRAM, 32GB+ RAM
- **Parameters:** Batch size 8 (adjustable), LR 1e-4 with cosine decay
- **Optimization:** AdamW with weight decay, mixed precision

### Lightning Module Requirements
- Standard training/validation steps
- Proper encoding of time deltas and auxiliary features
- Comprehensive logging and checkpointing

### Hyperparameter Tuning
- Focus on masking ratio, learning rate, loss weights
- Grid search for initial exploration

## Evaluation Framework

### Metrics
- **Pointwise:** RMSE, MAE, Bias, R²
- **Spatial:** Correlation, SSIM
- **Distribution:** KS test, Q-Q plots
- **Extremes:** Precision/recall, CSI

### Visualization Requirements
- Spatial comparison maps
- Time series for regional averages
- Distribution comparisons
- Error heatmaps

### Benchmarks
- Bilinear interpolation baseline
- Standard bias correction
- BCSD method

## Deployment

### Model Export Formats
- TorchScript
- ONNX
- PyTorch checkpoint

### Inference Requirements
- End-to-end processing pipeline
- Physical consistency checking
- NetCDF/GeoTIFF output generation

### Performance Targets
- Process 1 year daily data in <1 hour
- Stay within 16GB VRAM
- 20%+ improvement over baselines

## Project Plan

1. **Setup Phase (2 weeks)**
  - Environment configuration
  - Data pipeline implementation
  - Initial model loading

2. **Prototype Phase (2 weeks)**
  - Basic downscaling implementation 
  - Training/evaluation loops
  - Small-scale testing

3. **Development Phase (4 weeks)**
  - Scaling to larger datasets
  - Architecture optimization
  - Hyperparameter tuning

4. **Refinement Phase (2 weeks)**
  - Performance optimization
  - Documentation
  - Final evaluation

## Deliverables

- GitHub repository with documented code
- Trained model checkpoints
- Technical documentation
- Jupyter notebook examples
- Evaluation visualizations
