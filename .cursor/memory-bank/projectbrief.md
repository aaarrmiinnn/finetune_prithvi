# MERRA-2 to PRISM Downscaling Project Brief

## Project Goal
Create a downscaling pipeline that uses the Prithvi WxC foundation model to transform coarse MERRA-2 climate data into high-resolution outputs matching observed PRISM climatology using a fine-tuning approach.

## Core Requirements
1. **Fine-tune Prithvi WxC** on a downscaling task using 4 specific MERRA-2 variables
2. **Implement in PyTorch Lightning** for structured training and evaluation
3. **Process data** from NetCDF4 inputs to GeoTIFF/NetCDF4 outputs
4. **Evaluate performance** against baseline methods
5. **Document and deliver** reproducible code and model checkpoints

## Key Variables
- **Input (MERRA-2):** T2MMAX, T2MMEAN, T2MMIN, TPRECMAX
- **Target (PRISM):** Corresponding high-resolution temperature and precipitation variables

## Technical Stack Overview
- **Foundation Model:** Prithvi WxC (NASA/IBM geospatial model)
- **Framework:** PyTorch Lightning
- **Data Formats:** NetCDF4 (input), GeoTIFF/NetCDF4 (output)
- **Environment:** Python 3.8+, CUDA 11.0+
- **Hardware:** GPU with 16GB+ VRAM, 32GB+ RAM

## Deliverables
- GitHub repository with documented code
- Trained model checkpoints
- Technical documentation
- Jupyter notebook examples
- Evaluation visualizations

## Project Scope Boundaries
- Initial proof of concept with limited spatial and temporal extent
- Focus on specific temperature and precipitation variables
- Performance target: 20%+ improvement over baselines 