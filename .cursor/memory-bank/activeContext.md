# Active Context

## Current Focus
- Testing and executing the fine-tuning pipeline for Prithvi WxC on MERRA-2 to PRISM downscaling
- Verifying proper data loading and preprocessing with the current dataset
- Analyzing model performance and optimization opportunities
- Preparing for extended dataset collection and processing

## Recent Decisions
1. **Input Variable Selection:** Implemented the 4 key variables (T2MMAX, T2MMEAN, T2MMIN, TPRECMAX) with efficient data loading
2. **Framework Implementation:** PyTorch Lightning structure with model, dataloaders, and training loop completed
3. **Resolution Handling:** Implemented 4x upsampling (from MERRA-2 to PRISM resolution) with proper spatial alignment
4. **DEM Integration:** Added Digital Elevation Model as auxiliary input to enhance downscaling accuracy
5. **Project Structure:** Organized with modular components for data, models, training, and visualization

## Next Steps
1. **Execute Training:**
   - Run full training pipeline on the provided dataset
   - Monitor resource usage and training progress
   - Analyze initial results and model behavior

2. **Performance Optimization:**
   - Fine-tune hyperparameters for optimal performance
   - Evaluate memory usage and optimize batch size if needed
   - Consider gradient accumulation for larger effective batch sizes

3. **Extend Dataset:**
   - Acquire additional MERRA-2 and PRISM data for longer time periods
   - Implement efficient data storage and caching strategies
   - Expand spatial coverage for better generalization

## Current Challenges
1. **Performance Verification:** Need to execute training to verify model performance and convergence
2. **Limited Dataset:** Current dataset limited to 2 dates, which may limit model generalization
3. **Resource Requirements:** Need to monitor GPU memory and training time during execution

## Key Implementation Details
1. **Data Pipeline:**
   - MERRA-2 NetCDF4 loader with xarray for efficient processing
   - PRISM zip file handling with GeoTIFF extraction
   - DEM integration with proper spatial alignment
   - Normalization and patch extraction for efficient training

2. **Model Architecture:**
   - Prithvi WxC foundation model from Hugging Face
   - Input adapter for mapping 4 variables to Prithvi's expected format
   - Upsampling decoder for 4x resolution increase
   - DEM encoder for auxiliary input integration

3. **Training System:**
   - PyTorch Lightning module with clear training, validation, and test steps
   - Multi-component loss function (MAE, MSE, SSIM)
   - AdamW optimizer with cosine learning rate scheduling
   - Early stopping and model checkpointing

## Recent Implementation Achievements
1. **Complete Project Structure:**
   - Modular implementation with clear separation of concerns
   - Configuration-driven design with YAML config
   - Command-line interface for training and evaluation

2. **Data Systems:**
   - Efficient data loading and preprocessing
   - Caching to avoid repeated processing
   - Patch extraction for training on spatial regions
   - Train/validation/test splitting

3. **Visualization Tools:**
   - Comparison plots for MERRA-2, PRISM, and model predictions
   - Error analysis and visualization
   - Metrics tracking and visualization

## Key Patterns and Preferences
1. **Code Organization:**
   - Modular components with clear interfaces
   - Configuration-driven design
   - Comprehensive typing with type hints

2. **Development Style:**
   - Test-driven development for critical components
   - Experiment tracking for all model runs
   - Reproducible random seeds

3. **Documentation:**
   - Docstrings for all functions and classes
   - README files for major components
   - Example notebooks for key workflows

## Recent Learnings
1. **Prithvi WxC Structure:**
   - Dual-timestamp input format with time delta encoding
   - Masked reconstruction and forecasting pretraining objectives
   - Hierarchical attention mechanisms for spatial data

2. **Climate Data Processing:**
   - NetCDF4 handling best practices
   - Importance of consistent projections and grids
   - Normalization strategies for climate variables

3. **Implementation Insights:**
   - GPU memory optimization techniques
   - Efficient data loading patterns for geospatial data
   - Lightning training loop customizations 