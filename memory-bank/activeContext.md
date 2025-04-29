# Active Context

This document captures the current state of the MERRA-2 to PRISM downscaling project, including current work focus, recent developments, and key decisions.

## Current Work Focus

The project is currently focused on fine-tuning the Prithvi-100M model for climate downscaling, specifically:

1. **Dataset Implementation**: Developing and refining the MERRA2PRISMDataset class that handles:
   - Loading and preprocessing MERRA-2 and PRISM data
   - Incorporating DEM (Digital Elevation Model) data
   - Efficient patch extraction and normalization
   - Data augmentation via optional masking

2. **Model Architecture**: Implementing the PrithviDownscaler model that:
   - Adapts the transformer-based Prithvi architecture for downscaling
   - Incorporates DEM information via a specialized encoder
   - Uses progressive upsampling blocks for high-resolution output
   - Supports multi-variable prediction

3. **Training Pipeline**: Refining the PyTorch Lightning module that:
   - Manages the training, validation, and testing workflows
   - Implements custom loss functions combining MAE, MSE, and SSIM
   - Calculates comprehensive metrics for model evaluation
   - Handles checkpointing and learning rate scheduling

4. **Training Scripts**: Simplified to just two platform-specific scripts:
   - Mac-specific script (train_mac.sh) optimized for CPU training
   - Linux-specific script (train_linux.sh) optimized for GPU training with CUDA
   - All other complex or unnecessary scripts have been removed

## Recent Changes

The core components of the system have been implemented, including:

1. **Data Processing Pipeline**:
   - Completed MERRA2PRISMDataset implementation with efficient data loading
   - Implemented reprojection of MERRA-2 data to PRISM grid
   - Added support for DEM integration
   - Created efficient patch extraction and normalization utilities

2. **Model Architecture**:
   - Implemented PrithviDownscaler with transformer backbone
   - Added UpsampleBlock for progressive resolution enhancement
   - Implemented DEM encoder for elevation data processing
   - Configured for flexible input/output channel configuration

3. **Training Framework**:
   - Completed Lightning module implementation
   - Implemented combined loss function (MAE, MSE, SSIM)
   - Added comprehensive metric calculation
   - Set up optimizer with learning rate scheduling

4. **Training Scripts**:
   - Streamlined training workflow with two clean platform-specific scripts
   - Created train_mac.sh for Mac systems using CPU acceleration
   - Created train_linux.sh for Linux systems with GPU acceleration
   - Removed all other complex shell scripts (prithvi.sh, train.sh, etc.)
   - Made remaining scripts executable and ready to use

## Next Steps

The following tasks are prioritized for immediate implementation:

1. **Model Refinement**:
   - Fine-tune model hyperparameters (number of layers, heads, etc.)
   - Experiment with different loss function weights
   - Optimize memory usage for larger batch sizes

2. **Validation and Testing**:
   - Expand validation metrics to include spatial pattern analysis
   - Implement visualization tools for qualitative assessment
   - Compare performance across different geographical regions

3. **Performance Optimization**:
   - Optimize data loading for reduced training time
   - Implement distributed training for multi-GPU setups
   - Explore gradient checkpointing for memory efficiency

4. **Integration and Deployment**:
   - Create inference pipeline for new MERRA-2 data
   - Develop export functionality for downscaled outputs
   - Document API and usage patterns

## Active Decisions and Considerations

The following decisions and considerations are currently being evaluated:

1. **Model Architecture Decisions**:
   - Balance between model complexity and inference speed
   - Optimal number of transformer layers and attention heads
   - Integration strategy for DEM data (early fusion vs. late fusion)

2. **Training Strategy Decisions**:
   - Loss function weighting between MAE, MSE, and SSIM components
   - Learning rate scheduling strategy
   - Data augmentation techniques beyond random masking
   - Platform-specific training optimizations (Mac vs. Linux)

3. **Data Processing Decisions**:
   - Normalization strategy for different climate variables
   - Patch size and stride for optimal spatial context
   - Train/validation/test split methodology

4. **Evaluation Framework**:
   - Metrics prioritization for model selection
   - Validation strategy across different climate regions
   - Benchmarking against alternative downscaling methods

## Important Patterns and Preferences

The following patterns and preferences have emerged during development:

1. **Code Organization**:
   - Modular design with clear separation of concerns
   - Configuration-driven architecture with minimal hardcoding
   - Comprehensive docstrings and type hints
   - Simple platform-specific scripts rather than complex unified scripts
   - Minimalist approach to shell scripts - only keep what's necessary

2. **Data Handling**:
   - Preference for xarray and rasterio for climate and geospatial data
   - Careful management of coordinate systems and transformations
   - Emphasis on maintaining metadata throughout processing

3. **Model Design**:
   - Preference for transformer-based architectures over pure CNNs
   - Integration of domain-specific knowledge (e.g., elevation influences)
   - Modular components for easier experimentation

4. **Training Approach**:
   - Leveraging PyTorch Lightning for standardized training loops
   - Multi-metric evaluation rather than single-metric optimization
   - Regular checkpointing and experiment tracking
   - Platform-specific optimizations (CPU for Mac, GPU for Linux)
   - Direct, simple training scripts with clear parameters

## Learnings and Project Insights

Key insights gained during the project development include:

1. **Technical Insights**:
   - Transformer models effectively capture spatial relationships in climate data
   - DEM integration significantly improves downscaling performance
   - Combined loss functions better preserve both pixel accuracy and spatial patterns
   - Platform-specific training strategies yield better performance
   - Simpler shell scripts improve maintenance and usability

2. **Data Insights**:
   - MERRA-2 and PRISM alignment requires careful coordinate handling
   - Climate variables require different normalization strategies
   - Patch-based processing effectively balances context and memory constraints

3. **Training Insights**:
   - Mixed precision training essential for transformer models on GPU
   - Learning rate scheduling critical for stable convergence
   - Validation metrics should include both statistical and spatial pattern measures
   - Mac systems perform best with CPU optimizations rather than GPU (MPS) accelerations
   - Linux systems with CUDA provide significant training speed advantages

4. **Challenges and Solutions**:
   - Memory constraints addressed through patch-based processing
   - Complex loss landscape navigated via careful optimizer configuration
   - Data heterogeneity managed through variable-specific processing
   - Platform differences handled through dedicated training scripts
   - Script complexity reduced through elimination of unnecessary options 