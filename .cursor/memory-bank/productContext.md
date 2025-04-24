# Product Context

## Why This Project Exists
The MERRA-2 to PRISM downscaling project addresses the need for high-resolution climate data by leveraging modern foundation models. Climate data is often available at coarse resolutions (like MERRA-2's ~50km grid), which is insufficient for many applications requiring local detail. This project uses the Prithvi WxC foundation model to generate high-resolution climate data (at PRISM's 4km resolution) from coarse inputs.

## Problem Being Solved
1. **Resolution Mismatch:** MERRA-2 provides global coverage but at coarse resolution (~50km), while applications need higher resolution data
2. **Limited Observations:** High-resolution observed data like PRISM is only available for limited regions and time periods
3. **Traditional Downscaling Limitations:** Conventional statistical and dynamical downscaling methods often lack the ability to capture complex spatial patterns
4. **Computational Efficiency:** Need for efficient downscaling that can be applied to large datasets without prohibitive computational costs

## How It Should Work
1. **Foundation Model Approach:** Utilize Prithvi WxC's pre-trained knowledge of atmospheric patterns through transfer learning
2. **Fine-tuning Pipeline:** Take coarse MERRA-2 variables as input, produce high-resolution PRISM-like outputs
3. **Auxiliary Integration:** Incorporate elevation data (DEM) to enhance downscaling accuracy
4. **Patch-based Processing:** Process data in spatial patches for efficient training and inference
5. **Flexible Architecture:** Allow freezing/unfreezing of model components to optimize performance

## User Experience Goals
1. **Simple Configuration:** YAML-based configuration system for experiment customization
2. **Clear Visualization:** Tools to visualize and compare inputs, targets, and predictions
3. **Performance Metrics:** Comprehensive metrics to evaluate downscaling quality
4. **Reproducibility:** Deterministic training with seed control and version tracking
5. **Production Readiness:** Optimized for both research experiments and operational use

## Core Workflows

### Data Preparation Workflow
1. Load MERRA-2 NetCDF4 files with selected variables
2. Load corresponding PRISM GeoTIFF files
3. Align spatial and temporal dimensions
4. Apply normalization and preprocessing
5. Extract spatial patches for training

### Training Workflow
1. Configure experiment parameters via YAML
2. Initialize Prithvi WxC with optional layer freezing
3. Create data loaders with train/val/test splits
4. Execute training loop with monitoring and checkpointing
5. Evaluate performance against validation data

### Inference Workflow
1. Load trained model from checkpoint
2. Preprocess new MERRA-2 input data
3. Generate high-resolution predictions
4. Apply any post-processing needed
5. Export results in desired format

## Target Users
1. **Climate Scientists:** For research requiring high-resolution climate data
2. **Impact Modelers:** For downstream applications like hydrology, agriculture, etc.
3. **ML Researchers:** As a foundation for further climate AI research
4. **Operational Forecasters:** For enhancing resolution of operational products

## Solution Value
1. **Improved Resolution:** Transform 50km MERRA-2 data to 4km PRISM resolution
2. **Better Extreme Representation:** Capture temperature and precipitation extremes more accurately
3. **Physical Consistency:** Maintain relationships between variables using deep learning
4. **Transferability:** Create a framework extendable to other regions and variables

## Success Metrics
- **Technical:** 20%+ improvement over baseline methods in standard metrics
- **Scientific:** Better representation of spatial patterns and extreme events
- **Usability:** Complete documentation and example notebooks for users to extend the approach

## Current Alternatives
- **Statistical Downscaling:** Methods like BCSD (Bias Correction Spatial Disaggregation)
- **Dynamical Downscaling:** Running high-resolution regional climate models
- **Machine Learning:** Simpler CNN approaches without foundation model advantages

## Product Advantage
This approach leverages Prithvi WxC's pre-trained knowledge of atmospheric physics and patterns, acquired from 40 years of climate data across 160 variables. This foundation enables the model to generalize better than methods trained from scratch, particularly for extreme events and complex spatial patterns. 