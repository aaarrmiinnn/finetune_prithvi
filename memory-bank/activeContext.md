# Active Context

## Current State
1. Environment Setup:
   - Conda environment "prithvi" created with Python 3.8
   - Dependencies installed from requirements.txt
   - GDAL installation confirmed

2. Data Availability:
   - MERRA2 data present for March 1-2, 2025
   - PRISM data available for temperature and precipitation
   - DEM data ready in BIL format

3. Configuration:
   - Base configuration in src/config/config.yaml
   - Need to update model to use Prithvi-WxC-1.0-2300M
   - Mixed precision training enabled
   - Patch size set to 64x64
   - Mask ratio of 0.3 for training

## Recent Changes
- Initial project setup completed
- Environment configuration established
- Data files organized in appropriate directories
- Identified correct Prithvi model version (WxC-1.0-2300M)

## Next Steps
1. Model Setup:
   - Download Prithvi-WxC-1.0-2300M from HuggingFace
   - Configure model for downscaling task
   - Set up fine-tuning pipeline

2. Data Processing:
   - Verify data loading pipeline
   - Test patch creation
   - Validate DEM integration

3. Training Setup:
   - Test model initialization
   - Verify loss function implementation
   - Check logging configuration

4. Validation:
   - Implement evaluation metrics
   - Set up visualization tools
   - Create validation pipeline

## Active Decisions
1. Using Prithvi-WxC-1.0-2300M as the base model
2. Incorporating DEM as auxiliary input
3. Training with mixed precision for efficiency
4. Using multiple loss components (MAE, MSE, SSIM)

## Project Insights
1. Data Organization:
   - Clear separation of MERRA2, PRISM, and DEM data
   - Consistent file naming conventions
   - Hierarchical directory structure

2. Configuration Management:
   - YAML-based configuration for flexibility
   - Modular code organization
   - Clear separation of concerns

3. Development Approach:
   - Focus on reproducibility
   - Emphasis on code modularity
   - Integration of best practices 