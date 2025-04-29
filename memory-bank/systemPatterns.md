# System Patterns

This document outlines the system architecture, key technical decisions, design patterns, and component relationships for the MERRA-2 to PRISM downscaling project.

## System Architecture

The system follows a modular machine learning architecture designed for climate data downscaling, with the following major components:

### Data Pipeline
- **Data Loading**: Specialized loaders for MERRA-2 (coarse resolution) and PRISM (high resolution) datasets
- **Preprocessing**: Data normalization, coordinate reprojection, and patch extraction
- **DataModule**: PyTorch Lightning DataModule for efficient training/validation/test splits

### Model Architecture
- **Core Downscaler**: Based on IBM's Prithvi-100M architecture with transformer backbone
- **DEM Integration**: Digital Elevation Model encoder for topographic influence
- **UpsampleBlock**: Convolutional upsampling blocks for spatial resolution enhancement

### Training Framework
- **Lightning Module**: PyTorch Lightning module to handle training, validation, testing
- **Loss Functions**: Combination of MAE, MSE, and SSIM losses
- **Metrics**: Comprehensive evaluation metrics (RMSE, MAE, bias, R², SSIM)
- **Optimization**: Learning rate scheduling and gradient clipping

### Inference Pipeline
- **Model Loading**: Checkpoint loading mechanism
- **Prediction**: End-to-end inference workflow
- **Visualization**: Output visualization capabilities

## Key Technical Decisions

1. **Transformer-Based Architecture**: Using IBM's Prithvi model as backbone for its proven effectiveness in geospatial tasks
2. **Patch-Based Processing**: Data is processed in patches (64x64) to manage memory requirements and enable efficient batching
3. **Multi-Scale Evaluation**: Metrics are computed at both global and variable-specific levels
4. **Hybrid Loss Function**: Combining multiple losses (MAE, MSE, SSIM) for both pixel accuracy and structural similarity
5. **Integration of Elevation Data**: DEM encoder to capture topographic influences on climate variables
6. **Mixed Precision Training**: Using GPU-accelerated mixed precision for faster training
7. **Checkpoint Management**: Comprehensive model checkpointing for experiment tracking and reproducibility

## Design Patterns

1. **Factory Pattern**: Creation of dataloaders and model components
2. **Strategy Pattern**: Flexible loss function composition
3. **Adapter Pattern**: Harmonizing different data sources (MERRA-2, PRISM, DEM)
4. **Composite Pattern**: Building complex models from simpler components
5. **Observer Pattern**: Monitoring and logging training progress
6. **Template Method Pattern**: Common preprocessing steps with variable-specific implementations

## Component Relationships

```
                ┌───────────────┐
                │ Configuration │
                └───────┬───────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
│  DataModule   │ │   Model   │ │ LightningModule│
└───────┬───────┘ └─────┬─────┘ └───────┬───────┘
        │               │               │
┌───────▼───────┐       │               │
│    Dataset    │◄──────┘               │
└───────┬───────┘                       │
        │                               │
┌───────▼───────┐                       │
│ Preprocessing │                       │
└───────────────┘                       │
                                        │
                                        │
┌───────────────┐                       │
│     Losses    │◄──────────────────────┘
└───────────────┘
```

## Critical Implementation Paths

### Data Preparation
1. Load MERRA-2 and PRISM data from different file formats
2. Reproject MERRA-2 data to PRISM grid
3. Extract patches and create paired samples
4. Apply normalization based on training set statistics
5. Create efficient PyTorch DataLoaders

### Model Training
1. Initialize PrithviDownscaler model with configuration
2. Configure optimizer and learning rate scheduler
3. Execute training loop via PyTorch Lightning
   - Forward pass through model
   - Calculate combined loss
   - Compute and log metrics
   - Update weights
4. Validation on held-out set
5. Checkpoint saving based on validation metrics

### Inference
1. Load trained model from checkpoint
2. Preprocess input MERRA-2 data
3. Pass through model for high-resolution prediction
4. Post-process output (denormalize)
5. Evaluate predictions against PRISM ground truth

## Implementation Challenges and Solutions

1. **Memory Efficiency**: Patch-based processing and data loading optimization
2. **Coordinate Systems**: Careful reprojection between different grids
3. **Missing Data**: Handling of NaN values and masked regions
4. **Scale Differences**: Normalization strategies for different variables
5. **Evaluation Complexity**: Multi-metric approach for comprehensive assessment
6. **Reproducibility**: Seed setting and deterministic operations where possible

## Performance Considerations

1. **Bottlenecks**: Data loading and preprocessing are potential bottlenecks
2. **GPU Utilization**: Model designed to effectively utilize GPU acceleration
3. **Batch Size Optimization**: Balance between memory usage and training speed
4. **Caching**: Strategic caching of preprocessed data
5. **Distributed Training**: Support for multi-GPU training via PyTorch Lightning 