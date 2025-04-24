# System Patterns

## System Architecture

### High-Level Architecture
```
[MERRA-2 Data] → [Preprocessing] → [Data Module] → [Prithvi Fine-tuning] → [Postprocessing] → [PRISM-like Outputs]
```

### Core Components

1. **Data Pipeline** *(Implemented)*
   - **DataLoader:** Dedicated loaders for MERRA-2, PRISM, and DEM data
   - **Preprocessor:** Handles alignment, normalization, and quality control
   - **ClimateDataset:** PyTorch Dataset for efficient data handling and caching
   - **Patch extraction:** Creates training patches with context windows

2. **Model Architecture** *(Implemented)*
   - **InputAdapter:** Maps from 4 variables to Prithvi's input structure
   - **PrithviModel:** Encapsulated Prithvi WxC model with controllable frozen layers
   - **Upsampling Layers:** 4x upscaling to match PRISM resolution
   - **DEMEncoder:** Processes and integrates elevation data

3. **Training System** *(Implemented)*
   - **DownscalingModule:** Lightning module with standardized training workflow
   - **LossComponents:** MSE, MAE, and custom spatial loss functions
   - **Metrics:** RMSE, MAE, bias calculations during validation
   - **Callbacks:** Model checkpointing and TensorBoard logging

4. **Evaluation System** *(Implemented)*
   - **MetricCalculator:** RMSE, MAE, and bias metrics across spatial dimensions
   - **Visualizer:** Plot generation for spatial error maps and time series
   - **BaselineComparator:** Implements bilinear and cubic interpolation baselines

5. **Configuration System** *(Implemented)*
   - **ConfigManager:** YAML-based parameter management
   - **ArgumentParser:** Command-line configuration override
   - **ExperimentManager:** Experiment naming and tracking

## Implemented Patterns

### Data Flow Patterns
- **Modular Data Loading:** Separate loader classes for each data source
- **On-demand Processing:** Data loaded and processed as needed
- **Caching Strategy:** Preprocessed data cached to disk for efficiency
- **Patch-based Training:** Extracts spatial patches for optimal GPU usage

### Model Patterns
- **Foundation Model Integration:** Prithvi WxC integrated as backbone
- **Freezable Components:** Selectively freeze/unfreeze model parts
- **Auxiliary Data Fusion:** DEM elevation data incorporated via encoder
- **Multi-scale Feature Processing:** Features processed at multiple resolutions

### Training Patterns
- **PyTorch Lightning Structure:** Standardized training loop implementation
- **Configurable Loss Function:** Weighted combination of multiple losses
- **Mixed Precision Training:** Option for mixed precision to optimize memory
- **Progressive Learning Rate:** Learning rate scheduling for stable training

### Configuration Patterns
- **Hierarchical Configuration:** YAML with nested sections for organization
- **Run Mode System:** Different execution modes (train, test, explore)
- **Command-line Overrides:** Key parameters adjustable via command line
- **Environment Variables:** Support for environment-based configuration

## Key Interfaces

### Data Interface
```python
class ClimateDataset(Dataset):
    def __init__(self, merra_loader, prism_loader, dem_loader, config)
    def __len__()
    def __getitem__(idx)  # Returns (input_data, target_data) pairs
    def preprocess_data()  # Handles normalization and alignment
```

### Model Interface
```python
class DownscalingModel(nn.Module):
    def __init__(self, config)
    def forward(merra_data, dem_data=None)
    def freeze_backbone(freeze=True)
    def unfreeze_layers(layer_names)
```

### Training Interface
```python
class DownscalingModule(pl.LightningModule):
    def __init__(self, model, config)
    def forward(x, dem=None)
    def training_step(batch, batch_idx)
    def validation_step(batch, batch_idx)
    def test_step(batch, batch_idx)
    def configure_optimizers()
    def calculate_metrics(pred, target)
```

## Implementation Details

1. **Data Processing Implementation**
   - MERRA-2 NetCDF loading with xarray
   - PRISM GeoTIFF loading with rasterio
   - DEM BIL format loading with rasterio
   - Spatial alignment using projection transforms
   - Normalization based on variable statistics

2. **Model Implementation**
   - Prithvi WxC foundation model from Hugging Face
   - Custom input adapter for variable mapping
   - DEM encoder with convolutional layers
   - Upsampling decoder with 4x resolution increase
   - Skip connections for feature propagation

3. **Training Implementation**
   - AdamW optimizer with configurable parameters
   - OneCycleLR scheduler for learning rate management
   - EarlyStopping and ModelCheckpoint callbacks
   - TensorBoard logging for metrics tracking
   - Mixed precision support for memory efficiency 