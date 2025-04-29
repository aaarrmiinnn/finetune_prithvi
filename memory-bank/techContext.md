# Technical Context

This document outlines the technologies, development setup, technical constraints, dependencies, and tool usage patterns for the MERRA-2 to PRISM downscaling project.

## Technologies Used

### Core Libraries and Frameworks

1. **PyTorch**: Core deep learning framework
   - Version: Latest stable (1.10+)
   - Used for: Neural network definition, training, and inference
   - Key features: Dynamic computation graph, GPU acceleration

2. **PyTorch Lightning**: Training framework
   - Used for: Standardizing training loops, callbacks, multi-GPU training
   - Key benefits: Reduced boilerplate, built-in best practices

3. **xarray**: Labeled multidimensional arrays
   - Used for: Climate data handling with coordinate awareness
   - Key features: NetCDF compatibility, label-based indexing

4. **rioxarray/rasterio**: Geospatial data handling
   - Used for: Reading and manipulating georeferenced raster data
   - Key features: Coordinate transforms, reprojection

5. **NumPy**: Numerical computing
   - Used for: Data manipulation, array operations
   - Integrated with: PyTorch, xarray

6. **CUDA**: GPU computing platform
   - Used for: Accelerating model training and inference
   - Requirements: NVIDIA GPU with CUDA support

### Data Formats

1. **NetCDF**: Network Common Data Form
   - Used for: MERRA-2 input data
   - Handled via: xarray

2. **GeoTIFF**: Georeferenced TIFF files
   - Used for: PRISM target data, DEM data
   - Handled via: rasterio

3. **ZIP archives**: Compressed data files
   - Used for: Packaged PRISM data
   - Handled via: Python's zipfile module

## Development Setup

### Environment Requirements

1. **Python**: Version 3.8+ 
2. **CUDA Toolkit**: Compatible with PyTorch version
3. **Storage**: Large disk space for dataset storage (100GB+)
4. **RAM**: 16GB+ recommended for data processing
5. **GPU**: NVIDIA GPU with 8GB+ VRAM recommended

### Environment Management

- **conda/miniconda**: Recommended for environment isolation
- **requirements.txt**: Dependency specification

### Directory Structure

```
finetune_prithvi/
├── config/            # Configuration files
├── memory-bank/       # Project documentation
├── notebooks/         # Analysis and exploration notebooks
├── scripts/           # Utility and execution scripts
├── src/               # Source code
│   ├── data/          # Data loading and processing
│   ├── models/        # Model architecture
│   ├── trainers/      # Training implementations
│   └── utils/         # Utility functions
└── tests/             # Unit tests
```

## Technical Constraints

1. **Memory Limitations**:
   - Climate data can be very large (GB to TB)
   - Solution: Patch-based processing, efficient loading

2. **Computational Requirements**:
   - Transformer models are compute-intensive
   - Solution: Mixed precision training, gradient accumulation

3. **Data Resolution Differences**:
   - MERRA-2: ~50km resolution
   - PRISM: ~4km resolution
   - Solution: Careful reprojection and alignment

4. **Missing Data Handling**:
   - Climate data often has missing values, especially near coastlines
   - Solution: Masking, interpolation, and training accommodation

5. **Model Size**:
   - Prithvi-based model has significant parameter count
   - Solution: Efficient implementation, gradient checkpointing

## Dependencies

### Core Dependencies

```
torch>=1.10.0
pytorch-lightning>=1.5.0
xarray>=0.20.0
rioxarray>=0.8.0
rasterio>=1.2.0
numpy>=1.20.0
dask>=2021.11.0
netCDF4>=1.5.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
wandb>=0.12.0       # Optional for experiment tracking
```

### Development Dependencies

```
pytest>=6.0.0
flake8>=4.0.0
black>=21.10b0
isort>=5.9.0
jupyter>=1.0.0
```

## Tool Usage Patterns

### Data Processing Workflow

1. **Data Loading**:
   ```python
   # Load MERRA-2 data
   merra2_data, lats, lons = load_merra2_data(file_path, variables)
   
   # Load PRISM data
   prism_data, transform = load_prism_data(prism_dir, date, variables)
   
   # Load DEM data
   dem_data = load_dem_data(dem_path)
   ```

2. **Data Preprocessing**:
   ```python
   # Reproject MERRA-2 to PRISM grid
   aligned_merra2 = reproject_merra2_to_prism(
       merra2_data, merra2_lats, merra2_lons, prism_shape, prism_transform
   )
   
   # Normalize data
   normalized_data, stats = normalize_data(data)
   
   # Extract patches
   patches = extract_patches(data, patch_size=64, stride=32)
   ```

3. **Dataset Creation**:
   ```python
   dataset = MERRA2PRISMDataset(
       merra2_dir, prism_dir, dates, merra2_vars, prism_vars, dem_path
   )
   
   # Create data loaders
   train_loader, val_loader, test_loader = create_dataloaders(config)
   ```

### Training Workflow

1. **Model Initialization**:
   ```python
   model = PrithviDownscaler(
       input_channels=len(config['data']['input_vars']),
       output_channels=len(config['data']['target_vars']),
       hidden_dim=config['model']['hidden_dim'],
       num_heads=config['model']['num_heads'],
       num_layers=config['model']['num_layers'],
       mlp_ratio=config['model']['mlp_ratio'],
       use_dem=config['model']['use_dem']
   )
   ```

2. **Lightning Module Setup**:
   ```python
   lightning_module = DownscalingLightningModule(
       model=model,
       config=config
   )
   ```

3. **Training Execution**:
   ```python
   trainer = pl.Trainer(
       max_epochs=config['training']['max_epochs'],
       gpus=config['hardware']['gpus'],
       precision=config['training']['precision'],
       callbacks=[
           EarlyStopping(monitor='val_loss', patience=10),
           ModelCheckpoint(monitor='val_loss', save_top_k=3)
       ]
   )
   
   trainer.fit(lightning_module, datamodule)
   ```

### Inference Workflow

1. **Model Loading**:
   ```python
   lightning_module = DownscalingLightningModule.load_from_checkpoint(
       checkpoint_path,
       strict=False
   )
   model = lightning_module.model
   ```

2. **Prediction**:
   ```python
   model.eval()
   with torch.no_grad():
       predictions = model(inputs, dem)
   ```

3. **Post-processing**:
   ```python
   # Denormalize
   denormalized = denormalize_data(predictions, stats)
   
   # Save output
   save_predictions(denormalized, output_path, metadata)
   ```

## Best Practices

1. **Configuration Management**:
   - Use YAML/JSON config files for experiment parameters
   - Parse and validate configurations at runtime

2. **Checkpointing**:
   - Save models based on validation metrics
   - Include all necessary information for resuming training

3. **Logging**:
   - Use TensorBoard or W&B for experiment tracking
   - Log all relevant metrics, hyperparameters, and sample outputs

4. **Reproducibility**:
   - Set random seeds
   - Document software versions and environment
   - Save configuration with results

5. **Testing**:
   - Unit tests for core functionality
   - Integration tests for full data pipeline
   - Model correctness tests 