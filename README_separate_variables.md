# Separate Variable Training for MERRA2-PRISM Downscaling

This feature branch implements separate training for temperature and precipitation variables instead of training a single model for both. This approach allows for specialized model configurations tailored to each variable's unique characteristics.

## Why Train Variables Separately?

- **Specialized Optimization**: Temperature and precipitation have different statistical properties and spatial patterns
- **Different Loss Functions**: Precipitation benefits from MAE loss, while temperature works better with MSE
- **Hyperparameter Tuning**: Each variable can have its own learning rate, batch size, etc.
- **Simplified Debugging**: Easier to diagnose and fix issues when working with a single output variable
- **Variable-Specific Architecture**: Different hidden dimensions and model capacities for each variable

## Configuration Files

Two specialized configuration files are provided:

1. `src/config/temperature_config.yaml`: Optimized for temperature (tdmean) downscaling
2. `src/config/precipitation_config.yaml`: Optimized for precipitation (ppt) downscaling

Key differences in the configurations include:

| Parameter | Temperature | Precipitation | Reason |
|-----------|-------------|---------------|--------|
| hidden_dim | 64 | 96 | Precipitation patterns are more complex |
| batch_size | 2 | 1 | Temperature is smoother and can use larger batches |
| epochs | 10 | 15 | Precipitation requires more training time |
| learning rate | 1e-5 | 3e-6 | Lower learning rate for precipitation to avoid instability |
| loss weights | MSE: 1.0, MAE: 0.5 | MAE: 1.0, MSE: 0.1 | MAE better handles precipitation sparsity |
| SSIM weight | 0.1 | 0.2 | Spatial coherence more important for precipitation |

## Training the Models

Use the provided training script:

```bash
# Train both models (default)
./train_separate_variables.sh

# Train only temperature model
./train_separate_variables.sh --temperature

# Train only precipitation model
./train_separate_variables.sh --precipitation
```

The script will:
1. Create necessary directories
2. Validate cache files to prevent corrupted data
3. Train the models using the specialized configs
4. Save checkpoints in dedicated directories

## Inference with Separate Models

To run inference and combine predictions from both models:

```bash
python inference_separate_variables.py \
  --temp-checkpoint models/checkpoints/temperature/best.ckpt \
  --precip-checkpoint models/checkpoints/precipitation/best.ckpt \
  --use-test-data \
  --visualize
```

This will:
1. Load both models with their respective configurations
2. Run inference separately for each variable
3. Combine the predictions
4. Generate visualization comparing predictions to targets (if using test data)
5. Save the combined predictions and metadata

## Implementation Details

The separate variable implementation:

1. **Uses the same input data** for both models
2. **Creates specialized architecture capacity** for each variable
3. **Applies different loss functions** optimized for each variable
4. **Combines predictions at inference time** for a unified output
5. **Preserves the original model architecture** with the only change being the output channels

## Potential Improvements

- **Transfer Learning**: Train the temperature model first, then use it to initialize the precipitation model
- **Multi-Task Learning with Different Weights**: Alternative approach that keeps a single model but applies different loss weights
- **Ensemble Approaches**: Train multiple models for each variable and combine their predictions
- **Custom Metrics**: Develop specialized evaluation metrics for each variable 