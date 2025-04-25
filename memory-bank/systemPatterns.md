# System Patterns

## Architecture Overview

### Model Components
1. PrithviDownscaler
   - Main model class for downscaling MERRA2 to PRISM
   - Uses PrithviWxC as backbone
   - Includes input projection, feature extraction, and decoder components

2. PrithviWxC
   - Modified transformer-based backbone
   - Maintains spatial dimensions throughout
   - Uses dynamic positional embeddings
   - Customized patch embedding for our use case

3. Supporting Components
   - UpsampleBlock: Progressive upsampling with residual connections
   - PatchEmbedding: Convolutional embedding with spatial dimension preservation
   - TransformerBlock: Self-attention and MLP layers

### Data Flow
```
Input (MERRA2) -> Input Projection -> PrithviWxC Backbone -> Feature Extraction -> Decoder -> Output (PRISM)
```

## Key Technical Decisions

### Model Architecture
1. Spatial Dimension Handling
   - Maintain spatial dimensions through patch embedding
   - Use padding in convolutions to preserve size
   - Progressive upsampling in decoder

2. Memory Optimization
   - Reduced model size (hidden_dim: 256)
   - Smaller batch size (2)
   - MPS backend for Mac GPU acceleration

3. Training Strategy
   - Combined loss function (MAE + MSE)
   - AdamW optimizer with cosine learning rate schedule
   - Gradient clipping for stability

## Design Patterns

### Component Patterns
1. Modular Design
   - Separate modules for different functionalities
   - Clear interfaces between components
   - Reusable blocks (UpsampleBlock, TransformerBlock)

2. Configuration Management
   - YAML-based configuration
   - Hierarchical organization
   - Environment-specific settings

3. Data Processing
   - Patch-based processing
   - Normalization pipeline
   - Multi-variable handling

### Implementation Patterns
1. Forward Pass Flow
   ```
   1. Input projection
   2. Patch embedding
   3. Position embedding
   4. Transformer processing
   5. Feature extraction
   6. Progressive upsampling
   7. Final output
   ```

2. Training Loop
   ```
   1. Data loading
   2. Forward pass
   3. Loss calculation
   4. Backward pass
   5. Optimization step
   6. Validation
   ```

## Critical Implementation Paths

### Model Forward Pass
1. Input Processing
   - Project input to correct dimension
   - Create patches while preserving spatial info
   - Add position embeddings

2. Feature Extraction
   - Process through transformer blocks
   - Extract high-level features
   - Maintain spatial relationships

3. Upsampling
   - Progressive resolution increase
   - Feature refinement
   - Final output generation

### Training Process
1. Data Pipeline
   - Load MERRA2 and PRISM data
   - Create patches
   - Apply normalization

2. Optimization
   - Calculate losses
   - Update parameters
   - Adjust learning rate

3. Validation
   - Compute metrics
   - Log results
   - Save checkpoints 