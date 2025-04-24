# System Patterns

## Architecture Overview
The project follows a modular architecture with clear separation of concerns:

```
src/
├── config/         # Configuration management
├── data/           # Data loading and preprocessing
├── models/         # Model architecture definitions
├── trainers/       # Training logic
├── utils/          # Utility functions
├── visualizations/ # Visualization tools
└── main.py        # Entry point
```

## Key Design Patterns
1. Configuration Management:
   - YAML-based configuration
   - Hierarchical structure
   - Environment-aware settings

2. Data Pipeline:
   - Patch-based processing
   - Masked training strategy
   - Multi-resolution handling
   - Auxiliary input integration (DEM)

3. Model Architecture:
   - Pretrained Prithvi backbone
   - Optional encoder freezing
   - Upsampling pathway
   - Multi-scale feature fusion

4. Training System:
   - PyTorch Lightning Trainer
   - Mixed precision optimization
   - Cosine learning rate scheduling
   - Multi-metric loss function
   - Checkpoint management

## Critical Implementation Paths
1. Data Flow:
   ```
   MERRA2 (NetCDF4) → Patches → Model → Upsampled Output → PRISM Resolution
   DEM (BIL) ─────────┘
   ```

2. Training Loop:
   ```
   Load Data → Create Patches → Apply Masking → Forward Pass → 
   Multi-Loss Computation → Backward Pass → Optimization Step
   ```

3. Validation/Testing:
   ```
   Load Test Data → Full Resolution Forward Pass → 
   Compute Metrics → Log Results
   ``` 