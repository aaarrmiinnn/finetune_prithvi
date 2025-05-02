# Technical Context

## Development Environment
- Python 3.11.11
- Operating Systems: macOS (Darwin 24.4.0) and Linux support
- Hardware: Support for both CPU and GPU environments
- 64 CPU cores available in the development environment

## Core Technologies
1. Prithvi Model
   - Base model for fine-tuning
   - Requires specific checkpoint handling
   - Supports gradient checkpointing for memory efficiency

2. Training Infrastructure
   - Distributed training support
   - Multi-GPU capabilities
   - Dynamic resource allocation
   - Memory optimization features

## Key Dependencies
- Requirements managed via requirements.txt
- Core training scripts:
  - train_multi_gpu.sh (Multi-GPU training)
  - train_linux_memory_efficient.sh (Memory-optimized training)
  - train_linux.sh (Standard Linux training)
  - train_mac.sh (macOS support)

## Configuration System
- YAML-based configuration
- Dynamic parameter adjustment
- Hardware-aware settings
- Supports both training and model parameters

## File Structure
- src/: Core source code
- models/: Model checkpoints and artifacts
- data/: Training data
- logs/: Training logs and outputs
- docs/: Documentation
- cache/: Temporary files and caching

## Development Tools
- Version Control: Git
- Current Version: v1.0.0
- Feature Branch: feature/multi-gpu-training 