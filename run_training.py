#!/usr/bin/env python
"""Script to run training with NaN-resistant modifications."""
import os
import sys
import subprocess
import logging
import time
import platform
import shutil
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_runner.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_cuda_availability():
    """Check if CUDA is available using PyTorch."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"PyTorch CUDA is available: {cuda_available}")
            logger.info(f"CUDA Devices: {device_count}")
            logger.info(f"Current Device: {current_device}")
            logger.info(f"Device Name: {device_name}")
        else:
            logger.warning("PyTorch CUDA is not available")
        return cuda_available
    except ImportError:
        logger.error("PyTorch not installed or failed to import")
        return False
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {str(e)}")
        return False

def run_training():
    """Run the appropriate training script based on platform."""
    logger.info("Starting training with NaN-resistant modifications")
    
    # Ensure the logs and checkpoints directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    # Record the start time
    start_time = time.time()
    
    # Determine platform and select the appropriate script
    system = platform.system()
    if system == "Darwin":  # macOS
        script_path = "./train_mac.sh"
        logger.info("Detected macOS, using train_mac.sh")
    else:  # Linux or other
        script_path = "./train_linux.sh"
        logger.info(f"Detected {system}, using train_linux.sh")
        
        # Check if CUDA is available using PyTorch
        cuda_available = check_cuda_availability()
        if not cuda_available:
            logger.warning("CUDA not available via PyTorch. Training will continue but may not use GPU acceleration.")
            logger.info("Setting CUDA_VISIBLE_DEVICES= to disable GPU usage")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Make sure the script is executable
    os.chmod(script_path, 0o755)
    
    # Run the training script
    try:
        # Set environment variables including PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}"
        logger.info(f"Setting PYTHONPATH to include current directory: {env['PYTHONPATH']}")
        
        cmd = [script_path]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the process and stream output to the console
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env  # Pass the updated environment variables
        )
        
        # Stream the output
        for line in process.stdout:
            line = line.strip()
            logger.info(line)
            
            # Check for NaN-related issues
            if "NaN" in line or "Inf" in line:
                logger.warning(f"NaN/Inf detected: {line}")
                
            # Check for specific errors
            if "find_unused_parameters" in line and "unexpected keyword argument" in line:
                logger.error("PyTorch Lightning version incompatibility detected: 'find_unused_parameters' error")
                logger.info("Please update the config.yaml file to remove this parameter or upgrade PyTorch Lightning")
                
            # Check for GPU usage warnings
            if "GPU available but not used" in line:
                logger.error("GPU is available but not being used!")
                logger.info("Check your configuration to ensure GPU acceleration is properly enabled")
                
            # Check for device mismatch issues
            if "Expected all tensors to be on the same device" in line:
                logger.error("Device mismatch detected: tensors are on different devices")
                logger.info("This often happens when model device doesn't match the accelerator setting")
                logger.info("Stopping training to avoid further errors")
                
                # Try to kill the process to prevent further errors
                process.terminate()
                
                # Create a fixed config file that ensures device consistency
                fix_device_mismatch()
                
                logger.info("Created a fixed config file. Please try running again with: python run_training.py")
                break
        
        # Wait for the process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Training completed successfully")
        else:
            logger.error(f"Training failed with return code {return_code}")
        
    except Exception as e:
        logger.error(f"Error running training: {str(e)}")
        
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

def fix_device_mismatch():
    """Create a fixed config file that ensures device consistency."""
    logger.info("Creating a fixed config file to ensure device consistency")
    
    # Create a backup of the original config
    config_path = "src/config/config.yaml"
    backup_path = "src/config/config.yaml.bak"
    fixed_path = "src/config/config_fixed.yaml"
    
    # Create backup if it doesn't exist
    if not os.path.exists(backup_path):
        shutil.copy(config_path, backup_path)
        logger.info(f"Created backup of original config at {backup_path}")
    
    # Check for CUDA availability
    cuda_available = check_cuda_availability()
    
    # Read the config file
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Flag to track if we're in the model section
    in_model_section = False
    in_hardware_section = False
    in_cluster_section = False
    
    # Create fixed config
    with open(fixed_path, 'w') as f:
        for line in lines:
            # Check if we're entering different sections
            if line.strip() == "# Model Configuration":
                in_model_section = True
                in_hardware_section = False
                in_cluster_section = False
            elif line.strip() == "# Hardware":
                in_hardware_section = True
                in_model_section = False
                in_cluster_section = False
            elif line.strip() == "# Cluster Settings (for future use)":
                in_cluster_section = True
                in_model_section = False
                in_hardware_section = False
            elif line.strip().startswith("# ") and "Configuration" in line.strip():
                in_model_section = False
                in_hardware_section = False
                in_cluster_section = False
            
            # Handle specific settings based on CUDA availability
            if cuda_available:
                # If CUDA is available, set to use GPU
                if in_hardware_section and "accelerator:" in line:
                    f.write('  accelerator: "gpu"\n')
                    continue
                elif in_model_section and "device:" in line:
                    f.write('  device: "cuda"\n')
                    continue
                elif in_cluster_section and "enabled:" in line:
                    f.write('  enabled: true\n')
                    continue
                elif in_hardware_section and "devices:" in line:
                    f.write('  devices: 1\n')
                    continue
            else:
                # If CUDA is not available, ensure CPU is used
                if in_hardware_section and "accelerator:" in line:
                    f.write('  accelerator: "cpu"\n')
                    continue
                elif in_model_section and "device:" in line:
                    f.write('  device: "cpu"\n')
                    continue
            
            # Write the original line for other settings
            f.write(line)
    
    logger.info(f"Created fixed config file at {fixed_path}")
    logger.info(f"Please use this config file for your next training run")
    logger.info(f"Example: python src/main.py --config {fixed_path} --mode train --cluster")

if __name__ == "__main__":
    run_training() 