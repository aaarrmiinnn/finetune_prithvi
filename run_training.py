#!/usr/bin/env python
"""Script to run training with NaN-resistant modifications."""
import os
import sys
import subprocess
import logging
import time

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

def run_training():
    """Run the training script with NaN-resistant modifications."""
    logger.info("Starting training with NaN-resistant modifications")
    
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Record the start time
    start_time = time.time()
    
    # Run the training script
    try:
        cmd = ["python", "src/scripts/train.py", "--config", "src/config/config.yaml"]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the process and stream output to the console
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            line = line.strip()
            logger.info(line)
            
            # Check for NaN-related issues
            if "NaN" in line or "Inf" in line:
                logger.warning(f"NaN/Inf detected: {line}")
        
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

if __name__ == "__main__":
    run_training() 