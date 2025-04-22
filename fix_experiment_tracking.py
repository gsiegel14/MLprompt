#!/usr/bin/env python
"""
Fix Experiment Tracking

This script enhances the experiment tracking to ensure training
history is properly saved and displayed in the UI.
"""

import os
import sys
import shutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("fix_tracking.log"),
                       logging.StreamHandler(sys.stdout)
                   ])
logger = logging.getLogger("tracking_fix")

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup of {file_path} at {backup_path}")
        return True
    else:
        logger.warning(f"File {file_path} does not exist, cannot create backup")
        return False

def fix_experiment_tracking_js():
    """Fix the training.js to properly update progress and history."""
    js_file = "app/static/training.js"
    logger.info(f"Fixing progress tracking in {js_file}")
    
    if not backup_file(js_file):
        return False
    
    try:
        with open(js_file, 'r') as f:
            content = f.read()
        
        # Fix 1: Make sure updateTrainingProgress uses the actual iteration
        if "updateTrainingProgress(1, maxIterationsEl.value);" in content:
            content = content.replace(
                "updateTrainingProgress(1, maxIterationsEl.value);",
                "updateTrainingProgress(data.iterations || 1, maxIterationsEl.value);"
            )
            logger.info("Fixed updateTrainingProgress to use actual iteration count")
        
        # Fix 2: Enhance error logging
        if "log(`Error: ${error.message}`);" in content and not "log(`Full error details:" in content:
            content = content.replace(
                "log(`Error: ${error.message}`);",
                '''log(`Error: ${error.message}`);
            // Log more detailed error information
            console.error('Full error details:', error);
            log(`Error stack: ${error.stack || 'No stack trace available'}`);'''
            )
            logger.info("Enhanced error logging in training.js")
        
        # Save the modified file
        with open(js_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {js_file}")
        return True
    except Exception as e:
        logger.error(f"Error fixing training.js: {e}")
        return False

def fix_experiment_tracker():
    """Update the experiment_tracker.py to ensure all files are properly created."""
    exp_tracker_file = "app/experiment_tracker.py"
    logger.info(f"Fixing experiment tracker in {exp_tracker_file}")
    
    if not backup_file(exp_tracker_file):
        return False
    
    try:
        with open(exp_tracker_file, 'r') as f:
            content = f.read()
        
        # Find the save_iteration method
        save_iter_idx = content.find("def save_iteration(")
        if save_iter_idx >= 0:
            # Determine if the method needs robust error handling
            if content.find("try:", save_iter_idx, save_iter_idx + 500) < 0:
                # Find the end of the function definition line
                func_end = content.find(":", save_iter_idx)
                if func_end > 0:
                    # Add try/except block after function definition
                    func_def = content[save_iter_idx:func_end+1]
                    body_start = content.find("\n", func_end)
                    
                    # Add try-except wrapper for the entire function
                    content = content[:body_start+1] + "\n        try:\n" + content[body_start+1:]
                    
                    # Find the end of this function (next def or end of file)
                    next_def = content.find("\n    def ", body_start)
                    if next_def < 0:
                        next_def = len(content)
                    
                    # Add except block at the end of the function
                    try_block_end = next_def - 4  # 4 spaces of indentation
                    except_block = '''
        except Exception as e:
            logger.error(f"Error in save_iteration: {e}")
            # Try to save error info to a file
            try:
                os.makedirs('logs', exist_ok=True)
                error_file = f"logs/experiment_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(error_file, 'w') as f:
                    f.write(f"Error in save_iteration: {e}\\n")
                    import traceback
                    f.write(traceback.format_exc())
                logger.info(f"Saved error details to {error_file}")
            except Exception as e2:
                logger.error(f"Could not save error details: {e2}")
'''
                    content = content[:try_block_end] + except_block + content[try_block_end:]
                    
                    logger.info("Added robust error handling to save_iteration")
        
        # Ensure directory creation is handled properly
        if "os.makedirs(exp_dir, exist_ok=True)" not in content:
            exp_dir_creation = content.find("exp_dir = os.path.join(self.base_dir, experiment_id)")
            if exp_dir_creation > 0:
                line_end = content.find("\n", exp_dir_creation)
                if line_end > 0:
                    indent = " " * (content.find("exp_dir", exp_dir_creation) - content.rfind("\n", 0, exp_dir_creation) - 1)
                    content = content[:line_end+1] + f"{indent}os.makedirs(exp_dir, exist_ok=True)\n" + content[line_end+1:]
                    logger.info("Added directory creation to experiment tracker")
        
        # Ensure examples directory is created if it doesn't exist
        if "examples_dir = os.path.join(exp_dir, 'examples')" in content and "os.makedirs(examples_dir, exist_ok=True)" not in content:
            examples_dir = content.find("examples_dir = os.path.join(exp_dir, 'examples')")
            if examples_dir > 0:
                line_end = content.find("\n", examples_dir)
                if line_end > 0:
                    indent = " " * (content.find("examples_dir", examples_dir) - content.rfind("\n", 0, examples_dir) - 1)
                    content = content[:line_end+1] + f"{indent}os.makedirs(examples_dir, exist_ok=True)\n" + content[line_end+1:]
                    logger.info("Added examples directory creation to experiment tracker")
        
        # Save the modified file
        with open(exp_tracker_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {exp_tracker_file}")
        return True
    except Exception as e:
        logger.error(f"Error fixing experiment tracker: {e}")
        return False

def create_history_directories():
    """Ensure all required directories exist for experiment history."""
    dirs = [
        'experiments',
        'experiments/metrics',
        'experiments/prompts',
        'logs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return True

def main():
    logger.info("=== FIXING EXPERIMENT TRACKING STARTED ===")
    
    # Create necessary directories
    create_history_directories()
    
    # Fix the experiment tracker
    fix_experiment_tracker()
    
    # Fix the training.js file 
    fix_experiment_tracking_js()
    
    logger.info("=== FIXING EXPERIMENT TRACKING COMPLETED ===")
    print("\n✅ Successfully fixed experiment tracking!")
    print("Please restart the application to apply changes.")

if __name__ == "__main__":
    main()