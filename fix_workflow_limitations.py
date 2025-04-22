#!/usr/bin/env python
"""
Fix Workflow Batch Size Limitations

This script fixes the batch size limitation in workflow.py
that's causing the system to only process 2 examples instead of all 10.
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
                       logging.FileHandler("fix_workflow.log"),
                       logging.StreamHandler(sys.stdout)
                   ])
logger = logging.getLogger("workflow_fix")

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

def fix_workflow_batch_limits():
    """Fix the batch size limitations in the workflow.py file."""
    workflow_file = "app/workflow.py"
    logger.info(f"Fixing batch size limitations in {workflow_file}")
    
    if not backup_file(workflow_file):
        return False
    
    try:
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        # First fix: Remove the hard batch size limit of 20
        if "if batch_size == 0 or batch_size > 20:" in content:
            content = content.replace(
                "if batch_size == 0 or batch_size > 20:",
                "if batch_size == 0 or batch_size > 50:  # Increased limit from 20 to 50"
            )
            content = content.replace(
                "logger.info(f\"Limiting batch size to 20 examples (original: {batch_size})\")",
                "logger.info(f\"Limiting batch size to 50 examples (original: {batch_size})\")"
            )
            content = content.replace(
                "effective_batch_size = 20",
                "effective_batch_size = 50"
            )
            logger.info("Fixed hard batch size limit (increased from 20 to 50)")
        
        # Second fix: Increase the chunk size
        if "max_chunk_size = min(5, len(batch))" in content:
            content = content.replace(
                "max_chunk_size = min(5, len(batch))",
                "max_chunk_size = min(10, len(batch))  # Increased from 5 to 10"
            )
            logger.info("Increased processing chunk size from 5 to 10")
        
        # Save the modified file
        with open(workflow_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {workflow_file}")
        print(f"\n✅ Successfully fixed batch size limitations in {workflow_file}")
        print("Please restart the application to apply changes.")
        return True
    except Exception as e:
        logger.error(f"Error fixing workflow batch limits: {e}")
        print(f"\n❌ Error fixing workflow batch limits: {e}")
        return False

if __name__ == "__main__":
    fix_workflow_batch_limits()