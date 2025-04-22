#!/usr/bin/env python
"""
Fix Data Loading

This script fixes issues with loading training examples properly.
It will ensure the DataModule properly loads examples from files.
"""

import os
import sys
import json
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("fix_data_loading.log"),
                       logging.StreamHandler(sys.stdout)
                   ])
logger = logging.getLogger("data_fix")

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

def fix_data_module_loading():
    """Fix the data_module.py file to ensure examples are loaded properly."""
    data_module_file = "app/data_module.py"
    logger.info(f"Fixing data loading in {data_module_file}")
    
    if not backup_file(data_module_file):
        return False
    
    try:
        with open(data_module_file, 'r') as f:
            content = f.read()
        
        # Fix 1: Add a method to load examples from files if it doesn't exist
        if "_load_examples_from_file" not in content:
            # Find where to insert the new method
            init_end = content.find("def __init__")
            if init_end > 0:
                init_end = content.find("\n\n", init_end)
                if init_end > 0:
                    load_method = '''
    def _load_examples_from_file(self, file_path):
        """Load examples from a JSON file."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    examples = json.load(f)
                    if examples:
                        logger.debug(f"Loaded {len(examples)} examples from {file_path}")
                        return examples
                    else:
                        logger.warning(f"File {file_path} exists but contains no examples")
            except Exception as e:
                logger.error(f"Error loading examples from {file_path}: {e}")
        else:
            logger.warning(f"Examples file {file_path} does not exist")
        return []
'''
                    # Insert the new method after __init__
                    content = content[:init_end+2] + load_method + content[init_end+2:]
                    logger.info("Added method to load examples from file")
        
        # Fix 2: Modify __init__ to load examples from files
        init_start = content.find("def __init__")
        if init_start > 0:
            init_end = content.find("def ", init_start + 10)
            if init_end > 0:
                # Check if __init__ already loads examples
                if "self._load_examples_from_file" not in content[init_start:init_end]:
                    # Find the end of the __init__ method content
                    last_line = content.rfind("\n        ", init_start, init_end)
                    if last_line > 0:
                        load_code = '''
        # Load train and validation examples
        train_file = os.path.join(self.train_dir, 'current_train.json')
        validation_file = os.path.join(self.validation_dir, 'current_validation.json')
        
        # Try to load from current files first
        self.train_examples = self._load_examples_from_file(train_file)
        self.validation_examples = self._load_examples_from_file(validation_file)
        
        # If current files are empty, try examples.json
        if not self.train_examples:
            alt_train_file = os.path.join(self.train_dir, 'examples.json')
            self.train_examples = self._load_examples_from_file(alt_train_file)
            if self.train_examples:
                # Copy to current_train.json
                self._save_examples(self.train_examples, train_file)
                logger.info(f"Updated {train_file} with {len(self.train_examples)} examples")
        
        if not self.validation_examples:
            alt_validation_file = os.path.join(self.validation_dir, 'examples.json')
            self.validation_examples = self._load_examples_from_file(alt_validation_file)
            if self.validation_examples:
                # Copy to current_validation.json
                self._save_examples(self.validation_examples, validation_file)
                logger.info(f"Updated {validation_file} with {len(self.validation_examples)} examples")
        
        logger.info(f"Initialized DataModule with {len(self.train_examples)} train examples and {len(self.validation_examples)} validation examples")'''
                        content = content[:last_line+1] + load_code + content[last_line+1:]
                        logger.info("Modified __init__ method to load examples from files")
        
        # Fix 3: Enhance get_batch to reload examples if needed
        get_batch = content.find("def get_batch")
        if get_batch > 0:
            check_examples = content.find("if not examples:", get_batch)
            if check_examples > 0:
                reload_code = '''
        if not examples:
            logger.warning(f"No {'validation' if validation else 'training'} examples available when calling get_batch")
            # Try reloading examples from file
            if validation:
                examples_file = os.path.join(self.validation_dir, 'current_validation.json')
                self.validation_examples = self._load_examples_from_file(examples_file)
                examples = self.validation_examples
            else:
                examples_file = os.path.join(self.train_dir, 'current_train.json')
                self.train_examples = self._load_examples_from_file(examples_file)
                examples = self.train_examples
            
            logger.info(f"After reload attempt, have {len(examples)} {'validation' if validation else 'training'} examples")
            
            if not examples:'''
                
                content = content.replace("        if not examples:", reload_code)
                logger.info("Enhanced get_batch with reload capability")
                
        # Save the updated file
        with open(data_module_file, 'w') as f:
            f.write(content)
            
        logger.info(f"Successfully updated {data_module_file}")
        return True
    except Exception as e:
        logger.error(f"Error fixing data module loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_data_files():
    """Verify and fix data files if needed."""
    train_dir = "data/train"
    validation_dir = "data/validation"
    
    # Ensure directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    
    # Check train examples
    train_file = os.path.join(train_dir, "examples.json")
    validation_file = os.path.join(validation_dir, "examples.json")
    
    current_train = os.path.join(train_dir, "current_train.json")
    current_validation = os.path.join(validation_dir, "current_validation.json")
    
    files_to_check = [
        (train_file, "Train"),
        (validation_file, "Validation"),
        (current_train, "Current train"),
        (current_validation, "Current validation")
    ]
    
    for file_path, desc in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    examples = json.load(f)
                logger.info(f"{desc} file has {len(examples)} examples")
                
                # If current_* file is empty but examples.json has data, copy it
                if file_path == train_file and len(examples) > 0 and (not os.path.exists(current_train) or os.path.getsize(current_train) == 0):
                    with open(current_train, 'w') as f:
                        json.dump(examples, f, indent=2)
                    logger.info(f"Copied {len(examples)} examples to {current_train}")
                
                if file_path == validation_file and len(examples) > 0 and (not os.path.exists(current_validation) or os.path.getsize(current_validation) == 0):
                    with open(current_validation, 'w') as f:
                        json.dump(examples, f, indent=2)
                    logger.info(f"Copied {len(examples)} examples to {current_validation}")
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in {file_path}")
            except Exception as e:
                logger.error(f"Error checking {file_path}: {e}")
        else:
            logger.warning(f"{desc} file not found: {file_path}")
    
    return True

def main():
    logger.info("=== FIXING DATA LOADING STARTED ===")
    
    # Verify data files
    verify_data_files()
    
    # Fix data module
    fix_data_module_loading()
    
    logger.info("=== FIXING DATA LOADING COMPLETED ===")
    
    print("\n✅ Data loading fixes completed! Please restart the application.")
    print("The system should now properly load your examples during training.\n")

if __name__ == "__main__":
    main()