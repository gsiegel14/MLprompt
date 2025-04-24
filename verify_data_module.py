
#!/usr/bin/env python
"""
Verification script for DataModule

This script will test if the DataModule class is working correctly after fixes.
"""

import os
import sys
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.StreamHandler(sys.stdout)
                   ])
logger = logging.getLogger("verify_data_module")

def verify_data_module():
    """Verify that DataModule loads and functions correctly."""
    logger.info("=== VERIFYING DATA MODULE ===")
    
    # Import the DataModule
    try:
        from app.data_module import DataModule
        logger.info("✅ Successfully imported DataModule")
    except Exception as e:
        logger.error(f"❌ Failed to import DataModule: {e}")
        return False
    
    # Create an instance
    try:
        data_module = DataModule()
        logger.info("✅ Successfully created DataModule instance")
    except Exception as e:
        logger.error(f"❌ Failed to create DataModule instance: {e}")
        return False
    
    # Test train and validation examples
    try:
        train_examples = data_module.get_train_examples()
        validation_examples = data_module.get_validation_examples()
        logger.info(f"✅ Loaded {len(train_examples)} train examples and {len(validation_examples)} validation examples")
    except Exception as e:
        logger.error(f"❌ Failed to get examples: {e}")
        return False
    
    # Test batch retrieval
    try:
        batch = data_module.get_batch(batch_size=2, validation=False)
        logger.info(f"✅ Successfully retrieved batch of {len(batch)} examples")
    except Exception as e:
        logger.error(f"❌ Failed to get batch: {e}")
        return False
    
    # Test example splitting with sample data
    try:
        sample_data = [
            {"user_input": "Example 1", "ground_truth_output": "Output 1"},
            {"user_input": "Example 2", "ground_truth_output": "Output 2"},
            {"user_input": "Example 3", "ground_truth_output": "Output 3"},
            {"user_input": "Example 4", "ground_truth_output": "Output 4"}
        ]
        train, validation = data_module.split_examples(sample_data, train_ratio=0.75)
        logger.info(f"✅ Successfully split {len(sample_data)} examples into {len(train)} train and {len(validation)} validation")
    except Exception as e:
        logger.error(f"❌ Failed to split examples: {e}")
        return False
    
    logger.info("All DataModule tests passed!")
    return True

if __name__ == "__main__":
    if verify_data_module():
        print("\n✅ DataModule verification completed successfully!")
        print("The DataModule is now functioning correctly.")
    else:
        print("\n❌ DataModule verification failed.")
        print("Please check the logs for more information.")
