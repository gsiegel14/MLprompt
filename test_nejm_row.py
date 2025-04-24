
#!/usr/bin/env python3
"""
Test script to load the NEJM CSV file and use row 2 specifically
"""

import os
import json
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_nejm_csv_row2():
    """Load row 2 from the NEJM CSV file and display it"""
    
    # Check common locations for the NEJM CSV file
    possible_paths = [
        'data/nejm/cases.csv',
        'attached_assets/NEJM 160 Validation Database - NEJM 181.csv',
        'data/nejm/NEJM 160 Validation Database - NEJM 181.csv'
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            logger.info(f"Found NEJM CSV file at: {csv_path}")
            break
    
    if not csv_path:
        logger.error("NEJM CSV file not found in any of the expected locations")
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows and columns: {', '.join(df.columns)}")
        
        # Get row 2 (index 1 in zero-based indexing)
        if len(df) >= 2:
            row2 = df.iloc[1]
            logger.info("=== ROW 2 DATA ===")
            for col in df.columns:
                logger.info(f"{col}: {row2[col]}")
            
            # Create an example for testing with the 5-API workflow
            test_example = {
                "user_input": row2.get('Case', ''),
                "ground_truth_output": row2.get('Final Diagnosis', '')
            }
            
            # Save the example to a file for testing
            with open('data/test_validation/row2_example.json', 'w') as f:
                json.dump([test_example], f, indent=2)
            
            logger.info(f"Saved row 2 as test example to: data/test_validation/row2_example.json")
            return test_example
            
        else:
            logger.error(f"CSV file has only {len(df)} rows, cannot access row 2")
            return None
            
    except Exception as e:
        logger.error(f"Error loading NEJM CSV: {e}")
        return None

if __name__ == "__main__":
    # Create test_validation directory if it doesn't exist
    os.makedirs('data/test_validation', exist_ok=True)
    
    # Load row 2 and create test example
    example = load_nejm_csv_row2()
    
    if example:
        logger.info("\nTest example created successfully.")
        logger.info(f"User input: {example['user_input'][:100]}...")
        logger.info(f"Ground truth: {example['ground_truth_output']}")
        
        # Now run the 5-API workflow with this example
        logger.info("\nPrepared test example for 5-API workflow.")
        logger.info("You can now run: python test_five_api_workflow.py with this example")
    else:
        logger.error("Failed to create test example from row 2.")
