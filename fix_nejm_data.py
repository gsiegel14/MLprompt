#!/usr/bin/env python3
"""
Fix the NEJM case data by correctly processing from the CSV file.
This script takes NEJM case studies, ensures they are correctly processed,
and saves them to the correct training and validation files.
"""

import os
import json
import pandas as pd
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Process the CSV file
def fix_nejm_data(csv_path='attached_assets/NEJM 160 Validation Database - NEJM 181.csv', train_ratio=0.8):
    # Make sure directories exist
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/validation', exist_ok=True)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} cases from {csv_path}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return
    
    # Convert to the format expected by the application
    examples = []
    for _, row in df.iterrows():
        try:
            case_text = row['Case']
            diagnosis = row['Final Diagnosis']
            
            if pd.notna(case_text) and pd.notna(diagnosis):
                examples.append({
                    'user_input': case_text,
                    'ground_truth_output': diagnosis
                })
        except Exception as e:
            logger.error(f"Error processing row: {e}")
    
    logger.info(f"Processed {len(examples)} valid examples")
    
    # Set a fixed seed for deterministic splitting (ensures same split each time)
    random.seed(42)
    
    # Shuffle the examples
    random.shuffle(examples)
    
    # Use fixed counts to ensure we always get 127 training and 32 validation examples
    # This maintains consistency across runs
    train_count = 127
    validation_count = 32
    
    # Make sure we don't exceed the total examples available
    if len(examples) < (train_count + validation_count):
        # If we don't have enough examples, revert to ratio-based split
        split_index = int(len(examples) * train_ratio)
        train_examples = examples[:split_index]
        validation_examples = examples[split_index:]
        logger.warning(f"Not enough examples to create fixed split. Using ratio-based split instead.")
    else:
        # Use fixed counts
        train_examples = examples[:train_count]
        validation_examples = examples[train_count:train_count+validation_count]
    
    logger.info(f"Split into {len(train_examples)} training and {len(validation_examples)} validation examples")
    
    # Save to the application's data directory
    try:
        with open('data/train/examples.json', 'w') as f:
            json.dump(train_examples, f)
        with open('data/train/current_train.json', 'w') as f:
            json.dump(train_examples, f)
            
        with open('data/validation/examples.json', 'w') as f:
            json.dump(validation_examples, f)
        with open('data/validation/current_validation.json', 'w') as f:
            json.dump(validation_examples, f)
        
        logger.info("Successfully saved NEJM data files")
    except Exception as e:
        logger.error(f"Error saving data files: {e}")

if __name__ == "__main__":
    fix_nejm_data()
    logger.info("NEJM data processing complete.")