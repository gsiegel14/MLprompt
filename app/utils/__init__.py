"""
Utility functions for the prompt optimization platform.
"""

# Import from the new prompt_variables module
from .prompt_variables import (
    substitute_variables,
    get_variable_names,
    create_variables_dict,
    format_eval_data
)

# Import from the main utils module
import csv
import io
import logging
import pandas as pd
from typing import List, Dict, Any

# Re-export functions from the original utils.py
def is_allowed_file(filename):
    """Check if the file has an allowed extension."""
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_text_examples(text):
    """
    Parse example data from a text input.
    Expected format is CSV-like with each line having:
    user_input,ground_truth_output
    
    Args:
        text (str): The text containing examples
        
    Returns:
        list: List of dictionaries with user_input and ground_truth_output
    """
    examples = []
    
    if not text or not text.strip():
        return examples
    
    lines = text.strip().split('\n')
    
    for line in lines:
        if ',' not in line:
            # Skip lines without a comma separator
            continue
        
        # Split on the first comma only, in case ground_truth contains commas
        parts = line.split(',', 1)
        if len(parts) < 2:
            continue
            
        user_input = parts[0].strip()
        ground_truth_output = parts[1].strip()
        
        if user_input and ground_truth_output:
            examples.append({
                'user_input': user_input,
                'ground_truth_output': ground_truth_output
            })
    
    return examples

def parse_csv_file(file):
    """
    Parse example data from a CSV file.
    The CSV should have headers 'user_input' and 'ground_truth_output'.
    
    Args:
        file: The uploaded CSV file object
        
    Returns:
        list: List of dictionaries with user_input and ground_truth_output
    """
    examples = []
    logger = logging.getLogger(__name__)
    
    try:
        # Save the file position
        position = file.tell()
        
        # Reset file position to beginning
        file.seek(0)
        
        # Read the file content
        file_content = file.read()
        
        # Reset file position
        file.seek(position)
        
        # Try to decode as UTF-8
        try:
            content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            content = file_content.decode('latin-1')
        
        # Create a string buffer and use pandas to read CSV
        buffer = io.StringIO(content)
        df = pd.read_csv(buffer)
        
        # Check if required columns exist
        required_columns = ['user_input', 'ground_truth_output']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must have these columns: {', '.join(required_columns)}")
        
        # Convert DataFrame to list of dictionaries
        for _, row in df.iterrows():
            user_input = row['user_input']
            ground_truth_output = row['ground_truth_output']
            
            # Skip entries with missing values
            if pd.isna(user_input) or pd.isna(ground_truth_output):
                continue
                
            # Convert to string in case they are not
            user_input = str(user_input).strip()
            ground_truth_output = str(ground_truth_output).strip()
            
            if user_input and ground_truth_output:
                examples.append({
                    'user_input': user_input,
                    'ground_truth_output': ground_truth_output
                })
    
    except Exception as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise ValueError(f"Error parsing CSV file: {str(e)}")
    
    return examples

def split_train_validation(examples, train_ratio=0.8):
    """
    Split examples into training and validation sets.
    
    Args:
        examples (list): List of example dictionaries
        train_ratio (float): Ratio of examples to use for training
        
    Returns:
        tuple: (training_examples, validation_examples)
    """
    if not examples:
        return [], []
        
    # Calculate the split index
    split_idx = int(len(examples) * train_ratio)
    
    # Split the examples
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    return train_examples, val_examples

__all__ = [
    # Variable substitution functions
    'substitute_variables',
    'get_variable_names',
    'create_variables_dict',
    'format_eval_data',
    
    # Original utility functions
    'is_allowed_file',
    'parse_text_examples',
    'parse_csv_file',
    'split_train_validation'
]