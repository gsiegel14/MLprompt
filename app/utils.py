import csv
import io
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def is_allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

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
    
    try:
        # Try to parse as CSV
        for row in csv.reader(text.splitlines()):
            if len(row) >= 2:
                examples.append({
                    'user_input': row[0].strip(),
                    'ground_truth_output': row[1].strip()
                })
            elif len(row) == 1 and row[0].strip():
                # If there's only input, add it with empty ground truth
                examples.append({
                    'user_input': row[0].strip(),
                    'ground_truth_output': ''
                })
        
        if examples:
            return examples
        
        # If CSV parsing didn't work, try simple line splitting
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                examples.append({
                    'user_input': lines[i],
                    'ground_truth_output': lines[i+1]
                })
            else:
                examples.append({
                    'user_input': lines[i],
                    'ground_truth_output': ''
                })
        
        return examples
    except Exception as e:
        logger.error(f"Error parsing text examples: {e}")
        return []

def parse_csv_file(file):
    """
    Parse example data from a CSV file.
    The CSV should have headers 'user_input' and 'ground_truth_output'.
    
    Args:
        file: The uploaded CSV file object
        
    Returns:
        list: List of dictionaries with user_input and ground_truth_output
    """
    try:
        # Save the file contents to a buffer
        content = file.read().decode('utf-8')
        file_like_object = io.StringIO(content)
        
        # Try to read with pandas
        df = pd.read_csv(file_like_object)
        
        # Check column names
        required_columns = ['user_input', 'ground_truth_output']
        
        # If the dataframe doesn't have the exact column names, try to infer them
        if not all(col in df.columns for col in required_columns):
            # If there are exactly two columns, assume they are input and output
            if len(df.columns) == 2:
                df.columns = required_columns
            # If there's only one column, assume it's the input
            elif len(df.columns) == 1:
                df.columns = ['user_input']
                df['ground_truth_output'] = ''
            else:
                raise ValueError("CSV must have columns 'user_input' and 'ground_truth_output' or exactly two columns")
        
        # Convert dataframe to list of dictionaries
        examples = df.to_dict('records')
        
        # Ensure all entries have both required fields
        for example in examples:
            if 'user_input' not in example:
                example['user_input'] = ''
            if 'ground_truth_output' not in example:
                example['ground_truth_output'] = ''
                
        return examples
    except Exception as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise ValueError(f"Error parsing CSV file: {str(e)}")
