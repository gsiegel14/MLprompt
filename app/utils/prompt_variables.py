"""
Prompt Variable Substitution Utility

This module provides functions to substitute variables in prompts, such as:
- $USER_INPUT - User input from CSV or examples
- $EVAL_DATA_BASE - Evaluation data from the Hugging Face API call #2
- $EVAL_DATA_OPTIMIZED - Evaluation data from the Hugging Face API call #4
- $DATASET_ANSWERS_BASE - Answers to the dataset for prompt optimizer

Usage:
    processed_prompt = substitute_variables(prompt, variables_dict)
"""

import logging
import re

logger = logging.getLogger(__name__)

# Define regex pattern for finding variables
VARIABLE_PATTERN = r'\$([A-Z_]+)'

def substitute_variables(text, variables):
    """
    Substitute variables in a text with their values.
    
    Args:
        text (str): The text containing variables (e.g., $USER_INPUT)
        variables (dict): Dictionary mapping variable names to values
        
    Returns:
        str: Text with variables replaced by their values
    """
    if not text or not isinstance(text, str):
        return text
    
    def replace_var(match):
        var_name = match.group(1)
        if var_name in variables:
            logger.debug(f"Replacing ${var_name} with value of length {len(str(variables[var_name]))}")
            return str(variables[var_name])
        else:
            logger.warning(f"Variable ${var_name} not found in provided variables")
            return match.group(0)  # Return the original match if variable not found
    
    # Replace all variables in the text
    processed_text = re.sub(VARIABLE_PATTERN, replace_var, text)
    
    return processed_text

def get_variable_names(text):
    """
    Extract all variable names from a text.
    
    Args:
        text (str): The text containing variables (e.g., $USER_INPUT)
        
    Returns:
        list: List of variable names found in the text
    """
    if not text or not isinstance(text, str):
        return []
    
    matches = re.findall(VARIABLE_PATTERN, text)
    return matches

def create_variables_dict(user_input=None, eval_data_base=None, eval_data_optimized=None, dataset_answers_base=None, **kwargs):
    """
    Create a dictionary of variables for substitution.
    
    Args:
        user_input (str, optional): User input to substitute for $USER_INPUT
        eval_data_base (dict, optional): Base evaluation data for $EVAL_DATA_BASE
        eval_data_optimized (dict, optional): Optimized evaluation data for $EVAL_DATA_OPTIMIZED
        dataset_answers_base (list or dict, optional): Dataset answers to be used for prompt optimization
        **kwargs: Additional variables to add to the dictionary
        
    Returns:
        dict: Dictionary mapping variable names to values
    """
    variables = {}
    
    if user_input is not None:
        variables['USER_INPUT'] = user_input
    
    if eval_data_base is not None:
        if isinstance(eval_data_base, dict):
            formatted_eval_data = format_eval_data(eval_data_base)
            variables['EVAL_DATA_BASE'] = formatted_eval_data
        else:
            variables['EVAL_DATA_BASE'] = str(eval_data_base)
    
    if eval_data_optimized is not None:
        if isinstance(eval_data_optimized, dict):
            formatted_eval_data = format_eval_data(eval_data_optimized)
            variables['EVAL_DATA_OPTIMIZED'] = formatted_eval_data
        else:
            variables['EVAL_DATA_OPTIMIZED'] = str(eval_data_optimized)
    
    if dataset_answers_base is not None:
        if isinstance(dataset_answers_base, (list, dict)):
            formatted_data = format_dataset_answers(dataset_answers_base)
            variables['DATASET_ANSWERS_BASE'] = formatted_data
        else:
            variables['DATASET_ANSWERS_BASE'] = str(dataset_answers_base)
    
    # Add any additional variables
    for key, value in kwargs.items():
        variables[key.upper()] = value
    
    return variables

def format_eval_data(eval_data):
    """
    Format evaluation data as a readable string.
    
    Args:
        eval_data (dict): Evaluation data from Hugging Face API
        
    Returns:
        str: Formatted evaluation data as a string
    """
    if not eval_data:
        return "No evaluation data available"
    
    formatted_str = "Evaluation Metrics:\n"
    
    for metric_name, metric_value in eval_data.items():
        # Handle nested dictionaries (e.g., metric breakdowns)
        if isinstance(metric_value, dict):
            formatted_str += f"- {metric_name}:\n"
            for sub_name, sub_value in metric_value.items():
                formatted_value = f"{sub_value:.2f}" if isinstance(sub_value, float) else sub_value
                formatted_str += f"  - {sub_name}: {formatted_value}\n"
        else:
            # Handle direct metric values
            formatted_value = f"{metric_value:.2f}" if isinstance(metric_value, float) else metric_value
            formatted_str += f"- {metric_name}: {formatted_value}\n"
    
    return formatted_str

def format_dataset_answers(dataset_answers):
    """
    Format dataset answers as a readable string.
    
    Args:
        dataset_answers (list or dict): Dataset answers to be used for prompt optimization
        
    Returns:
        str: Formatted dataset answers as a string
    """
    if not dataset_answers:
        return "No dataset answers available"
    
    # Handle list of examples
    if isinstance(dataset_answers, list):
        formatted_str = "Dataset Examples:\n\n"
        
        for i, example in enumerate(dataset_answers, 1):
            if isinstance(example, dict):
                formatted_str += f"Example {i}:\n"
                
                # Include user input if available
                if 'user_input' in example:
                    formatted_str += f"Input: {example['user_input']}\n"
                
                # Include ground truth if available
                if 'ground_truth_output' in example:
                    formatted_str += f"Expected Output: {example['ground_truth_output']}\n"
                
                # Include model output if available
                if 'model_output' in example:
                    formatted_str += f"Model Output: {example['model_output']}\n"
                
                # Include any metrics if available
                if 'metrics' in example and isinstance(example['metrics'], dict):
                    formatted_str += "Metrics:\n"
                    for metric_name, metric_value in example['metrics'].items():
                        formatted_value = f"{metric_value:.2f}" if isinstance(metric_value, float) else metric_value
                        formatted_str += f"  - {metric_name}: {formatted_value}\n"
                
                formatted_str += "\n"
            else:
                # If it's just a string or simple value
                formatted_str += f"Example {i}: {example}\n\n"
    
    # Handle dictionary
    elif isinstance(dataset_answers, dict):
        formatted_str = "Dataset Summary:\n\n"
        
        for key, value in dataset_answers.items():
            if isinstance(value, dict):
                formatted_str += f"{key}:\n"
                for sub_key, sub_value in value.items():
                    formatted_value = f"{sub_value:.2f}" if isinstance(sub_value, float) else sub_value
                    formatted_str += f"  - {sub_key}: {formatted_value}\n"
            elif isinstance(value, list):
                formatted_str += f"{key}: {len(value)} items\n"
                # Show a preview of the first few items
                for i, item in enumerate(value[:3], 1):
                    item_preview = str(item)
                    if len(item_preview) > 50:
                        item_preview = item_preview[:50] + "..."
                    formatted_str += f"  {i}. {item_preview}\n"
                if len(value) > 3:
                    formatted_str += f"  ... and {len(value) - 3} more items\n"
            else:
                formatted_str += f"{key}: {value}\n"
    
    return formatted_str