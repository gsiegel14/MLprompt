"""
Functions for handling prompt variables and substitution.
"""

import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def substitute_variables(template: str, variables: Dict[str, str]) -> str:
    """
    Replace variables in a template with their values.
    
    Args:
        template (str): The prompt template containing variables like $VAR_NAME
        variables (dict): Dictionary of variable names to values
        
    Returns:
        str: The template with variables replaced by their values
    """
    result = template
    for var_name, var_value in variables.items():
        if var_value is None:
            var_value = ''
        result = result.replace(f"${var_name}", str(var_value))
    return result

def get_variable_names(template: str) -> List[str]:
    """
    Extract variable names from a template.
    
    Args:
        template (str): The prompt template containing variables like $VAR_NAME
        
    Returns:
        list: List of variable names found in the template
    """
    pattern = r'\$([A-Z_]+)'
    matches = re.findall(pattern, template)
    return matches

def create_variables_dict(var_names: List[str], default_value: str = '') -> Dict[str, str]:
    """
    Create a variables dictionary with default values.
    
    Args:
        var_names (list): List of variable names
        default_value (str): Default value for all variables
        
    Returns:
        dict: Dictionary of variable names to default values
    """
    return {name: default_value for name in var_names}

def format_eval_data(eval_data: Any, format_type: str = 'text') -> str:
    """
    Format evaluation data for display.
    
    Args:
        eval_data: The evaluation data to format
        format_type (str): The format type ('text' or 'html')
        
    Returns:
        str: Formatted evaluation data
    """
    if not eval_data:
        return ''
    
    if isinstance(eval_data, dict):
        if format_type == 'html':
            # Format as HTML
            formatted_str = "<div class='eval-data'>"
            for key, value in eval_data.items():
                formatted_str += f"<div class='eval-item'><strong>{key}:</strong> {value}</div>"
            formatted_str += "</div>"
        else:
            # Format as plain text
            formatted_str = ""
            for key, value in eval_data.items():
                formatted_str += f"{key}: {value}\n"
    elif isinstance(eval_data, list):
        if format_type == 'html':
            # Format as HTML list
            formatted_str = "<div class='eval-data'><ul>"
            for item in eval_data:
                formatted_str += f"<li>{item}</li>"
            formatted_str += "</ul></div>"
        else:
            # Format as plain text list
            formatted_str = ""
            for item in eval_data:
                formatted_str += f"- {item}\n"
    else:
        # If it's already a string, just return it
        formatted_str = str(eval_data)
    
    return formatted_str