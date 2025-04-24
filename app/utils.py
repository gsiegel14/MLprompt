"""
Utility functions for the ML Prompt Optimization Platform.
"""

import logging

logger = logging.getLogger(__name__)

def substitute_variables(template, variables):
    """
    Replace variables in a template with their values.
    
    Args:
        template (str): The prompt template containing variables like $VAR_NAME
        variables (dict): Dictionary of variable names to values
        
    Returns:
        str: The template with variables replaced by their values
    """
    if not template:
        logger.warning("Empty template provided to substitute_variables")
        return ""
        
    if not variables:
        logger.warning("No variables provided for template substitution")
        return template
        
    result = template
    for var_name, var_value in variables.items():
        if var_value is None:
            var_value = ""
        result = result.replace(f"${var_name}", str(var_value))
    
    return result