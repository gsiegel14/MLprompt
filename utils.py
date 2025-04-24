import re
import logging

def substitute_variables(text, variables_dict):
    """
    Replace variables in text with their values from variables_dict.
    Variables in text should be in format $VARIABLE_NAME.
    
    Args:
        text (str): The text containing variables to be substituted
        variables_dict (dict): Dictionary of variable names to their values
        
    Returns:
        str: Text with variables substituted
    """
    result = text
    for var_name, var_value in variables_dict.items():
        pattern = r'\$' + re.escape(var_name)
        result = re.sub(pattern, var_value, result)
    
    logging.debug(f"Substituted variables in text: {result}")
    return result
