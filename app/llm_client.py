import os
import logging
import google.generativeai as genai
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configure the Gemini API with the API key
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

def get_llm_response(system_prompt: str, user_input: str, output_prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get a response from the Google Gemini API.
    
    Args:
        system_prompt (str): The system prompt to guide the model
        user_input (str): The user input to process
        output_prompt (str): The output prompt format instructions
        config (dict): Configuration for Gemini API
        
    Returns:
        str: The model's response
    """
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please set this environment variable.")
    
    if config is None:
        config = {}
    
    # Default configuration
    model_name = config.get('model_name', 'gemini-1.5-pro')
    temperature = config.get('temperature', 0.0)
    top_p = config.get('top_p', 0.95)
    top_k = config.get('top_k', 40)
    max_output_tokens = config.get('max_output_tokens', 1024)
    
    try:
        # Get the model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }
        )
        
        # Construct the prompt
        combined_prompt = f"{system_prompt}\n\nUser Input: {user_input}\n\n{output_prompt}"
        
        # Generate response
        response = model.generate_content(combined_prompt)
        
        if hasattr(response, 'text'):
            return response.text
        else:
            # Handle different response formats
            return str(response)
            
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise e