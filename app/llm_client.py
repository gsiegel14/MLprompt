import os
import logging
import time
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
    Get a response from the Google Gemini API with retry logic.
    
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
    model_name = config.get('model_name', 'gemini-1.0-pro')  # Using free tier model
    temperature = config.get('temperature', 0.0)
    top_p = config.get('top_p', 0.95)
    top_k = config.get('top_k', 40)
    max_output_tokens = config.get('max_output_tokens', 1024)
    
    # Retry configuration
    max_retries = 3
    retry_delay = 2  # seconds between retries
    backoff_factor = 2  # exponential backoff
    
    # Initialize variables
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            # Log retry attempts if not first try
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries} for Gemini API call")
            
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
            
            # Generate response with timeout handling
            logger.info(f"Calling Gemini API with model {model_name}")
            response = model.generate_content(combined_prompt)
            
            # Check if we have a valid response
            if hasattr(response, 'text') and response.text:
                logger.info(f"Successfully received response from Gemini API (length: {len(response.text)} chars)")
                return response.text
            else:
                # Try to extract response from other formats
                response_str = str(response)
                if response_str and len(response_str) > 0:
                    logger.info(f"Received non-standard response format from Gemini API")
                    return response_str
                else:
                    raise ValueError("Empty response received from Gemini API")
                
        except Exception as e:
            last_error = e
            logger.warning(f"Error on API attempt {attempt+1}/{max_retries}: {str(e)}")
            
            # Check for specific error types
            error_type = type(e).__name__
            error_str = str(e).lower()
            
            # Don't retry certain errors
            if "invalid api key" in error_str or "authentication" in error_str:
                logger.error(f"API key error detected: {e}")
                raise ValueError(f"Invalid API key or authentication error: {e}")
            
            # Additional handling for rate limiting or server errors
            if "rate limit" in error_str or "429" in error_str:
                logger.warning("Rate limit detected, extending backoff...")
                retry_delay *= 2  # Double the delay for rate limits
            
            # Increment attempt counter
            attempt += 1
            
            if attempt < max_retries:
                # Calculate backoff delay
                current_delay = retry_delay * (backoff_factor ** (attempt - 1))
                logger.info(f"Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
            else:
                logger.error(f"Failed after {max_retries} attempts. Last error: {last_error}")
                raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts: {last_error}")
    
    # This should never be reached due to the raise in the loop, but just in case
    raise RuntimeError(f"Unexpected error in retry loop: {last_error}")