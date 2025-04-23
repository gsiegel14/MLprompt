import os
import logging
import time
import google.generativeai as genai
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configure the Gemini API with the API key
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:

import os
import logging
import time
import hashlib
import pickle
from pathlib import Path
import google.generativeai as genai
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Create cache directory
CACHE_DIR = Path("cache/llm_responses")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ENABLED = os.environ.get("ENABLE_LLM_CACHE", "1") == "1"

def _get_cache_key(system_prompt: str, user_input: str, output_prompt: str, config: Dict) -> str:
    """Generate a deterministic cache key for LLM requests."""
    # Combine all inputs that affect the output
    combined = f"{system_prompt}|||{user_input}|||{output_prompt}|||{str(config)}"
    # Create a hash to use as the cache key
    cache_key = hashlib.md5(combined.encode()).hexdigest()
    return cache_key

def _save_to_cache(cache_key: str, response: str) -> None:
    """Save a response to the cache."""
    if not CACHE_ENABLED:
        return
    
    try:
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)
        logger.debug(f"Saved response to cache: {cache_key}")
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")

def _load_from_cache(cache_key: str) -> Optional[str]:
    """Load a response from the cache if available."""
    if not CACHE_ENABLED:
        return None
    
    try:
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                response = pickle.load(f)
            logger.info(f"Loaded response from cache: {cache_key}")
            return response
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
    
    return None

    genai.configure(api_key=api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

def get_llm_response(system_prompt: str, user_input: str, output_prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get a response from the Google Gemini API with retry logic and caching.
    
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
    model_name = config.get('model_name', 'gemini-2.5-flash')  # Updated to Gemini 2.5 Flash
    temperature = config.get('temperature', 0.0)
    top_p = config.get('top_p', 0.95)
    top_k = config.get('top_k', 40)
    max_output_tokens = config.get('max_output_tokens', 1024)
    
    # Check if response is in cache
    skip_cache = config.get('skip_cache', False)
    if not skip_cache:
        cache_key = _get_cache_key(system_prompt, user_input, output_prompt, 
                                 {"model_name": model_name, "temperature": temperature, 
                                  "top_p": top_p, "top_k": top_k, "max_output_tokens": max_output_tokens})
        
        cached_response = _load_from_cache(cache_key)
        if cached_response:
            logger.info(f"Using cached response (key: {cache_key[:8]}...)")
            return cached_response
    
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
                response_text = response.text
                logger.info(f"Successfully received response from Gemini API (length: {len(response_text)} chars)")
                
                # Cache the response if it's valid and caching is enabled
                if not skip_cache:
                    _save_to_cache(cache_key, response_text)
                
                return response_text
            else:
                # Try to extract response from other formats
                response_str = str(response)
                if response_str and len(response_str) > 0:
                    logger.info(f"Received non-standard response format from Gemini API")
                    
                    # Cache this response format too
                    if not skip_cache:
                        _save_to_cache(cache_key, response_str)
                    
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

"""
LLM Client with memory-efficient response caching.

This module provides a client for interacting with Language Models,
with built-in caching to reduce API calls and costs.
"""

import os
import json
import hashlib
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configure cache directory
CACHE_DIR = os.path.join('cache', 'llm_responses')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(system_prompt: str, user_input: str, output_prompt: str) -> str:
    """Generate a deterministic cache key for the prompt combination."""
    # Create a combined string of all inputs
    combined = f"{system_prompt}|||{user_input}|||{output_prompt}"
    # Create a hash of the combined string
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def get_from_cache(cache_key: str) -> Optional[str]:
    """Retrieve a response from cache if it exists."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                logger.debug(f"Cache hit for key {cache_key}")
                return cached_data.get('response')
        except Exception as e:
            logger.warning(f"Failed to read from cache: {e}")
    return None

def save_to_cache(cache_key: str, response: str) -> None:
    """Save a response to the cache."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'response': response,
                'timestamp': time.time()
            }, f)
        logger.debug(f"Saved response to cache with key {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to write to cache: {e}")

def get_llm_response(system_prompt: str, user_input: str, output_prompt: str, config: Dict[str, Any]) -> str:
    """
    Get a response from the LLM, with caching to reduce API calls.
    
    Args:
        system_prompt: System prompt for the LLM
        user_input: User input to process
        output_prompt: Output formatting instructions
        config: Configuration for the LLM call
        
    Returns:
        Generated response from the LLM
    """
    # Check if caching is enabled in config
    enable_caching = config.get('enable_caching', True)
    
    if enable_caching:
        # Try to get from cache first
        cache_key = get_cache_key(system_prompt, user_input, output_prompt)
        cached_response = get_from_cache(cache_key)
        
        if cached_response:
            logger.info("Using cached LLM response")
            return cached_response
    
    # Not in cache or caching disabled, make API call
    try:
        # Placeholder for actual API call implementation
        # This should be replaced with your actual API call logic
        import google.generativeai as genai
        import os
        
        # Get API key from environment or config
        api_key = os.environ.get('GOOGLE_API_KEY', config.get('api_key', ''))
        
        if not api_key:
            raise ValueError("No API key provided for Gemini")
        
        # Configure the API client
        genai.configure(api_key=api_key)
        
        # Prepare model parameters
        model_name = config.get('model_name', 'gemini-1.5-pro')
        temperature = config.get('temperature', 0.0)
        max_output_tokens = config.get('max_output_tokens', 1024)
        top_p = config.get('top_p', 0.95)
        top_k = config.get('top_k', 40)
        
        # Set up the model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_output_tokens,
                'top_p': top_p,
                'top_k': top_k
            }
        )
        
        # Prepare the prompt
        prompt = f"{system_prompt}\n\n{user_input}\n\n{output_prompt}"
        
        # Make the API call with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = model.generate_content(prompt)
                elapsed_time = time.time() - start_time
                
                logger.debug(f"LLM API call completed in {elapsed_time:.2f} seconds")
                
                if response.text:
                    # Cache the successful response if caching is enabled
                    if enable_caching:
                        save_to_cache(cache_key, response.text)
                    return response.text
                else:
                    logger.warning("Empty response from LLM API")
                    return "Error: Empty response from model"
                    
            except Exception as e:
                logger.warning(f"LLM API call attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
        
        return "Error: Failed to get response after multiple attempts"
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        raise

    raise RuntimeError(f"Unexpected error in retry loop: {last_error}")