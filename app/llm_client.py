import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Initialize the Gemini API with the key from environment variables
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables. API calls will fail.")

def get_llm_response(system_prompt, user_input, output_prompt, config=None):
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
    if not config:
        # Default configuration
        config = {
            'model_name': 'gemini-1.5-pro-preview-0409',
            'temperature': 0.0,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 1024
        }
    
    model_name = config.get('model_name', 'gemini-1.5-pro')
    temperature = config.get('temperature', 0.0)
    top_p = config.get('top_p', 0.95)
    top_k = config.get('top_k', 40)
    max_output_tokens = config.get('max_output_tokens', 1024)
    
    logger.info(f"Processing request with system prompt: {system_prompt[:50]}...")
    logger.info(f"User input: {user_input[:50]}...")
    logger.info(f"Output prompt: {output_prompt[:50]}...")
    
    try:
        # Format the output prompt with the user input
        formatted_output_prompt = output_prompt.replace('{user_input}', user_input)
        
        # Combine the system prompt and the formatted output prompt
        combined_prompt = f"{system_prompt}\n\n{formatted_output_prompt}"
        
        # Check if API key is available
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        # Initialize the model
        model = genai.GenerativeModel(model_name)
        
        # Generate response
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        
        response = model.generate_content(
            combined_prompt,
            generation_config=generation_config
        )
        
        # Extract and return the response text
        if response.text:
            logger.info(f"Generated response: {response.text[:100]}...")
            return response.text
        else:
            logger.warning("Empty response received from Gemini API")
            return "No response generated."
            
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return f"Sorry, I couldn't process your request due to an error: {str(e)}"
