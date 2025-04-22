import os
import logging
import random

logger = logging.getLogger(__name__)

# Sample responses for testing without API
SAMPLE_RESPONSES = {
    "What is the capital of France?": "Paris",
    "How many planets are in our solar system?": "Eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune",
    "What's the square root of 64?": "8",
    "Who wrote Romeo and Juliet?": "William Shakespeare",
    "What is the formula for water?": "H2O"
}

def get_llm_response(system_prompt, user_input, output_prompt, config=None):
    """
    Get a response from the Vertex AI LLM.
    
    Args:
        system_prompt (str): The system prompt to guide the model
        user_input (str): The user input to process
        output_prompt (str): The output prompt format instructions
        config (dict): Configuration for Vertex AI
        
    Returns:
        str: The model's response
    """
    if not config:
        # Default configuration
        config = {
            'project_id': os.environ.get('GCP_PROJECT_ID', ''),
            'location': 'us-central1',
            'model_name': 'gemini-1.5-pro-preview-0409'
        }
    
    project_id = config.get('project_id')
    
    logger.info(f"Processing request with system prompt: {system_prompt[:50]}...")
    logger.info(f"User input: {user_input[:50]}...")
    logger.info(f"Output prompt: {output_prompt[:50]}...")
    
    # Check if we have GCP credentials
    if project_id and os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        logger.info("Detected GCP credentials, but using mock responses for demo")
    
    # IMPORTANT NOTE: 
    # In the actual implementation, this would call Vertex AI
    # The code below is a mock implementation for demonstration purposes only
    # To use the real API, you would need to set up GCP credentials and install the required packages

    try:
        # For demo purposes, use deterministic responses with some variation
        clean_input = user_input.strip().lower()
        
        # Check if we have a predefined response
        for key, response in SAMPLE_RESPONSES.items():
            if clean_input in key.lower():
                logger.info(f"Using predefined response for: {key}")
                return response
        
        # For other inputs, generate a simple response
        words = user_input.split()
        response_length = min(len(words) + 5, 20)
        
        # A more elaborate mock response based on the input
        if "what" in clean_input:
            return f"Based on your query about '{' '.join(words[:3])}...', the answer would depend on specific details."
        elif "how" in clean_input:
            return f"The process for '{' '.join(words[:3])}...' typically involves several steps that would be explained in detail."
        elif "why" in clean_input:
            return f"The reason for '{' '.join(words[:3])}...' is complex and would be thoroughly analyzed."
        else:
            return f"I've processed your input about '{' '.join(words[:3])}...' and would provide a detailed response."
            
    except Exception as e:
        logger.error(f"Error in mock LLM response: {e}")
        return f"Sorry, I couldn't process your request due to an error: {str(e)}"
