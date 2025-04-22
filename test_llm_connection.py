import os
import logging
import google.generativeai as genai
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def test_llm_response():
    """Test if we can get a response from the LLM for a simple medical case."""
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration with model: {config['gemini']['model_name']}")
    
    # Configure the API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return False
        
    genai.configure(api_key=api_key)
    
    # Define a simple test case
    test_case = """
    A 35-year-old man presents with a 3-day history of fever, headache, and a rash. 
    The rash started on his torso and spread to his extremities, including his palms and soles. 
    He reports hiking in a wooded area two weeks ago.
    """
    
    # Create a simple system prompt
    system_prompt = "You are an expert physician and diagnostician. Provide a differential diagnosis."
    
    # Combine the prompts
    prompt = f"{system_prompt}\n\nCase: {test_case}\n\nDifferential diagnosis:"
    
    try:
        # Set up the model with configuration
        model = genai.GenerativeModel(
            config['gemini']['model_name'],
            generation_config={
                "temperature": config['gemini']['temperature'],
                "top_p": config.get('gemini', {}).get('top_p', 0.95),
                "top_k": config.get('gemini', {}).get('top_k', 40),
                "max_output_tokens": config['gemini']['max_output_tokens'],
            }
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Log the response
        logger.info("LLM Response:")
        logger.info(response.text)
        
        return True
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing LLM medical response...")
    
    success = test_llm_response()
    
    if success:
        logger.info("✅ LLM medical response test successful!")
    else:
        logger.error("❌ LLM medical response test failed!")