import os
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_available_models():
    """List all available models in the Google Generative AI API."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        return False
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # List available models
        models = genai.list_models()
        logger.info("Available models:")
        
        for model in models:
            logger.info(f"- {model.name} (Supported generation methods: {model.supported_generation_methods})")
        
        return True
    except Exception as e:
        logger.error(f"API Error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Checking available Google Generative AI models...")
    success = list_available_models()
    
    if not success:
        logger.error("❌ Failed to retrieve model list - Check your API key and internet connection")