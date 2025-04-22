import os
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gemini_api():
    """Test the connection to the Google Gemini API."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        return False
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Test prompt
        model = genai.GenerativeModel('gemini-2.5-flash')  # Updated to Gemini 2.5 Flash
        response = model.generate_content("Hello, please respond with 'API is working!'")
        
        logger.info(f"API Test Response: {response.text}")
        return True
    except Exception as e:
        logger.error(f"API Test Error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing Google Gemini API integration...")
    
    success = test_gemini_api()
    
    if success:
        logger.info("✅ API test successful - Google Gemini API is working!")
    else:
        logger.error("❌ API test failed - Check your API key and internet connection")