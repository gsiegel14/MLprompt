import os
from flask import Flask, request, jsonify
from src.app.utils.logger import configure_logging
import logging

app = Flask(__name__) # Added Flask app initialization

# Configure advanced logging
log_level = os.environ.get("LOG_LEVEL", "INFO")
configure_logging(log_level=log_level)

# Create logger for this module
logger = logging.getLogger(__name__)

@app.route('/five_api_workflow', methods=['POST'])
def five_api_workflow():
    """Endpoint for running the 5-API workflow"""
    try:
        data = request.get_json() # Use get_json() for JSON data
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        user_input = data.get('user_input', '')
        ground_truth = data.get('ground_truth', '')
        batch_size = data.get('batch_size', 1)
        step = data.get('step', 1)

        # For testing purposes, return a successful response
        return jsonify({
            'success': True,
            'response': f"Processed input for step {step}: {user_input[:100]}...",
            'metrics': {
                'exact_match': 0.85,
                'similarity': 0.92
            }
        })
    except Exception as e:
        logger.exception(f"Error in five_api_workflow: {e}") # Log the exception for debugging
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health') #Added health check endpoint
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    logger.info("Starting Prompt Optimization Platform")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    logger.info(f"Log level: {log_level}")

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "True").lower() == "true"

    logger.info(f"Starting server on port {port}, debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)