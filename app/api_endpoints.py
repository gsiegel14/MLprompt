"""
API Endpoints for ML Prompt Optimization Platform

This module defines the REST API endpoints used by the platform:
1. Status endpoint
2. LLM API
3. Evaluation API
4. Optimization API
5. Hugging Face metrics API
6. Workflow API
7. Experiment history API
"""

import os
import json
import logging
import yaml
from flask import Flask, Blueprint, request, jsonify

# Create a Blueprint for API endpoints
api_blueprint = Blueprint('api', __name__)

logger = logging.getLogger(__name__)

# API Status Endpoint
@api_blueprint.route('/status', methods=['GET'])
def api_status():
    """Return the status of the API and its components."""
    try:
        return jsonify({
            'status': 'ok',
            'message': 'API is running correctly'
        })
    except Exception as e:
        logger.error(f"Error in API status endpoint: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# LLM API Endpoint
@api_blueprint.route('/llm/generate', methods=['POST'])
def llm_generate():
    """Generate a response using the LLM."""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        user_input = data.get('user_input', '')
        output_prompt = data.get('output_prompt', '')
        
        if not system_prompt or not user_input:
            return jsonify({'error': 'System prompt and user input are required'}), 400
        
        # Simulate API response for now
        model_response = f"Response to: {user_input}"
        
        return jsonify({
            'response': model_response,
            'model': 'gemini-1.5-flash'  # Default model
        })
    except Exception as e:
        logger.error(f"Error in LLM generate endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Variable Endpoint
@api_blueprint.route('/variables/list', methods=['GET'])
def list_variables():
    """List all available variables for prompt templates."""
    try:
        variables = {
            'BASE_PROMPTS': 'Collection of base prompts to use for optimization',
            'EVAL_DATA_BASE': 'Evaluation data for the base prompt performance',
            'EVAL_PROMPT': 'Prompt used for evaluation',
            'CONTEXT': 'Context information for evaluation',
            'USER_QUERY': 'User query for testing',
            'GROUND_TRUTH': 'Ground truth response for comparison',
            'OPTIMIZED_SYSTEMMESSAGE': 'Optimized system message after processing',
            'OPTIMIZED_OUTPUTPROMPT': 'Optimized output prompt after processing'
        }
        
        return jsonify({
            'variables': variables,
            'count': len(variables)
        })
    except Exception as e:
        logger.error(f"Error in list_variables endpoint: {e}")
        return jsonify({'error': str(e)}), 500