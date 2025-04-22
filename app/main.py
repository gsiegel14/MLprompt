import os
import logging
import yaml
import pandas as pd
from datetime import datetime
from flask import render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
from app import app
from app.llm_client import get_llm_response
from app.evaluator import calculate_score
from app.utils import parse_text_examples, parse_csv_file, is_allowed_file

logger = logging.getLogger(__name__)

# Load configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded configuration: {config}")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    # Default configuration
    config = {
        'vertex_ai': {
            'project_id': os.environ.get('GCP_PROJECT_ID', ''),
            'location': 'us-central1',
            'model_name': 'gemini-1.5-pro-preview-0409'
        }
    }

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('prompts', exist_ok=True)

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_evaluation():
    """Process inputs, run the model, and return evaluation results."""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        examples_format = data.get('examples_format', 'text')
        examples_content = data.get('examples_content', '')
        
        if not system_prompt or not output_prompt:
            return jsonify({'error': 'System prompt and output prompt are required'}), 400
        
        if not examples_content:
            return jsonify({'error': 'Example data is required'}), 400
        
        # Parse examples based on format
        if examples_format == 'text':
            examples = parse_text_examples(examples_content)
        else:
            # This should not happen as CSV is uploaded separately
            return jsonify({'error': 'Invalid examples format'}), 400
        
        if not examples:
            return jsonify({'error': 'No valid examples found'}), 400
        
        # Run evaluation for each example
        results = []
        for example in examples:
            user_input = example.get('user_input', '')
            ground_truth = example.get('ground_truth_output', '')
            
            if not user_input:
                continue
            
            try:
                # Call Vertex AI
                model_response = get_llm_response(
                    system_prompt, 
                    user_input, 
                    output_prompt,
                    config['vertex_ai']
                )
                
                # Calculate evaluation score
                score = calculate_score(model_response, ground_truth)
                
                results.append({
                    'user_input': user_input,
                    'ground_truth_output': ground_truth,
                    'model_response': model_response,
                    'score': score
                })
            except Exception as e:
                logger.error(f"Error processing example: {e}")
                results.append({
                    'user_input': user_input,
                    'ground_truth_output': ground_truth,
                    'model_response': f"Error: {str(e)}",
                    'score': 0
                })
        
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file uploads with example data."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and is_allowed_file(file.filename):
        try:
            examples = parse_csv_file(file)
            return jsonify({'examples': examples})
        except Exception as e:
            logger.error(f"Error parsing CSV file: {e}")
            return jsonify({'error': f'Error parsing CSV file: {str(e)}'}), 400
    else:
        return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400

@app.route('/save_prompts', methods=['POST'])
def save_prompts():
    """Save the current prompts to files."""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        
        if not system_prompt or not output_prompt:
            return jsonify({'error': 'System prompt and output prompt are required'}), 400
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save system prompt
        system_filename = f"system_prompt_{timestamp}.txt"
        with open(os.path.join('prompts', system_filename), 'w') as f:
            f.write(system_prompt)
        
        # Save output prompt
        output_filename = f"output_prompt_{timestamp}.txt"
        with open(os.path.join('prompts', output_filename), 'w') as f:
            f.write(output_prompt)
        
        return jsonify({
            'message': 'Prompts saved successfully',
            'system_filename': system_filename,
            'output_filename': output_filename
        })
    except Exception as e:
        logger.error(f"Error saving prompts: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
