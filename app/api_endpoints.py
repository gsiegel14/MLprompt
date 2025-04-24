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
from flask import request, jsonify
from app import app
from app.llm_client import get_llm_response
from app.evaluator import calculate_score, evaluate_batch
from app.optimizer import optimize_prompts, load_optimizer_prompt, get_optimization_strategies
from app.experiment_tracker import ExperimentTracker
from app.data_module import DataModule
from app.workflow import PromptOptimizationWorkflow
from app.huggingface_client import evaluate_metrics, validate_api_connection

logger = logging.getLogger(__name__)

# Initialize core components
experiment_tracker = ExperimentTracker()
data_module = DataModule()

# Load configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    config = {
        'gemini': {
            'model_name': 'gemini-1.5-flash',
            'temperature': 0.0,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 1024
        }
    }

prompt_workflow = PromptOptimizationWorkflow(data_module, experiment_tracker, config)

# API Status Endpoint
@app.route('/api/status', methods=['GET'])
def api_status():
    """Return the status of the API and its components."""
    try:
        # Check for Hugging Face API token
        hf_token_available = 'HUGGING_FACE_TOKEN' in os.environ or 'HUGGINGFACE_API_KEY' in os.environ
        
        # Check model configuration
        gemini_model = config.get('gemini', {}).get('model_name', 'unknown')
        
        # Get dataset stats
        train_count = len(data_module.get_train_examples())
        validation_count = len(data_module.get_validation_examples())
        
        # Get experiment count
        experiment_count = len(experiment_tracker.get_experiment_list())
        
        return jsonify({
            'status': 'ok',
            'components': {
                'llm_client': 'available',
                'data_module': 'available',
                'experiment_tracker': 'available'
            },
            'config': {
                'gemini_model': gemini_model,
                'huggingface_token_available': hf_token_available
            },
            'stats': {
                'train_examples': train_count,
                'validation_examples': validation_count,
                'experiment_count': experiment_count
            }
        })
    except Exception as e:
        logger.error(f"Error in API status endpoint: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# LLM API Endpoint
@app.route('/api/llm/generate', methods=['POST'])
def llm_generate():
    """Generate a response using the LLM."""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        user_input = data.get('user_input', '')
        output_prompt = data.get('output_prompt', '')
        
        if not system_prompt or not user_input:
            return jsonify({'error': 'System prompt and user input are required'}), 400
        
        # Call Gemini API
        model_response = get_llm_response(
            system_prompt, 
            user_input, 
            output_prompt,
            config.get('gemini', {})
        )
        
        return jsonify({
            'response': model_response,
            'model': config.get('gemini', {}).get('model_name', 'unknown')
        })
    except Exception as e:
        logger.error(f"Error in LLM generate endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Evaluation API Endpoint
@app.route('/api/evaluate', methods=['POST'])
def evaluate_api():
    """Evaluate the similarity between a model response and ground truth."""
    try:
        data = request.json
        model_response = data.get('model_response', '')
        ground_truth = data.get('ground_truth', '')
        
        if not model_response or not ground_truth:
            return jsonify({'error': 'Model response and ground truth are required'}), 400
        
        # Calculate evaluation score
        score = calculate_score(model_response, ground_truth)
        
        return jsonify({
            'score': score,
            'perfect_match': score >= config.get('evaluation', {}).get('perfect_threshold', 0.9)
        })
    except Exception as e:
        logger.error(f"Error in evaluate endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Optimization API Endpoint
@app.route('/api/optimize', methods=['POST'])
def optimize_api():
    """Optimize prompts based on examples and current prompts."""
    try:
        data = request.json
        current_system_prompt = data.get('current_system_prompt', '')
        current_output_prompt = data.get('current_output_prompt', '')
        examples = data.get('examples', [])
        optimizer_strategy = data.get('optimizer_strategy', 'reasoning_first')
        
        if not current_system_prompt or not examples:
            return jsonify({'error': 'System prompt and examples are required'}), 400
        
        # Load optimizer prompt based on strategy
        optimizer_prompt = load_optimizer_prompt(optimizer_strategy)
        
        # Run optimization
        optimization_result = optimize_prompts(
            current_system_prompt=current_system_prompt,
            current_output_prompt=current_output_prompt,
            examples=examples,
            optimizer_system_prompt=optimizer_prompt,
            strategy=optimizer_strategy
        )
        
        if not optimization_result:
            # If optimization fails, return the original prompts as a fallback
            return jsonify({
                'optimized_system_prompt': current_system_prompt,
                'optimized_output_prompt': current_output_prompt,
                'reasoning': 'Optimization failed, returning original prompts.',
                'analysis': 'Could not generate analysis.'
            })
            
        # Make sure all required fields are present
        if 'optimized_system_prompt' not in optimization_result:
            optimization_result['optimized_system_prompt'] = current_system_prompt
            
        if 'optimized_output_prompt' not in optimization_result:
            optimization_result['optimized_output_prompt'] = current_output_prompt
            
        if 'reasoning' not in optimization_result:
            optimization_result['reasoning'] = 'No reasoning provided.'
            
        if 'analysis' not in optimization_result:
            optimization_result['analysis'] = 'No analysis provided.'
            
        return jsonify(optimization_result)
    except Exception as e:
        logger.error(f"Error in optimize endpoint: {e}")
        # Return a more graceful error response with the original prompts
        return jsonify({
            'optimized_system_prompt': current_system_prompt,
            'optimized_output_prompt': current_output_prompt,
            'reasoning': f'Error during optimization: {str(e)}',
            'analysis': 'Could not generate analysis due to error.'
        })

# Hugging Face Metrics API Endpoint
@app.route('/api/metrics/huggingface', methods=['POST'])
def huggingface_metrics_api():
    """Compute metrics using Hugging Face's evaluation library."""
    try:
        data = request.json
        predictions = data.get('predictions', [])
        references = data.get('references', [])
        metrics = data.get('metrics', ['exact_match', 'bleu'])
        
        if not predictions or not references:
            return jsonify({'error': 'Predictions and references are required'}), 400
        
        if len(predictions) != len(references):
            return jsonify({'error': 'Predictions and references must have the same length'}), 400
        
        # Evaluate metrics
        results = evaluate_metrics(predictions, references, metrics)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in Hugging Face metrics endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Five-API Workflow Endpoint
@app.route('/api/workflow/five_api', methods=['POST'])
def five_api_workflow_api():
    """Run the complete 5-API workflow."""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        batch_size = data.get('batch_size', 5)
        optimizer_strategy = data.get('optimizer_strategy', 'reasoning_first')
        hf_metrics = data.get('hf_metrics', ['exact_match', 'bleu'])
        
        if not system_prompt:
            return jsonify({'error': 'System prompt is required'}), 400
        
        # Run the workflow
        result = prompt_workflow.run_five_api_workflow(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            batch_size=batch_size,
            optimizer_strategy=optimizer_strategy,
            hf_metrics=hf_metrics
        )
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in five_api_workflow endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Experiment History API Endpoint
@app.route('/api/experiments/list', methods=['GET'])
def list_experiments_api():
    """Get a list of all experiments."""
    try:
        # Get experiment list
        experiments = experiment_tracker.get_experiment_list()
        
        return jsonify({
            'experiments': experiments,
            'count': len(experiments)
        })
    except Exception as e:
        logger.error(f"Error in list_experiments endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Optimizer Prompts API Endpoints
@app.route('/api/optimizer_prompt', methods=['GET'])
def get_optimizer_prompt_api():
    """Get the current optimizer prompts."""
    try:
        # Get default optimizer prompt strategy
        strategy = request.args.get('strategy', 'reasoning_first')
        
        # Load system prompt and output prompt
        optimizer_strategies = get_optimization_strategies()
        
        if strategy not in optimizer_strategies:
            strategy = 'reasoning_first'  # Default fallback
            
        system_prompt = ""
        output_prompt = ""
        
        # Try to load from files
        try:
            with open(f"prompts/optimizer/Optimizer_systemmessage.md.txt", "r") as f:
                system_prompt = f.read()
                
            with open(f"prompts/optimizer/optimizer_output_prompt.txt", "r") as f:
                output_prompt = f.read()
        except Exception as e:
            logger.warning(f"Could not load optimizer prompt files: {e}")
            system_prompt = "You are an expert prompt engineer."
            output_prompt = "Analyze and optimize the given prompt."
        
        return jsonify({
            'system_prompt': system_prompt,
            'output_prompt': output_prompt,
            'strategy': strategy,
            'available_strategies': optimizer_strategies
        })
    except Exception as e:
        logger.error(f"Error in get_optimizer_prompt endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_optimizer_prompt', methods=['POST'])
def save_optimizer_prompt_api():
    """Save custom optimizer prompts."""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        
        if not system_prompt:
            return jsonify({'error': 'System prompt is required'}), 400
            
        # Create optimizer directory if it doesn't exist
        os.makedirs('prompts/optimizer', exist_ok=True)
        
        # Save files
        with open(f"prompts/optimizer/Optimizer_systemmessage.md.txt", "w") as f:
            f.write(system_prompt)
            
        if output_prompt:
            with open(f"prompts/optimizer/optimizer_output_prompt.txt", "w") as f:
                f.write(output_prompt)
        
        return jsonify({
            'success': True,
            'message': 'Optimizer prompts saved successfully'
        })
    except Exception as e:
        logger.error(f"Error in save_optimizer_prompt endpoint: {e}")
        return jsonify({'error': str(e)}), 500
        
# Metrics Summary API Endpoint
@app.route('/api/metrics_summary', methods=['GET'])
def metrics_summary_api():
    """Get a summary of metrics from recent experiments."""
    try:
        # Get experiment list (max 10 most recent)
        experiments = experiment_tracker.get_experiment_list()[-10:]
        
        # Initialize summary data
        summary = {
            'experiment_count': len(experiments),
            'average_improvement': 0.0,
            'metrics_by_experiment': {},
            'strategies_used': {}
        }
        
        if experiments:
            # Track total improvement
            total_improvement = 0.0
            improvements_count = 0
            
            # Process each experiment
            for exp_id in experiments:
                try:
                    iterations = experiment_tracker.get_experiment_iterations(exp_id)
                    if iterations and len(iterations) > 0:
                        first_iteration = iterations[0]
                        last_iteration = iterations[-1]
                        
                        strategy = first_iteration.get('strategy', 'unknown')
                        
                        # Track strategy usage
                        if strategy in summary['strategies_used']:
                            summary['strategies_used'][strategy] += 1
                        else:
                            summary['strategies_used'][strategy] = 1
                        
                        # Calculate improvement
                        first_score = first_iteration.get('metrics', {}).get('avg_score', 0)
                        last_score = last_iteration.get('metrics', {}).get('avg_score', 0)
                        improvement = last_score - first_score
                        
                        if improvement != 0:
                            total_improvement += improvement
                            improvements_count += 1
                        
                        # Add to metrics by experiment
                        summary['metrics_by_experiment'][exp_id] = {
                            'initial_score': first_score,
                            'final_score': last_score,
                            'improvement': improvement,
                            'iterations': len(iterations),
                            'strategy': strategy
                        }
                except Exception as inner_e:
                    logger.warning(f"Error processing experiment {exp_id}: {inner_e}")
                    continue
            
            # Calculate average improvement
            if improvements_count > 0:
                summary['average_improvement'] = total_improvement / improvements_count
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error in metrics_summary endpoint: {e}")
        return jsonify({
            'error': str(e),
            'experiment_count': 0,
            'average_improvement': 0.0
        }), 500

# Five API Workflow API Endpoint
@app.route('/api/five_api_workflow_info', methods=['GET'])
def five_api_workflow_info():
    """Return information for the five API workflow page."""
    try:
        # Get strategies for the UI
        strategies = get_optimization_strategies()
        
        return jsonify({
            'status': 'ok',
            'strategies': strategies,
            'message': 'Five API workflow info endpoint'
        })
    except Exception as e:
        logger.error(f"Error in five_api_workflow_info endpoint: {e}")
        return jsonify({'error': str(e)}), 500