import os
import logging
import yaml
import json
import pandas as pd
from datetime import datetime
from flask import render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app import app
from app.llm_client import get_llm_response
from app.evaluator import calculate_score
from app.optimizer import optimize_prompts, load_optimizer_prompt, get_optimization_strategies
from app.experiment_tracker import ExperimentTracker
from app.data_module import DataModule
from app.workflow import PromptOptimizationWorkflow
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
        'gemini': {
            'model_name': 'gemini-1.5-pro',
            'temperature': 0.0,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 1024
        }
    }

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)
os.makedirs('prompts', exist_ok=True)
os.makedirs('prompts/system', exist_ok=True)
os.makedirs('prompts/output', exist_ok=True)
os.makedirs('experiments', exist_ok=True)

# Initialize core components
experiment_tracker = ExperimentTracker()
data_module = DataModule()
prompt_workflow = PromptOptimizationWorkflow(data_module, experiment_tracker, config)

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/training')
def training():
    """Render the ML training interface."""
    return render_template('training.html')

@app.route('/history')
def history():
    """Render the experiment history page."""
    return render_template('history.html')

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
                # Call Gemini API
                model_response = get_llm_response(
                    system_prompt, 
                    user_input, 
                    output_prompt,
                    config.get('gemini', {})
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
        
        # Calculate aggregate metrics
        total_score = sum(r.get('score', 0) for r in results)
        perfect_matches = sum(1 for r in results if r.get('score', 0) >= 0.9)
        total = len(results)
        
        metrics = {
            "avg_score": total_score / total if total > 0 else 0,
            "perfect_matches": perfect_matches,
            "total_examples": total,
            "perfect_match_percent": (perfect_matches / total * 100) if total > 0 else 0
        }
        
        return jsonify({
            'results': results,
            'metrics': metrics
        })
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

@app.route('/load_dataset', methods=['GET'])
def load_dataset():
    """Load the current training and validation datasets."""
    try:
        dataset_type = request.args.get('type', 'train')
        
        if dataset_type == 'train':
            examples = data_module.get_train_examples()
            if not examples:
                # Try to load from file
                try:
                    with open(os.path.join('data/train', 'current_train.json'), 'r') as f:
                        examples = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load training examples from file: {e}")
        elif dataset_type == 'validation':
            examples = data_module.get_validation_examples()
            if not examples:
                # Try to load from file
                try:
                    with open(os.path.join('data/validation', 'current_validation.json'), 'r') as f:
                        examples = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load validation examples from file: {e}")
        elif dataset_type == 'nejm_train':
            # Load NEJM training dataset
            try:
                with open(os.path.join('data/train', 'current_train.json'), 'r') as f:
                    examples = json.load(f)
                logger.info(f"Loaded {len(examples)} NEJM training examples")
            except Exception as e:
                logger.error(f"Could not load NEJM training examples: {e}")
                return jsonify({'error': f"Could not load NEJM training examples: {str(e)}"}), 500
        elif dataset_type == 'nejm_validation':
            # Load NEJM validation dataset
            try:
                with open(os.path.join('data/validation', 'current_validation.json'), 'r') as f:
                    examples = json.load(f)
                logger.info(f"Loaded {len(examples)} NEJM validation examples")
            except Exception as e:
                logger.error(f"Could not load NEJM validation examples: {e}")
                return jsonify({'error': f"Could not load NEJM validation examples: {str(e)}"}), 500
        elif dataset_type == 'nejm_prompts':
            # Load NEJM specialized prompts
            try:
                system_prompt_path = os.path.join('prompts', 'system_prompt_advanced_medical.txt')
                output_prompt_path = os.path.join('prompts', 'output_prompt_advanced_medical.txt')
                
                system_prompt = ""
                output_prompt = ""
                
                if os.path.exists(system_prompt_path):
                    with open(system_prompt_path, 'r') as f:
                        system_prompt = f.read()
                else:
                    # Default medical system prompt if file doesn't exist
                    system_prompt = "You are a medical AI assistant with expertise in diagnostic reasoning based on clinical cases."
                
                if os.path.exists(output_prompt_path):
                    with open(output_prompt_path, 'r') as f:
                        output_prompt = f.read()
                else:
                    # Default medical output prompt if file doesn't exist
                    output_prompt = "Based on the clinical information provided, what is the most likely diagnosis?"
                
                return jsonify({
                    'prompts': {
                        'system_prompt': system_prompt,
                        'output_prompt': output_prompt
                    }
                })
            except Exception as e:
                logger.error(f"Error loading NEJM prompts: {e}")
                return jsonify({'error': f"Error loading NEJM prompts: {str(e)}"}), 500
        else:
            return jsonify({'error': 'Invalid dataset type. Use "train", "validation", "nejm_train", "nejm_validation", or "nejm_prompts"'}), 400
        
        # Create a CSV-like string from the examples for display
        csv_content = "user_input,ground_truth_output\n"
        for example in examples:
            user_input = example.get('user_input', '').replace(',', '\\,').replace('\n', ' ')
            ground_truth = example.get('ground_truth_output', '').replace(',', '\\,').replace('\n', ' ')
            csv_content += f"{user_input},{ground_truth}\n"
        
        return jsonify({
            'examples': examples,
            'csv_content': csv_content,
            'count': len(examples)
        })
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return jsonify({'error': str(e)}), 500

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
        with open(os.path.join('prompts/system', system_filename), 'w') as f:
            f.write(system_prompt)
        
        # Save output prompt
        output_filename = f"output_prompt_{timestamp}.txt"
        with open(os.path.join('prompts/output', output_filename), 'w') as f:
            f.write(output_prompt)
        
        return jsonify({
            'message': 'Prompts saved successfully',
            'system_filename': system_filename,
            'output_filename': output_filename
        })
    except Exception as e:
        logger.error(f"Error saving prompts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Optimize prompts using a second LLM based on evaluation results.
    """
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        examples = data.get('examples', [])
        optimizer_prompt = data.get('optimizer_prompt', None)
        
        if not system_prompt or not output_prompt:
            return jsonify({'error': 'System prompt and output prompt are required'}), 400
        
        if not examples:
            return jsonify({'error': 'Example results are required for optimization'}), 400
        
        # If optimizer prompt not provided, load the default
        if not optimizer_prompt:
            optimizer_prompt = load_optimizer_prompt()
        
        # Call the optimizer
        optimization_result = optimize_prompts(
            system_prompt,
            output_prompt,
            examples,
            optimizer_prompt
        )
        
        return jsonify(optimization_result)
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """
    Run a training iteration (evaluate -> optimize -> evaluate).
    """
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        examples_format = data.get('examples_format', 'text')
        examples_content = data.get('examples_content', '')
        optimizer_prompt = data.get('optimizer_prompt', None)
        experiment_id = data.get('experiment_id', None)
        iteration = data.get('iteration', 0)
        
        if not system_prompt or not output_prompt:
            return jsonify({'error': 'System prompt and output prompt are required'}), 400
        
        if not examples_content:
            return jsonify({'error': 'Example data is required'}), 400
        
        # Start a new experiment if needed
        if not experiment_id:
            experiment_id = experiment_tracker.start_experiment()
            iteration = 0
        
        # Parse examples
        if examples_format == 'text':
            examples = parse_text_examples(examples_content)
        else:
            return jsonify({'error': 'Invalid examples format'}), 400
        
        if not examples:
            return jsonify({'error': 'No valid examples found'}), 400
        
        # Step 1: Evaluate current prompts on examples
        evaluation_results = []
        for example in examples:
            user_input = example.get('user_input', '')
            ground_truth = example.get('ground_truth_output', '')
            
            if not user_input:
                continue
            
            try:
                model_response = get_llm_response(
                    system_prompt, 
                    user_input, 
                    output_prompt,
                    config.get('gemini', {})
                )
                
                score = calculate_score(model_response, ground_truth)
                
                evaluation_results.append({
                    'user_input': user_input,
                    'ground_truth_output': ground_truth,
                    'model_response': model_response,
                    'score': score
                })
            except Exception as e:
                logger.error(f"Error in evaluation: {e}")
                evaluation_results.append({
                    'user_input': user_input,
                    'ground_truth_output': ground_truth,
                    'model_response': f"Error: {str(e)}",
                    'score': 0
                })
        
        # Calculate metrics
        total_score = sum(r.get('score', 0) for r in evaluation_results)
        perfect_matches = sum(1 for r in evaluation_results if r.get('score', 0) >= 0.9)
        total = len(evaluation_results)
        
        initial_metrics = {
            "avg_score": total_score / total if total > 0 else 0,
            "perfect_matches": perfect_matches,
            "total_examples": total,
            "perfect_match_percent": (perfect_matches / total * 100) if total > 0 else 0
        }
        
        # Save the initial iteration
        experiment_tracker.save_iteration(
            experiment_id=experiment_id,
            iteration=iteration,
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            metrics=initial_metrics,
            examples=evaluation_results
        )
        
        # Step 2: Optimize prompts if there's room for improvement
        if initial_metrics["perfect_match_percent"] < 100:
            if not optimizer_prompt:
                optimizer_prompt = load_optimizer_prompt()
            
            optimization_result = optimize_prompts(
                system_prompt,
                output_prompt,
                evaluation_results,
                optimizer_prompt
            )
            
            new_system_prompt = optimization_result.get('system_prompt', system_prompt)
            new_output_prompt = optimization_result.get('output_prompt', output_prompt)
            reasoning = optimization_result.get('reasoning', '')
            
            # Step 3: Evaluate the optimized prompts
            optimized_results = []
            for example in examples:
                user_input = example.get('user_input', '')
                ground_truth = example.get('ground_truth_output', '')
                
                if not user_input:
                    continue
                
                try:
                    model_response = get_llm_response(
                        new_system_prompt, 
                        user_input, 
                        new_output_prompt,
                        config.get('gemini', {})
                    )
                    
                    score = calculate_score(model_response, ground_truth)
                    
                    optimized_results.append({
                        'user_input': user_input,
                        'ground_truth_output': ground_truth,
                        'model_response': model_response,
                        'score': score
                    })
                except Exception as e:
                    logger.error(f"Error in evaluation: {e}")
                    optimized_results.append({
                        'user_input': user_input,
                        'ground_truth_output': ground_truth,
                        'model_response': f"Error: {str(e)}",
                        'score': 0
                    })
            
            # Calculate new metrics
            new_total_score = sum(r.get('score', 0) for r in optimized_results)
            new_perfect_matches = sum(1 for r in optimized_results if r.get('score', 0) >= 0.9)
            new_total = len(optimized_results)
            
            optimized_metrics = {
                "avg_score": new_total_score / new_total if new_total > 0 else 0,
                "perfect_matches": new_perfect_matches,
                "total_examples": new_total,
                "perfect_match_percent": (new_perfect_matches / new_total * 100) if new_total > 0 else 0
            }
            
            # Save the optimized iteration
            experiment_tracker.save_iteration(
                experiment_id=experiment_id,
                iteration=iteration + 1,
                system_prompt=new_system_prompt,
                output_prompt=new_output_prompt,
                metrics=optimized_metrics,
                examples=optimized_results,
                optimizer_reasoning=reasoning
            )
            
            # Prepare the response
            response = {
                'experiment_id': experiment_id,
                'initial_iteration': iteration,
                'optimized_iteration': iteration + 1,
                'initial': {
                    'system_prompt': system_prompt,
                    'output_prompt': output_prompt,
                    'metrics': initial_metrics,
                    'results': evaluation_results
                },
                'optimized': {
                    'system_prompt': new_system_prompt,
                    'output_prompt': new_output_prompt,
                    'metrics': optimized_metrics,
                    'results': optimized_results,
                    'reasoning': reasoning
                },
                'improvement': optimized_metrics["avg_score"] - initial_metrics["avg_score"]
            }
        else:
            # No optimization needed
            response = {
                'experiment_id': experiment_id,
                'initial_iteration': iteration,
                'optimized_iteration': iteration,
                'initial': {
                    'system_prompt': system_prompt,
                    'output_prompt': output_prompt,
                    'metrics': initial_metrics,
                    'results': evaluation_results
                },
                'message': 'Perfect score achieved, no optimization needed.'
            }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in training: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/experiments', methods=['GET'])
def get_experiments():
    """Get list of all experiments."""
    try:
        experiments = experiment_tracker.load_experiment_history()
        return jsonify({'experiments': experiments})
    except Exception as e:
        logger.error(f"Error loading experiments: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/experiments/<experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    """Get details for a specific experiment."""
    try:
        iterations = experiment_tracker.get_iterations(experiment_id)
        return jsonify({'experiment_id': experiment_id, 'iterations': iterations})
    except Exception as e:
        logger.error(f"Error loading experiment {experiment_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimizer_prompt', methods=['GET'])
def get_optimizer_prompt():
    """Get the default optimizer prompt."""
    try:
        optimizer_prompt = load_optimizer_prompt()
        return jsonify({'optimizer_prompt': optimizer_prompt})
    except Exception as e:
        logger.error(f"Error loading optimizer prompt: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimizer_prompt', methods=['POST'])
def save_optimizer_prompt():
    """Save a custom optimizer prompt."""
    try:
        data = request.json
        optimizer_prompt = data.get('optimizer_prompt', '')
        
        if not optimizer_prompt:
            return jsonify({'error': 'Optimizer prompt is required'}), 400
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimizer_prompt_{timestamp}.txt"
        
        # Save to file
        with open(os.path.join('prompts', 'optimizer', filename), 'w') as f:
            f.write(optimizer_prompt)
        
        # Also update the default
        with open(os.path.join('prompts', 'optimizer', 'default_optimizer.txt'), 'w') as f:
            f.write(optimizer_prompt)
        
        return jsonify({
            'message': 'Optimizer prompt saved successfully',
            'filename': filename
        })
    except Exception as e:
        logger.error(f"Error saving optimizer prompt: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_optimization_strategies')
def get_available_strategies():
    """Get available optimization strategies."""
    try:
        strategies = get_optimization_strategies()
        return jsonify({'strategies': strategies})
    except Exception as e:
        logger.error(f"Error getting optimization strategies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/two_stage_train', methods=['POST'])
def two_stage_train():
    """
    Run the Two-Stage Training Cycle workflow.
    Phase 1: Primary LLM Inference & Evaluation
    Phase 2: Optimizer LLM Refinement
    """
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        examples = data.get('examples', [])
        max_iterations = int(data.get('max_iterations', 1))
        optimizer_strategy = data.get('optimizer_strategy', 'full_rewrite')
        optimizer_type = data.get('optimizer_type', 'general')
        
        if not system_prompt or not output_prompt:
            return jsonify({'error': 'System prompt and output prompt are required'}), 400
        
        if not examples:
            # If examples are not directly provided, we'll look for them in the data module
            if data_module.get_train_examples():
                logger.info("Using examples from data module")
            else:
                # Try to convert examples from the text content
                examples_content = data.get('examples_content', '')
                if examples_content:
                    examples = parse_text_examples(examples_content)
                    
                    # Split and save to data module
                    train_examples, validation_examples = data_module.split_examples(examples)
                    data_module._save_examples(train_examples, os.path.join('data', 'train', 'examples.json'))
                    data_module._save_examples(validation_examples, os.path.join('data', 'validation', 'examples.json'))
                else:
                    return jsonify({'error': 'No examples found. Please provide examples directly or load a dataset.'}), 400
        
        # Set early stopping patience from config or use default
        early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 2)
        
        # Run the training cycle
        result = prompt_workflow.run_training_cycle(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            max_iterations=max_iterations,
            early_stopping_patience=early_stopping_patience,
            optimizer_strategy=optimizer_strategy,
            optimizer_type=optimizer_type
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in two-stage training cycle: {e}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/validate_prompts', methods=['POST'])
def validate_prompts():
    """
    Run validation to compare different prompt versions on unseen data.
    """
    try:
        data = request.json
        prompt_versions = data.get('prompt_versions', [])
        
        if not prompt_versions:
            return jsonify({'error': 'Please specify which prompt versions to validate'}), 400
        
        # Run validation
        result = prompt_workflow.run_validation(prompt_versions)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in validation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
