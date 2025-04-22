import os
import logging
import yaml
import json
import pandas as pd
from datetime import datetime
from flask import render_template, request, jsonify, flash, redirect, url_for, send_from_directory
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
            'model_name': 'gemini-2.5-flash',
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

@app.route('/prompts/<path:filename>')
def serve_prompt_file(filename):
    """Serve prompt files from the prompts folder."""
    return send_from_directory('prompts', filename)

@app.route('/history')
def history():
    """Render the experiment history page."""
    return render_template('history.html')

@app.route('/todo')
def todo():
    """Render the todo list page."""
    return render_template('todo.html')

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

@app.route('/load_dataset', methods=['GET', 'POST'])
def load_dataset():
    """Load the current training and validation datasets."""
    try:
        # Support both GET and POST for flexibility
        if request.method == 'GET':
            dataset_type = request.args.get('type', 'train')
        else:
            data = request.json
            dataset_type = data.get('type', 'train')

        # Initialize empty examples list
        examples = []

        # Use in-memory cache for NEJM datasets to improve performance
        nejm_train_cache = getattr(app, 'nejm_train_cache', None)
        nejm_validation_cache = getattr(app, 'nejm_validation_cache', None)

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
            # Use cached data if available
            if nejm_train_cache and config.get('app', {}).get('enable_caching', True):
                examples = nejm_train_cache
                logger.debug(f"Using cached NEJM training examples ({len(examples)} examples)")
            else:
                # Load NEJM training dataset
                try:
                    # First try from examples.json, then fallback to current_train.json
                    try:
                        with open(os.path.join('data/train', 'examples.json'), 'r') as f:
                            examples = json.load(f)
                    except:
                        with open(os.path.join('data/train', 'current_train.json'), 'r') as f:
                            examples = json.load(f)

                    # Make sure we actually have examples
                    if not examples or len(examples) == 0:
                        # If we got an empty list, run the fix_nejm_data script
                        import sys
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("fix_nejm_data", "fix_nejm_data.py")
                        fix_nejm_module = importlib.util.module_from_spec(spec)

                        if spec and spec.loader:
                            spec.loader.exec_module(fix_nejm_module)
                            fix_nejm_module.fix_nejm_data()

                            # Now try loading again
                            with open(os.path.join('data/train', 'examples.json'), 'r') as f:
                                examples = json.load(f)
                        else:
                            logger.error("Could not load fix_nejm_data module")

                    # Cache the data for future requests
                    if examples:
                        app.nejm_train_cache = examples

                    logger.info(f"Loaded {len(examples)} NEJM training examples")
                except Exception as e:
                    logger.error(f"Could not load NEJM training examples: {e}")
                    return jsonify({'error': f"Could not load NEJM training examples: {str(e)}"}), 500
        elif dataset_type == 'nejm_validation':
            # Use cached data if available
            if nejm_validation_cache and config.get('app', {}).get('enable_caching', True):
                examples = nejm_validation_cache
                logger.debug(f"Using cached NEJM validation examples ({len(examples)} examples)")
            else:
                # Load NEJM validation dataset
                try:
                    # First try from examples.json, then fallback to current_validation.json
                    try:
                        with open(os.path.join('data/validation', 'examples.json'), 'r') as f:
                            examples = json.load(f)
                    except:
                        with open(os.path.join('data/validation', 'current_validation.json'), 'r') as f:
                            examples = json.load(f)

                    # Make sure we actually have examples
                    if not examples or len(examples) == 0:
                        # If we got an empty list, run the fix_nejm_data script
                        import sys
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("fix_nejm_data", "fix_nejm_data.py")
                        fix_nejm_module = importlib.util.module_from_spec(spec)

                        if spec and spec.loader:
                            spec.loader.exec_module(fix_nejm_module)
                            fix_nejm_module.fix_nejm_data()

                            # Now try loading again
                            with open(os.path.join('data/validation', 'examples.json'), 'r') as f:
                                examples = json.load(f)
                        else:
                            logger.error("Could not load fix_nejm_data module")

                    # Cache the data for future requests
                    if examples:
                        app.nejm_validation_cache = examples

                    logger.info(f"Loaded {len(examples)} NEJM validation examples")
                except Exception as e:
                    logger.error(f"Could not load NEJM validation examples: {e}")
                    return jsonify({'error': f"Could not load NEJM validation examples: {str(e)}"}), 500
        elif dataset_type == 'nejm_prompts':
            # Load NEJM specialized prompts
            try:
                system_prompt_path = 'prompts/system_prompt_advanced_medical.txt'
                output_prompt_path = 'prompts/output_prompt_advanced_medical.txt'

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

        # For NEJM datasets, don't include full csv_content for performance reasons
        # Just return a truncated version and the original examples
        if dataset_type in ['nejm_train', 'nejm_validation']:
            # Only include first 10 examples in csv_content to save bandwidth and processing time
            short_examples = examples[:10]
            csv_content = "user_input,ground_truth_output\n"

            for example in short_examples:
                # Use shorter version of text for display only
                user_input = example.get('user_input', '')[:200] + '...' if len(example.get('user_input', '')) > 200 else example.get('user_input', '')
                ground_truth = example.get('ground_truth_output', '')[:50] + '...' if len(example.get('ground_truth_output', '')) > 50 else example.get('ground_truth_output', '')

                # Replace commas and newlines
                user_input = user_input.replace(',', '\\,').replace('\n', ' ')
                ground_truth = ground_truth.replace(',', '\\,').replace('\n', ' ')

                csv_content += f"{user_input},{ground_truth}\n"

            logger.info(f"Returning {len(examples)} examples with truncated CSV content")
            return jsonify({
                'examples': examples,
                'csv_content': csv_content,
                'count': len(examples),
                'truncated': True
            })
        else:
            # For smaller datasets, include full CSV content
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

        # Ensure directories exist
        os.makedirs('prompts/system', exist_ok=True)
        os.makedirs('prompts/output', exist_ok=True)

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
        # Create a training log file for this session
        import traceback
        from datetime import datetime
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"logs/training_{log_timestamp}.log"
        os.makedirs("logs", exist_ok=True)

        # Log start of training with detailed request info
        logger.info(f"========== TRAINING RUN STARTED AT {log_timestamp} ==========")

        # Step 0: Parse and validate request data
        logger.info("STEP 0: Parsing and validating request data")
        data = request.json
        if not data:
            error_msg = "No JSON data received in request"
            logger.error(error_msg)
            with open(log_file_path, 'w') as f:
                f.write(f"ERROR: {error_msg}\n")
            return jsonify({'error': error_msg}), 400

        # Extract request parameters with detailed logging
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        examples_format = data.get('examples_format', 'text')
        examples_content = data.get('examples_content', '')
        optimizer_prompt = data.get('optimizer_prompt', None)
        experiment_id = data.get('experiment_id', None)
        iteration = data.get('iteration', 0)
        optimizer_strategy = data.get('optimizer_strategy', 'reasoning_first')

        # Log request details
        logger.info(f"Request details: format={examples_format}, strategy={optimizer_strategy}, iteration={iteration}")
        logger.info(f"System prompt length: {len(system_prompt)} chars")
        logger.info(f"Output prompt length: {len(output_prompt)} chars")

        # Validate required parameters
        if not system_prompt or not output_prompt:
            error_msg = "System prompt and output prompt are required"
            logger.error(error_msg)
            with open(log_file_path, 'w') as f:
                f.write(f"ERROR: {error_msg}\n")
            return jsonify({'error': error_msg}), 400

        if not examples_content:
            error_msg = "Example data is required"
            logger.error(error_msg)
            with open(log_file_path, 'w') as f:
                f.write(f"ERROR: {error_msg}\n")
            return jsonify({'error': error_msg}), 400

        # Create a new experiment if needed
        logger.info("STEP 1: Creating or loading experiment")
        if not experiment_id:
            experiment_id = experiment_tracker.start_experiment()
            iteration = 0
            logger.info(f"Created new experiment with ID: {experiment_id}")
        else:
            logger.info(f"Using existing experiment ID: {experiment_id}")

        # Parse examples with detailed error handling
        logger.info("STEP 2: Parsing example data")
        try:
            if examples_format == 'text':
                examples = parse_text_examples(examples_content)
                logger.info(f"Successfully parsed {len(examples)} examples from text")
            else:
                error_msg = "Invalid examples format"
                logger.error(error_msg)
                with open(log_file_path, 'a') as f:
                    f.write(f"ERROR: {error_msg}\n")
                return jsonify({'error': error_msg}), 400
        except Exception as e:
            error_msg = f"Failed to parse examples: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_msg}\n{traceback.format_exc()}\n")
            return jsonify({'error': error_msg}), 400

        if not examples:
            error_msg = "No valid examples found in input"
            logger.error(error_msg)
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_msg}\n")
            return jsonify({'error': error_msg}), 400

        # Step 3: Initial evaluation of current prompts
        logger.info(f"STEP 3: Evaluating initial prompts on {len(examples)} examples")
        evaluation_results = []
        success_count = 0
        error_count = 0

        for i, example in enumerate(examples):
            user_input = example.get('user_input', '')
            ground_truth = example.get('ground_truth_output', '')

            if not user_input:
                logger.warning(f"Skipping example {i+1}: No user input")
                continue

            logger.info(f"Processing example {i+1}/{len(examples)}")
            try:
                # Call LLM API with detailed logging
                logger.info(f"Calling LLM API for example {i+1}")
                model_response = get_llm_response(
                    system_prompt, 
                    user_input, 
                    output_prompt,
                    config.get('gemini', {})
                )

                # Calculate score
                logger.info(f"Calculating score for example {i+1}")
                score = calculate_score(model_response, ground_truth)

                # Record result
                evaluation_results.append({
                    'user_input': user_input,
                    'ground_truth_output': ground_truth,
                    'model_response': model_response,
                    'score': score
                })

                # Log brief summary of result
                logger.info(f"Example {i+1} processed with score: {score:.4f}")
                success_count += 1

            except Exception as e:
                error_msg = f"Error processing example {i+1}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                with open(log_file_path, 'a') as f:
                    f.write(f"ERROR: {error_msg}\n{traceback.format_exc()}\n")

                evaluation_results.append({
                    'user_input': user_input,
                    'ground_truth_output': ground_truth,
                    'model_response': f"Error: {str(e)}",
                    'score': 0
                })
                error_count += 1

        logger.info(f"Initial evaluation complete: {success_count} successful, {error_count} errors")

        # Calculate metrics with detailed logging
        logger.info("STEP 4: Calculating initial metrics")
        try:
            total_score = sum(r.get('score', 0) for r in evaluation_results)
            perfect_matches = sum(1 for r in evaluation_results if r.get('score', 0) >= 0.9)
            total = len(evaluation_results)

            if total == 0:
                error_msg = "No examples were successfully processed"
                logger.error(error_msg)
                with open(log_file_path, 'a') as f:
                    f.write(f"ERROR: {error_msg}\n")
                return jsonify({'error': error_msg}), 500

            initial_metrics = {
                "avg_score": total_score / total if total > 0 else 0,
                "perfect_matches": perfect_matches,
                "total_examples": total,
                "perfect_match_percent": (perfect_matches / total * 100) if total > 0 else 0
            }

            logger.info(f"Initial metrics: avg_score={initial_metrics['avg_score']:.4f}, "
                       f"perfect_matches={perfect_matches}/{total}")
        except Exception as e:
            error_msg = f"Error calculating metrics: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_msg}\n{traceback.format_exc()}\n")
            return jsonify({'error': error_msg}), 500

        # Save the initial iteration with detailed logging
        logger.info("STEP 5: Saving initial iteration")
        try:
            experiment_tracker.save_iteration(
                experiment_id=experiment_id,
                iteration=iteration,
                system_prompt=system_prompt,
                output_prompt=output_prompt,
                metrics=initial_metrics,
                examples=evaluation_results
            )
            logger.info(f"Successfully saved iteration {iteration} for experiment {experiment_id}")
        except Exception as e:
            error_msg = f"Error saving initial iteration: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_msg}\n{traceback.format_exc()}\n")
            # Continue despite error to try to complete the process

        # Step 6: Optimize prompts if there's room for improvement
        logger.info("STEP 6: Optimizing prompts")
        try:
            if initial_metrics["perfect_match_percent"] < 100:
                logger.info(f"Room for improvement: {initial_metrics['perfect_match_percent']:.2f}% perfect matches")

                # Load optimizer prompt if not provided
                if not optimizer_prompt:
                    logger.info(f"Loading optimizer prompt with strategy: {optimizer_strategy}")
                    optimizer_prompt = load_optimizer_prompt(optimizer_strategy)
                    logger.info(f"Loaded optimizer prompt: {len(optimizer_prompt)} chars")

                # Call optimizer with detailed logging
                logger.info("Calling optimize_prompts function")
                logger.info(f"Using optimization strategy: {optimizer_strategy}")
                optimization_result = optimize_prompts(
                    system_prompt,
                    output_prompt,
                    evaluation_results,
                    optimizer_prompt,
                    optimizer_strategy
                )

                # Extract optimization results
                new_system_prompt = optimization_result.get('system_prompt', system_prompt)
                new_output_prompt = optimization_result.get('output_prompt', output_prompt)
                reasoning = optimization_result.get('reasoning', '')

                logger.info("Optimization complete")
                logger.info(f"New system prompt length: {len(new_system_prompt)} chars")
                logger.info(f"New output prompt length: {len(new_output_prompt)} chars")

                # Step 7: Evaluate the optimized prompts
                logger.info("STEP 7: Evaluating optimized prompts")
                optimized_results = []
                opt_success_count = 0
                opt_error_count = 0

                for i, example in enumerate(examples):
                    user_input = example.get('user_input', '')
                    ground_truth = example.get('ground_truth_output', '')

                    if not user_input:
                        continue

                    logger.info(f"Processing example {i+1}/{len(examples)} with optimized prompts")
                    try:
                        # Call LLM with optimized prompts
                        model_response = get_llm_response(
                            new_system_prompt, 
                            user_input, 
                            new_output_prompt,
                            config.get('gemini', {})
                        )

                        # Calculate score
                        score = calculate_score(model_response, ground_truth)

                        # Record result
                        optimized_results.append({
                            'user_input': user_input,
                            'ground_truth_output': ground_truth,
                            'model_response': model_response,
                            'score': score
                        })

                        logger.info(f"Optimized example {i+1} processed with score: {score:.4f}")
                        opt_success_count += 1

                    except Exception as e:
                        error_msg = f"Error processing optimized example {i+1}: {str(e)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        with open(log_file_path, 'a') as f:
                            f.write(f"ERROR: {error_msg}\n{traceback.format_exc()}\n")

                        optimized_results.append({
                            'user_input': user_input,
                            'ground_truth_output': ground_truth,
                            'model_response': f"Error: {str(e)}",
                            'score': 0
                        })
                        opt_error_count += 1

                logger.info(f"Optimized evaluation complete: {opt_success_count} successful, {opt_error_count} errors")

                # Calculate new metrics
                logger.info("STEP 8: Calculating optimized metrics")
                new_total_score = sum(r.get('score', 0) for r in optimized_results)
                new_perfect_matches = sum(1 for r in optimized_results if r.get('score', 0) >= 0.9)
                new_total = len(optimized_results)

                optimized_metrics = {
                    "avg_score": new_total_score / new_total if new_total > 0 else 0,
                    "perfect_matches": new_perfect_matches,
                    "total_examples": new_total,
                    "perfect_match_percent": (new_perfect_matches / new_total * 100) if new_total > 0 else 0
                }

                logger.info(f"Optimized metrics: avg_score={optimized_metrics['avg_score']:.4f}, "
                           f"perfect_matches={new_perfect_matches}/{new_total}")

                # Compare metrics
                improvement = optimized_metrics["avg_score"] - initial_metrics["avg_score"]
                logger.info(f"Score improvement: {improvement:.4f} ({improvement*100:.2f}%)")

                # Step 9: Save the optimized iteration
                logger.info("STEP 9: Saving optimized iteration")
                try:
                    experiment_tracker.save_iteration(
                        experiment_id=experiment_id,
                        iteration=iteration + 1,
                        system_prompt=new_system_prompt,
                        output_prompt=new_output_prompt,
                        metrics=optimized_metrics,
                        examples=optimized_results,
                        optimizer_reasoning=reasoning
                    )
                    logger.info(f"Successfully saved optimized iteration {iteration+1} for experiment {experiment_id}")
                except Exception as e:
                    error_msg = f"Error saving optimized iteration: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    with open(log_file_path, 'a') as f:
                        f.write(f"ERROR: {error_msg}\n{traceback.format_exc()}\n")

                                # Step 10: Prepare the response
                logger.info("STEP 10: Preparing response")
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
                    'improvement': improvement
                }
            else:
                # No optimization needed
                logger.info("Perfect score achieved, no optimization needed")
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
        except Exception as e:
            error_msg = f"Error in optimization process: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_msg}\n{traceback.format_exc()}\n")

            # Return partial results if we at least have initial evaluation
            if 'initial_metrics' in locals():
                response = {
                    'experiment_id': experiment_id,
                    'initial_iteration': iteration,
                    'initial': {
                        'system_prompt': system_prompt,
                        'output_prompt': output_prompt,
                        'metrics': initial_metrics,
                        'results': evaluation_results
                    },
                    'error': f"Optimization failed: {str(e)}"
                }
            else:
                return jsonify({'error': error_msg}), 500

        # Log completion of training run
        logger.info(f"========== TRAINING RUN COMPLETED SUCCESSFULLY ==========")
        with open(log_file_path, 'a') as f:
            f.write(f"Training run completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment ID: {experiment_id}\n")
            if 'improvement' in response:
                f.write(f"Improvement: {response['improvement']:.4f}\n")

        return jsonify(response)
    except Exception as e:
        # Catch-all error handler with detailed traceback
        error_msg = f"Critical error in training process: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        # Try to write to log file if not already done
        try:
            with open(log_file_path, 'a') as f:
                f.write(f"CRITICAL ERROR: {error_msg}\n{traceback.format_exc()}\n")
        except:
            # If log file writing fails, just continue
            pass

        return jsonify({'error': error_msg}), 500

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
def get_experiment_details(experiment_id):
    """Get detailed information about a specific experiment."""
    tracker = ExperimentTracker()
    iterations = tracker.get_iterations(experiment_id)

    if not iterations:
        return jsonify({'error': f'Experiment {experiment_id} not found or has no iterations'})

    return jsonify({'iterations': iterations})

@app.route('/experiments/<experiment_id>/examples/<int:iteration>')
def get_experiment_examples(experiment_id, iteration):
    """Get examples for a specific iteration of an experiment."""
    try:
        # Access the experiment directory
        exp_dir = os.path.join('experiments', experiment_id)
        if not os.path.exists(exp_dir):
            return jsonify({'error': f'Experiment {experiment_id} not found'})

        # Check for examples directory
        examples_dir = os.path.join(exp_dir, 'examples')
        if not os.path.exists(examples_dir):
            return jsonify({'error': 'No examples directory found for this experiment'})

        # Find example file for the iteration
        examples_file = os.path.join(examples_dir, f'examples_{iteration}.json')
        if not os.path.exists(examples_file):
            return jsonify({'error': f'No examples found for iteration {iteration}'})

        # Load examples
        with open(examples_file, 'r') as f:
            examples = json.load(f)

        return jsonify({'examples': examples})
    except Exception as e:
        logger.error(f"Error retrieving examples: {e}")
        return jsonify({'error': f'Error retrieving examples: {str(e)}'})

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

@app.route('/reset_nejm_cache', methods=['POST'])
def reset_nejm_cache():
    """Reset the NEJM dataset cache."""
    try:
        # Clear the cached datasets if they exist
        if hasattr(app, 'nejm_train_cache'):
            delattr(app, 'nejm_train_cache')
            logger.info("Cleared NEJM training dataset cache")

        if hasattr(app, 'nejm_validation_cache'):
            delattr(app, 'nejm_validation_cache')
            logger.info("Cleared NEJM validation dataset cache")

        return jsonify({'message': 'NEJM dataset cache cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing NEJM cache: {e}")
        return jsonify({'error': f"Error clearing NEJM cache: {str(e)}"}), 500

@app.route('/regenerate_nejm_data', methods=['POST'])
def regenerate_nejm_data():
    """Regenerate NEJM datasets by running the fix_nejm_data.py script."""
    try:
        # Import and run fix_nejm_data.py to regenerate datasets
        import sys
        import importlib.util

        logger.info("Loading fix_nejm_data.py module")

        spec = importlib.util.spec_from_file_location("fix_nejm_data", "fix_nejm_data.py")

        if spec and spec.loader:
            fix_nejm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fix_nejm_module)

            logger.info("Running fix_nejm_data.fix_nejm_data()")
            fix_nejm_module.fix_nejm_data()

            logger.info("NEJM datasets regenerated successfully")
            return jsonify({'message': 'NEJM datasets regenerated successfully'})
        else:
            logger.error("Could not load fix_nejm_data module")
            return jsonify({'error': 'Could not load fix_nejm_data module'}), 500

    except Exception as e:
        logger.error(f"Error regenerating NEJM datasets: {e}")
        return jsonify({'error': f"Error regenerating NEJM datasets: {str(e)}"}), 500

@app.route('/two_stage_train', methods=['POST'])
def two_stage_train():
    """
    Run the Two-Stage Training Cycle workflow.
    Phase 1: Primary LLM Inference & Evaluation
    Phase 2: Optimizer LLM Refinement
    """
    # Create a training log file for this session
    import traceback
    from datetime import datetime
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"logs/two_stage_{log_timestamp}.log"

    # Ensure all required directories exist
    required_dirs = [
        "logs",
        "data/train",
        "data/validation",
        "experiments",
        "prompts"
    ]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensuring directory exists: {directory}")

    # Write to log file
    with open(log_file_path, 'w') as f:
        f.write(f"=== TWO-STAGE TRAINING SESSION STARTED AT {log_timestamp} ===\n")

    logger.info(f"========== TWO-STAGE TRAINING RUN STARTED AT {log_timestamp} ==========")

    try:
        # Step 0: Parse request data
        logger.info("STEP 0: Parsing and validating request data")

        if not request.json:
            error_msg = "No JSON data received in request"
            logger.error(error_msg)
            with open(log_file_path, 'w') as f:
                f.write(f"ERROR: {error_msg}\n")
            return jsonify({'error': error_msg}), 400

        data = request.json
        system_prompt = data.get('system_prompt', '')
        output_prompt = data.get('output_prompt', '')
        examples_content = data.get('examples_content', '')
        max_iterations = int(data.get('max_iterations', 1))
        batch_size = int(data.get('batch_size', 0))
        optimizer_strategy = data.get('optimizer_strategy', 'reasoning_first')
        optimizer_type = data.get('optimizer_type', 'reasoning_first')

        # Log request parameters
        logger.info(f"Request parameters:")
        logger.info(f"  - max_iterations: {max_iterations}")
        logger.info(f"  - batch_size: {batch_size}")
        logger.info(f"  - optimizer_strategy: {optimizer_strategy}")
        logger.info(f"  - optimizer_type: {optimizer_type}")
        logger.info(f"  - system_prompt length: {len(system_prompt)} chars")
        logger.info(f"  - output_prompt length: {len(output_prompt)} chars")
        logger.info(f"  - examples_content length: {len(examples_content)} chars")

        # Validate required parameters
        if not system_prompt or not output_prompt:
            error_msg = "System prompt and output prompt are required"
            logger.error(error_msg)
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_msg}\n")
            return jsonify({'error': error_msg}), 400

        # Parse examples from content
        logger.info("STEP 1: Parsing examples")
        examples = []
        try:
            if examples_content:
                examples = parse_text_examples(examples_content)
                logger.info(f"Successfully parsed {len(examples)} examples from text content")
            else:
                examples = data.get('examples', [])
                logger.info(f"Using {len(examples)} examples from direct input")
        except Exception as e:
            error_msg = f"Failed to parse examples: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_msg}\n{traceback.format_exc()}\n")
            return jsonify({'error': error_msg}), 400

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

        # Run the training cycle with enhanced error handling
        try:
            # Add memory management monitoring before training starts
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage before training: {initial_memory:.2f} MB")

            # Set timeout for the training cycle
            import signal
            from contextlib import contextmanager

            @contextmanager
            def timeout(time):
                # Register a function to raise a TimeoutError on the signal
                signal.signal(signal.SIGALRM, lambda: (_ for _ in ()).throw(TimeoutError("Training cycle timed out")))
                signal.alarm(time)
                try:
                    yield
                finally:
                    # Unregister the signal so it won't be triggered
                    signal.signal(signal.SIGALRM, signal.SIG_IGN)
                    signal.alarm(0)

            # Run with timeout protection (30 minutes max)
            with timeout(1800):  # 30 minutes in seconds
                result = prompt_workflow.run_training_cycle(
                    system_prompt=system_prompt,
                    output_prompt=output_prompt,
                    max_iterations=max_iterations,
                    early_stopping_patience=early_stopping_patience,
                    batch_size=batch_size,
                    optimizer_strategy=optimizer_strategy,
                    optimizer_type=optimizer_type
                )

                # Record final memory usage
                final_memory = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage after training: {final_memory:.2f} MB (change: {final_memory - initial_memory:.2f} MB)")

                # Force garbage collection at the end
                import gc
                gc.collect()
                logger.info("Forced garbage collection after training")

                return jsonify(result)

        except TimeoutError as e:
            error_message = "Training cycle timed out after 30 minutes. Try reducing batch size or number of iterations."
            logger.error(f"Timeout error in training: {error_message}")
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_message}\n")
            return jsonify({'error': error_message, 'type': 'timeout'}), 500

        except MemoryError as e:
            error_message = "Out of memory during training. Try reducing batch size or using shorter examples."
            logger.error(f"Memory error in training: {error_message}")
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_message}\n")
            # Force garbage collection
            gc.collect()
            return jsonify({'error': error_message, 'type': 'memory'}), 500

        except Exception as e:
            error_message = f"Error in training cycle: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            with open(log_file_path, 'a') as f:
                f.write(f"ERROR: {error_message}\n{traceback.format_exc()}\n")
            return jsonify({'error': error_message, 'type': 'general'}), 500

    except Exception as e:
        # Outer try-except to handle any errors in the error handling code itself
        logger.error(f"Critical error in two-stage training cycle: {e}")
        logger.error(traceback.format_exc())
        with open(log_file_path, 'a') as f:
            f.write(f"CRITICAL ERROR: {str(e)}\n{traceback.format_exc()}\n")
        return jsonify({'error': f"A critical error occurred: {str(e)}", 'type': 'critical'}), 500

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

@app.route('/experiments/<experiment_id>/iterations/<int:iteration>/examples')
def get_iteration_examples(experiment_id, iteration):
    """Get examples for a specific iteration of an experiment."""
    try:
        experiment_tracker = ExperimentTracker()
        exp_dir = os.path.join(experiment_tracker.base_dir, experiment_id)

        if not os.path.exists(exp_dir):
            return jsonify({'error': f'Experiment {experiment_id} not found'}), 404

        # Look for examples file
        examples_dir = os.path.join(exp_dir, 'examples')
        examples_file = os.path.join(examples_dir, f"examples_{iteration}.json")

        if not os.path.exists(examples_file):
            return jsonify({'error': f'No examples found for iteration {iteration}'}), 404

        # Load examples
        with open(examples_file, 'r') as f:
            examples = json.load(f)

        return jsonify({'examples': examples})
    except Exception as e:
        logger.error(f"Error loading examples: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)