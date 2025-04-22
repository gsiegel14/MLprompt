import os
import json
import logging
from app.llm_client import get_llm_response
from app.evaluator import calculate_score, evaluate_batch
from app.optimizer import optimize_prompts, load_optimizer_prompt
from app.experiment_tracker import ExperimentTracker
from app.data_module import DataModule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_integration():
    """Test the connection to the Google Gemini API."""
    logger.info("Testing API integration...")
    
    # Simple test prompt
    system_prompt = "You are a helpful assistant that provides concise answers."
    user_input = "What is the capital of France?"
    output_prompt = "Provide a one-word answer."
    
    try:
        response = get_llm_response(system_prompt, user_input, output_prompt)
        logger.info(f"API Integration Test - Response: {response}")
        return True, response
    except Exception as e:
        logger.error(f"API Integration Test - Error: {e}")
        return False, str(e)

def test_evaluation():
    """Test the evaluation functionality."""
    logger.info("Testing evaluation functionality...")
    
    # Test example
    model_response = "Paris"
    ground_truth = "Paris"
    
    score = calculate_score(model_response, ground_truth)
    logger.info(f"Evaluation Test - Score: {score} for response '{model_response}' against ground truth '{ground_truth}'")
    
    # Test batch evaluation
    examples = [
        {"model_response": "Paris", "ground_truth_output": "Paris"},
        {"model_response": "France", "ground_truth_output": "Paris"},
        {"model_response": "Paris, the capital of France", "ground_truth_output": "Paris"}
    ]
    
    metrics = evaluate_batch(examples)
    logger.info(f"Batch Evaluation Test - Metrics: {metrics}")
    
    return True, metrics

def test_optimization():
    """Test the optimization functionality."""
    logger.info("Testing optimization functionality...")
    
    # Setup test data
    system_prompt = "You are a helpful assistant that provides geographical information."
    output_prompt = "Answer the question concisely."
    
    examples = [
        {
            "user_input": "What is the capital of France?",
            "ground_truth_output": "Paris",
            "model_response": "The capital of France is Paris, which is also the largest city in the country.",
            "score": 0.7
        },
        {
            "user_input": "What is the capital of Italy?",
            "ground_truth_output": "Rome",
            "model_response": "Italy's capital city is Rome, known for its ancient ruins and beautiful architecture.",
            "score": 0.6
        },
        {
            "user_input": "What is the capital of Germany?",
            "ground_truth_output": "Berlin",
            "model_response": "Berlin",
            "score": 1.0
        }
    ]
    
    # Load optimizer prompt
    optimizer_prompt = load_optimizer_prompt()
    
    try:
        # Test different optimization strategies
        for strategy in ['full_rewrite', 'targeted_edit', 'example_addition']:
            logger.info(f"Testing optimization with strategy: {strategy}")
            
            result = optimize_prompts(
                system_prompt,
                output_prompt,
                examples,
                optimizer_prompt,
                strategy
            )
            
            logger.info(f"Optimization Test ({strategy}) - New System Prompt: {result['system_prompt']}")
            logger.info(f"Optimization Test ({strategy}) - New Output Prompt: {result['output_prompt']}")
            logger.info(f"Optimization Test ({strategy}) - Reasoning: {result['reasoning'][:200]}...")
        
        return True, result
    except Exception as e:
        logger.error(f"Optimization Test - Error: {e}")
        return False, str(e)

def test_experiment_tracking():
    """Test the experiment tracking functionality."""
    logger.info("Testing experiment tracking...")
    
    # Initialize experiment tracker
    experiment_tracker = ExperimentTracker()
    
    # Start a new experiment
    experiment_id = experiment_tracker.start_experiment()
    logger.info(f"Experiment Tracking Test - Created experiment: {experiment_id}")
    
    # Save iterations
    system_prompt = "You are a helpful assistant that provides geographical information."
    output_prompt = "Answer the question with only the name of the capital city."
    
    metrics = {
        "avg_score": 0.7,
        "perfect_matches": 1,
        "total_examples": 3,
        "perfect_match_percent": 33.3
    }
    
    examples = [
        {
            "user_input": "What is the capital of France?",
            "ground_truth_output": "Paris",
            "model_response": "The capital of France is Paris.",
            "score": 0.7
        },
        {
            "user_input": "What is the capital of Italy?",
            "ground_truth_output": "Rome",
            "model_response": "Rome is the capital of Italy.",
            "score": 0.6
        },
        {
            "user_input": "What is the capital of Germany?",
            "ground_truth_output": "Berlin",
            "model_response": "Berlin",
            "score": 1.0
        }
    ]
    
    # Save iteration 0
    success = experiment_tracker.save_iteration(
        experiment_id=experiment_id,
        iteration=0,
        system_prompt=system_prompt,
        output_prompt=output_prompt,
        metrics=metrics,
        examples=examples
    )
    
    logger.info(f"Experiment Tracking Test - Saved iteration 0: {success}")
    
    # Save iteration 1 with improved prompts
    improved_system_prompt = "You are a helpful assistant that provides concise geographical information about capital cities."
    improved_output_prompt = "Respond with only the name of the capital city. Do not include any additional information."
    
    improved_metrics = {
        "avg_score": 0.9,
        "perfect_matches": 2,
        "total_examples": 3,
        "perfect_match_percent": 66.7
    }
    
    success = experiment_tracker.save_iteration(
        experiment_id=experiment_id,
        iteration=1,
        system_prompt=improved_system_prompt,
        output_prompt=improved_output_prompt,
        metrics=improved_metrics,
        examples=examples,
        optimizer_reasoning="Improved the prompts to focus on conciseness and specificity."
    )
    
    logger.info(f"Experiment Tracking Test - Saved iteration 1: {success}")
    
    # Load the experiment history
    iterations = experiment_tracker.get_iterations(experiment_id)
    logger.info(f"Experiment Tracking Test - Loaded {len(iterations)} iterations for experiment {experiment_id}")
    
    return True, iterations

def test_complete_training_loop():
    """Test a complete training loop with real inputs."""
    logger.info("Testing complete training loop...")
    
    # Initialize components
    data_module = DataModule()
    experiment_tracker = ExperimentTracker()
    
    # Create test data
    examples = [
        {
            "user_input": "What is the capital of France?",
            "ground_truth_output": "Paris"
        },
        {
            "user_input": "What is the capital of Italy?",
            "ground_truth_output": "Rome"
        },
        {
            "user_input": "What is the capital of Germany?",
            "ground_truth_output": "Berlin"
        },
        {
            "user_input": "What is the capital of Spain?",
            "ground_truth_output": "Madrid"
        },
        {
            "user_input": "What is the capital of Japan?",
            "ground_truth_output": "Tokyo"
        }
    ]
    
    # Split data into train/validation
    train_examples, val_examples = data_module.split_examples(examples, 0.8)
    logger.info(f"Training Loop - Split data into {len(train_examples)} train and {len(val_examples)} validation examples")
    
    # Initial prompts
    system_prompt = "You are a helpful assistant that provides geographical information."
    output_prompt = "Answer the question."
    
    # Start a new experiment
    experiment_id = experiment_tracker.start_experiment()
    logger.info(f"Training Loop - Started experiment: {experiment_id}")
    
    # Training loop
    max_iterations = 2
    current_iteration = 0
    best_score = 0
    
    while current_iteration < max_iterations:
        logger.info(f"Training Loop - Starting iteration {current_iteration}")
        
        # Step 1: Evaluate current prompts on training examples
        evaluation_results = []
        for example in train_examples:
            user_input = example.get('user_input', '')
            ground_truth = example.get('ground_truth_output', '')
            
            try:
                # Call Gemini API
                model_response = get_llm_response(
                    system_prompt, 
                    user_input, 
                    output_prompt
                )
                
                # Calculate score
                score = calculate_score(model_response, ground_truth)
                
                evaluation_results.append({
                    'user_input': user_input,
                    'ground_truth_output': ground_truth,
                    'model_response': model_response,
                    'score': score
                })
                
                logger.info(f"Training Loop - Example: '{user_input}' -> '{model_response}' (Expected: '{ground_truth}', Score: {score:.2f})")
            except Exception as e:
                logger.error(f"Training Loop - Error evaluating example: {e}")
        
        # Calculate metrics
        metrics = evaluate_batch(evaluation_results)
        logger.info(f"Training Loop - Iteration {current_iteration} metrics: {metrics}")
        
        # Save the iteration
        experiment_tracker.save_iteration(
            experiment_id=experiment_id,
            iteration=current_iteration,
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            metrics=metrics,
            examples=evaluation_results
        )
        
        # Step 2: Optimize prompts if there's room for improvement
        if metrics["perfect_match_percent"] < 100:
            logger.info(f"Training Loop - Optimizing prompts for iteration {current_iteration}")
            
            optimizer_prompt = load_optimizer_prompt()
            
            optimization_result = optimize_prompts(
                system_prompt,
                output_prompt,
                evaluation_results,
                optimizer_prompt,
                'full_rewrite'  # Using full rewrite strategy
            )
            
            new_system_prompt = optimization_result.get('system_prompt', system_prompt)
            new_output_prompt = optimization_result.get('output_prompt', output_prompt)
            reasoning = optimization_result.get('reasoning', '')
            
            logger.info(f"Training Loop - New system prompt: {new_system_prompt}")
            logger.info(f"Training Loop - New output prompt: {new_output_prompt}")
            logger.info(f"Training Loop - Optimization reasoning: {reasoning[:200]}...")
            
            # Step 3: Evaluate the optimized prompts on validation set
            validation_results = []
            for example in val_examples:
                user_input = example.get('user_input', '')
                ground_truth = example.get('ground_truth_output', '')
                
                try:
                    model_response = get_llm_response(
                        new_system_prompt, 
                        user_input, 
                        new_output_prompt
                    )
                    
                    score = calculate_score(model_response, ground_truth)
                    
                    validation_results.append({
                        'user_input': user_input,
                        'ground_truth_output': ground_truth,
                        'model_response': model_response,
                        'score': score
                    })
                    
                    logger.info(f"Training Loop - Validation: '{user_input}' -> '{model_response}' (Expected: '{ground_truth}', Score: {score:.2f})")
                except Exception as e:
                    logger.error(f"Training Loop - Error validating example: {e}")
            
            # Calculate validation metrics
            validation_metrics = evaluate_batch(validation_results)
            logger.info(f"Training Loop - Validation metrics: {validation_metrics}")
            
            # Accept improvements if validation score is better
            if validation_metrics["avg_score"] > best_score:
                logger.info(f"Training Loop - Accepting improved prompts (validation score: {validation_metrics['avg_score']:.2f} > {best_score:.2f})")
                
                system_prompt = new_system_prompt
                output_prompt = new_output_prompt
                best_score = validation_metrics["avg_score"]
                
                # Save the optimized iteration
                experiment_tracker.save_iteration(
                    experiment_id=experiment_id,
                    iteration=current_iteration + 1,
                    system_prompt=system_prompt,
                    output_prompt=output_prompt,
                    metrics=validation_metrics,
                    examples=validation_results,
                    optimizer_reasoning=reasoning
                )
            else:
                logger.info(f"Training Loop - Rejecting prompt changes (validation score: {validation_metrics['avg_score']:.2f} <= {best_score:.2f})")
        else:
            logger.info(f"Training Loop - Perfect score achieved, no optimization needed")
        
        # Increment iteration
        current_iteration += 1
    
    # Training complete
    logger.info(f"Training Loop - Training complete after {current_iteration} iterations")
    logger.info(f"Training Loop - Final system prompt: {system_prompt}")
    logger.info(f"Training Loop - Final output prompt: {output_prompt}")
    
    # Get the experiment history
    iterations = experiment_tracker.get_iterations(experiment_id)
    logger.info(f"Training Loop - Experiment history: {len(iterations)} iterations")
    
    return True, {
        "experiment_id": experiment_id,
        "iterations": len(iterations),
        "final_system_prompt": system_prompt,
        "final_output_prompt": output_prompt,
        "best_score": best_score
    }

def run_tests():
    """Run all tests."""
    test_results = {}
    
    # Test API integration
    success, result = test_api_integration()
    test_results["api_integration"] = {"success": success, "result": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)}
    
    if not success:
        logger.error("API integration test failed, stopping further tests")
        return test_results
    
    # Test evaluation
    success, result = test_evaluation()
    test_results["evaluation"] = {"success": success, "result": result}
    
    # Test optimization
    success, result = test_optimization()
    test_results["optimization"] = {"success": success, "result": "Optimization results (truncated)" if success else str(result)}
    
    # Test experiment tracking
    success, result = test_experiment_tracking()
    test_results["experiment_tracking"] = {"success": success, "result": f"{len(result)} iterations tracked" if success else str(result)}
    
    # Test complete training loop
    success, result = test_complete_training_loop()
    test_results["complete_training_loop"] = {"success": success, "result": result}
    
    return test_results

if __name__ == "__main__":
    logger.info("Starting backend tests...")
    results = run_tests()
    
    # Print summary
    logger.info("==========================================")
    logger.info("Test Results Summary:")
    for test_name, test_result in results.items():
        status = "✅ PASSED" if test_result["success"] else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if isinstance(test_result["result"], dict):
            for key, value in test_result["result"].items():
                logger.info(f"  - {key}: {value}")
        else:
            logger.info(f"  Result: {test_result['result']}")
    logger.info("==========================================")
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Test results saved to test_results.json")