#!/usr/bin/env python3
"""
Comprehensive Debugging Script for Prompt Engineering Platform

This script tests each component of the platform with a small set of 10 test cases.
It verifies functionality of:
1. Data loading and preprocessing
2. LLM connections (Gemini API)
3. Prompt evaluation and metrics calculation
4. Prompt optimization
5. Backend API endpoints
6. Experiment tracking and history management
7. Memory usage monitoring
8. NEJM data loading and caching

Usage:
    python debug_platform.py [--verbose] [--fix-nejm] [--test-api-only]
"""

import argparse
import gc
import json
import logging
import os
import psutil
import random
import sys
import time
import traceback
from datetime import datetime
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_platform.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("debug")

# Test data - 10 sample medical cases to use for testing
# Format: [user_input, ground_truth_output]
TEST_CASES = [
    {
        "user_input": "A 45-year-old man presents with severe chest pain that radiates to his left arm and jaw. The pain started suddenly while he was exercising.",
        "ground_truth_output": "Acute myocardial infarction (heart attack)"
    },
    {
        "user_input": "A 35-year-old woman with progressively worsening headaches over the past month, now accompanied by early morning vomiting and blurred vision.",
        "ground_truth_output": "Brain tumor"
    },
    {
        "user_input": "A 62-year-old man with a history of smoking presents with a persistent cough, blood in sputum, weight loss, and fatigue for the past 3 months.",
        "ground_truth_output": "Lung cancer"
    },
    {
        "user_input": "A 28-year-old woman presents with a rash that started on her face and has spread to her chest and arms. She also reports joint pain and fatigue.",
        "ground_truth_output": "Systemic lupus erythematosus"
    },
    {
        "user_input": "A 50-year-old man with a history of alcohol abuse presents with yellowing of the skin and eyes, abdominal swelling, and confusion.",
        "ground_truth_output": "Liver cirrhosis"
    },
    {
        "user_input": "A 7-year-old child with a high fever, sore throat, and a bright red rash that feels like sandpaper. The rash started on the neck and chest.",
        "ground_truth_output": "Scarlet fever"
    },
    {
        "user_input": "A 68-year-old woman with gradually worsening memory problems, confusion, and difficulty performing familiar tasks.",
        "ground_truth_output": "Alzheimer's disease"
    },
    {
        "user_input": "A 40-year-old man with recurrent episodes of severe abdominal pain, especially after fatty meals, with pain radiating to the back.",
        "ground_truth_output": "Gallstones"
    },
    {
        "user_input": "A 30-year-old woman experiencing excessive thirst, frequent urination, increased hunger, and unexplained weight loss.",
        "ground_truth_output": "Type 1 diabetes mellitus"
    },
    {
        "user_input": "A 55-year-old man with progressive shortness of breath, chronic cough with sputum production, and decreased exercise tolerance over several years.",
        "ground_truth_output": "Chronic obstructive pulmonary disease (COPD)"
    }
]

# Sample prompts to use for testing
SAMPLE_SYSTEM_PROMPT = """You are an expert physician and master diagnostician who can analyze complex medical cases. 
Given a patient presentation, provide the most likely diagnosis.
Follow a step-by-step reasoning process to arrive at your conclusion.
Consider the patient's demographics, symptoms, risk factors, and clinical presentation.
If you believe you need additional information to make a diagnosis, state what tests you would order, but still provide your most likely diagnosis based on the available information."""

SAMPLE_OUTPUT_PROMPT = """Provide your final diagnosis in a clear, concise manner.
Format your answer as:
"The most likely diagnosis is: [DIAGNOSIS]"
"""

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Debug the Prompt Engineering Platform")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--fix-nejm", action="store_true", help="Run fix_nejm_data before tests")
    parser.add_argument("--test-api-only", action="store_true", help="Only test API connections")
    
    # Add options to test specific components
    parser.add_argument("--test-data", action="store_true", help="Test data module")
    parser.add_argument("--test-llm", action="store_true", help="Test LLM client")
    parser.add_argument("--test-evaluator", action="store_true", help="Test evaluator")
    parser.add_argument("--test-optimizer", action="store_true", help="Test optimizer")
    parser.add_argument("--test-tracker", action="store_true", help="Test experiment tracker")
    parser.add_argument("--test-nejm", action="store_true", help="Test NEJM data")
    parser.add_argument("--test-memory", action="store_true", help="Test memory management")
    
    return parser.parse_args()

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def create_test_data_file():
    """Create a test CSV file with the test cases."""
    filename = "debug_test_cases.csv"
    with open(filename, "w") as f:
        f.write("user_input,ground_truth_output\n")
        for case in TEST_CASES:
            f.write(f"\"{case['user_input']}\",\"{case['ground_truth_output']}\"\n")
    logger.info(f"Created test data file: {filename}")
    return filename

def import_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        logger.error(f"Could not load module: {module_name} from {file_path}")
        return None
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_api_key():
    """Verify the API key is available."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return None
    return api_key

def test_api_connection():
    """Test connection to the Gemini API."""
    logger.info("=== Testing API Connection ===")
    try:
        # Import the test_api_key module
        test_api = import_from_path("test_api_key", "./test_api_key.py")
        if not test_api:
            return False

        # Call the test function
        logger.info("Testing Gemini API connection...")
        result = test_api.test_gemini_api()
        
        if result:
            logger.info("✅ API connection successful")
            return True
        else:
            logger.error("❌ API connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ API test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_data_module():
    """Test the data_module functionality."""
    logger.info("=== Testing Data Module ===")
    try:
        # Import the data_module 
        sys.path.append("./app")
        data_module = None
        
        try:
            from app.data_module import DataModule
            data_module = DataModule()
            logger.info("Imported DataModule directly")
        except ImportError:
            # Try alternate import method
            data_module_spec = import_from_path("data_module", "./app/data_module.py")
            if data_module_spec:
                data_module = data_module_spec.DataModule()
                logger.info("Imported DataModule via spec")
            
        if not data_module:
            logger.error("❌ Failed to import DataModule")
            return False
            
        # Create test data
        test_file = create_test_data_file()
        
        # Load examples from test file
        logger.info("Loading examples from test file...")
        train_examples, val_examples = data_module.load_examples_from_csv(test_file)
        
        # Verify examples were loaded correctly
        if len(train_examples) + len(val_examples) != 10:
            logger.error(f"❌ Expected 10 examples, got {len(train_examples) + len(val_examples)}")
            return False
            
        logger.info(f"✅ Loaded {len(train_examples)} training examples and {len(val_examples)} validation examples")
        
        # Test batch creation
        logger.info("Testing batch creation...")
        batch = data_module.get_batch(batch_size=3)
        
        if len(batch) != 3:
            logger.error(f"❌ Expected batch size 3, got {len(batch)}")
            return False
            
        logger.info("✅ Batch creation successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ Data module test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False
        
def test_llm_client():
    """Test the LLM client functionality."""
    logger.info("=== Testing LLM Client ===")
    try:
        # Import the llm_client
        sys.path.append("./app")
        
        try:
            from app.llm_client import get_llm_response
            logger.info("Imported get_llm_response directly")
        except ImportError:
            # Try alternate import method
            llm_client = import_from_path("llm_client", "./app/llm_client.py")
            if not llm_client:
                logger.error("❌ Failed to import llm_client")
                return False
            get_llm_response = llm_client.get_llm_response
            logger.info("Imported get_llm_response via spec")
        
        # Load config to get the correct model name
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            model_name = config.get('gemini', {}).get('model_name', 'gemini-1.5-flash')
            logger.info(f"Using model from config: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load config.yaml, using default model: {e}")
            model_name = 'gemini-1.5-flash'
        
        # Test with a simple example
        logger.info("Testing LLM response generation...")
        user_input = TEST_CASES[0]["user_input"]
        
        # Create a custom config with the correct model name
        custom_config = {'model_name': model_name}
        
        start_time = time.time()
        response = get_llm_response(
            system_prompt=SAMPLE_SYSTEM_PROMPT,
            user_input=user_input,
            output_prompt=SAMPLE_OUTPUT_PROMPT,
            config=custom_config
        )
        end_time = time.time()
        
        if not response or len(response) < 10:
            logger.error(f"❌ LLM response too short or empty: {response}")
            return False
            
        logger.info(f"✅ LLM response generated in {end_time - start_time:.2f} seconds")
        logger.info(f"Response preview: {response[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ LLM client test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_evaluator():
    """Test the evaluator functionality."""
    logger.info("=== Testing Evaluator ===")
    try:
        # Import the evaluator
        sys.path.append("./app")
        
        try:
            from app.evaluator import calculate_score, evaluate_batch
            logger.info("Imported evaluator functions directly")
        except ImportError:
            # Try alternate import method
            evaluator = import_from_path("evaluator", "./app/evaluator.py")
            if not evaluator:
                logger.error("❌ Failed to import evaluator")
                return False
            calculate_score = evaluator.calculate_score
            evaluate_batch = evaluator.evaluate_batch
            logger.info("Imported evaluator functions via spec")
        
        # Test score calculation
        logger.info("Testing score calculation...")
        ground_truth = "The most likely diagnosis is: Acute myocardial infarction (heart attack)"
        model_response = "The most likely diagnosis is: Acute myocardial infarction"
        
        score = calculate_score(model_response, ground_truth)
        
        if score < 0 or score > 1:
            logger.error(f"❌ Invalid score: {score}, should be between 0 and 1")
            return False
            
        logger.info(f"✅ Score calculation successful: {score}")
        
        # Test batch evaluation
        logger.info("Testing batch evaluation...")
        examples = [
            {
                "ground_truth_output": "The most likely diagnosis is: Acute myocardial infarction (heart attack)",
                "model_response": "The most likely diagnosis is: Acute myocardial infarction"
            },
            {
                "ground_truth_output": "The most likely diagnosis is: Brain tumor",
                "model_response": "The most likely diagnosis is: Brain tumor"
            }
        ]
        
        metrics = evaluate_batch(examples)
        
        if "avg_score" not in metrics:
            logger.error(f"❌ Missing avg_score in metrics: {metrics}")
            return False
            
        logger.info(f"✅ Batch evaluation successful: avg_score={metrics['avg_score']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Evaluator test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_optimizer():
    """Test the optimizer functionality."""
    logger.info("=== Testing Optimizer ===")
    try:
        # Import the optimizer
        sys.path.append("./app")
        
        try:
            from app.optimizer import optimize_prompts, load_optimizer_prompt
            logger.info("Imported optimizer functions directly")
        except ImportError:
            # Try alternate import method
            optimizer = import_from_path("optimizer", "./app/optimizer.py")
            if not optimizer:
                logger.error("❌ Failed to import optimizer")
                return False
            optimize_prompts = optimizer.optimize_prompts
            load_optimizer_prompt = optimizer.load_optimizer_prompt
            logger.info("Imported optimizer functions via spec")
        
        # Test optimizer prompt loading
        logger.info("Testing optimizer prompt loading...")
        optimizer_prompt = load_optimizer_prompt("reasoning_first")
        
        if not optimizer_prompt or len(optimizer_prompt) < 100:
            logger.error(f"❌ Optimizer prompt too short or empty: {optimizer_prompt}")
            return False
            
        logger.info(f"✅ Optimizer prompt loaded successfully ({len(optimizer_prompt)} chars)")
        
        # Create examples with scores for optimization
        examples = [
            {
                "user_input": TEST_CASES[0]["user_input"],
                "ground_truth_output": TEST_CASES[0]["ground_truth_output"],
                "model_response": "The diagnosis is myocardial infarction.",
                "score": 0.7
            },
            {
                "user_input": TEST_CASES[1]["user_input"],
                "ground_truth_output": TEST_CASES[1]["ground_truth_output"],
                "model_response": "The patient has a brain tumor.",
                "score": 0.6
            }
        ]
        
        # Only test optimization if API connection works
        if test_api_connection():
            # Test prompt optimization
            logger.info("Testing prompt optimization...")
            
            start_time = time.time()
            result = optimize_prompts(
                current_system_prompt=SAMPLE_SYSTEM_PROMPT,
                current_output_prompt=SAMPLE_OUTPUT_PROMPT,
                examples=examples,
                strategy="reasoning_first"
            )
            end_time = time.time()
            
            if not result or "new_system_prompt" not in result:
                logger.error(f"❌ Missing new_system_prompt in optimization result: {result}")
                return False
                
            logger.info(f"✅ Prompt optimization successful in {end_time - start_time:.2f} seconds")
            logger.info(f"New system prompt preview: {result['new_system_prompt'][:100]}...")
        else:
            logger.warning("⚠️ Skipping optimizer test due to API connection failure")
        
        return True
    except Exception as e:
        logger.error(f"❌ Optimizer test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_experiment_tracker():
    """Test the experiment tracker functionality."""
    logger.info("=== Testing Experiment Tracker ===")
    try:
        # Import the experiment_tracker
        sys.path.append("./app")
        
        try:
            from app.experiment_tracker import ExperimentTracker
            logger.info("Imported ExperimentTracker directly")
        except ImportError:
            # Try alternate import method
            experiment_tracker = import_from_path("experiment_tracker", "./app/experiment_tracker.py")
            if not experiment_tracker:
                logger.error("❌ Failed to import experiment_tracker")
                return False
            ExperimentTracker = experiment_tracker.ExperimentTracker
            logger.info("Imported ExperimentTracker via spec")
        
        # Create a test experiment tracker
        logger.info("Testing experiment tracking...")
        tracker = ExperimentTracker(base_dir='debug_experiments')
        
        # Start an experiment
        experiment_id = tracker.start_experiment()
        
        if not experiment_id:
            logger.error("❌ Failed to start experiment")
            return False
            
        logger.info(f"✅ Started experiment: {experiment_id}")
        
        # Save an iteration
        metrics = {"avg_score": 0.75, "perfect_matches": 3, "total_examples": 5}
        examples = [
            {
                "user_input": TEST_CASES[0]["user_input"],
                "ground_truth_output": TEST_CASES[0]["ground_truth_output"],
                "model_response": "The diagnosis is myocardial infarction.",
                "score": 0.7
            }
        ]
        
        tracker.save_iteration(
            experiment_id=experiment_id,
            iteration=0,
            system_prompt=SAMPLE_SYSTEM_PROMPT,
            output_prompt=SAMPLE_OUTPUT_PROMPT,
            metrics=metrics,
            examples=examples,
            optimizer_reasoning="This is a test reasoning"
        )
        
        # Load experiments - there might be more than one
        history = tracker.load_experiment_history(experiment_id)
        
        if not history:
            logger.error("❌ No experiments found in history")
            return False
        
        # Find our experiment in the history
        found_experiment = False
        for exp in history:
            if exp.get('experiment_id') == experiment_id:
                found_experiment = True
                break
                
        if not found_experiment:
            logger.error(f"❌ Couldn't find experiment {experiment_id} in history")
            return False
            
        logger.info(f"✅ Loaded experiment history successfully with {len(history)} experiments")
        
        # Load iterations
        iterations = tracker.get_iterations(experiment_id)
        
        if not iterations or len(iterations) != 1:
            logger.error(f"❌ Expected 1 iteration, got {len(iterations) if iterations else 0}")
            return False
            
        logger.info(f"✅ Loaded {len(iterations)} iteration(s) successfully")
        
        # Clean up test directory
        import shutil
        if os.path.exists('debug_experiments'):
            shutil.rmtree('debug_experiments')
            logger.info("Cleaned up test experiments directory")
        
        return True
    except Exception as e:
        logger.error(f"❌ Experiment tracker test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_nejm_data():
    """Test NEJM data loading and processing."""
    logger.info("=== Testing NEJM Data Processing ===")
    try:
        # Import the fix_nejm_data module
        fix_nejm = import_from_path("fix_nejm_data", "./fix_nejm_data.py")
        if not fix_nejm:
            logger.error("❌ Failed to import fix_nejm_data")
            return False
            
        # Check if NEJM CSV file exists
        csv_path = 'attached_assets/NEJM 160 Validation Database - NEJM 181.csv'
        if not os.path.exists(csv_path):
            logger.error(f"❌ NEJM CSV file not found: {csv_path}")
            return False
            
        logger.info(f"✅ NEJM CSV file found: {csv_path}")
        
        # Test fix_nejm_data function
        logger.info("Testing fix_nejm_data function...")
        
        try:
            fix_nejm.fix_nejm_data(csv_path)
            logger.info("✅ fix_nejm_data function executed successfully")
        except Exception as e:
            logger.error(f"❌ fix_nejm_data function failed: {e}")
            logger.error(traceback.format_exc())
            return False
            
        # Check if the data files were created - they could be in either location
        possible_train_paths = [
            'data/nejm_train.json',
            'data/train/nejm_train.json',
            'data/train/examples.json'
        ]
        
        possible_validation_paths = [
            'data/nejm_validation.json',
            'data/validation/nejm_validation.json',
            'data/validation/examples.json'
        ]
        
        train_path = None
        for path in possible_train_paths:
            if os.path.exists(path):
                train_path = path
                logger.info(f"✅ Found NEJM training data file: {train_path}")
                break
                
        validation_path = None
        for path in possible_validation_paths:
            if os.path.exists(path):
                validation_path = path
                logger.info(f"✅ Found NEJM validation data file: {validation_path}")
                break
        
        if not train_path:
            logger.error("❌ NEJM training data file not found in any of the expected locations")
            logger.error(f"Checked: {possible_train_paths}")
            return False
            
        if not validation_path:
            logger.error("❌ NEJM validation data file not found in any of the expected locations")
            logger.error(f"Checked: {possible_validation_paths}")
            return False
            
        # Load and verify data files
        try:
            with open(train_path, 'r') as f:
                train_data = json.load(f)
                
            with open(validation_path, 'r') as f:
                validation_data = json.load(f)
                
            logger.info(f"✅ Loaded {len(train_data)} training examples and {len(validation_data)} validation examples")
            
            # Check the expected counts
            expected_train = 127
            expected_validation = 32
            
            if len(train_data) != expected_train:
                logger.warning(f"⚠️ Expected {expected_train} training examples, got {len(train_data)}")
            
            if len(validation_data) != expected_validation:
                logger.warning(f"⚠️ Expected {expected_validation} validation examples, got {len(validation_data)}")
                
            # Verify example format
            if train_data and not all(key in train_data[0] for key in ['user_input', 'ground_truth_output']):
                logger.error("❌ Training data has incorrect format")
                return False
                
            if validation_data and not all(key in validation_data[0] for key in ['user_input', 'ground_truth_output']):
                logger.error("❌ Validation data has incorrect format")
                return False
                
            logger.info("✅ NEJM data format verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load NEJM data files: {e}")
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"❌ NEJM data test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_memory_management():
    """Test memory management during processing."""
    logger.info("=== Testing Memory Management ===")
    try:
        # Import workflow module if available
        sys.path.append("./app")
        workflow_module = None
        
        try:
            from app.workflow import PromptWorkflow
            workflow_module = PromptWorkflow
            logger.info("Imported PromptWorkflow directly")
        except ImportError:
            try:
                workflow = import_from_path("workflow", "./app/workflow.py")
                if workflow:
                    workflow_module = workflow.PromptWorkflow
                    logger.info("Imported PromptWorkflow via spec")
            except:
                logger.warning("⚠️ PromptWorkflow not found, skipping some memory tests")
                
        # Track initial memory usage
        log_memory_usage()
        
        # Create a large dataset
        large_data = [TEST_CASES[0].copy() for _ in range(100)]
        logger.info(f"Created test dataset with {len(large_data)} examples")
        
        # Track memory after dataset creation
        log_memory_usage()
        
        # Perform garbage collection
        gc.collect()
        logger.info("Performed garbage collection")
        
        # Track memory after garbage collection
        log_memory_usage()
        
        # Test batch processing if workflow module is available
        if workflow_module:
            # Create a workflow instance
            workflow = workflow_module()
            
            # Process in batches
            batch_size = 10
            num_batches = len(large_data) // batch_size
            
            logger.info(f"Processing {num_batches} batches of size {batch_size}...")
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch = large_data[start_idx:end_idx]
                
                # Simulate batch processing
                logger.info(f"Processing batch {i+1}/{num_batches}...")
                time.sleep(0.1)  # Simulate processing time
                
                # Track memory after each batch
                log_memory_usage()
                
                # Force garbage collection between batches
                if i % 2 == 0:
                    gc.collect()
                    logger.info(f"Performed garbage collection after batch {i+1}")
                    log_memory_usage()
        
        logger.info("✅ Memory management test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory management test failed with error: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    logger.info("=== Starting Debugging Script ===")
    logger.info(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    
    # Check if necessary environment variables are set
    if not get_api_key():
        logger.error("GOOGLE_API_KEY not set. Set this environment variable before running tests.")
        return
    
    # Run fix_nejm_data if requested
    if args.fix_nejm:
        logger.info("Running fix_nejm_data...")
        try:
            fix_nejm = import_from_path("fix_nejm_data", "./fix_nejm_data.py")
            if fix_nejm:
                fix_nejm.fix_nejm_data()
                logger.info("✅ fix_nejm_data completed successfully")
            else:
                logger.error("❌ Failed to import fix_nejm_data")
        except Exception as e:
            logger.error(f"❌ fix_nejm_data failed with error: {e}")
            logger.error(traceback.format_exc())
    
    # Keep track of test results
    results = {}
    
    # Test API connection first
    api_ok = test_api_connection()
    results["API Connection"] = api_ok
    
    # If only testing API, stop here
    if args.test_api_only:
        logger.info("=== API Test Only Mode ===")
        logger.info(f"API Connection: {'✅ PASS' if api_ok else '❌ FAIL'}")
        return
    
    # Check if specific components are to be tested
    specific_tests = any([
        args.test_data, args.test_llm, args.test_evaluator,
        args.test_optimizer, args.test_tracker, args.test_nejm,
        args.test_memory
    ])
    
    # Run tests based on command line options
    try:
        # Test data module
        if not specific_tests or args.test_data:
            data_ok = test_data_module()
            results["Data Module"] = data_ok
        
        # Test LLM client
        if not specific_tests or args.test_llm:
            llm_ok = test_llm_client()
            results["LLM Client"] = llm_ok
        
        # Test evaluator
        if not specific_tests or args.test_evaluator:
            eval_ok = test_evaluator()
            results["Evaluator"] = eval_ok
        
        # Test optimizer
        if not specific_tests or args.test_optimizer:
            optimizer_ok = test_optimizer()
            results["Optimizer"] = optimizer_ok
        
        # Test experiment tracker
        if not specific_tests or args.test_tracker:
            tracker_ok = test_experiment_tracker()
            results["Experiment Tracker"] = tracker_ok
        
        # Test NEJM data
        if not specific_tests or args.test_nejm:
            nejm_ok = test_nejm_data()
            results["NEJM Data"] = nejm_ok
        
        # Test memory management
        if not specific_tests or args.test_memory:
            memory_ok = test_memory_management()
            results["Memory Management"] = memory_ok
        
    except Exception as e:
        logger.error(f"Uncaught exception during testing: {e}")
        logger.error(traceback.format_exc())
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The platform is functioning properly.")
    else:
        logger.info("\n⚠️ Some tests failed. See the log for details.")
    
    # Final memory usage
    log_memory_usage()
    
if __name__ == "__main__":
    main()