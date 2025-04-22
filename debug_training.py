#!/usr/bin/env python
"""
Debug Training Process

This script diagnoses issues with the training process, including:
1. Example parsing and processing
2. Batch size limitations
3. Workflow execution
4. Progress tracking and reporting

Usage:
    python debug_training.py [--verbose] [--examples N]
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime
from pprint import pformat
import yaml

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug_training.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger("training_debug")

# Import app modules
try:
    from app.data_module import DataModule
    from app.experiment_tracker import ExperimentTracker
    from app.workflow import PromptOptimizationWorkflow
    from app.utils import parse_text_examples
    logger.info("Successfully imported app modules")
except Exception as e:
    logger.error(f"Error importing app modules: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration: {pformat(config)}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.error(traceback.format_exc())
        return {}

def diagnose_data_parsing(examples_text, detailed=False):
    """
    Diagnose issues with parsing example data.
    
    Args:
        examples_text (str): Raw examples text
        detailed (bool): Whether to print detailed diagnostics
    """
    logger.info(f"=== DIAGNOSING DATA PARSING ===")
    logger.info(f"Examples text length: {len(examples_text)} characters")
    
    # Check for common formatting issues
    line_count = len(examples_text.splitlines())
    logger.info(f"Line count: {line_count}")
    
    if detailed and line_count > 0:
        logger.debug("First few lines of examples:")
        for i, line in enumerate(examples_text.splitlines()[:5]):
            logger.debug(f"  Line {i+1}: {line[:100]}...")
    
    # Try parsing the examples
    try:
        examples = parse_text_examples(examples_text)
        logger.info(f"Successfully parsed {len(examples)} examples")
        
        if detailed and len(examples) > 0:
            logger.debug("First example parsed:")
            logger.debug(pformat(examples[0]))
        
        return examples
    except Exception as e:
        logger.error(f"Error parsing examples: {e}")
        logger.error(traceback.format_exc())
        return []

def diagnose_batch_processing(data_module, examples, batch_size=10):
    """
    Diagnose issues with batch processing.
    
    Args:
        data_module (DataModule): The data module instance
        examples (list): List of parsed examples
        batch_size (int): Batch size to test
    """
    logger.info(f"=== DIAGNOSING BATCH PROCESSING ===")
    logger.info(f"Total examples: {len(examples)}")
    logger.info(f"Testing batch size: {batch_size}")
    
    # Save examples to data module
    try:
        train_examples, validation_examples = data_module.split_examples(examples)
        logger.info(f"Split examples into {len(train_examples)} training and {len(validation_examples)} validation examples")
        
        # Try getting a batch
        batch = data_module.get_batch(batch_size=batch_size, validation=False)
        logger.info(f"Retrieved batch size: {len(batch)}")
        
        # Check for size mismatch
        if len(batch) < batch_size and len(batch) < len(train_examples):
            logger.warning(f"Batch size ({len(batch)}) is smaller than requested ({batch_size}) and available ({len(train_examples)})")
            logger.info("Checking for internal batch size limitations in workflow.py...")
            
            # Check for hardcoded limits in the workflow
            with open('app/workflow.py', 'r') as f:
                workflow_code = f.read()
                if "max_chunk_size" in workflow_code or "effective_batch_size" in workflow_code:
                    logger.warning("Possible batch size limitation found in workflow.py")
                    
                    # Look for specific lines with limits
                    for i, line in enumerate(workflow_code.splitlines()):
                        if "batch_size" in line and ("limit" in line.lower() or "max" in line.lower() or "=" in line):
                            logger.warning(f"Line {i+1}: {line.strip()}")
        
        return batch
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        logger.error(traceback.format_exc())
        return []

def diagnose_workflow_execution(workflow, system_prompt, output_prompt, batch_size=5, max_iterations=1):
    """
    Diagnose issues with workflow execution.
    
    Args:
        workflow (PromptOptimizationWorkflow): Workflow instance
        system_prompt (str): System prompt
        output_prompt (str): Output prompt
        batch_size (int): Batch size to use
        max_iterations (int): Number of iterations to run
    """
    logger.info(f"=== DIAGNOSING WORKFLOW EXECUTION ===")
    logger.info(f"System prompt length: {len(system_prompt)}")
    logger.info(f"Output prompt length: {len(output_prompt)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max iterations: {max_iterations}")
    
    try:
        result = workflow.run_training_cycle(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            max_iterations=max_iterations,
            batch_size=batch_size,
            optimizer_strategy="reasoning_first"
        )
        
        logger.info(f"Workflow execution completed successfully")
        logger.info(f"Result: {pformat(result)}")
        return result
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        logger.error(traceback.format_exc())
        return None

def diagnose_experiment_history(experiment_tracker, experiment_id):
    """
    Diagnose issues with experiment history.
    
    Args:
        experiment_tracker (ExperimentTracker): Experiment tracker instance
        experiment_id (str): Experiment ID to check
    """
    logger.info(f"=== DIAGNOSING EXPERIMENT HISTORY ===")
    logger.info(f"Checking experiment: {experiment_id}")
    
    try:
        # Check if experiment directory exists
        exp_dir = os.path.join(experiment_tracker.base_dir, experiment_id)
        if not os.path.exists(exp_dir):
            logger.error(f"Experiment directory does not exist: {exp_dir}")
            return None
        
        logger.info(f"Experiment directory exists: {exp_dir}")
        
        # Check iteration files
        iterations = []
        for i in range(10):  # Check up to 10 iterations
            iter_file = os.path.join(exp_dir, f"iteration_{i}.json")
            if os.path.exists(iter_file):
                try:
                    with open(iter_file, 'r') as f:
                        data = json.load(f)
                    logger.info(f"Found iteration file: {iter_file}")
                    iterations.append(data)
                except Exception as e:
                    logger.error(f"Error reading iteration file {iter_file}: {e}")
        
        logger.info(f"Found {len(iterations)} iteration files")
        
        # Get full history through API
        history = experiment_tracker.get_iterations(experiment_id)
        logger.info(f"Got {len(history)} iterations from tracker API")
        
        return history
    except Exception as e:
        logger.error(f"Error checking experiment history: {e}")
        logger.error(traceback.format_exc())
        return None

def diagnose_ui_reporting():
    """
    Diagnose issues with UI reporting by checking frontend code.
    """
    logger.info(f"=== DIAGNOSING UI REPORTING ===")
    
    try:
        with open('app/static/training.js', 'r') as f:
            js_code = f.read()
        
        # Check for progress reporting functions
        if "updateTrainingProgress" in js_code:
            logger.info("Found updateTrainingProgress function")
        else:
            logger.error("updateTrainingProgress function not found")
        
        # Check for history updating functions
        if "updateChart" in js_code:
            logger.info("Found updateChart function")
        else:
            logger.error("updateChart function not found")
        
        # Check for error handling
        if "catch(error =>" in js_code:
            logger.info("Found error handling code")
        else:
            logger.error("Error handling code not found")
        
        # Check response processing
        progress_update_snippets = []
        for i, line in enumerate(js_code.splitlines()):
            if "updateTrainingProgress" in line:
                start = max(0, i-5)
                end = min(len(js_code.splitlines()), i+5)
                snippet = "\n".join(js_code.splitlines()[start:end])
                progress_update_snippets.append((i+1, snippet))
        
        if progress_update_snippets:
            logger.info(f"Found {len(progress_update_snippets)} progress update snippets:")
            for line_num, snippet in progress_update_snippets[:2]:  # Show first 2
                logger.info(f"Line {line_num}:\n{snippet}\n---")
        
        return True
    except Exception as e:
        logger.error(f"Error analyzing UI reporting: {e}")
        logger.error(traceback.format_exc())
        return False

def count_examples_in_dataset():
    """Count examples in all dataset files."""
    logger.info(f"=== CHECKING DATASET FILES ===")
    
    dataset_files = [
        ('data/train/examples.json', 'Training examples'),
        ('data/train/current_train.json', 'Current training examples'),
        ('data/validation/examples.json', 'Validation examples'),
        ('data/validation/current_validation.json', 'Current validation examples')
    ]
    
    for file_path, description in dataset_files:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"{description}: {len(data)} examples in {file_path}")
            else:
                logger.warning(f"{description}: File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error reading {description} file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Debug training process issues")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--examples', type=int, default=10, help='Number of examples to test with')
    args = parser.parse_args()
    
    logger.info("=== TRAINING PROCESS DEBUGGING STARTED ===")
    logger.info(f"Debug script version: 1.0.0")
    logger.info(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Verbose mode: {args.verbose}")
    logger.info(f"Example count: {args.examples}")
    
    # Load config
    config = load_config()
    
    # Initialize components
    try:
        data_module = DataModule()
        experiment_tracker = ExperimentTracker()
        workflow = PromptOptimizationWorkflow(data_module, experiment_tracker, config)
        logger.info("Core components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Count examples in dataset files
    count_examples_in_dataset()
    
    # Create test examples
    logger.info("Creating test examples...")
    test_examples_text = """user_input,ground_truth_output
A 45-year-old man presents with crushing chest pain radiating to the left arm that started 1 hour ago. He has a history of hypertension and smoking. Vital signs: BP 160/95, HR 110, RR 22. ECG shows ST elevation in leads II, III, and aVF.,Acute inferior myocardial infarction
A 60-year-old woman presents with sudden onset of weakness and numbness on the left side of her face and left arm that started 2 hours ago. She has a history of atrial fibrillation and is on warfarin. Vital signs: BP 170/90, HR 80, RR 16. CT scan of the head shows no bleeding.,Ischemic stroke (cerebrovascular accident)
A 35-year-old woman presents with severe headache that began suddenly while exercising. She describes it as "the worst headache of my life." Neurological exam shows nuchal rigidity and photophobia. CT scan of the head shows subarachnoid blood.,Subarachnoid hemorrhage
A 50-year-old man presents with gradually worsening shortness of breath over the past 2 weeks. He has a history of congestive heart failure. Physical exam reveals bilateral lower extremity edema and crackles in both lung bases. BNP is elevated.,Acute exacerbation of congestive heart failure
A 25-year-old woman presents with fever, right flank pain, and dysuria for the past 2 days. Urinalysis shows pyuria and bacteriuria. CBC reveals leukocytosis.,Pyelonephritis
A 40-year-old man presents with abdominal pain that started periumbilically and moved to the right lower quadrant over 12 hours. He reports anorexia and nausea. On exam, he has rebound tenderness at McBurney's point. WBC is elevated.,Acute appendicitis
A 30-year-old woman presents with severe, colicky right flank pain radiating to the groin. She reports nausea and gross hematuria. Urinalysis shows microscopic hematuria. CT scan shows a 5mm stone in the right ureter.,Ureterolithiasis (kidney stone)
A 55-year-old man with a history of alcoholism presents with severe upper abdominal pain radiating to the back, nausea, and vomiting for the past 12 hours. On exam, he has epigastric tenderness. Labs show elevated lipase and amylase.,Acute pancreatitis
A 70-year-old woman presents with fever, productive cough with yellow sputum, and dyspnea for the past 3 days. She has a history of COPD. Chest X-ray shows a right lower lobe infiltrate.,Community-acquired pneumonia
A 65-year-old man with a history of hypertension and diabetes presents with sudden onset of severe, tearing chest pain radiating to the back. CT angiogram shows a dilated aortic root with an intimal flap.,Aortic dissection
"""
    
    # Parse test examples
    examples = diagnose_data_parsing(test_examples_text, args.verbose)
    
    # Load sample prompts for testing
    logger.info("Loading sample prompts...")
    system_prompt = """You are a highly skilled medical diagnostician with extensive clinical experience. Your task is to analyze patient cases and provide the most likely diagnosis based on the information presented."""
    
    output_prompt = """Based on the clinical information provided, determine the most likely diagnosis. Consider the patient's symptoms, medical history, physical examination findings, and any available test results. Provide only the final diagnosis without explanation."""
    
    # Test batch processing
    batch = diagnose_batch_processing(data_module, examples, batch_size=args.examples)
    
    # Test workflow execution with a small number of iterations
    if args.verbose:
        logger.info("Testing workflow execution...")
        experiment_result = diagnose_workflow_execution(
            workflow, 
            system_prompt, 
            output_prompt, 
            batch_size=2,  # Test with small batch size
            max_iterations=1  # Test with single iteration
        )
        
        # Check experiment history
        if experiment_result and 'experiment_id' in experiment_result:
            exp_id = experiment_result['experiment_id']
            diagnose_experiment_history(experiment_tracker, exp_id)
    
    # Diagnose UI reporting
    diagnose_ui_reporting()
    
    logger.info("=== TRAINING PROCESS DEBUGGING COMPLETED ===")
    logger.info("Check debug_training.log for detailed results")

if __name__ == "__main__":
    main()