import os
import json
import logging
from app.workflow import PromptOptimizationWorkflow
from app.data_module import DataModule
from app.experiment_tracker import ExperimentTracker
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_minimal_test():
    """Run a minimal test with just a few examples and simpler prompts."""
    # Initialize components
    data_module = DataModule()
    experiment_tracker = ExperimentTracker()
    
    # Create minimal configuration
    config = {
        'gemini': {
            'model_name': 'gemini-1.5-flash',
            'temperature': 0.0,
            'max_output_tokens': 1024
        },
        'optimizer': {
            'model_name': 'gemini-1.5-flash',
            'temperature': 0.7,
            'max_output_tokens': 2048
        }
    }
    
    workflow = PromptOptimizationWorkflow(data_module, experiment_tracker, config)
    
    # Create minimal system and output prompts
    system_prompt = """You are an expert physician and diagnostician.
Your task is to provide a differential diagnosis based on the presented case."""

    output_prompt = """Analyze the case carefully and provide your differential diagnosis. 
List the possible diagnoses in order of likelihood."""
    
    try:
        # Load a small subset of examples for testing
        with open('data/train/examples.json', 'r') as f:
            all_examples = json.load(f)
            
        # Just use 3 examples for the test
        test_examples = all_examples[:3]
        
        # Set the data module to use these test files
        data_module.train_examples = test_examples
        data_module.validation_examples = test_examples
        
        logger.info(f"Test running with {len(test_examples)} examples")
        
        # Run a single training iteration
        result = workflow.run_training_cycle(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            max_iterations=1,
            optimizer_strategy='full_rewrite',  # Using simpler strategy for test
            optimizer_type='general'  # Using general optimizer for test
        )
        
        logger.info(f"Training result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    logger.info("Starting minimal test training run...")
    result = run_minimal_test()
    
    if "error" in result:
        logger.error(f"Test failed: {result['error']}")
    else:
        logger.info("Test completed successfully!")
        logger.info(f"Experiment ID: {result.get('experiment_id')}")
        logger.info(f"Best score: {result.get('best_score')}")