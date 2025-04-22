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

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def load_prompt(filename):
    """Load a prompt from a file."""
    try:
        with open(os.path.join('prompts', filename), 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt {filename}: {e}")
        return ""

def run_small_test():
    """Run a small test with just a few examples."""
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config}")
    
    # Initialize components
    data_module = DataModule()
    experiment_tracker = ExperimentTracker()
    workflow = PromptOptimizationWorkflow(data_module, experiment_tracker, config)
    
    # Load medical prompts
    system_prompt = load_prompt('system_prompt_advanced_medical.txt')
    output_prompt = load_prompt('output_prompt_advanced_medical.txt')
    
    logger.info(f"System prompt length: {len(system_prompt)} characters")
    logger.info(f"Output prompt length: {len(output_prompt)} characters")
    
    # Load a small subset of examples for testing
    try:
        with open('data/train/examples.json', 'r') as f:
            all_examples = json.load(f)
            
        # Just use 5 examples for the test
        test_examples = all_examples[:5]
        
        # Save test examples temporarily
        os.makedirs('data/test', exist_ok=True)
        with open('data/test/examples.json', 'w') as f:
            json.dump(test_examples, f)
            
        # Save the same examples as validation to track improvement
        os.makedirs('data/test_validation', exist_ok=True)
        with open('data/test_validation/examples.json', 'w') as f:
            json.dump(test_examples, f)
            
        # Set the data module to use these test files
        data_module.train_examples = test_examples
        data_module.validation_examples = test_examples
        
        logger.info(f"Test running with {len(test_examples)} examples")
        
        # Run a single training iteration
        result = workflow.run_training_cycle(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            max_iterations=1,
            optimizer_strategy='reasoning_first',
            optimizer_type='reasoning_first'
        )
        
        logger.info(f"Training result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    logger.info("Starting small test training run...")
    result = run_small_test()
    
    if "error" in result:
        logger.error(f"Test failed: {result['error']}")
    else:
        logger.info("Test completed successfully!")
        logger.info(f"Experiment ID: {result.get('experiment_id')}")
        logger.info(f"Best score: {result.get('best_score')}")