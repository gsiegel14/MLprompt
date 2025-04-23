#!/usr/bin/env python3
"""
Quick test for the 5-step workflow architecture

This script validates the structure of the workflow without making full API calls:
1. Google Vertex API #1: Primary LLM inference (mocked)
2. Hugging Face API: First external validation (local implementation)
3. Google Vertex API #2: Optimizer LLM for prompt refinement (mocked)
4. Google Vertex API #3: Optimizer LLM reruns on original dataset (mocked)
5. Hugging Face API: Second external validation on refined outputs (local implementation)
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("architecture_test")

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import necessary modules with error handling
try:
    from app.workflow import PromptOptimizationWorkflow
    from app.data_module import DataModule
    from app.experiment_tracker import ExperimentTracker
    
    # Import the configuration
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Mock for the LLM client to avoid actual API calls
class MockLLMClient:
    @staticmethod
    def get_llm_response(*args, **kwargs):
        """Mock implementation that returns a placeholder response."""
        return "This is a mock LLM response for testing the workflow architecture."

# Mock for the HuggingFace client
class MockHuggingFaceClient:
    @staticmethod
    def evaluate_metrics(predictions, references, metrics=None):
        """Mock implementation that returns placeholder metrics."""
        if metrics is None:
            metrics = ["exact_match", "bleu"]
            
        return {metric: 0.75 for metric in metrics}
    
    @staticmethod
    def validate_api_connection():
        """Mock implementation that simulates a successful connection."""
        return True

def patch_modules():
    """Patch the necessary modules to use mocks instead of actual API calls."""
    import app.llm_client
    import app.huggingface_client
    
    # Store original functions
    original_get_llm_response = app.llm_client.get_llm_response
    original_evaluate_metrics = app.huggingface_client.evaluate_metrics
    original_validate_api_connection = app.huggingface_client.validate_api_connection
    
    # Replace with mocks
    app.llm_client.get_llm_response = MockLLMClient.get_llm_response
    app.huggingface_client.evaluate_metrics = MockHuggingFaceClient.evaluate_metrics
    app.huggingface_client.validate_api_connection = MockHuggingFaceClient.validate_api_connection
    
    logger.info("Patched modules with mock implementations")
    
    # Return a function to restore original functionality
    def restore():
        app.llm_client.get_llm_response = original_get_llm_response
        app.huggingface_client.evaluate_metrics = original_evaluate_metrics
        app.huggingface_client.validate_api_connection = original_validate_api_connection
        logger.info("Restored original module functionality")
    
    return restore

def test_workflow_architecture():
    """Test the workflow architecture with mock implementations."""
    # Initialize components
    data_module = DataModule()
    experiment_tracker = ExperimentTracker()
    workflow = PromptOptimizationWorkflow(data_module, experiment_tracker, config)
    
    # Test parameters
    test_system_prompt = "You are an expert diagnostician. Analyze the symptoms and provide a diagnosis."
    test_output_prompt = "List the most likely diagnoses based on the given symptoms."
    test_batch_size = 1  # Small batch for quick testing
    test_optimizer_strategy = "reasoning_first"
    test_hf_metrics = ["exact_match", "bleu"]
    
    # Run the workflow
    logger.info("=== TESTING 5-STEP WORKFLOW ARCHITECTURE ===")
    try:
        results = workflow.run_four_api_workflow(
            system_prompt=test_system_prompt,
            output_prompt=test_output_prompt,
            batch_size=test_batch_size,
            optimizer_strategy=test_optimizer_strategy,
            hf_metrics=test_hf_metrics
        )
        
        # Check results structure
        if results and isinstance(results, dict):
            logger.info("✓ Workflow execution completed successfully")
            logger.info(f"✓ Experiment ID: {results.get('experiment_id', 'UNKNOWN')}")
            
            # Check that all expected sections are present
            expected_keys = [
                'experiment_id', 
                'internal_metrics', 
                'huggingface_metrics',
                'prompts',
                'examples_count',
                'validation_count'
            ]
            
            missing_keys = [key for key in expected_keys if key not in results]
            if missing_keys:
                logger.warning(f"✗ Some expected keys are missing: {missing_keys}")
            else:
                logger.info("✓ All expected result keys are present")
            
            # Save results to file for inspection
            output_path = f"test_outputs/architecture_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("test_outputs", exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✓ Results saved to {output_path}")
            
            return True
        else:
            if isinstance(results, dict) and 'error' in results:
                logger.error(f"✗ Workflow execution failed: {results.get('error')}")
            else:
                logger.error("✗ Workflow execution returned invalid results")
            return False
    except Exception as e:
        import traceback
        logger.error(f"✗ Exception during workflow execution: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run the workflow architecture test."""
    # Patch modules with mocks
    restore_modules = patch_modules()
    
    try:
        # Run the test
        success = test_workflow_architecture()
        
        if success:
            logger.info("=== WORKFLOW ARCHITECTURE TEST PASSED ===")
            return 0
        else:
            logger.error("=== WORKFLOW ARCHITECTURE TEST FAILED ===")
            return 1
    finally:
        # Always restore original functionality
        restore_modules()

if __name__ == "__main__":
    sys.exit(main())