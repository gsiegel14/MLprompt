#!/usr/bin/env python3
"""
Quick Integration Test Script for ML Prompt Optimization Platform

This script specifically tests the full 4-API workflow integration with a timeout:
1. Ensures all API endpoints are reachable
2. Tests a simple workflow run with a single example
3. Uses a timeout to prevent hanging

Usage:
    python test_integration_quick.py
"""

import os
import sys
import signal
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integration_test")

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Test data
TEST_SYSTEM_PROMPT = """You are an expert medical diagnostician. Analyze the presented case and provide a differential diagnosis."""
TEST_OUTPUT_PROMPT = """List the top 3 most likely diagnoses in order of probability, with brief explanations."""

# Set up timeout mechanism
def timeout_handler(signum, frame):
    """Handle timeout by raising an exception."""
    raise TimeoutError("Test execution timed out")

def test_workflow_integration():
    """Test the 4-API workflow integration with a single example."""
    try:
        from app.data_module import DataModule
        from app.experiment_tracker import ExperimentTracker
        from app.workflow import PromptOptimizationWorkflow
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Initializing components...")
        data_module = DataModule()
        experiment_tracker = ExperimentTracker()
        workflow = PromptOptimizationWorkflow(data_module, experiment_tracker, config)
        
        logger.info("Starting 4-API workflow test...")
        # Register timeout handler (60 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # Set 60-second alarm
        
        try:
            # Run the workflow with minimal batch size
            result = workflow.run_four_api_workflow(
                system_prompt=TEST_SYSTEM_PROMPT,
                output_prompt=TEST_OUTPUT_PROMPT,
                batch_size=1,
                optimizer_strategy="reasoning_first",
                hf_metrics=["exact_match", "bleu"]
            )
            
            # Cancel the alarm
            signal.alarm(0)
            
            if result and 'error' not in result:
                experiment_id = result.get('experiment_id', "")
                logger.info(f"✓ Workflow integration test completed successfully - experiment ID: {experiment_id}")
                logger.info(f"✓ Processed {result.get('examples_count', 0)} training examples")
                logger.info(f"✓ Processed {result.get('validation_count', 0)} validation examples")
                return True
            else:
                error = result.get('error', "Unknown error") if isinstance(result, dict) else "Invalid result structure"
                logger.error(f"✗ Workflow integration test failed: {error}")
                return False
        except TimeoutError:
            logger.error("✗ Workflow integration test timed out after 60 seconds")
            return False
        
    except Exception as e:
        logger.error(f"✗ Workflow integration test failed with exception: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run the quick integration test."""
    logger.info("=== STARTING QUICK INTEGRATION TEST ===")
    
    success = test_workflow_integration()
    
    # Summary
    logger.info("=== TEST SUMMARY ===")
    if success:
        logger.info("✓ Integration test PASSED")
        return 0
    else:
        logger.error("✗ Integration test FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())