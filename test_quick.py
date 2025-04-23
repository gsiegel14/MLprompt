#!/usr/bin/env python3
"""
Quick Test Script for ML Prompt Optimization Platform

This script runs quick tests for each component individually without the full workflow:
1. Tests LLM client with a single example
2. Tests evaluator with a simple comparison
3. Tests the simplified Hugging Face metrics implementation

Usage:
    python test_quick.py
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quick_test")

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Test data
TEST_SYSTEM_PROMPT = """You are an expert medical diagnostician. Analyze the presented case and provide a differential diagnosis."""
TEST_OUTPUT_PROMPT = """List the top 3 most likely diagnoses in order of probability, with brief explanations."""
TEST_USER_INPUT = """A 45-year-old male presents with sudden onset chest pain, radiating to the left arm, accompanied by sweating and shortness of breath. He has a history of hypertension and diabetes."""

def test_llm_client():
    """Test the LLM client for generating responses."""
    try:
        from app.llm_client import get_llm_response
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Testing LLM client...")
        response = get_llm_response(
            TEST_SYSTEM_PROMPT,
            TEST_USER_INPUT,
            TEST_OUTPUT_PROMPT,
            config.get('gemini', {})
        )
        
        if response and len(response) > 100:
            logger.info(f"✓ LLM client generated response of length {len(response)}")
            logger.info(f"Preview: {response[:150]}...")
            return True
        else:
            logger.error(f"✗ LLM client generated short or empty response: {response}")
            return False
    except Exception as e:
        logger.error(f"✗ LLM client test failed with exception: {str(e)}")
        return False

def test_evaluator():
    """Test the evaluator for computing metrics."""
    try:
        from app.evaluator import calculate_score
        
        logger.info("Testing evaluator...")
        
        # Test data
        model_response = "The patient is likely experiencing a heart attack due to chest pain, radiation to left arm, and risk factors."
        expected_output = "Acute Myocardial Infarction (Heart Attack)"
        
        # Calculate score
        score = calculate_score(model_response, expected_output)
        
        logger.info(f"✓ Evaluator calculated score: {score}")
        return True
    except Exception as e:
        logger.error(f"✗ Evaluator test failed with exception: {str(e)}")
        return False

def test_huggingface_local():
    """Test the local Hugging Face metrics implementation."""
    try:
        from app.huggingface_client import evaluate_metrics
        
        logger.info("Testing local Hugging Face metrics implementation...")
        
        # Test data
        predictions = ["The patient is likely experiencing a heart attack."]
        references = ["Acute Myocardial Infarction (Heart Attack)"]
        
        # Evaluate metrics
        results = evaluate_metrics(predictions, references, ["exact_match", "bleu"])
        
        logger.info(f"✓ Hugging Face metrics: {results}")
        return True
    except Exception as e:
        logger.error(f"✗ Hugging Face metrics test failed with exception: {str(e)}")
        return False

def test_optimizer_imports():
    """Test that the optimizer imports correctly."""
    try:
        from app.optimizer import load_optimizer_prompt
        
        logger.info("Testing optimizer imports...")
        
        # Load prompt
        prompt = load_optimizer_prompt('general')
        
        if prompt and len(prompt) > 100:
            logger.info(f"✓ Optimizer prompt loaded successfully (length: {len(prompt)})")
            return True
        else:
            logger.error(f"✗ Optimizer prompt is empty or too short: {prompt}")
            return False
    except Exception as e:
        logger.error(f"✗ Optimizer import test failed with exception: {str(e)}")
        return False

def main():
    """Run quick tests for key components."""
    logger.info("=== STARTING QUICK COMPONENT TESTS ===")
    
    # Track results
    results = {
        "llm_client": False,
        "evaluator": False,
        "huggingface_local": False,
        "optimizer_imports": False
    }
    
    # Run tests
    results["llm_client"] = test_llm_client()
    results["evaluator"] = test_evaluator()
    results["huggingface_local"] = test_huggingface_local()
    results["optimizer_imports"] = test_optimizer_imports()
    
    # Summary
    logger.info("=== TEST SUMMARY ===")
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    overall_status = "PASSED" if all_passed else "FAILED"
    logger.info(f"Overall status: {overall_status}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())