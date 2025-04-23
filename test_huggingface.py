#!/usr/bin/env python3
"""
Test script for Hugging Face API integration

This script tests the Hugging Face API integration by:
1. Validating the API connection
2. Computing metrics for a simple test case
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
logger = logging.getLogger("hf_test")

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import necessary modules
try:
    from app.huggingface_client import validate_api_connection, evaluate_metrics
    logger.info("Successfully imported Hugging Face client")
except ImportError as e:
    logger.error(f"Failed to import Hugging Face client: {e}")
    sys.exit(1)

def test_huggingface_connection():
    """Test connection to the Hugging Face API."""
    try:
        validate_api_connection()
        logger.info("✓ Hugging Face API connection successful!")
        return True
    except Exception as e:
        logger.error(f"✗ Hugging Face API connection failed: {e}")
        return False

def test_huggingface_metrics():
    """Test computing metrics with the Hugging Face API."""
    # Simple test case
    predictions = ["The patient has pneumonia."]
    references = ["The patient is diagnosed with pneumonia."]
    metrics = ["exact_match", "bleu"]
    
    try:
        results = evaluate_metrics(predictions, references, metrics)
        logger.info(f"✓ Hugging Face metrics computed successfully: {results}")
        return True
    except Exception as e:
        logger.error(f"✗ Hugging Face metrics computation failed: {e}")
        return False

def main():
    """Run the Hugging Face integration tests."""
    logger.info("=== HUGGING FACE INTEGRATION TEST ===")
    
    # Test connection
    connection_success = test_huggingface_connection()
    
    # If connection successful, test metrics
    if connection_success:
        metrics_success = test_huggingface_metrics()
    else:
        metrics_success = False
    
    # Overall success
    if connection_success and metrics_success:
        logger.info("✓ All Hugging Face integration tests passed!")
        return 0
    else:
        logger.error("✗ Some Hugging Face integration tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())