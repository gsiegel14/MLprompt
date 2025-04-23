#!/usr/bin/env python3
"""
Test Runner Script for ML Prompt Optimization Platform

This script provides options to run different test sets:
1. API tests - Tests HTTP API endpoints using requests
2. Component tests - Tests workflow components directly
3. All tests - Runs both of the above

Usage:
    python run_tests.py [--api] [--components] [--all] [--verbose]
"""

import os
import sys
import argparse
import logging
import time
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_runner")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test Runner for ML Prompt Optimization Platform")
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--api", action="store_true", help="Run API endpoint tests")
    test_group.add_argument("--components", action="store_true", help="Run workflow component tests")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    
    # Additional options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Default to --all if no test option specified
    if not (args.api or args.components or args.all):
        args.all = True
    
    return args

def run_api_tests():
    """Run the API endpoint tests."""
    logger.info("=== RUNNING API ENDPOINT TESTS ===")
    
    try:
        # Import the API tester module
        import test_api_endpoints
        
        # Run the tests
        start_time = time.time()
        result = test_api_endpoints.main()
        duration = time.time() - start_time
        
        logger.info(f"API tests completed in {duration:.2f}s with exit code {result}")
        return result == 0
    except Exception as e:
        logger.error(f"Error running API tests: {str(e)}")
        return False

def run_component_tests():
    """Run the workflow component tests."""
    logger.info("=== RUNNING WORKFLOW COMPONENT TESTS ===")
    
    try:
        # Import the component tester module
        import test_workflow_components
        
        # Run the tests
        start_time = time.time()
        result = test_workflow_components.main()
        duration = time.time() - start_time
        
        logger.info(f"Component tests completed in {duration:.2f}s with exit code {result}")
        return result == 0
    except Exception as e:
        logger.error(f"Error running component tests: {str(e)}")
        return False

def main():
    """Run the selected tests."""
    args = parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests_run": [],
        "all_passed": True
    }
    
    # Run selected tests
    if args.api or args.all:
        logger.info("Running API endpoint tests...")
        api_success = run_api_tests()
        results["tests_run"].append({"name": "api_tests", "success": api_success})
        results["all_passed"] &= api_success
    
    if args.components or args.all:
        logger.info("Running workflow component tests...")
        component_success = run_component_tests()
        results["tests_run"].append({"name": "component_tests", "success": component_success})
        results["all_passed"] &= component_success
    
    # Generate summary
    logger.info("=== TEST SUMMARY ===")
    for test in results["tests_run"]:
        status = "PASSED" if test["success"] else "FAILED"
        logger.info(f"{test['name']}: {status}")
    
    overall_status = "PASSED" if results["all_passed"] else "FAILED"
    logger.info(f"Overall status: {overall_status}")
    
    # Save results to file
    output_path = f"test_outputs/test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("test_outputs", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test summary saved to {output_path}")
    
    return 0 if results["all_passed"] else 1

if __name__ == "__main__":
    sys.exit(main())