#!/usr/bin/env python3
"""
Minimal test for the 5-step workflow architecture

This script validates the high-level structure of the workflow without executing it
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
logger = logging.getLogger("minimal_test")

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def check_workflow_structure():
    """
    Check that the workflow structure includes all 5 required steps
    by examining the source code (without executing it).
    """
    # Find the workflow.py file
    workflow_path = os.path.join('app', 'workflow.py')
    if not os.path.exists(workflow_path):
        logger.error(f"✗ Could not find workflow.py at {workflow_path}")
        return False
    
    # Read the file content
    with open(workflow_path, 'r') as f:
        content = f.read()
    
    # Check for the required phases
    required_phases = [
        "PHASE 1: Google Vertex API #1 - Primary LLM Inference",
        "PHASE 2: Hugging Face API - First External Validation",
        "PHASE 3: Google Vertex API #2 - Optimizer LLM for Prompt Refinement",
        "PHASE 4: Google Vertex API #3 - Rerun with Optimized Prompts",
        "PHASE 5: Hugging Face API - Second External Validation"
    ]
    
    missing_phases = []
    for phase in required_phases:
        if phase not in content:
            missing_phases.append(phase)
    
    if missing_phases:
        logger.error(f"✗ Missing required phases: {missing_phases}")
        return False
    else:
        logger.info("✓ All required phases are present in the workflow")
    
    # Check for the methods for each phase
    required_methods = [
        "run_four_api_workflow",        # Main workflow method
        "get_llm_response",             # For API #1 and #3
        "evaluate_metrics",             # For Hugging Face integration
        "optimize_prompts",             # For API #2
        "save_validation_results"       # For saving final results
    ]
    
    missing_methods = []
    for method in required_methods:
        if method not in content:
            missing_methods.append(method)
    
    if missing_methods:
        logger.error(f"✗ Missing required methods: {missing_methods}")
        return False
    else:
        logger.info("✓ All required methods are referenced in the workflow")
    
    # Check for necessary imports
    required_imports = [
        "from app.llm_client import get_llm_response",
        "from app.evaluator import",
        "from app.optimizer import optimize_prompts",
        "from app.huggingface_client import"
    ]
    
    missing_imports = []
    for imp in required_imports:
        if imp not in content:
            missing_imports.append(imp)
    
    if missing_imports:
        logger.error(f"✗ Missing required imports: {missing_imports}")
        return False
    else:
        logger.info("✓ All required imports are present in the workflow")
    
    return True

def check_huggingface_integration():
    """
    Check that the Hugging Face client integration is properly implemented.
    """
    huggingface_path = os.path.join('app', 'huggingface_client.py')
    if not os.path.exists(huggingface_path):
        logger.error(f"✗ Could not find huggingface_client.py at {huggingface_path}")
        return False
    
    # Read the file content
    with open(huggingface_path, 'r') as f:
        content = f.read()
    
    # Check for the required functions
    required_functions = [
        "def evaluate_metrics",
        "def validate_api_connection",
        "def compute_bleu_score",
        "def compute_exact_match"
    ]
    
    missing_functions = []
    for func in required_functions:
        if func not in content:
            missing_functions.append(func)
    
    if missing_functions:
        logger.error(f"✗ Missing required functions in Hugging Face client: {missing_functions}")
        return False
    else:
        logger.info("✓ All required functions are present in the Hugging Face client")
    
    # Check for token validation
    if "HUGGING_FACE_TOKEN" not in content:
        logger.error("✗ No reference to HUGGING_FACE_TOKEN in Hugging Face client")
        return False
    else:
        logger.info("✓ Hugging Face token reference is present")
    
    return True

def main():
    """Run the minimal workflow structure test."""
    logger.info("=== STARTING MINIMAL WORKFLOW STRUCTURE TEST ===")
    
    # Check workflow structure
    workflow_ok = check_workflow_structure()
    logger.info(f"Workflow structure check: {'PASSED' if workflow_ok else 'FAILED'}")
    
    # Check Hugging Face integration
    huggingface_ok = check_huggingface_integration()
    logger.info(f"Hugging Face integration check: {'PASSED' if huggingface_ok else 'FAILED'}")
    
    # Overall result
    if workflow_ok and huggingface_ok:
        logger.info("=== MINIMAL WORKFLOW STRUCTURE TEST PASSED ===")
        return 0
    else:
        logger.error("=== MINIMAL WORKFLOW STRUCTURE TEST FAILED ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())