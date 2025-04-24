#!/usr/bin/env python3
"""
Comprehensive Test Script for 5-API Workflow Backend

This script thoroughly tests each stage of the 5-API workflow:
1. Load the base prompts into the API workflow
2. Upload optimizer prompts
3. Test each of the 5 steps using a sample medical case

Input: "I have a 34-year-old with a history of PE. What is the differential diagnosis?"  
Ground Truth: Mention of PE (Pulmonary Embolism)

Usage:
    python test_five_api_workflow.py
"""

import os
import json
import time
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("five_api_workflow_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base URL for API endpoints
BASE_URL = "http://localhost:5000"  # Change if running on a different port

# Test case input and ground truth
TEST_INPUT = "I have a 34-year-old with a history of PE. What is the differential diagnosis?"
GROUND_TRUTH = "Mention of PE"

# Wait time between API calls (in seconds)
WAIT_TIME = 2

def log_step(message):
    """Print a nicely formatted step message"""
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP: {message}")
    logger.info("=" * 80)

def load_prompts(prompt_type):
    """Load prompt files from disk"""
    if prompt_type == "base":
        # Load base prompts
        try:
            with open("prompts/Base Prompts/Base_system_message.md", "r") as f:
                system_prompt = f.read()
            with open("prompts/Base Prompts/Base_output_prompt.md", "r") as f:
                output_prompt = f.read()
            return system_prompt, output_prompt
        except FileNotFoundError as e:
            logger.error(f"Error loading base prompts: {e}")
            return None, None
    elif prompt_type == "nejm":
        # Load NEJM prompts
        try:
            with open("prompts/nejm_system_prompt.txt", "r") as f:
                system_prompt = f.read()
            with open("prompts/nejm_output_prompt.txt", "r") as f:
                output_prompt = f.read()
            return system_prompt, output_prompt
        except FileNotFoundError as e:
            logger.error(f"Error loading NEJM prompts: {e}")
            return None, None
    elif prompt_type == "optimizer":
        # Load optimizer prompts
        try:
            with open("prompts/optimizer_system_prompt.txt", "r") as f:
                system_prompt = f.read()
            with open("prompts/optimizer_output_prompt.txt", "r") as f:
                output_prompt = f.read()
            return system_prompt, output_prompt
        except FileNotFoundError as e:
            logger.error(f"Error loading optimizer prompts: {e}")
            # Use default optimizer prompts from API if files not found
            logger.info("Using default optimizer prompts from API")
            return None, None
    else:
        logger.error(f"Unknown prompt type: {prompt_type}")
        return None, None

def test_load_base_prompts():
    """Test loading base prompts via API"""
    log_step("Loading base prompts")
    
    # First try loading from API
    try:
        response = requests.get(f"{BASE_URL}/load_dataset?type=base_prompts")
        if response.status_code == 200:
            data = response.json()
            if "prompts" in data and "system_prompt" in data["prompts"] and "output_prompt" in data["prompts"]:
                logger.info("Successfully loaded base prompts from API")
                system_prompt = data["prompts"]["system_prompt"]
                output_prompt = data["prompts"]["output_prompt"]
                logger.info(f"System prompt length: {len(system_prompt)} characters")
                logger.info(f"Output prompt length: {len(output_prompt)} characters")
                return system_prompt, output_prompt
    except requests.RequestException as e:
        logger.error(f"Error loading base prompts from API: {e}")
    
    # If API fails, load from files
    logger.info("Loading base prompts from files")
    system_prompt, output_prompt = load_prompts("base")
    if system_prompt and output_prompt:
        logger.info("Successfully loaded base prompts from files")
        logger.info(f"System prompt length: {len(system_prompt)} characters")
        logger.info(f"Output prompt length: {len(output_prompt)} characters")
        return system_prompt, output_prompt
    
    logger.error("Failed to load base prompts")
    return None, None

def test_load_optimizer_prompts():
    """Test loading optimizer prompts"""
    log_step("Loading optimizer prompts")
    
    # First check if API has default optimizer prompts
    try:
        response = requests.get(f"{BASE_URL}/api/optimizer_prompt")
        if response.status_code == 200:
            data = response.json()
            if "system_prompt" in data and "output_prompt" in data:
                logger.info("Successfully loaded optimizer prompts from API")
                system_prompt = data["system_prompt"]
                output_prompt = data["output_prompt"]
                logger.info(f"Optimizer system prompt length: {len(system_prompt)} characters")
                logger.info(f"Optimizer output prompt length: {len(output_prompt)} characters")
                return system_prompt, output_prompt
    except requests.RequestException as e:
        logger.error(f"Error loading optimizer prompts from API: {e}")
    
    # If API fails, load from files
    logger.info("Loading optimizer prompts from files")
    system_prompt, output_prompt = load_prompts("optimizer")
    if system_prompt and output_prompt:
        # Save the optimizer prompts to the API
        try:
            save_data = {
                "system_prompt": system_prompt,
                "output_prompt": output_prompt
            }
            response = requests.post(
                f"{BASE_URL}/api/save_optimizer_prompt",
                json=save_data
            )
            if response.status_code == 200:
                logger.info("Successfully saved optimizer prompts to API")
        except requests.RequestException as e:
            logger.error(f"Error saving optimizer prompts to API: {e}")
        
        return system_prompt, output_prompt
    
    logger.error("Failed to load optimizer prompts, using defaults if available")
    return None, None

def test_api_workflow_step(step_number, data):
    """Test a specific step in the 5-API workflow"""
    log_step(f"Testing Step {step_number} of 5-API Workflow")
    
    endpoint = f"{BASE_URL}/five_api_workflow"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(endpoint, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Step {step_number} completed successfully")
            
            # Log summarized response data
            if "response" in result:
                response_text = result["response"]
                summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
                logger.info(f"Response summary: {summary}")
            
            # Check for expected fields
            expected_fields = ["success", "response"]
            missing_fields = [field for field in expected_fields if field not in result]
            if missing_fields:
                logger.warning(f"Response is missing expected fields: {missing_fields}")
            
            return result
        else:
            logger.error(f"Step {step_number} failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"Request error in step {step_number}: {e}")
        return None

def run_full_five_api_workflow(system_prompt, output_prompt):
    """Run the full 5-API workflow with all steps"""
    log_step("Running Full 5-API Workflow Test")
    
    # Pre-workflow test data
    test_data = {
        "system_prompt": system_prompt,
        "output_prompt": output_prompt,
        "user_input": TEST_INPUT,
        "ground_truth": GROUND_TRUTH
    }
    
    # Step 1: Initial LLM inference with Primary LLM
    logger.info("Step 1: Initial LLM inference with Primary LLM (Google Vertex API)")
    step1_data = test_data.copy()
    step1_data["step"] = 1
    step1_result = test_api_workflow_step(1, step1_data)
    if not step1_result:
        logger.error("Step 1 failed, cannot continue workflow")
        return False
    time.sleep(WAIT_TIME)
    
    # Step 2: External validation with Hugging Face
    logger.info("Step 2: External validation with Hugging Face API")
    step2_data = test_data.copy()
    step2_data["step"] = 2
    step2_data["previous_response"] = step1_result.get("response", "")
    step2_result = test_api_workflow_step(2, step2_data)
    if not step2_result:
        logger.error("Step 2 failed, cannot continue workflow")
        return False
    time.sleep(WAIT_TIME)
    
    # Step 3: Optimizer LLM for prompt refinement
    logger.info("Step 3: Optimizer LLM for prompt refinement (Google Vertex API)")
    step3_data = test_data.copy()
    step3_data["step"] = 3
    step3_data["previous_response"] = step1_result.get("response", "")
    step3_data["evaluation_result"] = step2_result.get("response", "")
    step3_result = test_api_workflow_step(3, step3_data)
    if not step3_result:
        logger.error("Step 3 failed, cannot continue workflow")
        return False
    time.sleep(WAIT_TIME)
    
    # Step 4: Optimizer LLM reruns on original dataset
    logger.info("Step 4: Optimizer LLM reruns on original dataset (Google Vertex API)")
    step4_data = test_data.copy()
    step4_data["step"] = 4
    step4_data["optimized_system_prompt"] = step3_result.get("optimized_system_prompt", system_prompt)
    step4_data["optimized_output_prompt"] = step3_result.get("optimized_output_prompt", output_prompt)
    step4_result = test_api_workflow_step(4, step4_data)
    if not step4_result:
        logger.error("Step 4 failed, cannot continue workflow")
        return False
    time.sleep(WAIT_TIME)
    
    # Step 5: Second external validation on refined outputs
    logger.info("Step 5: Second external validation with Hugging Face API")
    step5_data = test_data.copy()
    step5_data["step"] = 5
    step5_data["original_response"] = step1_result.get("response", "")
    step5_data["optimized_response"] = step4_result.get("response", "")
    step5_result = test_api_workflow_step(5, step5_data)
    if not step5_result:
        logger.error("Step 5 failed")
        return False
    
    # Workflow successful
    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Original Output: {step1_result.get('response', '')[:150]}...")
    logger.info(f"Optimized Output: {step4_result.get('response', '')[:150]}...")
    logger.info(f"Comparison: {step5_result.get('response', '')[:150]}...")
    
    # Final validation
    if "improvement" in step5_result:
        improvement = step5_result["improvement"]
        logger.info(f"Improvement: {improvement}")
    
    return True

def main():
    """Main test function"""
    logger.info("\n" + "#" * 80)
    logger.info("## 5-API WORKFLOW COMPREHENSIVE TEST SCRIPT")
    logger.info("## Testing all components of the 5-API workflow backend")
    logger.info(f"## Test Input: {TEST_INPUT}")
    logger.info(f"## Ground Truth: {GROUND_TRUTH}")
    logger.info("#" * 80 + "\n")
    
    # Test 1: Load Base Prompts
    system_prompt, output_prompt = test_load_base_prompts()
    if not system_prompt or not output_prompt:
        logger.error("Test failed: Could not load base prompts")
        return False
    
    # Test 2: Load Optimizer Prompts
    optimizer_system, optimizer_output = test_load_optimizer_prompts()
    if not optimizer_system or not optimizer_output:
        logger.warning("Could not load optimizer prompts, will use defaults provided by API")
    
    # Test 3: Run the full 5-API workflow
    success = run_full_five_api_workflow(system_prompt, output_prompt)
    
    if success:
        logger.info("\n" + "#" * 80)
        logger.info("## TEST COMPLETED SUCCESSFULLY")
        logger.info("#" * 80)
        return True
    else:
        logger.error("\n" + "#" * 80)
        logger.error("## TEST FAILED")
        logger.error("#" * 80)
        return False

if __name__ == "__main__":
    main()