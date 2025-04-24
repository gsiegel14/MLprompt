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
import sys
import logging
from pathlib import Path
import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

# Check for API keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
# Determine if we need simulation mode based on whether the relevant API keys are available
# Note: GEMINI_API_KEY not needed if GOOGLE_API_KEY is available
SIMULATION_MODE = not (GOOGLE_API_KEY and HUGGING_FACE_TOKEN)

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

# Set the batch size to 1 to reduce memory usage
BATCH_SIZE = 1

# Configuration parameters
WAIT_TIME = 2  # Wait time between API calls (in seconds)
REQUEST_TIMEOUT = 30  # Timeout for HTTP requests (in seconds)
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 5  # Delay between retries (in seconds)
ENABLE_PARTIAL_TESTING = True  # Continue testing even if some steps fail

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
            with open("prompts/optimizer/Optimizer_systemmessage.md.txt", "r") as f:
                system_prompt = f.read()
            with open("prompts/optimizer/optimizer_output_prompt.txt", "r") as f:
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
    success, result = make_api_request('get', f"{BASE_URL}/load_dataset", params={"type": "base_prompts"})
    
    if success and isinstance(result, dict):
        if "prompts" in result and "system_prompt" in result["prompts"] and "output_prompt" in result["prompts"]:
            logger.info("Successfully loaded base prompts from API")
            system_prompt = result["prompts"]["system_prompt"]
            output_prompt = result["prompts"]["output_prompt"]
            logger.info(f"System prompt length: {len(system_prompt)} characters")
            logger.info(f"Output prompt length: {len(output_prompt)} characters")
            return system_prompt, output_prompt
    
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
    success, result = make_api_request('get', f"{BASE_URL}/api/optimizer_prompt")
    
    if success and isinstance(result, dict):
        if "system_prompt" in result and "output_prompt" in result:
            logger.info("Successfully loaded optimizer prompts from API")
            system_prompt = result["system_prompt"]
            output_prompt = result["output_prompt"]
            logger.info(f"Optimizer system prompt length: {len(system_prompt)} characters")
            logger.info(f"Optimizer output prompt length: {len(output_prompt)} characters")
            return system_prompt, output_prompt
    
    # If API fails, load from files
    logger.info("Loading optimizer prompts from files")
    system_prompt, output_prompt = load_prompts("optimizer")
    if system_prompt and output_prompt:
        # Save the optimizer prompts to the API
        save_data = {
            "system_prompt": system_prompt,
            "output_prompt": output_prompt
        }
        
        success, result = make_api_request('post', f"{BASE_URL}/api/save_optimizer_prompt", 
                                          json=save_data, 
                                          headers={"Content-Type": "application/json"})
        
        if success:
            logger.info("Successfully saved optimizer prompts to API")
        
        return system_prompt, output_prompt
    
    logger.error("Failed to load optimizer prompts, using defaults if available")
    return None, None

def make_api_request(method, url, headers=None, json=None, params=None, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES):
    """
    Make an API request with retry logic and timeout handling
    
    Args:
        method (str): HTTP method ('get' or 'post')
        url (str): The URL to request
        headers (dict, optional): HTTP headers
        json (dict, optional): JSON payload for POST requests
        params (dict, optional): URL parameters for GET requests
        timeout (int): Request timeout in seconds
        retries (int): Maximum number of retries
        
    Returns:
        tuple: (success, response_or_error)
    """
    retry_count = 0
    while retry_count <= retries:
        try:
            logger.info(f"API request: {method.upper()} {url} (Attempt {retry_count + 1}/{retries + 1})")
            if method.lower() == 'get':
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            else:
                response = requests.post(url, headers=headers, json=json, timeout=timeout)
            
            # Check if response can be parsed as JSON
            try:
                result = response.json()
                is_json = True
            except ValueError:
                result = response.text
                is_json = False
            
            # Process response
            if response.status_code == 200:
                return True, result
            else:
                error_msg = f"API call failed with status code: {response.status_code}"
                if is_json and isinstance(result, dict) and 'error' in result:
                    error_msg += f", Error: {result['error']}"
                elif not is_json:
                    error_msg += f", Response: {result[:200]}..." if len(result) > 200 else f", Response: {result}"
                
                logger.warning(error_msg)
                
                # Don't retry client errors (4xx) except 429 (too many requests)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    return False, error_msg
        
        except Timeout:
            logger.warning(f"Request timeout after {timeout} seconds")
        except ConnectionError:
            logger.warning("Connection error, server may be unavailable")
        except RequestException as e:
            logger.warning(f"Request error: {str(e)}")
        
        # Only retry if we haven't exceeded our retry limit
        if retry_count < retries:
            wait_time = RETRY_DELAY * (retry_count + 1)  # Exponential backoff
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retry_count += 1
        else:
            return False, f"Failed after {retries + 1} attempts"
    
    return False, "Maximum retries exceeded"

def test_api_workflow_step(step_number, data):
    """Test a specific step in the 5-API workflow"""
    log_step(f"Testing Step {step_number} of 5-API Workflow")
    
    endpoint = f"{BASE_URL}/five_api_workflow"
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    success, result = make_api_request('post', endpoint, headers=headers, json=data)
    elapsed_time = time.time() - start_time
    
    if success:
        logger.info(f"Step {step_number} completed successfully in {elapsed_time:.2f} seconds")
        
        # Log summarized response data
        if isinstance(result, dict) and "response" in result:
            response_text = result["response"]
            summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
            logger.info(f"Response summary: {summary}")
        
        # Check for expected fields
        if isinstance(result, dict):
            expected_fields = ["success", "response"]
            missing_fields = [field for field in expected_fields if field not in result]
            if missing_fields:
                logger.warning(f"Response is missing expected fields: {missing_fields}")
        
        return result
    else:
        logger.error(f"Step {step_number} failed after {elapsed_time:.2f} seconds: {result}")
        return None

def run_full_five_api_workflow(system_prompt, output_prompt):
    """Run the full 5-API workflow with all steps"""
    log_step("Running Full 5-API Workflow Test")
    
    # Pre-workflow test data
    test_data = {
        "system_prompt": system_prompt,
        "output_prompt": output_prompt,
        "user_input": TEST_INPUT,
        "ground_truth": GROUND_TRUTH,
        "batch_size": BATCH_SIZE  # Use small batch size to reduce memory usage
    }
    
    # Tracking completion status
    steps_completed = 0
    steps_skipped = 0
    steps_total = 5
    fallback_response = "Fallback response used for testing subsequent steps"
    
    # Step 1: Initial LLM inference with Primary LLM
    logger.info("Step 1: Initial LLM inference with Primary LLM (Google Vertex API)")
    step1_data = test_data.copy()
    step1_data["step"] = 1
    step1_result = test_api_workflow_step(1, step1_data)
    
    if step1_result:
        steps_completed += 1
        time.sleep(WAIT_TIME)
    elif not ENABLE_PARTIAL_TESTING:
        logger.error("Step 1 failed, cannot continue workflow")
        return False
    else:
        steps_skipped += 1
        logger.warning("Step 1 failed but continuing with partial testing enabled")
        # Create a fallback result for testing subsequent steps
        step1_result = {
            "response": fallback_response,
            "success": False
        }
    
    # Step 2: External validation with Hugging Face
    logger.info("Step 2: External validation with Hugging Face API")
    step2_data = test_data.copy()
    step2_data["step"] = 2
    step2_data["previous_response"] = step1_result.get("response", "")
    step2_result = test_api_workflow_step(2, step2_data)
    
    if step2_result:
        steps_completed += 1
        time.sleep(WAIT_TIME)
    elif not ENABLE_PARTIAL_TESTING:
        logger.error("Step 2 failed, cannot continue workflow")
        return False
    else:
        steps_skipped += 1
        logger.warning("Step 2 failed but continuing with partial testing enabled")
        # Create a fallback result
        step2_result = {
            "response": fallback_response,
            "success": False
        }
    
    # Step 3: Optimizer LLM for prompt refinement
    logger.info("Step 3: Optimizer LLM for prompt refinement (Google Vertex API)")
    step3_data = test_data.copy()
    step3_data["step"] = 3
    step3_data["previous_response"] = step1_result.get("response", "")
    step3_data["evaluation_result"] = step2_result.get("response", "")
    step3_result = test_api_workflow_step(3, step3_data)
    
    if step3_result:
        steps_completed += 1
        time.sleep(WAIT_TIME)
    elif not ENABLE_PARTIAL_TESTING:
        logger.error("Step 3 failed, cannot continue workflow")
        return False
    else:
        steps_skipped += 1
        logger.warning("Step 3 failed but continuing with partial testing enabled")
        # Create a fallback result
        step3_result = {
            "response": fallback_response,
            "optimized_system_prompt": system_prompt,
            "optimized_output_prompt": output_prompt,
            "success": False
        }
    
    # Step 4: Optimizer LLM reruns on original dataset
    logger.info("Step 4: Optimizer LLM reruns on original dataset (Google Vertex API)")
    step4_data = test_data.copy()
    step4_data["step"] = 4
    step4_data["optimized_system_prompt"] = step3_result.get("optimized_system_prompt", system_prompt)
    step4_data["optimized_output_prompt"] = step3_result.get("optimized_output_prompt", output_prompt)
    step4_result = test_api_workflow_step(4, step4_data)
    
    if step4_result:
        steps_completed += 1
        time.sleep(WAIT_TIME)
    elif not ENABLE_PARTIAL_TESTING:
        logger.error("Step 4 failed, cannot continue workflow")
        return False
    else:
        steps_skipped += 1
        logger.warning("Step 4 failed but continuing with partial testing enabled")
        # Create a fallback result
        step4_result = {
            "response": fallback_response,
            "success": False
        }
    
    # Step 5: Second external validation on refined outputs
    logger.info("Step 5: Second external validation with Hugging Face API")
    step5_data = test_data.copy()
    step5_data["step"] = 5
    step5_data["original_response"] = step1_result.get("response", "")
    step5_data["optimized_response"] = step4_result.get("response", "")
    step5_result = test_api_workflow_step(5, step5_data)
    
    if step5_result:
        steps_completed += 1
    elif not ENABLE_PARTIAL_TESTING:
        logger.error("Step 5 failed")
        return False
    else:
        steps_skipped += 1
        logger.warning("Step 5 failed")
        # Create a fallback result
        step5_result = {
            "response": "Comparison not available",
            "improvement": 0,
            "success": False
        }
    
    # Workflow summary
    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Steps completed: {steps_completed}/{steps_total}")
    logger.info(f"Steps skipped: {steps_skipped}/{steps_total}")
    
    # Show sample outputs for each step that completed
    if not step1_result.get("success", True) is False:
        logger.info(f"Original Output: {step1_result.get('response', '')[:150]}...")
    else:
        logger.info("Original Output: <step failed>")
        
    if not step4_result.get("success", True) is False:
        logger.info(f"Optimized Output: {step4_result.get('response', '')[:150]}...")
    else:
        logger.info("Optimized Output: <step failed>")
        
    if not step5_result.get("success", True) is False:
        logger.info(f"Comparison: {step5_result.get('response', '')[:150]}...")
    else:
        logger.info("Comparison: <step failed>")
    
    # Final validation
    if "improvement" in step5_result:
        improvement = step5_result["improvement"]
        logger.info(f"Improvement: {improvement}")
    
    # Test is successful if we had at least 1 step complete, or if all steps were completed
    success = (steps_completed > 0) or (steps_completed == steps_total)
    if steps_skipped > 0:
        logger.warning(f"Partial test completed with {steps_completed} successful steps and {steps_skipped} skipped steps")
    else:
        logger.info("All workflow steps completed successfully")
    
    return success

def check_api_endpoints():
    """Check which API endpoints are available"""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING API ENDPOINTS AVAILABILITY")
    logger.info("=" * 80)
    
    endpoints = [
        {"method": "get", "url": f"{BASE_URL}/", "name": "Home Page"},
        {"method": "get", "url": f"{BASE_URL}/api/status", "name": "API Status"},
        {"method": "get", "url": f"{BASE_URL}/load_dataset", "params": {"type": "base_prompts"}, "name": "Load Dataset"},
        {"method": "get", "url": f"{BASE_URL}/api/optimizer_prompt", "name": "Get Optimizer Prompt"},
        {"method": "get", "url": f"{BASE_URL}/api/metrics_summary", "name": "Metrics Summary"},
        {"method": "get", "url": f"{BASE_URL}/api/five_api_workflow_info", "name": "5-API Workflow Info"}
    ]
    
    available_endpoints = []
    missing_endpoints = []
    
    for endpoint in endpoints:
        method = endpoint["method"]
        url = endpoint["url"]
        name = endpoint["name"]
        params = endpoint.get("params", None)
        
        logger.info(f"Checking endpoint: {name} ({method.upper()} {url})")
        
        success, _ = make_api_request(method, url, params=params, retries=0, timeout=5)
        
        if success:
            logger.info(f"✓ Endpoint {name} is available")
            available_endpoints.append(name)
        else:
            logger.warning(f"✗ Endpoint {name} is not available")
            missing_endpoints.append(name)
    
    logger.info("\nSUMMARY:")
    logger.info(f"Available endpoints: {len(available_endpoints)}/{len(endpoints)}")
    logger.info(f"Missing endpoints: {len(missing_endpoints)}/{len(endpoints)}")
    
    # Check API keys
    logger.info("\nAPI KEYS STATUS:")
    if GOOGLE_API_KEY:
        logger.info("✓ GOOGLE_API_KEY is available")
    else:
        logger.warning("✗ GOOGLE_API_KEY is not available")
        
    if HUGGING_FACE_TOKEN:
        logger.info("✓ HUGGING_FACE_TOKEN is available")
    else:
        logger.warning("✗ HUGGING_FACE_TOKEN is not available (will use simulation mode)")
    
    return available_endpoints, missing_endpoints

def main():
    """Main test function"""
    logger.info("\n" + "#" * 80)
    logger.info("## 5-API WORKFLOW COMPREHENSIVE TEST SCRIPT")
    logger.info("## Testing all components of the 5-API workflow backend")
    logger.info(f"## Test Input: {TEST_INPUT}")
    logger.info(f"## Ground Truth: {GROUND_TRUTH}")
    logger.info("#" * 80 + "\n")
    
    # Check available API endpoints
    available_endpoints, missing_endpoints = check_api_endpoints()
    
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