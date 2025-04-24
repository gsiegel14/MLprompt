
#!/usr/bin/env python3
"""
Test script for running the 5-API workflow with row 2 from NEJM CSV.
Modified version of test_five_api_workflow.py targeting the specific example.
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
SIMULATION_MODE = not (GOOGLE_API_KEY and HUGGING_FACE_TOKEN)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("row2_api_workflow_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base URL for API endpoints
BASE_URL = "http://localhost:5000"

# Set the batch size to 1 to reduce memory usage
BATCH_SIZE = 1

# Configuration parameters
WAIT_TIME = 2  # Wait time between API calls (in seconds)
REQUEST_TIMEOUT = 30  # Timeout for HTTP requests (in seconds)
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 5  # Delay between retries (in seconds)
ENABLE_PARTIAL_TESTING = True  # Continue testing even if some steps fail

def load_test_example():
    """Load the row 2 test example"""
    example_path = 'data/test_validation/row2_example.json'
    if not os.path.exists(example_path):
        logger.error(f"Test example not found at {example_path}")
        logger.info("Please run test_nejm_row.py first to create the example")
        return None, None
    
    try:
        with open(example_path, 'r') as f:
            examples = json.load(f)
        
        if examples and len(examples) > 0:
            example = examples[0]
            test_input = example.get('user_input', '')
            ground_truth = example.get('ground_truth_output', '')
            
            logger.info(f"Loaded test example from row 2:")
            logger.info(f"Input: {test_input[:150]}...")
            logger.info(f"Ground truth: {ground_truth}")
            
            return test_input, ground_truth
        else:
            logger.error("No examples found in test file")
            return None, None
    except Exception as e:
        logger.error(f"Error loading test example: {e}")
        return None, None

def make_api_request(method, url, headers=None, json=None, params=None, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES):
    """Make an API request with retry logic and timeout handling"""
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
    else:
        logger.error(f"Unknown prompt type: {prompt_type}")
        return None, None

def test_api_workflow(test_input, ground_truth, system_prompt, output_prompt):
    """Test the 5-API workflow with the row 2 example"""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING 5-API WORKFLOW WITH ROW 2 FROM NEJM DATASET")
    logger.info("=" * 80)
    
    # Step 1: Initial LLM inference with Primary LLM
    logger.info("Step 1: Initial LLM inference with Primary LLM (Google Vertex API)")
    step1_data = {
        "system_prompt": system_prompt,
        "output_prompt": output_prompt,
        "user_input": test_input,
        "ground_truth": ground_truth,
        "batch_size": BATCH_SIZE,
        "step": 1
    }
    
    endpoint = f"{BASE_URL}/five_api_workflow"
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    success, result = make_api_request('post', endpoint, headers=headers, json=step1_data)
    elapsed_time = time.time() - start_time
    
    if success:
        logger.info(f"Step 1 completed successfully in {elapsed_time:.2f} seconds")
        
        # Log summarized response data
        if isinstance(result, dict) and "response" in result:
            response_text = result["response"]
            summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
            logger.info(f"Response summary: {summary}")
        
        # Save the full response to a file
        if isinstance(result, dict) and "response" in result:
            os.makedirs("row2_results", exist_ok=True)
            with open("row2_results/initial_response.txt", "w") as f:
                f.write(result["response"])
            logger.info("Saved initial response to row2_results/initial_response.txt")
    else:
        logger.error(f"Step 1 failed after {elapsed_time:.2f} seconds: {result}")
        return None

def main():
    """Main test function"""
    logger.info("\n" + "#" * 80)
    logger.info("## NEJM ROW 2 TEST SCRIPT")
    logger.info("## Testing the 5-API workflow with row 2 from NEJM dataset")
    logger.info("#" * 80 + "\n")
    
    # Load the test example
    test_input, ground_truth = load_test_example()
    if not test_input or not ground_truth:
        logger.error("Could not load test example. Please run test_nejm_row.py first.")
        return False
    
    # Load the NEJM prompts
    system_prompt, output_prompt = load_prompts("nejm")
    if not system_prompt or not output_prompt:
        logger.warning("Could not load NEJM prompts, trying base prompts")
        system_prompt, output_prompt = load_prompts("base")
        if not system_prompt or not output_prompt:
            logger.error("Could not load any prompts. Test cannot continue.")
            return False
    
    # Run the workflow with the example
    test_api_workflow(test_input, ground_truth, system_prompt, output_prompt)
    
    logger.info("\n" + "#" * 80)
    logger.info("## TEST COMPLETED")
    logger.info("#" * 80)
    return True

if __name__ == "__main__":
    main()
