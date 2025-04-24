
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
import subprocess
import signal
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
BASE_URL = "http://0.0.0.0:5000"

# Global variable to store server process
server_process = None

def start_api_server():
    """Start the API server if it's not already running"""
    global server_process
    
    # Check if server is already running
    if check_server_status():
        logger.info("API server is already running")
        return True
    
    logger.info("Starting API server...")
    try:
        # Start the server as a subprocess
        server_process = subprocess.Popen(
            ["gunicorn", "--bind", "0.0.0.0:5000", "--reuse-port", "--reload", "main:app"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # To create a new process group for termination
        )
        
        # Wait for server to start (max 10 seconds)
        start_time = time.time()
        while time.time() - start_time < 10:
            if check_server_status():
                logger.info("API server started successfully")
                return True
            time.sleep(0.5)
        
        logger.error("Failed to start API server within timeout period")
        stop_api_server()
        return False
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return False

def stop_api_server():
    """Stop the API server if it was started by this script"""
    global server_process
    
    if server_process:
        logger.info("Stopping API server...")
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait(timeout=5)
            logger.info("API server stopped")
        except Exception as e:
            logger.error(f"Error stopping API server: {e}")
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
            except:
                pass
        finally:
            server_process = None

# Function to check if the server is running
def check_server_status():
    """Check if the API server is running and accessible"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("API server is running and responding to health checks")
            return True
        else:
            logger.warning(f"API server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.debug("API server is not running or not accessible")
        return False
    except Exception as e:
        logger.error(f"Error checking server status: {e}")
        return False

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
            return None, None
    else:
        logger.error(f"Unknown prompt type: {prompt_type}")
        return None, None

def test_api_workflow(test_input, ground_truth, system_prompt, output_prompt):
    """Test the 5-API workflow with the row 2 example"""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING 5-API WORKFLOW WITH ROW 2 FROM NEJM DATASET")
    logger.info("=" * 80)

    # Create results directory
    os.makedirs("row2_results", exist_ok=True)

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
    success, step1_result = make_api_request('post', endpoint, headers=headers, json=step1_data)
    elapsed_time = time.time() - start_time

    if not success:
        logger.error(f"Step 1 failed after {elapsed_time:.2f} seconds: {step1_result}")
        if not ENABLE_PARTIAL_TESTING:
            return None
        # Create fallback result for testing subsequent steps
        step1_result = {
            "response": "Fallback response for testing subsequent steps",
            "success": False
        }
    else:
        logger.info(f"Step 1 completed successfully in {elapsed_time:.2f} seconds")

        # Log summarized response data
        if isinstance(step1_result, dict) and "response" in step1_result:
            response_text = step1_result["response"]
            summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
            logger.info(f"Response summary: {summary}")

        # Save the full response to a file
        if isinstance(step1_result, dict) and "response" in step1_result:
            with open("row2_results/initial_response.txt", "w") as f:
                f.write(step1_result["response"])
            logger.info("Saved initial response to row2_results/initial_response.txt")
    
    time.sleep(WAIT_TIME)  # Wait before next API call
    
    # Step 2: External validation with Hugging Face
    logger.info("Step 2: External validation with Hugging Face API")
    step2_data = {
        "system_prompt": system_prompt,
        "output_prompt": output_prompt,
        "user_input": test_input,
        "ground_truth": ground_truth,
        "batch_size": BATCH_SIZE,
        "step": 2,
        "previous_response": step1_result.get("response", "")
    }
    
    start_time = time.time()
    success, step2_result = make_api_request('post', endpoint, headers=headers, json=step2_data)
    elapsed_time = time.time() - start_time
    
    if not success:
        logger.error(f"Step 2 failed after {elapsed_time:.2f} seconds: {step2_result}")
        if not ENABLE_PARTIAL_TESTING:
            return None
        # Create fallback result for testing subsequent steps
        step2_result = {
            "response": "Fallback evaluation result",
            "success": False
        }
    else:
        logger.info(f"Step 2 completed successfully in {elapsed_time:.2f} seconds")
        
        # Log evaluation result
        if isinstance(step2_result, dict) and "response" in step2_result:
            response_text = step2_result["response"]
            summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
            logger.info(f"Evaluation summary: {summary}")
            
            # Save evaluation result
            with open("row2_results/evaluation_result.txt", "w") as f:
                f.write(step2_result["response"])
            logger.info("Saved evaluation result to row2_results/evaluation_result.txt")
    
    time.sleep(WAIT_TIME)  # Wait before next API call
    
    # Step 3: Optimizer LLM for prompt refinement
    logger.info("Step 3: Optimizer LLM for prompt refinement (Google Vertex API)")
    
    # Load optimizer prompts if they aren't already loaded
    optimizer_system, optimizer_output = load_prompts("optimizer")
    
    step3_data = {
        "system_prompt": system_prompt,
        "output_prompt": output_prompt,
        "optimizer_system_prompt": optimizer_system,
        "optimizer_output_prompt": optimizer_output,
        "user_input": test_input,
        "ground_truth": ground_truth,
        "batch_size": BATCH_SIZE,
        "step": 3,
        "previous_response": step1_result.get("response", ""),
        "evaluation_result": step2_result.get("response", "")
    }
    
    start_time = time.time()
    success, step3_result = make_api_request('post', endpoint, headers=headers, json=step3_data)
    elapsed_time = time.time() - start_time
    
    if not success:
        logger.error(f"Step 3 failed after {elapsed_time:.2f} seconds: {step3_result}")
        if not ENABLE_PARTIAL_TESTING:
            return None
        # Create fallback result for testing subsequent steps
        step3_result = {
            "response": "Fallback optimizer result",
            "optimized_system_prompt": system_prompt,
            "optimized_output_prompt": output_prompt,
            "success": False
        }
    else:
        logger.info(f"Step 3 completed successfully in {elapsed_time:.2f} seconds")
        
        # Save optimized prompts
        if isinstance(step3_result, dict):
            optimized_system = step3_result.get("optimized_system_prompt", system_prompt)
            optimized_output = step3_result.get("optimized_output_prompt", output_prompt)
            
            with open("row2_results/optimized_system_prompt.txt", "w") as f:
                f.write(optimized_system)
            with open("row2_results/optimized_output_prompt.txt", "w") as f:
                f.write(optimized_output)
            
            logger.info("Saved optimized prompts to row2_results directory")
    
    time.sleep(WAIT_TIME)  # Wait before next API call
    
    # Step 4: Optimizer LLM reruns on original dataset
    logger.info("Step 4: Optimizer LLM reruns with optimized prompts (Google Vertex API)")
    
    optimized_system = step3_result.get("optimized_system_prompt", system_prompt)
    optimized_output = step3_result.get("optimized_output_prompt", output_prompt)
    
    step4_data = {
        "system_prompt": system_prompt,
        "output_prompt": output_prompt,
        "optimized_system_prompt": optimized_system,
        "optimized_output_prompt": optimized_output,
        "user_input": test_input,
        "ground_truth": ground_truth,
        "batch_size": BATCH_SIZE,
        "step": 4
    }
    
    start_time = time.time()
    success, step4_result = make_api_request('post', endpoint, headers=headers, json=step4_data)
    elapsed_time = time.time() - start_time
    
    if not success:
        logger.error(f"Step 4 failed after {elapsed_time:.2f} seconds: {step4_result}")
        if not ENABLE_PARTIAL_TESTING:
            return None
        # Create fallback result for testing subsequent steps
        step4_result = {
            "response": "Fallback optimized response",
            "success": False
        }
    else:
        logger.info(f"Step 4 completed successfully in {elapsed_time:.2f} seconds")
        
        # Save optimized response
        if isinstance(step4_result, dict) and "response" in step4_result:
            response_text = step4_result["response"]
            summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
            logger.info(f"Optimized response summary: {summary}")
            
            with open("row2_results/optimized_response.txt", "w") as f:
                f.write(step4_result["response"])
            logger.info("Saved optimized response to row2_results/optimized_response.txt")
    
    time.sleep(WAIT_TIME)  # Wait before next API call
    
    # Step 5: Second external validation on refined outputs
    logger.info("Step 5: Second external validation with Hugging Face API")
    
    step5_data = {
        "system_prompt": system_prompt,
        "output_prompt": output_prompt,
        "optimized_system_prompt": optimized_system,
        "optimized_output_prompt": optimized_output,
        "user_input": test_input,
        "ground_truth": ground_truth,
        "batch_size": BATCH_SIZE,
        "step": 5,
        "original_response": step1_result.get("response", ""),
        "optimized_response": step4_result.get("response", "")
    }
    
    start_time = time.time()
    success, step5_result = make_api_request('post', endpoint, headers=headers, json=step5_data)
    elapsed_time = time.time() - start_time
    
    if not success:
        logger.error(f"Step 5 failed after {elapsed_time:.2f} seconds: {step5_result}")
        if not ENABLE_PARTIAL_TESTING:
            return None
        # Create fallback result
        step5_result = {
            "response": "Fallback comparison result",
            "improvement": 0,
            "success": False
        }
    else:
        logger.info(f"Step 5 completed successfully in {elapsed_time:.2f} seconds")
        
        # Save comparison result
        if isinstance(step5_result, dict):
            comparison = step5_result.get("response", "No comparison available")
            improvement = step5_result.get("improvement", 0)
            
            with open("row2_results/comparison_result.txt", "w") as f:
                f.write(f"Comparison Result:\n{comparison}\n\nImprovement: {improvement}")
            
            logger.info(f"Improvement score: {improvement}")
            logger.info("Saved comparison result to row2_results/comparison_result.txt")
    
    # Workflow summary
    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW SUMMARY")
    logger.info("=" * 80)
    
    # Create summary of all steps
    steps = [
        ("Step 1 (Initial LLM)", step1_result.get("success", False)),
        ("Step 2 (Evaluation)", step2_result.get("success", False)),
        ("Step 3 (Optimizer)", step3_result.get("success", False)),
        ("Step 4 (Optimized LLM)", step4_result.get("success", False)),
        ("Step 5 (Comparison)", step5_result.get("success", False))
    ]
    
    steps_completed = sum(1 for _, success in steps if success)
    
    logger.info(f"Steps completed: {steps_completed}/5")
    for step_name, success in steps:
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{step_name}: {status}")
    
    # Save all results to a summary file
    with open("row2_results/workflow_summary.json", "w") as f:
        json.dump({
            "test_input": test_input[:500] + "..." if len(test_input) > 500 else test_input,
            "ground_truth": ground_truth,
            "steps_completed": steps_completed,
            "step_statuses": {name: success for name, success in steps},
            "improvement": step5_result.get("improvement", 0) if isinstance(step5_result, dict) else 0
        }, f, indent=2)
    
    logger.info(f"Complete workflow summary saved to row2_results/workflow_summary.json")
    return steps_completed == 5

def main():
    """Main test function"""
    logger.info("\n" + "#" * 80)
    logger.info("## NEJM ROW 2 TEST SCRIPT")
    logger.info("## Testing the 5-API workflow with row 2 from NEJM dataset")
    logger.info("#" * 80 + "\n")

    # Start the API server
    if not start_api_server():
        logger.error("Failed to start API server. Cannot continue with tests.")
        return False

    try:
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
        success = test_api_workflow(test_input, ground_truth, system_prompt, output_prompt)

        logger.info("\n" + "#" * 80)
        logger.info("## TEST COMPLETED")
        logger.info("#" * 80)
        return success
    finally:
        # Only stop the server if we started it
        if server_process:
            stop_api_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        # Make sure to clean up server process
        stop_api_server()
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        # Make sure to clean up server process
        stop_api_server()
