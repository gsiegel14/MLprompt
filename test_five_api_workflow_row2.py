import logging
import json
import requests
import os
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:5000"  # Replace with your API server URL

def start_api_server():
    """Start the API server using a subprocess (replace with your server start command)"""
    try:
        process = subprocess.Popen(['python', 'your_api_server.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Replace 'your_api_server.py' with your actual server script
        #Give the server some time to start
        import time
        time.sleep(5)
        return process
    except FileNotFoundError:
        logger.error("API server script not found.  Please ensure 'your_api_server.py' exists and is executable.")
        return False
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        return False

def stop_api_server(process):
    """Stop the API server"""
    try:
        process.terminate()
        process.wait()
        logger.info("API server stopped successfully.")
    except Exception as e:
        logger.error(f"Error stopping API server: {str(e)}")

def check_server_status():
    """Check if the API server is running and accessible"""
    try:
        # Try the health endpoint first
        try:
            logger.info(f"Checking health endpoint at {BASE_URL}/health")
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            logger.info(f"Health endpoint response: {response.status_code}")
            if response.status_code == 200:
                logger.info("API server is running and responding to health checks")
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Health endpoint check failed: {str(e)}")
            # If health endpoint fails, try the root endpoint
            pass

        # Try root endpoint as fallback
        logger.info(f"Checking root endpoint at {BASE_URL}")
        response = requests.get(BASE_URL, timeout=2)
        logger.info(f"Root endpoint response: {response.status_code}")
        if response.status_code == 200 or response.status_code == 302:
            logger.info("API server is running (root endpoint responsive)")
            return True
        else:
            logger.warning(f"API server returned status code: {response.status_code}")

        # Check API endpoint directly
        logger.info(f"Checking API endpoint at {BASE_URL}/api/workflow/status")
        response = requests.get(f"{BASE_URL}/api/workflow/status", timeout=2)
        logger.info(f"API endpoint response: {response.status_code}")
        if response.status_code == 200:
            logger.info("API server workflow endpoint is responsive")
            return True

        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during server status check: {e}")
        return False


def load_test_data():
    """Load test data from JSON file"""
    file_path = "data/test_validation/row2_example.json"
    logger.info(f"Loading test data from {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"Test data file not found: {file_path}")
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        # Verify data structure
        if not isinstance(data, dict):
            logger.error(f"Invalid data format: expected dict, got {type(data)}")
            return None

        # Check for required fields
        required_fields = ["input", "expected_output"]
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field in test data: {field}")
                return None

        logger.info(f"Successfully loaded test data with fields: {', '.join(data.keys())}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON test data: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return None

def test_workflow_step(step, data):
    # Test API workflow step
    try:
        logger.info(f"Sending request to {BASE_URL}/api/workflow/step/{step}")
        logger.info(f"Request data: {data}")

        response = requests.post(f"{BASE_URL}/api/workflow/step/{step}", json=data, timeout=30)

        logger.info(f"Response status code: {response.status_code}")

        if response.status_code != 200:
            logger.warning(f"Step {step} failed with status code {response.status_code}")
            logger.warning(f"Response: {response.text}")
            return None

        result = response.json()
        logger.info(f"Step {step} succeeded with response: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Step {step} failed with exception: {str(e)}")
        logger.error(f"Exception details: {type(e).__name__}")
        return None

def run_workflow_test():
    """Run the complete five-step workflow test"""
    # Start API server if not already running
    server_process = None
    server_started = False
    try:
        # Try to check if server is already running
        if check_server_status():
            logger.info("API server is already running")
            server_started = True
        else:
            logger.info("Server not running, attempting to start it...")
            server_process = start_api_server()
            if server_process:
                logger.info("Successfully started API server")
                server_started = True
            else:
                logger.error("Failed to start API server, cannot run test")
                return False

        # Load test data
        test_data = load_test_data()
        if not test_data:
            logger.error("Failed to load test data, cannot run test")
            if server_started and server_process:
                logger.info("Stopping API server...")
                stop_api_server(server_process)  # Ensure we stop the server
            return False

        # Verify API endpoints are accessible
        try:
            logger.info("Verifying API endpoints...")
            response = requests.get(f"{BASE_URL}/api/workflow/status", timeout=5)
            logger.info(f"API endpoint verification status: {response.status_code}")
            if response.status_code != 200:
                logger.warning(f"API endpoints may not be properly set up: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"API endpoint verification failed: {str(e)}")


        # Run workflow steps
        successful_steps = 0
        skipped_steps = 0
        for step in range(1, 6):
            result = test_workflow_step(step, test_data.get("input", {})) #Use default empty dict if 'input' is missing
            if result is not None:
                successful_steps +=1
            else:
                skipped_steps += 1
        # Summarize results
        logger.info("WORKFLOW SUMMARY")
        logger.info("================================================================================")
        logger.info(f"Steps completed: {successful_steps}/5")
        logger.info(f"Steps skipped: {skipped_steps}/5")
        if successful_steps == 5:
            logger.info("TEST PASSED")
        else:
            logger.error("TEST FAILED")

        if server_started and server_process:
            logger.info("Stopping API server...")
            stop_api_server(server_process)

        return successful_steps == 5

    except Exception as e:
        logger.exception(f"An unexpected error occurred during workflow test: {e}")
        if server_started and server_process:
            logger.info("Stopping API server...")
            stop_api_server(server_process)
        return False



if __name__ == "__main__":
    if run_workflow_test():
        print("Workflow test passed successfully!")
    else:
        print("Workflow test failed.")