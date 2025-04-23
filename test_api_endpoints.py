#!/usr/bin/env python3
"""
Comprehensive API Endpoint and Workflow Test Script

This script tests the API endpoints and workflow of the ML prompt optimization platform:
1. Tests direct API endpoints with HTTP requests
2. Tests workflow components individually
3. Verifies data flow between components

Usage:
    python test_api_endpoints.py
"""

import os
import sys
import json
import logging
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_api_endpoints.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_test")

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Base URL for API requests
BASE_URL = "http://localhost:5000"

# Test data
TEST_SYSTEM_PROMPT = """You are an expert medical diagnostician. Analyze the presented case and provide a differential diagnosis."""
TEST_OUTPUT_PROMPT = """List the top 3 most likely diagnoses in order of probability, with brief explanations."""
TEST_USER_INPUT = """A 45-year-old male presents with sudden onset chest pain, radiating to the left arm, accompanied by sweating and shortness of breath. He has a history of hypertension and diabetes."""
TEST_EXPECTED_OUTPUT = """Based on the symptoms and history, the most likely diagnoses are:

1. Acute Myocardial Infarction (Heart Attack): Most likely given the sudden chest pain radiating to the left arm, sweating, and shortness of breath. Risk factors include age, male gender, hypertension, and diabetes.

2. Unstable Angina: Similar presentation to a heart attack but without permanent heart damage. The patient's risk factors make this a strong possibility if cardiac enzymes are negative.

3. Aortic Dissection: Less common but serious consideration, especially with the history of hypertension. The sudden onset of severe pain is consistent with this diagnosis."""

class APITester:
    """Test the API endpoints and workflow integration."""
    
    def __init__(self):
        """Initialize the tester."""
        self.session = requests.Session()
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "api_tests": {},
            "workflow_tests": {},
            "success_count": 0,
            "failure_count": 0
        }
    
    def _record_result(self, test_name: str, success: bool, message: str, category: str = "api_tests", data: Optional[Any] = None):
        """Record a test result."""
        self.test_results[category][test_name] = {
            "success": success,
            "message": message,
            "data": data
        }
        
        if success:
            self.test_results["success_count"] += 1
            logger.info(f"✓ {test_name}: {message}")
        else:
            self.test_results["failure_count"] += 1
            logger.error(f"✗ {test_name}: {message}")
    
    def test_home_page(self):
        """Test that the home page is accessible."""
        try:
            response = self.session.get(f"{BASE_URL}/")
            
            if response.status_code == 200:
                self._record_result(
                    "home_page", 
                    True, 
                    f"Home page accessible - status code: {response.status_code}"
                )
            else:
                self._record_result(
                    "home_page", 
                    False, 
                    f"Home page returned status code: {response.status_code}"
                )
        except Exception as e:
            self._record_result("home_page", False, f"Exception: {str(e)}")
    
    def test_api_status(self):
        """Test the API status endpoint."""
        try:
            response = self.session.get(f"{BASE_URL}/api/status")
            
            if response.status_code == 200:
                data = response.json()
                self._record_result(
                    "api_status", 
                    True, 
                    f"API status endpoint accessible - status: {data.get('status', 'unknown')}",
                    data=data
                )
            else:
                self._record_result(
                    "api_status", 
                    False, 
                    f"API status endpoint returned status code: {response.status_code}"
                )
        except Exception as e:
            self._record_result("api_status", False, f"Exception: {str(e)}")
    
    def test_llm_api(self):
        """Test the LLM API endpoint."""
        try:
            payload = {
                "system_prompt": TEST_SYSTEM_PROMPT,
                "user_input": TEST_USER_INPUT,
                "output_prompt": TEST_OUTPUT_PROMPT
            }
            
            response = self.session.post(
                f"{BASE_URL}/api/llm/generate",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                
                if response_text and len(response_text) > 50:
                    self._record_result(
                        "llm_api", 
                        True, 
                        f"LLM API endpoint returned response of length {len(response_text)}",
                        data={"response_preview": response_text[:100] + "..."}
                    )
                else:
                    self._record_result(
                        "llm_api", 
                        False, 
                        f"LLM API endpoint returned short or empty response: {response_text}"
                    )
            else:
                self._record_result(
                    "llm_api", 
                    False, 
                    f"LLM API endpoint returned status code: {response.status_code}"
                )
        except Exception as e:
            self._record_result("llm_api", False, f"Exception: {str(e)}")
    
    def test_evaluation_api(self):
        """Test the evaluation API endpoint."""
        try:
            payload = {
                "model_response": "The patient is likely experiencing a heart attack.",
                "ground_truth": "Acute Myocardial Infarction (Heart Attack)"
            }
            
            response = self.session.post(
                f"{BASE_URL}/api/evaluate",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                score = data.get("score", 0)
                
                self._record_result(
                    "evaluation_api", 
                    True, 
                    f"Evaluation API endpoint returned score: {score}",
                    data=data
                )
            else:
                self._record_result(
                    "evaluation_api", 
                    False, 
                    f"Evaluation API endpoint returned status code: {response.status_code}"
                )
        except Exception as e:
            self._record_result("evaluation_api", False, f"Exception: {str(e)}")
    
    def test_optimization_api(self):
        """Test the prompt optimization API endpoint."""
        try:
            payload = {
                "current_system_prompt": TEST_SYSTEM_PROMPT,
                "current_output_prompt": TEST_OUTPUT_PROMPT,
                "examples": [
                    {
                        "user_input": TEST_USER_INPUT,
                        "ground_truth_output": TEST_EXPECTED_OUTPUT,
                        "model_response": "The patient might have a heart condition.",
                        "score": 0.5
                    }
                ],
                "optimizer_strategy": "reasoning_first"
            }
            
            response = self.session.post(
                f"{BASE_URL}/api/optimize",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                optimized_system_prompt = data.get("optimized_system_prompt", "")
                optimized_output_prompt = data.get("optimized_output_prompt", "")
                
                if optimized_system_prompt and optimized_output_prompt:
                    self._record_result(
                        "optimization_api", 
                        True, 
                        f"Optimization API endpoint returned improved prompts",
                        data={
                            "system_prompt_length": len(optimized_system_prompt),
                            "output_prompt_length": len(optimized_output_prompt)
                        }
                    )
                else:
                    self._record_result(
                        "optimization_api", 
                        False, 
                        f"Optimization API endpoint returned incomplete results"
                    )
            else:
                self._record_result(
                    "optimization_api", 
                    False, 
                    f"Optimization API endpoint returned status code: {response.status_code}"
                )
        except Exception as e:
            self._record_result("optimization_api", False, f"Exception: {str(e)}")
    
    def test_huggingface_metrics_api(self):
        """Test the Hugging Face metrics API endpoint."""
        try:
            payload = {
                "predictions": ["The patient is having a heart attack."],
                "references": ["Acute Myocardial Infarction (Heart Attack)"],
                "metrics": ["exact_match", "bleu"]
            }
            
            response = self.session.post(
                f"{BASE_URL}/api/metrics/huggingface",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                self._record_result(
                    "huggingface_metrics_api", 
                    True, 
                    f"Hugging Face metrics API endpoint returned results",
                    data=data
                )
            else:
                self._record_result(
                    "huggingface_metrics_api", 
                    False, 
                    f"Hugging Face metrics API endpoint returned status code: {response.status_code}"
                )
        except Exception as e:
            self._record_result("huggingface_metrics_api", False, f"Exception: {str(e)}")
    
    def test_four_api_workflow(self):
        """Test the 4-API workflow endpoint."""
        try:
            payload = {
                "system_prompt": TEST_SYSTEM_PROMPT,
                "output_prompt": TEST_OUTPUT_PROMPT,
                "batch_size": 1,
                "optimizer_strategy": "reasoning_first",
                "hf_metrics": ["exact_match", "bleu"]
            }
            
            response = self.session.post(
                f"{BASE_URL}/api/workflow/four_api",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                experiment_id = data.get("experiment_id", "")
                
                if experiment_id:
                    self._record_result(
                        "four_api_workflow", 
                        True, 
                        f"4-API workflow endpoint executed successfully - experiment ID: {experiment_id}",
                        category="workflow_tests",
                        data={
                            "experiment_id": experiment_id,
                            "has_metrics": "huggingface_metrics" in data
                        }
                    )
                else:
                    self._record_result(
                        "four_api_workflow", 
                        False, 
                        f"4-API workflow endpoint returned incomplete results",
                        category="workflow_tests"
                    )
            else:
                self._record_result(
                    "four_api_workflow", 
                    False, 
                    f"4-API workflow endpoint returned status code: {response.status_code}",
                    category="workflow_tests"
                )
        except Exception as e:
            self._record_result(
                "four_api_workflow", 
                False, 
                f"Exception: {str(e)}",
                category="workflow_tests"
            )
    
    def test_experiment_history_api(self):
        """Test the experiment history API endpoint."""
        try:
            response = self.session.get(f"{BASE_URL}/api/experiments/list")
            
            if response.status_code == 200:
                data = response.json()
                experiments = data.get("experiments", [])
                
                self._record_result(
                    "experiment_history_api", 
                    True, 
                    f"Experiment history API endpoint returned {len(experiments)} experiments",
                    data={"count": len(experiments)}
                )
            else:
                self._record_result(
                    "experiment_history_api", 
                    False, 
                    f"Experiment history API endpoint returned status code: {response.status_code}"
                )
        except Exception as e:
            self._record_result("experiment_history_api", False, f"Exception: {str(e)}")
    
    def run_all_tests(self):
        """Run all API endpoint tests."""
        logger.info("=== STARTING API ENDPOINT TESTS ===")
        
        # Basic connectivity tests
        self.test_home_page()
        self.test_api_status()
        
        # Core API endpoints
        self.test_llm_api()
        self.test_evaluation_api()
        self.test_optimization_api()
        self.test_huggingface_metrics_api()
        
        # History API
        self.test_experiment_history_api()
        
        # Workflow test
        logger.info("=== STARTING WORKFLOW TESTS ===")
        self.test_four_api_workflow()
        
        # Summary
        logger.info("=== TEST SUMMARY ===")
        logger.info(f"Total tests: {self.test_results['success_count'] + self.test_results['failure_count']}")
        logger.info(f"Passed: {self.test_results['success_count']}")
        logger.info(f"Failed: {self.test_results['failure_count']}")
        
        # Save results to file
        output_path = f"test_outputs/api_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("test_outputs", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"Test results saved to {output_path}")
        
        return self.test_results["success_count"] > 0 and self.test_results["failure_count"] == 0

def main():
    """Run the API endpoint tests."""
    tester = APITester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("=== ALL API TESTS PASSED ===")
        return 0
    else:
        logger.error("=== SOME API TESTS FAILED ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())