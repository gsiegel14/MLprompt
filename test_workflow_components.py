#!/usr/bin/env python3
"""
Direct Workflow Component Testing Script

This script tests individual workflow components directly without using the HTTP API:
1. Tests LLM client for generating responses
2. Tests evaluator for computing metrics 
3. Tests optimizer for improving prompts
4. Tests Hugging Face client for external validation
5. Tests the full workflow with all components

Usage:
    python test_workflow_components.py
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_workflow_components.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("component_test")

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import required modules
try:
    from app.llm_client import get_llm_response
    from app.evaluator import calculate_score, evaluate_batch
    from app.optimizer import optimize_prompts, load_optimizer_prompt
    from app.data_module import DataModule
    from app.experiment_tracker import ExperimentTracker
    from app.workflow import PromptOptimizationWorkflow
    from app.huggingface_client import evaluate_metrics, validate_api_connection
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Test data
TEST_SYSTEM_PROMPT = """You are an expert medical diagnostician. Analyze the presented case and provide a differential diagnosis."""
TEST_OUTPUT_PROMPT = """List the top 3 most likely diagnoses in order of probability, with brief explanations."""
TEST_USER_INPUT = """A 45-year-old male presents with sudden onset chest pain, radiating to the left arm, accompanied by sweating and shortness of breath. He has a history of hypertension and diabetes."""
TEST_EXPECTED_OUTPUT = """Based on the symptoms and history, the most likely diagnoses are:

1. Acute Myocardial Infarction (Heart Attack): Most likely given the sudden chest pain radiating to the left arm, sweating, and shortness of breath. Risk factors include age, male gender, hypertension, and diabetes.

2. Unstable Angina: Similar presentation to a heart attack but without permanent heart damage. The patient's risk factors make this a strong possibility if cardiac enzymes are negative.

3. Aortic Dissection: Less common but serious consideration, especially with the history of hypertension. The sudden onset of severe pain is consistent with this diagnosis."""

class WorkflowComponentTester:
    """Test individual workflow components directly."""
    
    def __init__(self):
        """Initialize the tester."""
        self.data_module = DataModule()
        self.experiment_tracker = ExperimentTracker()
        self.workflow = PromptOptimizationWorkflow(
            self.data_module, 
            self.experiment_tracker,
            config
        )
        
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "component_tests": {},
            "workflow_test": {},
            "success_count": 0,
            "failure_count": 0
        }
    
    def _record_result(self, test_name: str, success: bool, message: str, category: str = "component_tests", data: Optional[Any] = None):
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
    
    def test_llm_client(self):
        """Test the LLM client for generating responses."""
        try:
            start_time = time.time()
            
            response = get_llm_response(
                TEST_SYSTEM_PROMPT,
                TEST_USER_INPUT,
                TEST_OUTPUT_PROMPT,
                config.get('gemini', {})
            )
            
            duration = time.time() - start_time
            
            if response and len(response) > 100:
                self._record_result(
                    "llm_client", 
                    True, 
                    f"LLM client generated response of length {len(response)} in {duration:.2f}s",
                    data={"response_preview": response[:100] + "..."}
                )
                return response
            else:
                self._record_result(
                    "llm_client", 
                    False, 
                    f"LLM client generated short or empty response: {response}"
                )
                return None
        except Exception as e:
            self._record_result("llm_client", False, f"Exception: {str(e)}")
            return None
    
    def test_evaluator(self, model_response=None):
        """Test the evaluator for computing metrics."""
        try:
            if model_response is None:
                model_response = "The patient is likely experiencing a heart attack due to chest pain, radiation to left arm, and risk factors."
            
            # Calculate individual score
            score = calculate_score(model_response, TEST_EXPECTED_OUTPUT)
            
            # Evaluate batch
            batch_result = evaluate_batch([
                {
                    'model_response': model_response,
                    'ground_truth_output': TEST_EXPECTED_OUTPUT
                }
            ])
            
            self._record_result(
                "evaluator", 
                True, 
                f"Evaluator calculated score: {score}, batch metrics generated",
                data={
                    "score": score,
                    "batch_metrics": batch_result
                }
            )
            return score
        except Exception as e:
            self._record_result("evaluator", False, f"Exception: {str(e)}")
            return None
    
    def test_optimizer(self, model_response=None, score=None):
        """Test the optimizer for improving prompts."""
        try:
            if model_response is None:
                model_response = "The patient is likely experiencing a heart attack due to chest pain, radiation to left arm, and risk factors."
            
            if score is None:
                score = 0.6
            
            # Load optimizer prompt
            optimizer_prompt = load_optimizer_prompt('general')
            
            # Prepare examples
            examples = [
                {
                    'user_input': TEST_USER_INPUT,
                    'ground_truth_output': TEST_EXPECTED_OUTPUT,
                    'model_response': model_response,
                    'score': score
                }
            ]
            
            # Run optimization
            start_time = time.time()
            optimization_result = optimize_prompts(
                current_system_prompt=TEST_SYSTEM_PROMPT,
                current_output_prompt=TEST_OUTPUT_PROMPT,
                examples=examples,
                optimizer_system_prompt=optimizer_prompt,
                strategy="reasoning_first"
            )
            duration = time.time() - start_time
            
            if optimization_result:
                optimized_system_prompt = optimization_result.get('optimized_system_prompt', "")
                optimized_output_prompt = optimization_result.get('optimized_output_prompt', "")
                
                self._record_result(
                    "optimizer", 
                    True, 
                    f"Optimizer generated improved prompts in {duration:.2f}s",
                    data={
                        "system_prompt_length": len(optimized_system_prompt),
                        "output_prompt_length": len(optimized_output_prompt),
                        "system_prompt_preview": optimized_system_prompt[:100] + "..."
                    }
                )
                return optimization_result
            else:
                self._record_result(
                    "optimizer", 
                    False, 
                    "Optimizer failed to generate improved prompts"
                )
                return None
        except Exception as e:
            self._record_result("optimizer", False, f"Exception: {str(e)}")
            return None
    
    def test_huggingface_client(self, model_response=None):
        """Test the Hugging Face client for external validation."""
        try:
            # First, validate API connection
            connection_result = validate_api_connection()
            
            if not connection_result:
                self._record_result(
                    "huggingface_client", 
                    False, 
                    "Failed to connect to Hugging Face API"
                )
                return None
            
            if model_response is None:
                model_response = "The patient is likely experiencing a heart attack due to chest pain, radiation to left arm, and risk factors."
            
            # Evaluate with HF metrics
            metrics_result = evaluate_metrics(
                [model_response],
                [TEST_EXPECTED_OUTPUT],
                ["exact_match", "bleu"]
            )
            
            self._record_result(
                "huggingface_client", 
                True, 
                "Hugging Face client validated connection and computed metrics",
                data=metrics_result
            )
            return metrics_result
        except Exception as e:
            self._record_result("huggingface_client", False, f"Exception: {str(e)}")
            return None
    
    def test_workflow_integration(self):
        """Test the full workflow integration with all components."""
        try:
            # Get a small batch of examples
            batch = self.data_module.get_batch(batch_size=1, validation=False)
            
            if not batch:
                self._record_result(
                    "workflow_integration", 
                    False, 
                    "No examples available for testing",
                    category="workflow_test"
                )
                return None
            
            # Run the full workflow
            start_time = time.time()
            results = self.workflow.run_four_api_workflow(
                system_prompt=TEST_SYSTEM_PROMPT,
                output_prompt=TEST_OUTPUT_PROMPT,
                batch_size=1,
                optimizer_strategy="reasoning_first",
                hf_metrics=["exact_match", "bleu"]
            )
            duration = time.time() - start_time
            
            if results and isinstance(results, dict) and 'error' not in results:
                experiment_id = results.get('experiment_id', "")
                internal_metrics = results.get('internal_metrics', {})
                huggingface_metrics = results.get('huggingface_metrics', {})
                
                self._record_result(
                    "workflow_integration", 
                    True, 
                    f"Full workflow executed successfully in {duration:.2f}s - experiment ID: {experiment_id}",
                    category="workflow_test",
                    data={
                        "experiment_id": experiment_id,
                        "has_internal_metrics": bool(internal_metrics),
                        "has_huggingface_metrics": bool(huggingface_metrics),
                        "examples_processed": results.get('examples_count', 0)
                    }
                )
                return results
            else:
                error = results.get('error', "Unknown error") if isinstance(results, dict) else "Invalid result structure"
                self._record_result(
                    "workflow_integration", 
                    False, 
                    f"Full workflow execution failed: {error}",
                    category="workflow_test"
                )
                return None
        except Exception as e:
            import traceback
            self._record_result(
                "workflow_integration", 
                False, 
                f"Exception: {str(e)}",
                category="workflow_test",
                data={"traceback": traceback.format_exc()}
            )
            return None
    
    def run_all_tests(self):
        """Run all component tests."""
        logger.info("=== STARTING WORKFLOW COMPONENT TESTS ===")
        
        # Test individual components
        model_response = self.test_llm_client()
        score = self.test_evaluator(model_response)
        self.test_optimizer(model_response, score)
        self.test_huggingface_client(model_response)
        
        # Test full workflow integration
        logger.info("=== STARTING WORKFLOW INTEGRATION TEST ===")
        self.test_workflow_integration()
        
        # Summary
        logger.info("=== TEST SUMMARY ===")
        logger.info(f"Total tests: {self.test_results['success_count'] + self.test_results['failure_count']}")
        logger.info(f"Passed: {self.test_results['success_count']}")
        logger.info(f"Failed: {self.test_results['failure_count']}")
        
        # Save results to file
        output_path = f"test_outputs/component_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("test_outputs", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"Test results saved to {output_path}")
        
        return self.test_results["success_count"] > 0 and self.test_results["failure_count"] == 0

def main():
    """Run the workflow component tests."""
    tester = WorkflowComponentTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("=== ALL COMPONENT TESTS PASSED ===")
        return 0
    else:
        logger.error("=== SOME COMPONENT TESTS FAILED ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())