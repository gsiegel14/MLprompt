#!/usr/bin/env python3
"""
Comprehensive Test Script for 5-Step Workflow Integration

This script tests the entire 5-step workflow sequence:
1. Google Vertex API #1: Primary LLM inference
2. Hugging Face API: First external validation
3. Google Vertex API #2: Optimizer LLM for prompt refinement
4. Google Vertex API #3: Optimizer LLM reruns on original dataset
5. Hugging Face API: Second external validation on refined outputs

Each step is tested individually and then as part of the complete workflow
to ensure proper integration and data passing between steps.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_workflow_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("workflow_test")

# Import necessary modules
try:
    # Add the app directory to the path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
    
    # Import core components
    from app.llm_client import get_llm_response
    from app.evaluator import evaluate_batch, calculate_score
    from app.optimizer import optimize_prompts, load_optimizer_prompt
    from app.data_module import DataModule
    from app.experiment_tracker import ExperimentTracker
    from app.workflow import PromptOptimizationWorkflow
    from app.huggingface_client import evaluate_metrics, validate_api_connection
    
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

class WorkflowIntegrationTester:
    """Tests each step of the workflow integration."""
    
    def __init__(self):
        """Initialize the tester with necessary components."""
        self.data_module = DataModule()
        self.experiment_tracker = ExperimentTracker()
        self.workflow = PromptOptimizationWorkflow(
            self.data_module, 
            self.experiment_tracker,
            config
        )
        
        # Create test directory
        os.makedirs("test_outputs", exist_ok=True)
        
        # Test parameters
        self.test_system_prompt = (
            "You are an expert medical diagnostician. Your task is to analyze patient symptoms "
            "and medical history to provide differential diagnoses."
        )
        self.test_output_prompt = (
            "Please provide a ranked list of the most likely diagnoses based on the symptoms "
            "and history provided. For each diagnosis, explain your reasoning."
        )
        self.test_batch_size = 3
        self.test_hf_metrics = ["exact_match", "bleu"]
        self.test_optimizer_strategy = "reasoning_first"
        
        # Store test results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "steps": {},
            "overall_success": False
        }
        
        logger.info("WorkflowIntegrationTester initialized")
    
    def save_results(self):
        """Save test results to file."""
        output_path = f"test_outputs/workflow_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Test results saved to {output_path}")
        return output_path
        
    def test_step1_primary_llm(self):
        """Test Step 1: Google Vertex API - Primary LLM inference."""
        logger.info("TESTING STEP 1: Primary LLM Inference")
        try:
            # Get examples for testing
            batch = self.data_module.get_batch(batch_size=self.test_batch_size, validation=False)
            if not batch:
                logger.error("No examples available for testing")
                self.results["steps"]["step1"] = {
                    "success": False,
                    "message": "No examples available for testing"
                }
                return None
                
            logger.info(f"Got batch of {len(batch)} examples")
            
            # Process examples with Primary LLM
            predictions = []
            references = []
            inputs = []
            
            for example in batch:
                user_input = example.get('user_input', '')
                ground_truth = example.get('ground_truth_output', '')
                
                # Call the Primary LLM
                model_response = get_llm_response(
                    self.test_system_prompt,
                    user_input,
                    self.test_output_prompt,
                    config.get('gemini', {})
                )
                
                predictions.append(model_response)
                references.append(ground_truth)
                inputs.append(user_input)
                
                logger.info(f"Example processed - response length: {len(model_response)}")
            
            step1_results = {
                "batch_size": len(batch),
                "predictions": predictions,
                "references": references,
                "inputs": inputs,
                "success": True,
                "message": f"Successfully processed {len(predictions)} examples"
            }
            
            self.results["steps"]["step1"] = step1_results
            logger.info("Step 1 completed successfully")
            return step1_results
            
        except Exception as e:
            logger.error(f"Error in Step 1: {e}")
            logger.error(traceback.format_exc())
            self.results["steps"]["step1"] = {
                "success": False,
                "message": f"Error: {str(e)}"
            }
            return None
    
    def test_step2_huggingface_validation(self, step1_results):
        """Test Step 2: Hugging Face API - First external validation."""
        logger.info("TESTING STEP 2: Hugging Face API - First External Validation")
        try:
            if not step1_results or not step1_results.get("success"):
                logger.error("Cannot run Step 2 because Step 1 failed")
                self.results["steps"]["step2"] = {
                    "success": False,
                    "message": "Cannot run because Step 1 failed"
                }
                return None
            
            # Check Hugging Face API connection
            try:
                validate_api_connection()
                logger.info("Hugging Face API connection validated")
            except Exception as e:
                logger.error(f"Hugging Face API connection failed: {e}")
                self.results["steps"]["step2"] = {
                    "success": False,
                    "message": f"Hugging Face API connection failed: {str(e)}"
                }
                return None
            
            predictions = step1_results["predictions"]
            references = step1_results["references"]
            
            # Evaluate with Hugging Face metrics
            hf_metrics_results = evaluate_metrics(
                predictions,
                references,
                self.test_hf_metrics
            )
            
            logger.info(f"Hugging Face metrics: {hf_metrics_results}")
            
            # Also calculate internal metrics for completeness
            internal_metrics = evaluate_batch([
                {
                    'model_response': pred,
                    'ground_truth_output': ref
                } for pred, ref in zip(predictions, references)
            ])
            
            logger.info(f"Internal metrics: {internal_metrics}")
            
            step2_results = {
                "hf_metrics": hf_metrics_results,
                "internal_metrics": internal_metrics,
                "success": True,
                "message": "Successfully evaluated with Hugging Face metrics"
            }
            
            self.results["steps"]["step2"] = step2_results
            logger.info("Step 2 completed successfully")
            return step2_results
            
        except Exception as e:
            logger.error(f"Error in Step 2: {e}")
            logger.error(traceback.format_exc())
            self.results["steps"]["step2"] = {
                "success": False,
                "message": f"Error: {str(e)}"
            }
            return None
    
    def test_step3_optimizer_llm(self, step1_results):
        """Test Step 3: Google Vertex API - Optimizer LLM for prompt refinement."""
        logger.info("TESTING STEP 3: Optimizer LLM for Prompt Refinement")
        try:
            if not step1_results or not step1_results.get("success"):
                logger.error("Cannot run Step 3 because Step 1 failed")
                self.results["steps"]["step3"] = {
                    "success": False,
                    "message": "Cannot run because Step 1 failed"
                }
                return None
            
            # Load the appropriate optimizer prompt
            optimizer_prompt = load_optimizer_prompt('general')
            logger.info(f"Loaded optimizer prompt - length: {len(optimizer_prompt)}")
            
            # Prepare examples with results for optimization
            predictions = step1_results["predictions"]
            references = step1_results["references"]
            inputs = step1_results["inputs"]
            
            examples_for_optimizer = []
            for i in range(len(predictions)):
                examples_for_optimizer.append({
                    'user_input': inputs[i],
                    'ground_truth_output': references[i],
                    'model_response': predictions[i],
                    'score': calculate_score(predictions[i], references[i])
                })
            
            logger.info(f"Prepared {len(examples_for_optimizer)} examples for optimizer")
            
            # Optimize the prompts
            optimization_result = optimize_prompts(
                current_system_prompt=self.test_system_prompt,
                current_output_prompt=self.test_output_prompt,
                examples=examples_for_optimizer,
                optimizer_prompt=optimizer_prompt,
                optimizer_strategy=self.test_optimizer_strategy
            )
            
            if not optimization_result:
                logger.error("Optimization failed")
                self.results["steps"]["step3"] = {
                    "success": False,
                    "message": "Optimization failed"
                }
                return None
                
            optimized_system_prompt = optimization_result.get('optimized_system_prompt', self.test_system_prompt)
            optimized_output_prompt = optimization_result.get('optimized_output_prompt', self.test_output_prompt)
            
            logger.info(f"Optimization complete - new system prompt length: {len(optimized_system_prompt)} chars")
            logger.info(f"Optimization complete - new output prompt length: {len(optimized_output_prompt)} chars")
            
            step3_results = {
                "optimized_system_prompt": optimized_system_prompt,
                "optimized_output_prompt": optimized_output_prompt,
                "examples_for_optimizer": examples_for_optimizer,
                "success": True,
                "message": "Successfully optimized prompts"
            }
            
            self.results["steps"]["step3"] = step3_results
            logger.info("Step 3 completed successfully")
            return step3_results
            
        except Exception as e:
            logger.error(f"Error in Step 3: {e}")
            logger.error(traceback.format_exc())
            self.results["steps"]["step3"] = {
                "success": False,
                "message": f"Error: {str(e)}"
            }
            return None
    
    def test_step4_rerun_optimized(self, step1_results, step3_results):
        """Test Step 4: Google Vertex API - Rerun with optimized prompts."""
        logger.info("TESTING STEP 4: Rerun with Optimized Prompts")
        try:
            if not step1_results or not step1_results.get("success") or not step3_results or not step3_results.get("success"):
                logger.error("Cannot run Step 4 because previous steps failed")
                self.results["steps"]["step4"] = {
                    "success": False,
                    "message": "Cannot run because previous steps failed"
                }
                return None
            
            optimized_system_prompt = step3_results["optimized_system_prompt"]
            optimized_output_prompt = step3_results["optimized_output_prompt"]
            inputs = step1_results["inputs"]
            
            # Process the same examples with the optimized prompts
            optimized_predictions = []
            
            for i, user_input in enumerate(inputs):
                # Call the Primary LLM with optimized prompts
                optimized_response = get_llm_response(
                    optimized_system_prompt,
                    user_input,
                    optimized_output_prompt,
                    config.get('gemini', {})
                )
                
                optimized_predictions.append(optimized_response)
                logger.info(f"Example {i+1} reprocessed with optimized prompts - response length: {len(optimized_response)}")
            
            step4_results = {
                "optimized_predictions": optimized_predictions,
                "original_references": step1_results["references"],
                "success": True,
                "message": f"Successfully reprocessed {len(optimized_predictions)} examples with optimized prompts"
            }
            
            self.results["steps"]["step4"] = step4_results
            logger.info("Step 4 completed successfully")
            return step4_results
            
        except Exception as e:
            logger.error(f"Error in Step 4: {e}")
            logger.error(traceback.format_exc())
            self.results["steps"]["step4"] = {
                "success": False,
                "message": f"Error: {str(e)}"
            }
            return None
    
    def test_step5_second_validation(self, step1_results, step3_results, step4_results):
        """Test Step 5: Hugging Face API - Second external validation on refined outputs."""
        logger.info("TESTING STEP 5: Second External Validation")
        try:
            if (not step1_results or not step1_results.get("success") or 
                not step3_results or not step3_results.get("success") or
                not step4_results or not step4_results.get("success")):
                logger.error("Cannot run Step 5 because previous steps failed")
                self.results["steps"]["step5"] = {
                    "success": False,
                    "message": "Cannot run because previous steps failed"
                }
                return None
            
            # Check Hugging Face API connection
            try:
                validate_api_connection()
                logger.info("Hugging Face API connection validated")
            except Exception as e:
                logger.error(f"Hugging Face API connection failed: {e}")
                self.results["steps"]["step5"] = {
                    "success": False,
                    "message": f"Hugging Face API connection failed: {str(e)}"
                }
                return None
            
            original_predictions = step1_results["predictions"]
            optimized_predictions = step4_results["optimized_predictions"]
            references = step1_results["references"]
            
            # Evaluate both sets with Hugging Face metrics
            original_hf_metrics = evaluate_metrics(
                original_predictions,
                references,
                self.test_hf_metrics
            )
            
            optimized_hf_metrics = evaluate_metrics(
                optimized_predictions,
                references,
                self.test_hf_metrics
            )
            
            logger.info(f"Original Hugging Face metrics: {original_hf_metrics}")
            logger.info(f"Optimized Hugging Face metrics: {optimized_hf_metrics}")
            
            # Compare the two sets of metrics
            improved = all(optimized_hf_metrics.get(metric, 0) >= original_hf_metrics.get(metric, 0) 
                          for metric in original_hf_metrics)
            
            step5_results = {
                "original_hf_metrics": original_hf_metrics,
                "optimized_hf_metrics": optimized_hf_metrics,
                "improved": improved,
                "success": True,
                "message": "Successfully evaluated optimized prompts with Hugging Face metrics"
            }
            
            self.results["steps"]["step5"] = step5_results
            logger.info("Step 5 completed successfully")
            logger.info(f"Metrics comparison shows {'improvement' if improved else 'no improvement'}")
            return step5_results
            
        except Exception as e:
            logger.error(f"Error in Step 5: {e}")
            logger.error(traceback.format_exc())
            self.results["steps"]["step5"] = {
                "success": False,
                "message": f"Error: {str(e)}"
            }
            return None
    
    def test_full_workflow(self):
        """Test the complete 5-step workflow integration."""
        logger.info("==== STARTING FULL WORKFLOW INTEGRATION TEST ====")
        
        # Test with the workflow implementation
        logger.info("Testing complete workflow with PromptOptimizationWorkflow class...")
        try:
            full_results = self.workflow.run_four_api_workflow(
                system_prompt=self.test_system_prompt,
                output_prompt=self.test_output_prompt,
                batch_size=self.test_batch_size,
                optimizer_strategy=self.test_optimizer_strategy,
                hf_metrics=self.test_hf_metrics
            )
            
            if not full_results or 'error' in full_results:
                logger.error(f"Full workflow test failed: {full_results.get('error', 'Unknown error')}")
                self.results["full_workflow"] = {
                    "success": False,
                    "message": f"Error: {full_results.get('error', 'Unknown error')}"
                }
            else:
                logger.info("Full workflow test completed successfully")
                logger.info(f"Experiment ID: {full_results.get('experiment_id')}")
                logger.info(f"Internal metrics: {full_results.get('internal_metrics')}")
                logger.info(f"Hugging Face metrics: {full_results.get('huggingface_metrics')}")
                
                self.results["full_workflow"] = {
                    "success": True,
                    "experiment_id": full_results.get('experiment_id'),
                    "metrics": {
                        "internal": full_results.get('internal_metrics'),
                        "huggingface": full_results.get('huggingface_metrics')
                    },
                    "message": "Successfully completed full workflow test"
                }
        except Exception as e:
            logger.error(f"Error in full workflow test: {e}")
            logger.error(traceback.format_exc())
            self.results["full_workflow"] = {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    def run_all_tests(self):
        """Run all tests in sequence and report results."""
        logger.info("==== STARTING ALL TESTS ====")
        
        # Step 1: Primary LLM inference
        step1_results = self.test_step1_primary_llm()
        
        # Step 2: First external validation
        step2_results = self.test_step2_huggingface_validation(step1_results)
        
        # Step 3: Optimizer LLM for refinement
        step3_results = self.test_step3_optimizer_llm(step1_results)
        
        # Step 4: Rerun with optimized prompts
        step4_results = self.test_step4_rerun_optimized(step1_results, step3_results)
        
        # Step 5: Second external validation
        step5_results = self.test_step5_second_validation(step1_results, step3_results, step4_results)
        
        # Test the full workflow integration
        self.test_full_workflow()
        
        # Calculate overall success
        steps_success = all(
            self.results["steps"].get(step, {}).get("success", False)
            for step in ["step1", "step2", "step3", "step4", "step5"]
        )
        full_workflow_success = self.results.get("full_workflow", {}).get("success", False)
        
        self.results["overall_success"] = steps_success and full_workflow_success
        
        # Save results to file
        output_path = self.save_results()
        
        # Print summary
        logger.info("==== TEST SUMMARY ====")
        logger.info(f"Overall success: {self.results['overall_success']}")
        for step, result in self.results["steps"].items():
            logger.info(f"{step}: {'✓' if result.get('success') else '✗'} - {result.get('message')}")
        logger.info(f"Full workflow: {'✓' if self.results.get('full_workflow', {}).get('success') else '✗'}")
        logger.info(f"Results saved to: {output_path}")
        
        return self.results


def main():
    """Run the workflow integration tests."""
    logger.info("Starting workflow integration tests")
    tester = WorkflowIntegrationTester()
    results = tester.run_all_tests()
    
    if results["overall_success"]:
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Some tests failed. See log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())