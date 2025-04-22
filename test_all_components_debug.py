#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Script with Real API Calls and Debug Output

This script tests all platform components with real API calls using 10 medical cases,
displaying detailed debugging output for each component including input/output pairs.

Usage:
    python test_all_components_debug.py

This script will:
1. Test all components individually with real data
2. Show detailed outputs from each component
3. Run an end-to-end workflow simulation
4. Save detailed logs to a report file
"""

import os
import sys
import json
import yaml
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

# Test case data - 10 medical cases with input and expected output
MEDICAL_CASES = [
    {
        "user_input": "A 45-year-old male presents with sudden onset crushing chest pain radiating to his left arm and jaw. Pain started 2 hours ago while resting. He is diaphoretic with BP 160/95, pulse 110, respirations 22. ECG shows ST-segment elevation in leads II, III, and aVF.",
        "ground_truth_output": "Acute myocardial infarction (specifically, inferior wall STEMI)"
    },
    {
        "user_input": "A 68-year-old female with a history of smoking presents with progressive shortness of breath over 3 months, weight loss, and a persistent cough. Chest X-ray shows a 3 cm mass in the right upper lobe. CT reveals mediastinal lymphadenopathy.",
        "ground_truth_output": "Lung cancer (likely non-small cell lung carcinoma)"
    },
    {
        "user_input": "A 25-year-old woman presents with fatigue, cold intolerance, weight gain, and constipation for the past 6 months. Physical exam reveals dry skin, brittle hair, and slowed reflexes. TSH is elevated at 12 mIU/L with low T4.",
        "ground_truth_output": "Hypothyroidism"
    },
    {
        "user_input": "A 55-year-old male with history of hypertension and smoking presents with sudden onset of worst headache of his life, vomiting, and neck stiffness. BP is 180/110, confusion is present on exam. CT scan shows blood in the subarachnoid space.",
        "ground_truth_output": "Subarachnoid hemorrhage"
    },
    {
        "user_input": "A 30-year-old female presents with episodic palpitations, tremors, weight loss despite increased appetite, heat intolerance, and anxiety. Examination reveals tachycardia, fine tremor, and exophthalmos. TSH is suppressed and free T4 is elevated.",
        "ground_truth_output": "Graves' disease"
    },
    {
        "user_input": "A 60-year-old male with a history of alcoholism presents with confusion, jaundice, ascites, and spider angiomas. Lab work shows elevated liver enzymes, low albumin, and prolonged PT/INR. Ultrasound reveals a nodular liver with splenomegaly.",
        "ground_truth_output": "Cirrhosis"
    },
    {
        "user_input": "A 22-year-old female presents with malar rash, joint pain, fatigue, and photosensitivity. Lab results show positive ANA, anti-dsDNA antibodies, and low complement levels. CBC shows mild leukopenia and lymphopenia.",
        "ground_truth_output": "Systemic lupus erythematosus"
    },
    {
        "user_input": "An 8-year-old boy presents with fever, sore throat, tender cervical lymphadenopathy, and tonsillar exudates. Rapid strep test is positive. No cough or rhinorrhea is present.",
        "ground_truth_output": "Streptococcal pharyngitis (Strep throat)"
    },
    {
        "user_input": "A 70-year-old male with a 40 pack-year smoking history presents with chronic productive cough, progressive dyspnea, and recurrent respiratory infections. Spirometry shows FEV1/FVC < 0.7 and FEV1 45% of predicted.",
        "ground_truth_output": "Chronic obstructive pulmonary disease (COPD)"
    },
    {
        "user_input": "A 50-year-old female with a history of hypertension reports severe headaches, palpitations, and excessive sweating. BP is 190/110, and plasma metanephrines are elevated. CT abdomen reveals a 3 cm right adrenal mass.",
        "ground_truth_output": "Pheochromocytoma"
    }
]

class ComponentTester:
    """Base class for individual component testing with detailed output reporting."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.passed = False
        self.error_message = None
        self.test_outputs = {}
        self.duration_ms = 0
    
    def run_test(self) -> bool:
        """Run the test with timing and error handling."""
        logger.info(f"\n{'=' * 80}\nTesting {self.component_name}\n{'=' * 80}")
        
        start_time = time.time()
        try:
            self._run_implementation()
            self.passed = True
            logger.info(f"{self.component_name} test PASSED")
        except Exception as e:
            self.error_message = str(e)
            logger.error(f"{self.component_name} test FAILED: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"{self.component_name} test completed in {self.duration_ms} ms")
        
        return self.passed
    
    def _run_implementation(self):
        """Implement specific test logic in subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_result(self) -> Dict[str, Any]:
        """Get the test result as a dictionary."""
        return {
            "component": self.component_name,
            "passed": self.passed,
            "error": self.error_message,
            "duration_ms": self.duration_ms,
            "outputs": self.test_outputs
        }


class ConfigTester(ComponentTester):
    """Test configuration loading."""
    
    def __init__(self):
        super().__init__("Configuration")
    
    def _run_implementation(self):
        # Load the configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['gemini', 'app', 'optimizer', 'evaluation', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required section '{section}' missing from config")
        
        # Debug output - show key configuration values
        self.test_outputs["gemini_model"] = config['gemini']['model_name']
        self.test_outputs["temperature"] = config['gemini']['temperature']
        self.test_outputs["optimizer_model"] = config['optimizer']['model_name']
        self.test_outputs["optimizer_strategies"] = config['optimizer']['strategies']
        self.test_outputs["evaluation_metrics"] = config['evaluation']['metrics']
        self.test_outputs["max_iterations"] = config['training']['default_max_iterations']
        
        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  Gemini model: {self.test_outputs['gemini_model']}")
        logger.info(f"  Optimizer model: {self.test_outputs['optimizer_model']}")
        logger.info(f"  Available strategies: {', '.join(self.test_outputs['optimizer_strategies'])}")


class DataModuleTester(ComponentTester):
    """Test data module with sample medical cases."""
    
    def __init__(self):
        super().__init__("Data Module")
    
    def _run_implementation(self):
        from app.data_module import DataModule
        
        # Initialize data module
        data_module = DataModule(base_dir='data')
        
        # Format cases as CSV
        cases_csv = "\n".join([
            f"{case['user_input']},{case['ground_truth_output']}"
            for case in MEDICAL_CASES
        ])
        
        # Load examples from text content
        train_examples, validation_examples = data_module.load_examples_from_text(cases_csv)
        
        # Check that examples were loaded correctly
        if len(train_examples) + len(validation_examples) != len(MEDICAL_CASES):
            raise ValueError(f"Expected {len(MEDICAL_CASES)} total examples, got {len(train_examples) + len(validation_examples)}")
        
        # Test batch retrieval
        batch = data_module.get_batch(batch_size=5)  # Get 5 examples
        
        # Save test outputs
        self.test_outputs["train_count"] = len(train_examples)
        self.test_outputs["validation_count"] = len(validation_examples)
        self.test_outputs["batch_size"] = len(batch)
        self.test_outputs["sample_train_example"] = train_examples[0] if train_examples else None
        self.test_outputs["sample_batch"] = batch[0] if batch else None
        
        logger.info(f"Data module loaded {len(train_examples)} training and {len(validation_examples)} validation examples")
        if train_examples:
            logger.info(f"Sample train example input: {train_examples[0]['user_input'][:50]}...")


class LLMClientTester(ComponentTester):
    """Test LLM client with real API calls."""
    
    def __init__(self):
        super().__init__("LLM Client")
    
    def _run_implementation(self):
        from app.llm_client import get_llm_response
        
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test basic system and user prompts
        system_prompt = "You are a medical diagnosis assistant. Provide a concise diagnosis."
        user_input = MEDICAL_CASES[0]["user_input"]  # Use the first test case
        output_prompt = "Give a single diagnosis that best explains the symptoms, without explanations."
        
        # Make actual API call
        start_time = time.time()
        response = get_llm_response(
            system_prompt=system_prompt,
            user_input=user_input,
            output_prompt=output_prompt,
            config=config.get('gemini', {})
        )
        api_call_time = round((time.time() - start_time) * 1000, 2)
        
        # Check that we got a valid response
        if not response or len(response) < 5:
            raise ValueError(f"Invalid or empty response from LLM: {response}")
        
        # Record test outputs
        self.test_outputs["model_used"] = config['gemini']['model_name']
        self.test_outputs["api_call_time_ms"] = api_call_time
        self.test_outputs["input"] = user_input[:100] + "..." if len(user_input) > 100 else user_input
        self.test_outputs["response"] = response
        self.test_outputs["system_prompt"] = system_prompt
        self.test_outputs["output_prompt"] = output_prompt
        
        logger.info(f"LLM client returned response in {api_call_time} ms:")
        logger.info(f"  Response: {response}")


class EvaluatorTester(ComponentTester):
    """Test evaluator on medical diagnosis cases."""
    
    def __init__(self):
        super().__init__("Evaluator")
    
    def _run_implementation(self):
        from app.evaluator import calculate_score, evaluate_batch
        
        # Create a batch of test examples with ground truth and model responses
        examples = []
        for i, case in enumerate(MEDICAL_CASES[:5]):  # Test with 5 cases
            ground_truth = case["ground_truth_output"]
            
            # Add a perfect match
            if i == 0:
                examples.append({
                    "ground_truth_output": ground_truth,
                    "model_response": ground_truth,  # Exact match
                    "score": 1.0
                })
            # Add a good partial match
            elif i == 1:
                partial_response = ground_truth.split("(")[0].strip()  # Just first part
                score = calculate_score(partial_response, ground_truth)
                examples.append({
                    "ground_truth_output": ground_truth,
                    "model_response": partial_response,
                    "score": score
                })
            # Add a mediocre match
            elif i == 2:
                mediocre_response = f"The patient likely has {ground_truth} but additional tests are needed."
                score = calculate_score(mediocre_response, ground_truth)
                examples.append({
                    "ground_truth_output": ground_truth,
                    "model_response": mediocre_response,
                    "score": score
                })
            # Add a poor match
            elif i == 3:
                poor_response = "The differential diagnosis includes multiple possibilities."
                score = calculate_score(poor_response, ground_truth)
                examples.append({
                    "ground_truth_output": ground_truth,
                    "model_response": poor_response,
                    "score": score
                })
            # Add a completely wrong match
            else:
                wrong_response = "The patient is healthy with no significant findings."
                score = calculate_score(wrong_response, ground_truth)
                examples.append({
                    "ground_truth_output": ground_truth,
                    "model_response": wrong_response,
                    "score": score
                })
        
        # Evaluate the batch
        metrics = evaluate_batch(examples)
        
        # Record test outputs
        self.test_outputs["examples"] = examples
        self.test_outputs["metrics"] = metrics
        self.test_outputs["avg_score"] = metrics["avg_score"]
        self.test_outputs["perfect_matches"] = metrics["perfect_matches"]
        
        # Log each example score
        logger.info(f"Evaluator tested with {len(examples)} examples:")
        for i, example in enumerate(examples):
            logger.info(f"  Example {i+1}: " +
                      f"Score {example['score']:.2f}, " +
                      f"Ground truth: '{example['ground_truth_output']}', " +
                      f"Response: '{example['model_response']}'")
        logger.info(f"  Average score: {metrics['avg_score']:.2f}")
        logger.info(f"  Perfect matches: {metrics['perfect_matches']}/{metrics['total_examples']}")


class OptimizerTester(ComponentTester):
    """Test optimizer using real API calls."""
    
    def __init__(self):
        super().__init__("Optimizer")
    
    def _run_implementation(self):
        from app.optimizer import (
            load_optimizer_prompt, 
            select_examples_for_optimizer,
            optimize_prompts
        )
        
        # Test basic prompts
        current_system_prompt = (
            "You are a medical diagnosis assistant. Provide diagnoses based on patient information."
        )
        current_output_prompt = (
            "Give a single diagnosis that best explains the symptoms."
        )
        
        # Create example results
        examples = []
        for i, case in enumerate(MEDICAL_CASES[:5]):  # Use first 5 cases
            examples.append({
                "user_input": case["user_input"],
                "ground_truth_output": case["ground_truth_output"],
                "model_response": f"Simulated model response for case {i}",
                "score": 0.5 + (i * 0.1)  # Varying scores from 0.5 to 0.9
            })
        
        # Load optimizer prompt
        optimizer_prompt = load_optimizer_prompt(optimizer_type='reasoning_first')
        if not optimizer_prompt:
            raise ValueError("Failed to load optimizer prompt")
        
        # Optimize prompts with real API call
        start_time = time.time()
        optimization_result = optimize_prompts(
            current_system_prompt=current_system_prompt,
            current_output_prompt=current_output_prompt,
            examples=examples,
            optimizer_system_prompt=optimizer_prompt,
            strategy="reasoning_first"
        )
        api_call_time = round((time.time() - start_time) * 1000, 2)
        
        # Check for required keys
        required_keys = ["system_prompt", "output_prompt", "reasoning"]
        for key in required_keys:
            if key not in optimization_result:
                raise ValueError(f"Required key '{key}' not found in optimization result")
        
        # Check for changes in the prompts
        system_prompt_changed = optimization_result["system_prompt"] != current_system_prompt
        output_prompt_changed = optimization_result["output_prompt"] != current_output_prompt
        has_reasoning = bool(optimization_result.get("reasoning")) and len(optimization_result["reasoning"]) > 50
        
        if not (system_prompt_changed or output_prompt_changed or has_reasoning):
            raise ValueError("Optimization didn't produce any meaningful changes")
        
        # Record test outputs
        self.test_outputs["api_call_time_ms"] = api_call_time
        self.test_outputs["original_system_prompt"] = current_system_prompt
        self.test_outputs["optimized_system_prompt"] = optimization_result["system_prompt"]
        self.test_outputs["original_output_prompt"] = current_output_prompt
        self.test_outputs["optimized_output_prompt"] = optimization_result["output_prompt"]
        self.test_outputs["reasoning"] = optimization_result["reasoning"][:500] + "..." if len(optimization_result["reasoning"]) > 500 else optimization_result["reasoning"]
        self.test_outputs["system_prompt_changed"] = system_prompt_changed
        self.test_outputs["output_prompt_changed"] = output_prompt_changed
        
        logger.info(f"Optimizer completed in {api_call_time} ms")
        logger.info(f"  System prompt changed: {system_prompt_changed}")
        logger.info(f"  Output prompt changed: {output_prompt_changed}")
        
        if system_prompt_changed:
            logger.info(f"  Original system prompt: {current_system_prompt}")
            logger.info(f"  Optimized system prompt: {optimization_result['system_prompt']}")
        
        if output_prompt_changed:
            logger.info(f"  Original output prompt: {current_output_prompt}")
            logger.info(f"  Optimized output prompt: {optimization_result['output_prompt']}")
        
        if has_reasoning:
            reasoning_snippet = optimization_result["reasoning"][:200] + "..." if len(optimization_result["reasoning"]) > 200 else optimization_result["reasoning"]
            logger.info(f"  Optimizer reasoning: {reasoning_snippet}")


class ExperimentTrackerTester(ComponentTester):
    """Test experiment tracking functionality."""
    
    def __init__(self):
        super().__init__("Experiment Tracker")
    
    def _run_implementation(self):
        from app.experiment_tracker import ExperimentTracker
        
        # Create test directory
        test_dir = os.path.join('test_outputs', 'test_experiments')
        os.makedirs(test_dir, exist_ok=True)
        
        # Initialize experiment tracker
        tracker = ExperimentTracker(base_dir=test_dir)
        
        # Start a new experiment
        experiment_id = tracker.start_experiment()
        if not experiment_id:
            raise ValueError("Failed to start experiment")
        
        # Test saving multiple iterations
        system_prompts = [
            "You are a medical diagnosis expert. Provide accurate diagnoses for medical cases.",
            "You are a skilled diagnostician with expertise in all medical specialties. Analyze cases thoroughly.",
            "You are an expert physician with decades of clinical experience. Diagnose patients accurately."
        ]
        
        output_prompts = [
            "Provide a single, concise diagnosis.",
            "Give a specific diagnosis that explains all symptoms. Be precise and concise.",
            "Offer the most likely diagnosis with high confidence. Be specific and accurate."
        ]
        
        metrics = [
            {"avg_score": 0.65, "perfect_matches": 2, "total_examples": 10, "perfect_match_percent": 20.0},
            {"avg_score": 0.78, "perfect_matches": 4, "total_examples": 10, "perfect_match_percent": 40.0},
            {"avg_score": 0.89, "perfect_matches": 7, "total_examples": 10, "perfect_match_percent": 70.0}
        ]
        
        reasonings = [
            "Initial prompt needs improvement in specificity.",
            "Better prompt but still lacks structured diagnostic approach.",
            "Significant improvement with structured reasoning approach."
        ]
        
        # Save test iterations
        for i in range(3):
            saved = tracker.save_iteration(
                experiment_id=experiment_id,
                iteration=i+1,
                system_prompt=system_prompts[i],
                output_prompt=output_prompts[i],
                metrics=metrics[i],
                examples=MEDICAL_CASES[:2],  # Just use 2 examples for testing
                optimizer_reasoning=reasonings[i]
            )
            
            if not saved:
                raise ValueError(f"Failed to save iteration {i+1}")
        
        # Test getting iterations
        iterations = tracker.get_iterations(experiment_id)
        if len(iterations) != 3:
            raise ValueError(f"Expected 3 iterations, got {len(iterations)}")
        
        # Test loading experiment history
        history = tracker.load_experiment_history(experiment_id)
        if not history:
            raise ValueError("Failed to load experiment history")
        
        # Record test outputs
        self.test_outputs["experiment_id"] = experiment_id
        self.test_outputs["iterations_saved"] = len(iterations)
        self.test_outputs["final_metrics"] = iterations[-1]["metrics"]
        self.test_outputs["iterations"] = [{
            "iteration": it["iteration"],
            "system_prompt": it["system_prompt"][:50] + "..." if len(it["system_prompt"]) > 50 else it["system_prompt"],
            "metrics": it["metrics"]
        } for it in iterations]
        
        logger.info(f"Experiment tracking test completed with experiment_id: {experiment_id}")
        logger.info(f"  Saved {len(iterations)} iterations")
        logger.info(f"  Final metrics: avg_score={iterations[-1]['metrics']['avg_score']:.2f}, " +
                  f"perfect_matches={iterations[-1]['metrics']['perfect_matches']}")
        
        # Clean up test directory
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)


class EndToEndWorkflowTester(ComponentTester):
    """Test a complete workflow from data loading to optimization."""
    
    def __init__(self):
        super().__init__("End-to-End Workflow")
    
    def _run_implementation(self):
        from app.data_module import DataModule
        from app.llm_client import get_llm_response
        from app.evaluator import calculate_score, evaluate_batch
        from app.optimizer import optimize_prompts, load_optimizer_prompt
        from app.experiment_tracker import ExperimentTracker
        
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Step 1: Set up data
        logger.info("Step 1: Setting up test data")
        data_module = DataModule(base_dir='data')
        
        # Convert cases to CSV
        cases_csv = "\n".join([
            f"{case['user_input']},{case['ground_truth_output']}"
            for case in MEDICAL_CASES
        ])
        
        # Load examples from text content
        train_examples, validation_examples = data_module.load_examples_from_text(cases_csv, train_ratio=0.7)
        logger.info(f"  Loaded {len(train_examples)} training and {len(validation_examples)} validation examples")
        
        # Step 2: Set up prompts
        logger.info("Step 2: Setting up initial prompts")
        system_prompt = "You are a medical diagnosis assistant. Your task is to provide a specific diagnosis based on patient information."
        output_prompt = "Provide a concise, specific diagnosis that best explains all the symptoms and findings."
        
        # Step 3: Initialize experiment tracker
        logger.info("Step 3: Initializing experiment tracker")
        test_dir = os.path.join('test_outputs', 'e2e_test')
        os.makedirs(test_dir, exist_ok=True)
        tracker = ExperimentTracker(base_dir=test_dir)
        experiment_id = tracker.start_experiment()
        logger.info(f"  Experiment ID: {experiment_id}")
        
        # Step 4: Evaluate initial prompts on a limited batch
        logger.info("Step 4: Evaluating initial prompts")
        batch_size = 3  # Limit to 3 examples for testing
        batch = train_examples[:batch_size]
        
        evaluation_results = []
        for example in batch:
            user_input = example.get('user_input', '')
            ground_truth = example.get('ground_truth_output', '')
            
            # Get model response
            model_response = get_llm_response(
                system_prompt=system_prompt,
                user_input=user_input,
                output_prompt=output_prompt,
                config=config.get('gemini', {})
            )
            
            # Calculate score
            score = calculate_score(model_response, ground_truth)
            
            evaluation_results.append({
                'user_input': user_input,
                'ground_truth_output': ground_truth,
                'model_response': model_response,
                'score': score
            })
        
        # Calculate metrics
        metrics = evaluate_batch(evaluation_results)
        logger.info(f"  Initial evaluation: avg_score={metrics['avg_score']:.2f}, " +
                  f"perfect_matches={metrics['perfect_matches']}/{metrics['total_examples']}")
        
        # Step 5: Save initial iteration
        logger.info("Step 5: Saving initial results")
        tracker.save_iteration(
            experiment_id=experiment_id,
            iteration=1,
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            metrics=metrics,
            examples=evaluation_results,
            optimizer_reasoning=""
        )
        
        # Step 6: Optimize prompts
        logger.info("Step 6: Optimizing prompts")
        optimizer_prompt = load_optimizer_prompt('reasoning_first')
        optimization_result = optimize_prompts(
            current_system_prompt=system_prompt,
            current_output_prompt=output_prompt,
            examples=evaluation_results,
            optimizer_system_prompt=optimizer_prompt,
            strategy="reasoning_first"
        )
        
        optimized_system_prompt = optimization_result["system_prompt"]
        optimized_output_prompt = optimization_result["output_prompt"]
        optimizer_reasoning = optimization_result["reasoning"]
        
        logger.info(f"  Optimization complete")
        logger.info(f"  Original system prompt: {system_prompt}")
        logger.info(f"  Optimized system prompt: {optimized_system_prompt}")
        logger.info(f"  Original output prompt: {output_prompt}")
        logger.info(f"  Optimized output prompt: {optimized_output_prompt}")
        
        # Step 7: Evaluate optimized prompts
        logger.info("Step 7: Evaluating optimized prompts")
        
        optimized_evaluation_results = []
        for example in batch:
            user_input = example.get('user_input', '')
            ground_truth = example.get('ground_truth_output', '')
            
            # Get model response with optimized prompts
            optimized_response = get_llm_response(
                system_prompt=optimized_system_prompt,
                user_input=user_input,
                output_prompt=optimized_output_prompt,
                config=config.get('gemini', {})
            )
            
            # Calculate score
            score = calculate_score(optimized_response, ground_truth)
            
            optimized_evaluation_results.append({
                'user_input': user_input,
                'ground_truth_output': ground_truth,
                'model_response': optimized_response,
                'score': score
            })
        
        # Calculate metrics for optimized prompts
        optimized_metrics = evaluate_batch(optimized_evaluation_results)
        logger.info(f"  Optimized evaluation: avg_score={optimized_metrics['avg_score']:.2f}, " +
                  f"perfect_matches={optimized_metrics['perfect_matches']}/{optimized_metrics['total_examples']}")
        
        # Step 8: Save optimized iteration
        logger.info("Step 8: Saving optimized results")
        tracker.save_iteration(
            experiment_id=experiment_id,
            iteration=2,
            system_prompt=optimized_system_prompt,
            output_prompt=optimized_output_prompt,
            metrics=optimized_metrics,
            examples=optimized_evaluation_results,
            optimizer_reasoning=optimizer_reasoning
        )
        
        # Record test outputs
        self.test_outputs["experiment_id"] = experiment_id
        self.test_outputs["initial_metrics"] = metrics
        self.test_outputs["optimized_metrics"] = optimized_metrics
        self.test_outputs["score_improvement"] = optimized_metrics["avg_score"] - metrics["avg_score"]
        self.test_outputs["initial_examples"] = [{
            "user_input": ex["user_input"][:50] + "..." if len(ex["user_input"]) > 50 else ex["user_input"],
            "ground_truth": ex["ground_truth_output"],
            "response": ex["model_response"],
            "score": ex["score"]
        } for ex in evaluation_results]
        self.test_outputs["optimized_examples"] = [{
            "user_input": ex["user_input"][:50] + "..." if len(ex["user_input"]) > 50 else ex["user_input"],
            "ground_truth": ex["ground_truth_output"],
            "response": ex["model_response"],
            "score": ex["score"]
        } for ex in optimized_evaluation_results]
        
        logger.info("End-to-end workflow test completed successfully")
        
        # Clean up test directory
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)


def run_all_tests():
    """Run all component tests and return results."""
    testers = [
        ConfigTester(),
        DataModuleTester(),
        LLMClientTester(),
        EvaluatorTester(),
        OptimizerTester(),
        ExperimentTrackerTester(),
        EndToEndWorkflowTester()
    ]
    
    results = []
    for tester in testers:
        passed = tester.run_test()
        results.append(tester.get_result())
    
    passed_count = sum(1 for result in results if result["passed"])
    
    logger.info("\n" + "=" * 80)
    logger.info(f"All tests completed: {passed_count}/{len(results)} components passed")
    
    for result in results:
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        logger.info(f"{result['component']}: {status}")
        
        if not result["passed"]:
            logger.info(f"  Error: {result['error']}")
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join('test_outputs', f'detailed_test_results_{timestamp}.json')
    os.makedirs('test_outputs', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "summary": {
                "total_tests": len(results),
                "passed_tests": passed_count,
                "success_rate": passed_count / len(results) if results else 0
            },
            "results": results
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    # Make sure output directory exists
    os.makedirs('test_outputs', exist_ok=True)
    
    # Run all tests
    run_all_tests()