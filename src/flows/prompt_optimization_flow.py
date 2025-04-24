"""
Prompt Optimization Flow using Prefect 2.0 with PostgreSQL database integration
"""
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import uuid

from prefect import flow, task, get_run_logger
from prefect.deployments import run_deployment

from src.app.models.prompt_state import PromptState
from src.app.clients.vertex_client import VertexAIClient
from src.app.clients.hf_evaluator import EvaluatorService
from src.app.database.db import SessionLocal
from src.app.repositories.prompt_repository import PromptRepository
from src.app.repositories.experiment_repository import ExperimentRepository
from src.app.repositories.dataset_repository import DatasetRepository
from src.app.repositories.ml_settings_repository import MLSettingsRepository

logger = logging.getLogger(__name__)

@task(name="load_prompt_state")
def load_prompt_state(system_prompt_path: str, output_prompt_path: str) -> PromptState:
    """Load prompt state from files"""
    run_logger = get_run_logger()
    run_logger.info(f"Loading prompts from {system_prompt_path} and {output_prompt_path}")

    try:
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read().strip()

        with open(output_prompt_path, 'r') as f:
            output_prompt = f.read().strip()

        # Create prompt state
        prompt_state = PromptState(
            system_prompt=system_prompt,
            output_prompt=output_prompt
        )

        # Try to save to database
        try:
            db = SessionLocal()
            repo = PromptRepository(db)
            db_prompt = repo.create(
                system_prompt=system_prompt,
                output_prompt=output_prompt
            )
            prompt_state.id = str(db_prompt.id)
            run_logger.info(f"Saved prompt to database with ID: {prompt_state.id}")
        except Exception as e:
            run_logger.warning(f"Failed to save prompt to database: {str(e)}")
        finally:
            db.close()

        return prompt_state
    except Exception as e:
        run_logger.error(f"Error loading prompt state: {str(e)}")
        raise

@task(name="load_dataset")
def load_dataset(dataset_path: str, batch_size: int = 5, sample_k: int = 3) -> List[Dict]:
    """Load dataset from file"""
    run_logger = get_run_logger()
    run_logger.info(f"Loading dataset from {dataset_path}")

    import json
    import random

    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # Sample examples if needed
        if sample_k > 0 and sample_k < len(dataset):
            dataset = random.sample(dataset, min(sample_k, len(dataset)))

        # Try to save to database
        try:
            db = SessionLocal()
            repo = DatasetRepository(db)
            dataset_name = os.path.basename(dataset_path)
            db_dataset = repo.create(
                name=dataset_name,
                file_path=dataset_path,
                row_count=len(dataset)
            )
            run_logger.info(f"Saved dataset to database with ID: {db_dataset.id}")
        except Exception as e:
            run_logger.warning(f"Failed to save dataset to database: {str(e)}")
        finally:
            db.close()

        return dataset[:batch_size]
    except Exception as e:
        run_logger.error(f"Error loading dataset: {str(e)}")
        raise

@task(name="primary_llm_inference")
def primary_llm_inference(prompt_state: PromptState, examples: List[Dict]) -> List[Dict]:
    """Run primary LLM inference on examples"""
    run_logger = get_run_logger()
    run_logger.info(f"Running primary LLM inference on {len(examples)} examples")

    try:
        db = SessionLocal()
        ml_settings_repo = MLSettingsRepository(db)
        model_config = ml_settings_repo.get_default()
        db.close()

        # Configure client with settings from database
        llm_client = VertexAIClient()
        if model_config:
            run_logger.info(f"Using model configuration: {model_config.name}")
            llm_client.configure(
                model_name=model_config.primary_model,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                top_p=model_config.top_p,
                top_k=model_config.top_k
            )

        results = []

        for example in examples:
            user_input = example.get("input", "")
            expected_output = example.get("expected_output", "")

            response = llm_client.generate(
                system_prompt=prompt_state.system_prompt,
                user_prompt=user_input,
                output_prompt=prompt_state.output_prompt
            )

            results.append({
                "input": user_input,
                "expected_output": expected_output,
                "actual_output": response,
                "prompt_id": prompt_state.id
            })

        return results
    except Exception as e:
        run_logger.error(f"Error in primary LLM inference: {str(e)}")
        raise

@task(name="evaluate_metrics")
def evaluate_metrics(examples: List[Dict], metric_names: List[str]) -> Dict[str, float]:
    """Evaluate metrics using HuggingFace Evaluator"""
    run_logger = get_run_logger()
    run_logger.info(f"Evaluating metrics: {', '.join(metric_names)}")

    try:
        evaluator = EvaluatorService()

        predictions = [example["actual_output"] for example in examples]
        references = [example["expected_output"] for example in examples]

        # Calculate all requested metrics
        metrics_results = {}
        for metric_name in metric_names:
            metric_score = evaluator.calculate_metric(
                metric_name=metric_name,
                predictions=predictions,
                references=references
            )
            metrics_results[metric_name] = metric_score

        # Calculate average score
        if metrics_results:
            avg_score = sum(metrics_results.values()) / len(metrics_results)
            metrics_results["avg_score"] = avg_score

        run_logger.info(f"Metrics: {metrics_results}")
        return metrics_results
    except Exception as e:
        run_logger.error(f"Error evaluating metrics: {str(e)}")
        raise

@task(name="optimize_prompt")
def optimize_prompt(
    prompt_state: PromptState,
    examples: List[Dict],
    metrics: Dict[str, float],
    optimizer_strategy: str = "reasoning_first"
) -> PromptState:
    """Optimize prompt using LLM"""
    run_logger = get_run_logger()
    run_logger.info(f"Optimizing prompt using strategy: {optimizer_strategy}")

    try:
        db = SessionLocal()
        ml_settings_repo = MLSettingsRepository(db)
        model_config = ml_settings_repo.get_default()
        db.close()

        # Configure optimizer client with settings from database
        optimizer_client = VertexAIClient()
        if model_config:
            run_logger.info(f"Using optimizer configuration: {model_config.name}")
            optimizer_client.configure(
                model_name=model_config.optimizer_model,
                temperature=0.7,  # Use higher temperature for optimization
                max_tokens=model_config.max_tokens * 2,  # Need more tokens for reasoning
                top_p=0.95
            )

        # Prepare optimizer input
        optimizer_prompt = get_optimizer_prompt(optimizer_strategy)

        # Format examples and metrics for the optimizer
        formatted_examples = format_examples_for_optimizer(examples)
        formatted_metrics = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        # Build the full optimizer input
        optimizer_input = f"""
CURRENT SYSTEM PROMPT:
{prompt_state.system_prompt}

CURRENT OUTPUT PROMPT:
{prompt_state.output_prompt}

EXAMPLES AND CURRENT RESULTS:
{formatted_examples}

METRICS:
{formatted_metrics}

Please analyze and improve both prompts based on the examples and metrics above.
"""

        # Run the optimizer
        response = optimizer_client.generate(
            system_prompt=optimizer_prompt,
            user_prompt=optimizer_input
        )

        # Parse the response to get new prompts
        new_system_prompt, new_output_prompt, reasoning = parse_optimizer_response(
            response, prompt_state.system_prompt, prompt_state.output_prompt
        )

        # Create optimized prompt state
        optimized_prompt = PromptState(
            system_prompt=new_system_prompt,
            output_prompt=new_output_prompt,
            parent_id=prompt_state.id
        )

        # Save to database
        try:
            db = SessionLocal()
            repo = PromptRepository(db)
            metadata = {
                "optimizer_strategy": optimizer_strategy,
                "metrics": metrics,
                "reasoning": reasoning
            }
            db_prompt = repo.create(
                system_prompt=new_system_prompt,
                output_prompt=new_output_prompt,
                parent_id=prompt_state.id,
                metadata=metadata
            )
            optimized_prompt.id = str(db_prompt.id)
            run_logger.info(f"Saved optimized prompt to database with ID: {optimized_prompt.id}")
        except Exception as e:
            run_logger.warning(f"Failed to save optimized prompt to database: {str(e)}")
        finally:
            db.close()

        return optimized_prompt
    except Exception as e:
        run_logger.error(f"Error optimizing prompt: {str(e)}")
        raise

def get_optimizer_prompt(optimizer_strategy: str) -> str:
    """Get the appropriate optimizer prompt based on strategy"""
    if optimizer_strategy == "reasoning_first":
        # Try to load from file first
        try:
            with open("prompts/optimizer/reasoning_first.txt", "r") as f:
                return f.read().strip()
        except:
            pass

        # Default optimizer prompt
        return """You are an expert prompt engineer specializing in optimizing system and output prompts for LLMs.
Your task is to analyze examples, results, and metrics, then suggest improvements to both prompts.
First, analyze exactly what's going wrong with the current responses compared to expected outputs.
Then, suggest specific changes to both prompts to improve performance.
Provide your response in this format:

REASONING: [Your detailed analysis of what's working and what's not]

IMPROVED SYSTEM PROMPT:
[Your improved system prompt]

IMPROVED OUTPUT PROMPT:
[Your improved output prompt]"""
    else:
        return """You are an expert prompt engineer. Analyze the current prompts, examples, and metrics.
Then suggest improved versions of both the system prompt and output prompt.

Format your response as:

IMPROVED SYSTEM PROMPT:
[Your improved system prompt]

IMPROVED OUTPUT PROMPT:
[Your improved output prompt]

REASONING:
[Brief explanation of your changes]"""

def format_examples_for_optimizer(examples: List[Dict]) -> str:
    """Format examples for the optimizer input"""
    formatted = []

    for i, example in enumerate(examples, 1):
        formatted.append(f"EXAMPLE {i}:")
        formatted.append(f"Input: {example['input']}")
        formatted.append(f"Expected: {example['expected_output']}")
        formatted.append(f"Actual: {example['actual_output']}")
        formatted.append("")

    return "\n".join(formatted)

def parse_optimizer_response(response: str, original_system_prompt: str, original_output_prompt: str) -> tuple:
    """Parse the optimizer response to extract new prompts and reasoning"""
    system_prompt = original_system_prompt
    output_prompt = original_output_prompt
    reasoning = ""

    # Extract reasoning
    if "REASONING:" in response:
        reasoning_parts = response.split("REASONING:")
        if len(reasoning_parts) > 1:
            reasoning_text = reasoning_parts[1].strip()
            next_header = None
            for header in ["IMPROVED SYSTEM PROMPT:", "SYSTEM PROMPT:", "IMPROVED OUTPUT PROMPT:", "OUTPUT PROMPT:"]:
                if header in reasoning_text:
                    next_header = reasoning_text.find(header)
                    break

            if next_header:
                reasoning = reasoning_text[:next_header].strip()
            else:
                reasoning = reasoning_text

    # Extract system prompt
    for header in ["IMPROVED SYSTEM PROMPT:", "SYSTEM PROMPT:"]:
        if header in response:
            parts = response.split(header)
            if len(parts) > 1:
                system_text = parts[1].strip()
                next_header = None
                for h in ["IMPROVED OUTPUT PROMPT:", "OUTPUT PROMPT:", "REASONING:"]:
                    if h in system_text:
                        next_header = system_text.find(h)
                        break

                if next_header:
                    system_prompt = system_text[:next_header].strip()
                else:
                    system_prompt = system_text

    # Extract output prompt
    for header in ["IMPROVED OUTPUT PROMPT:", "OUTPUT PROMPT:"]:
        if header in response:
            parts = response.split(header)
            if len(parts) > 1:
                output_text = parts[1].strip()
                next_header = None
                for h in ["IMPROVED SYSTEM PROMPT:", "SYSTEM PROMPT:", "REASONING:"]:
                    if h in output_text:
                        next_header = output_text.find(h)
                        break

                if next_header:
                    output_prompt = output_text[:next_header].strip()
                else:
                    output_prompt = output_text

    return system_prompt, output_prompt, reasoning

@flow(name="Prompt Optimization Flow", description="Five-step prompt optimization workflow")
def prompt_optimization_flow(
    system_prompt_path: str,
    output_prompt_path: str,
    dataset_path: str,
    experiment_id: Optional[str] = None,
    metric_names: List[str] = ["exact_match", "bleu", "rouge"],
    target_metric: str = "avg_score",
    target_threshold: float = 0.85,
    patience: int = 3,
    max_iterations: int = 5,
    batch_size: int = 5,
    sample_k: int = 3,
    optimizer_strategy: str = "reasoning_first",
    model_config_id: Optional[str] = None,
) -> Dict:
    """
    Main flow for prompt optimization using the 5-step process

    Args:
        system_prompt_path: Path to system prompt file
        output_prompt_path: Path to output prompt file
        dataset_path: Path to dataset file
        experiment_id: Optional ID for experiment tracking in database
        metric_names: List of metrics to evaluate
        target_metric: Target metric to optimize
        target_threshold: Threshold to reach for early stopping
        patience: Number of iterations without improvement before stopping
        max_iterations: Maximum number of iterations
        batch_size: Number of examples to process in each batch
        sample_k: Number of examples to sample for optimization
        optimizer_strategy: Strategy for optimization
        model_config_id: Optional ID for ML model configuration

    Returns:
        Dict containing results and metrics
    """
    # Get database session if we need to store results
    db_session = None
    if experiment_id:
        from sqlalchemy.orm import Session
        from src.app.database.db import SessionLocal
        db_session = SessionLocal()

    try:
        run_logger = get_run_logger()
        run_logger.info(f"Starting prompt optimization flow for {dataset_path}")

        # Load initial state
        prompt_state = load_prompt_state(system_prompt_path, output_prompt_path)
        examples = load_dataset(dataset_path, batch_size, sample_k)

        # Get or create experiment  (Modified to use db_session)
        experiment_repo = ExperimentRepository(db_session)
        if experiment_id:
            experiment = experiment_repo.get_by_id(experiment_id)
            if not experiment:
                run_logger.error(f"Experiment with ID {experiment_id} not found")
                raise ValueError(f"Experiment with ID {experiment_id} not found")
            run_logger.info(f"Using existing experiment: {experiment.name} (ID: {experiment_id})")
            experiment_repo.update_status(experiment_id, "running") #Using db_session
        else:
            dataset_name = os.path.basename(dataset_path)
            dataset_repo = DatasetRepository(db_session)
            dataset = dataset_repo.get_by_name(dataset_name)
            if not dataset:
                dataset = dataset_repo.create(
                    name=dataset_name,
                    file_path=dataset_path,
                    row_count=len(examples)
                )
            experiment = experiment_repo.create(
                name=f"Optimization {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                initial_prompt_id=prompt_state.id,
                dataset_id=str(dataset.id),
                metrics=metric_names,
                max_epochs=max_iterations,
                target_threshold=target_threshold
            )
            experiment_id = str(experiment.id)
            run_logger.info(f"Created new experiment: {experiment.name} (ID: {experiment_id})")

        # Optimization loop
        best_metric = 0
        best_prompt = prompt_state
        no_improvement_count = 0

        for iteration in range(max_iterations):
            run_logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")

            # Step 1: Primary LLM Inference
            examples_with_results = primary_llm_inference(prompt_state, examples)

            # Step 2: HuggingFace Evaluation
            metrics = evaluate_metrics(examples_with_results, metric_names)

            # Save metrics to database (Modified to use db_session)
            metrics_record = experiment_repo.add_metrics_record(
                experiment_id=experiment_id,
                epoch=iteration + 1,
                metrics=metrics,
                prompt_id=prompt_state.id
            )
            run_logger.info(f"Saved metrics record for epoch {iteration + 1}")

            # Check if target_metric exists in metrics
            current_metric = metrics.get(target_metric, 0)
            run_logger.info(f"Current {target_metric}: {current_metric}, Best: {best_metric}, Target: {target_threshold}")

            # Update best metric and prompt if improved
            if current_metric > best_metric:
                best_metric = current_metric
                best_prompt = prompt_state
                no_improvement_count = 0
                experiment_repo.update_best_prompt(experiment_id, prompt_state.id) #Using db_session
            else:
                no_improvement_count += 1

            # Early stopping if target reached or no improvement for 'patience' iterations
            if current_metric >= target_threshold:
                run_logger.info(f"Target threshold {target_threshold} reached. Stopping optimization.")
                break
            if no_improvement_count >= patience:
                run_logger.info(f"No improvement for {patience} iterations. Stopping optimization.")
                break

            # Step 3: Optimize Prompt
            prompt_state = optimize_prompt(prompt_state, examples_with_results, metrics, optimizer_strategy)

            # Wait a bit to avoid rate limiting
            time.sleep(2)

        # Final status update (Modified to use db_session)
        experiment_repo.update_status(experiment_id, "completed")
        experiment_repo.update_best_prompt(experiment_id, best_prompt.id) #Using db_session

        run_logger.info(f"Prompt optimization completed. Best {target_metric}: {best_metric}")

        return {
            "experiment_id": experiment_id,
            "best_prompt_id": best_prompt.id,
            "best_metric": best_metric,
            "target_metric": target_metric,
            "iterations_completed": min(max_iterations, iteration + 1),
            "target_reached": best_metric >= target_threshold
        }
    except Exception as e:
        run_logger.error(f"Error during prompt optimization: {str(e)}")
        if db_session:
            db_session.rollback()
        raise
    finally:
        if db_session:
            db_session.close()

# Prefect entry point for deployment
def start_optimization_flow(experiment_id: str):
    """Start a prompt optimization flow from an experiment ID"""
    # Get experiment details from database
    db = SessionLocal()
    experiment_repo = ExperimentRepository(db)
    prompt_repo = PromptRepository(db)
    dataset_repo = DatasetRepository(db)

    try:
        experiment = experiment_repo.get_by_id(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment with ID {experiment_id} not found")

        initial_prompt = prompt_repo.get_by_id(str(experiment.initial_prompt_id))
        if not initial_prompt:
            raise ValueError(f"Initial prompt with ID {experiment.initial_prompt_id} not found")

        dataset = dataset_repo.get_by_id(str(experiment.dataset_id))
        if not dataset:
            raise ValueError(f"Dataset with ID {experiment.dataset_id} not found")

        # Save prompt files
        temp_dir = os.path.join("temp", experiment_id)
        os.makedirs(temp_dir, exist_ok=True)

        system_prompt_path = os.path.join(temp_dir, "system_prompt.txt")
        output_prompt_path = os.path.join(temp_dir, "output_prompt.txt")

        with open(system_prompt_path, "w") as f:
            f.write(initial_prompt.system_prompt)

        with open(output_prompt_path, "w") as f:
            f.write(initial_prompt.output_prompt)

        # Start the flow
        return prompt_optimization_flow(
            system_prompt_path=system_prompt_path,
            output_prompt_path=output_prompt_path,
            dataset_path=dataset.file_path,
            metric_names=experiment.metrics,
            target_metric="avg_score",  # Default
            target_threshold=experiment.target_threshold,
            max_iterations=experiment.max_epochs,
            experiment_id=experiment_id
        )
    finally:
        db.close()

if __name__ == "__main__":
    # Example usage for local testing
    result = prompt_optimization_flow(
        system_prompt_path="prompts/system/medical_diagnosis.txt",
        output_prompt_path="prompts/output/medical_diagnosis.txt",
        dataset_path="data/train/examples.json",
        metric_names=["exact_match", "bleu"],
        target_metric="avg_score",
        max_iterations=3
    )

    print(f"Optimization result: {result}")