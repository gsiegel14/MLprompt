
"""
Prefect tasks for evaluation operations
"""
from prefect import task
from typing import Dict, List, Any, Optional
from app.evaluator import calculate_score, evaluate_batch
from app.huggingface_client import evaluate_metrics

@task(name="hf-eval-baseline", retries=2)
def hf_eval_baseline(dataset_dict, metric_names=None):
    """Evaluate baseline predictions with both internal and HF metrics"""
    if metric_names is None:
        metric_names = ["exact_match", "bleu"]
    
    # Extract predictions and references for HF metrics
    predictions = [item.get("model_response", "") for item in dataset_dict]
    references = [item.get("ground_truth_output", "") for item in dataset_dict]
    
    # Compute HF metrics
    hf_metrics = evaluate_metrics(predictions, references, metric_names)
    
    # Compute internal metrics
    internal_metrics = evaluate_batch([
        {
            'model_response': pred,
            'ground_truth_output': ref
        } for pred, ref in zip(predictions, references)
    ])
    
    # Merge metrics
    metrics = {**hf_metrics, **internal_metrics}
    
    # Return both the dataset and the metrics
    return {
        "dataset": dataset_dict,
        "metrics": metrics
    }

@task(name="hf-eval-refined", retries=2)
def hf_eval_refined(dataset_dict, metric_names=None):
    """Evaluate refined predictions with both internal and HF metrics"""
    if metric_names is None:
        metric_names = ["exact_match", "bleu"]
    
    # Extract predictions and references for HF metrics
    predictions = [item.get("refined_response", "") for item in dataset_dict]
    references = [item.get("ground_truth_output", "") for item in dataset_dict]
    
    # Compute HF metrics
    hf_metrics = evaluate_metrics(predictions, references, metric_names)
    
    # Compute internal metrics
    internal_metrics = evaluate_batch([
        {
            'model_response': pred,
            'ground_truth_output': ref
        } for pred, ref in zip(predictions, references)
    ])
    
    # Merge metrics
    metrics = {**hf_metrics, **internal_metrics}
    
    # Return both the dataset and the metrics
    return {
        "dataset": dataset_dict,
        "metrics": metrics
    }
