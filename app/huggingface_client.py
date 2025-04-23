"""
Hugging Face API Client for Evaluation Metrics

This module provides functions to interact with Hugging Face's Inference API
for evaluation metrics on prompt engineering tasks.
"""

import os
import logging
import requests
import time
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

# Get the Hugging Face token from environment variables
HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")

def validate_hf_token():
    """
    Validate that the Hugging Face token is set.
    
    Raises:
        ValueError: If the token is not set
    """
    if not HF_TOKEN:
        raise ValueError("HUGGING_FACE_TOKEN is not set. Please set this environment variable.")

def evaluate_metrics(
    predictions: List[str], 
    references: List[str], 
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate predictions against references using Hugging Face metrics.
    
    Args:
        predictions (List[str]): List of model predictions
        references (List[str]): List of ground truth references
        metrics (List[str], optional): List of metrics to evaluate. Defaults to ["exact_match", "bleu"].
    
    Returns:
        Dict[str, Any]: Dictionary containing evaluation results
    """
    validate_hf_token()
    
    if metrics is None:
        metrics = ["exact_match", "bleu"]
    
    if len(predictions) != len(references):
        raise ValueError(f"Number of predictions ({len(predictions)}) must match number of references ({len(references)})")
    
    if len(predictions) == 0:
        return {metric: 0.0 for metric in metrics}
    
    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # We'll use the Hugging Face Evaluate API endpoint
    url = "https://api-inference.huggingface.co/models/evaluate-metric"
    
    results = {}
    
    # Process each metric independently as some might fail
    for metric in metrics:
        payload = {
            "data": {
                "predictions": predictions,
                "references": references
            },
            "metric": metric
        }
        
        # Implement retry logic
        max_retries = 3
        retry_delay = 2
        success = False
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt+1}/{max_retries} for metric {metric}")
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    metric_result = response.json()
                    
                    # Different metrics have different response formats
                    if isinstance(metric_result, dict):
                        # Some metrics return multiple values
                        for key, value in metric_result.items():
                            results[f"{metric}_{key}" if key != metric else metric] = value
                    else:
                        # Some metrics return a single value
                        results[metric] = metric_result
                    
                    success = True
                    break
                elif response.status_code == 429:
                    # Rate limit - wait longer
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit for HF API. Waiting {wait_time}s before retry.")
                    time.sleep(wait_time)
                else:
                    # Other errors
                    logger.warning(f"Error from HF API for metric {metric}: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"Exception during HF API call for metric {metric}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        if not success:
            # If all retries failed, add a placeholder result
            results[metric] = None
            logger.error(f"Failed to get results for metric {metric} after {max_retries} attempts")
    
    return results

def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score for a list of predictions and references.
    
    Args:
        predictions (List[str]): List of model predictions
        references (List[str]): List of ground truth references
    
    Returns:
        float: BLEU score
    """
    results = evaluate_metrics(predictions, references, ["bleu"])
    return results.get("bleu", 0.0)

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match score for a list of predictions and references.
    
    Args:
        predictions (List[str]): List of model predictions
        references (List[str]): List of ground truth references
    
    Returns:
        float: Exact match score
    """
    results = evaluate_metrics(predictions, references, ["exact_match"])
    return results.get("exact_match", 0.0)

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE score for a list of predictions and references.
    
    Args:
        predictions (List[str]): List of model predictions
        references (List[str]): List of ground truth references
    
    Returns:
        Dict[str, float]: Dictionary with rouge-1, rouge-2, and rouge-l scores
    """
    results = evaluate_metrics(predictions, references, ["rouge"])
    
    # Extract ROUGE scores which are usually nested
    rouge_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                rouge_results[f"{key}_{subkey}"] = subvalue
        else:
            rouge_results[key] = value
    
    return rouge_results

def validate_api_connection() -> bool:
    """
    Validate that we can connect to the Hugging Face API.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        validate_hf_token()
        
        # Test with a minimal example
        results = evaluate_metrics(["test"], ["test"], ["exact_match"])
        
        # If we get here, connection is working
        logger.info("Successfully connected to Hugging Face API")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Hugging Face API: {str(e)}")
        return False