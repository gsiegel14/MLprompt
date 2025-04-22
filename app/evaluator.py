import re
import difflib
import logging
import yaml
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {
            'evaluation': {
                'metrics': ['exact_match', 'semantic_similarity', 'keyword_match'],
                'perfect_threshold': 0.9
            }
        }

def calculate_score(model_response: str, ground_truth_output: str) -> float:
    """
    Calculate a simple evaluation score between model response and ground truth.
    
    Args:
        model_response (str): The response from the LLM
        ground_truth_output (str): The expected output
        
    Returns:
        float: Score between 0 and 1
    """
    if not model_response or not ground_truth_output:
        return 0.0
    
    # Clean both texts (lowercase, remove extra spaces)
    model_response_clean = re.sub(r'\s+', ' ', model_response.lower().strip())
    ground_truth_clean = re.sub(r'\s+', ' ', ground_truth_output.lower().strip())
    
    # Check for exact match
    if model_response_clean == ground_truth_clean:
        return 1.0
    
    # Check how similar they are using difflib
    similarity_ratio = difflib.SequenceMatcher(None, model_response_clean, ground_truth_clean).ratio()
    
    # Additional checks for common patterns
    contains_score = 0.0
    
    # Check if the model response contains key parts of the ground truth
    # Split ground truth into "words" and check if they exist in the response
    ground_truth_words = set(ground_truth_clean.split())
    model_response_words = set(model_response_clean.split())
    
    if ground_truth_words:
        matches = ground_truth_words.intersection(model_response_words)
        contains_score = len(matches) / len(ground_truth_words)
    
    # Weighted score (70% sequence similarity, 30% key word matching)
    weighted_score = 0.7 * similarity_ratio + 0.3 * contains_score
    
    return weighted_score

def evaluate_batch(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run evaluation metrics on a batch of examples.
    
    Args:
        examples (list): List of dictionaries with ground_truth_output and model_response
        
    Returns:
        dict: Evaluation metrics
    """
    if not examples:
        return {
            "avg_score": 0.0,
            "perfect_matches": 0,
            "total_examples": 0,
            "perfect_match_percent": 0.0
        }
    
    # Load the configuration
    config = load_config()
    eval_config = config.get('evaluation', {})
    perfect_threshold = eval_config.get('perfect_threshold', 0.9)
    
    total_score = 0.0
    perfect_matches = 0
    
    for example in examples:
        model_response = example.get('model_response', '')
        ground_truth = example.get('ground_truth_output', '')
        
        if not model_response or not ground_truth:
            score = 0.0
        else:
            score = calculate_score(model_response, ground_truth)
        
        # Add the score to the example if not already there
        if 'score' not in example:
            example['score'] = score
        
        total_score += score
        
        # Consider a match "perfect" if score > threshold
        if score >= perfect_threshold:
            perfect_matches += 1
    
    total_examples = len(examples)
    
    return {
        "avg_score": total_score / total_examples if total_examples > 0 else 0.0,
        "perfect_matches": perfect_matches,
        "total_examples": total_examples,
        "perfect_match_percent": (perfect_matches / total_examples) * 100 if total_examples > 0 else 0.0
    }