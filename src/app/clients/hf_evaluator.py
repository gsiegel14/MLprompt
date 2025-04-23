
"""
HuggingFace evaluator for calculating metrics on model outputs
"""
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class EvaluatorService:
    """Service for evaluating model outputs using HuggingFace metrics"""
    
    def __init__(self):
        """Initialize the evaluator service"""
        try:
            import evaluate
            self.evaluate = evaluate
            logger.info("HuggingFace evaluate library initialized")
        except ImportError:
            logger.warning("HuggingFace evaluate library not installed. Metrics will be limited.")
            self.evaluate = None
    
    def evaluate(self, 
                predictions: List[str], 
                references: List[str], 
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate predictions against references using specified metrics
        
        Args:
            predictions: List of model prediction strings
            references: List of reference (ground truth) strings
            metrics: List of metric names to compute (default: exact_match, bleu)
            
        Returns:
            Dictionary mapping metric names to scores
        """
        if not self.evaluate:
            # Fallback to simple metrics if HuggingFace evaluate is not available
            return self._calculate_simple_metrics(predictions, references)
        
        if metrics is None:
            metrics = ["exact_match", "bleu"]
        
        results = {}
        
        # Process metrics one by one
        for metric_name in metrics:
            try:
                # Load the metric
                metric = self.evaluate.load(metric_name)
                
                # Compute the metric
                score = metric.compute(predictions=predictions, references=references)
                
                # Add to results
                if isinstance(score, dict):
                    # Some metrics return dictionaries with multiple values
                    for k, v in score.items():
                        results[f"{metric_name}_{k}"] = v
                else:
                    results[metric_name] = score
            except Exception as e:
                logger.error(f"Error computing metric {metric_name}: {str(e)}")
                results[metric_name] = None
        
        return results
    
    def _calculate_simple_metrics(self, 
                                predictions: List[str], 
                                references: List[str]) -> Dict[str, float]:
        """
        Calculate simple metrics without using HuggingFace
        
        Args:
            predictions: List of model prediction strings
            references: List of reference (ground truth) strings
            
        Returns:
            Dictionary with simple metrics (exact_match, token_match)
        """
        # Calculate exact match score
        exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        exact_match_score = exact_matches / len(predictions) if predictions else 0
        
        # Calculate token match (percentage of prediction tokens in reference)
        token_matches = []
        for p, r in zip(predictions, references):
            p_tokens = set(p.lower().split())
            r_tokens = set(r.lower().split())
            if not p_tokens:
                token_matches.append(0)
            else:
                matches = len(p_tokens.intersection(r_tokens))
                token_matches.append(matches / len(p_tokens))
        
        token_match_score = sum(token_matches) / len(token_matches) if token_matches else 0
        
        return {
            "exact_match_score": exact_match_score,
            "token_match_score": token_match_score
        }
