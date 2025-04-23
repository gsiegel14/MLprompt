
"""
Prefect tasks for logging, artifacts, and decision making
"""
from prefect import task, get_run_logger
import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict, Any

@task(name="compare-and-log")
def compare_and_log(baseline_metrics, refined_metrics, state_dict, 
                   refined_state_dict, iteration, target_metric="avg_score", 
                   target_threshold=0.9, patience=3, no_improve_count=0):
    """Compare metrics and decide whether to continue or stop"""
    logger = get_run_logger()
    
    # Get the primary metric value from each set
    baseline_value = baseline_metrics.get(target_metric, 0.0)
    refined_value = refined_metrics.get(target_metric, 0.0)
    
    # Create a comparison table for logging
    metrics_table = []
    for metric in set(list(baseline_metrics.keys()) + list(refined_metrics.keys())):
        baseline = baseline_metrics.get(metric, 0.0)
        refined = refined_metrics.get(metric, 0.0)
        change = refined - baseline
        metrics_table.append({
            "Metric": metric,
            "Baseline": f"{baseline:.4f}",
            "Refined": f"{refined:.4f}",
            "Change": f"{change:+.4f}"
        })
    
    # Log the metrics table
    logger.info(f"Metrics comparison for iteration {iteration}:")
    for row in metrics_table:
        logger.info(f"{row['Metric']}: {row['Baseline']} → {row['Refined']} ({row['Change']})")
    
    # Log prompt comparison
    logger.info(f"System prompt change: {len(state_dict['system_prompt'])} → {len(refined_state_dict['system_prompt'])} chars")
    logger.info(f"Output prompt change: {len(state_dict['output_prompt'])} → {len(refined_state_dict['output_prompt'])} chars")
    
    # Save metrics to file
    os.makedirs("experiments/metrics", exist_ok=True)
    with open(f"experiments/metrics/iteration_{iteration}.json", "w") as f:
        json.dump({
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "baseline": baseline_metrics,
            "refined": refined_metrics,
            "target_metric": target_metric,
            "baseline_value": baseline_value,
            "refined_value": refined_value
        }, f, indent=2)
    
    # Decision logic
    improved = refined_value > baseline_value
    reached_threshold = refined_value >= target_threshold
    
    if improved:
        logger.info(f"Iteration {iteration}: {target_metric} improved from {baseline_value:.4f} to {refined_value:.4f}")
        # Reset the counter if we improved
        no_improve_count = 0
        use_refined = True
    else:
        logger.info(f"Iteration {iteration}: {target_metric} did not improve ({baseline_value:.4f} → {refined_value:.4f})")
        no_improve_count += 1
        use_refined = False
    
    # Check early stopping conditions
    should_stop = no_improve_count >= patience or reached_threshold
    
    if should_stop:
        if reached_threshold:
            logger.info(f"Reached target threshold {target_threshold:.4f} for {target_metric}")
        elif no_improve_count >= patience:
            logger.info(f"No improvement for {no_improve_count} iterations, stopping early")
    
    return {
        "improved": improved, 
        "use_refined": use_refined,
        "should_stop": should_stop,
        "no_improve_count": no_improve_count,
        "target_value": refined_value if use_refined else baseline_value
    }
