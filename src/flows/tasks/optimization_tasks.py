
"""
Prefect tasks for optimizer LLM operations
"""
from prefect import task
import json
from typing import List, Dict, Any
from app.optimizer import optimize_prompts, load_optimizer_prompt
from app.models.prompt_state import PromptState

@task(name="vertex-optimizer-refine", retries=2, retry_delay_seconds=60)
def vertex_optimizer_refine(state_dict, dataset_dict, baseline_metrics, 
                           optimizer_strategy="reasoning_first", sample_k=5):
    """Call optimizer LLM to get new prompts"""
    prompt_state = PromptState(**state_dict)
    
    # Load the appropriate optimizer prompt
    optimizer_prompt = load_optimizer_prompt(optimizer_strategy)
    
    # Select a sample of examples for optimization
    # Sort by score (ascending) and take the k worst examples
    for item in dataset_dict:
        item_score = calculate_score(
            item.get("model_response", ""), 
            item.get("ground_truth_output", "")
        )
        item["score"] = item_score
    
    sorted_examples = sorted(dataset_dict, key=lambda x: x.get("score", 0.0))
    worst_examples = sorted_examples[:sample_k]
    
    # Call the optimizer
    optimization_result = optimize_prompts(
        current_system_prompt=prompt_state.system_prompt,
        current_output_prompt=prompt_state.output_prompt,
        examples=worst_examples,
        optimizer_system_prompt=optimizer_prompt,
        strategy=optimizer_strategy
    )
    
    if not optimization_result:
        # Fall back to existing prompts
        optimization_result = {
            "system_prompt": prompt_state.system_prompt,
            "output_prompt": prompt_state.output_prompt,
            "reasoning": "Optimization failed, using existing prompts"
        }
    
    # Create new prompt state as child of current one
    new_prompt_state = PromptState(
        system_prompt=optimization_result.get("system_prompt", prompt_state.system_prompt),
        output_prompt=optimization_result.get("output_prompt", prompt_state.output_prompt),
        parent_id=prompt_state.id,
        version=prompt_state.version + 1,
        metadata={
            "baseline_metrics": baseline_metrics,
            "optimizer_strategy": optimizer_strategy,
            "optimization_reasoning": optimization_result.get("reasoning", "")
        }
    )
    
    return new_prompt_state.dict()
