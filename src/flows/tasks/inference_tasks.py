
"""
Prefect tasks for LLM inference operations
"""
from prefect import task
import pandas as pd
from typing import Dict, List, Any, Optional
from src.app.llm_client import get_llm_response
from src.app.models.prompt_state import PromptState

@task(name="vertex-primary-inference", retries=3, retry_delay_seconds=60)
def vertex_primary_inference(state_dict, dataset_dict, batch_size=10):
    """Run primary inference with LLM"""
    prompt_state = PromptState(**state_dict)
    
    # Process in batches to optimize API calls and memory usage
    for i in range(0, len(dataset_dict), batch_size):
        batch = dataset_dict[i:i+batch_size]
        
        # Process each example in the batch
        for example in batch:
            user_input = example.get('user_input', '')
            
            # Call the Primary LLM
            model_response = get_llm_response(
                prompt_state.system_prompt,
                user_input,
                prompt_state.output_prompt
            )
            
            # Add the response to the example
            example["model_response"] = model_response
    
    return dataset_dict

@task(name="vertex-refined-inference", retries=3, retry_delay_seconds=60)
def vertex_refined_inference(refined_state_dict, dataset_dict, batch_size=10):
    """Run inference with refined prompts"""
    refined_prompt_state = PromptState(**refined_state_dict)
    
    # Process in batches to optimize API calls and memory usage
    for i in range(0, len(dataset_dict), batch_size):
        batch = dataset_dict[i:i+batch_size]
        
        # Process each example in the batch
        for example in batch:
            user_input = example.get('user_input', '')
            
            # Call LLM with refined prompts
            refined_response = get_llm_response(
                refined_prompt_state.system_prompt,
                user_input,
                refined_prompt_state.output_prompt
            )
            
            # Add the refined response to the example
            example["refined_response"] = refined_response
    
    return dataset_dict
