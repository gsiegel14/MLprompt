
"""
Prefect tasks for data loading and state management
"""
from prefect import task
import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional
from src.app.models.prompt_state import PromptState
from src.app.config import settings

@task(name="load-state", retries=2, retry_delay_seconds=30)
def load_state(system_prompt_path, output_prompt_path, dataset_path, state_path=None):
    """Load or initialize PromptState & training data"""
    # If state_path is provided, load existing state
    if state_path and os.path.exists(state_path):
        with open(state_path, 'r') as f:
            prompt_state_dict = json.load(f)
            prompt_state = PromptState(**prompt_state_dict)
    else:
        # Otherwise create a new state from the provided paths
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read().strip()
        
        with open(output_prompt_path, "r") as f:
            output_prompt = f.read().strip()
            
        prompt_state = PromptState(
            system_prompt=system_prompt,
            output_prompt=output_prompt
        )
    
    # Load dataset
    if dataset_path.endswith('.csv'):
        dataset = pd.read_csv(dataset_path)
        data_dict = dataset.to_dict(orient="records")
    elif dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    return {
        "prompt_state": prompt_state.dict(),
        "dataset": data_dict
    }

@task(name="save-state", retries=2)
def save_state(prompt_state_dict, iteration, bucket_name=None):
    """Save prompt state to GCS or local file system"""
    prompt_state = PromptState(**prompt_state_dict)
    
    if bucket_name:
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            
            blob_name = f"prompt_states/{prompt_state.id}_v{iteration}.json"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(
                data=json.dumps(prompt_state.dict()),
                content_type='application/json'
            )
            path = f"gs://{bucket_name}/{blob_name}"
        except Exception as e:
            print(f"Error saving to GCS: {str(e)}")
            # Fall back to local storage
            os.makedirs("data/prompt_states", exist_ok=True)
            path = f"data/prompt_states/{prompt_state.id}_v{iteration}.json"
            with open(path, 'w') as f:
                json.dump(prompt_state.dict(), f)
    else:
        os.makedirs("data/prompt_states", exist_ok=True)
        path = f"data/prompt_states/{prompt_state.id}_v{iteration}.json"
        with open(path, 'w') as f:
            json.dump(prompt_state.dict(), f)
            
    return path
