
"""
API endpoints for dataset management
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
import logging
import os
import json
import csv
import io
from typing import List, Dict, Any, Optional

from src.api.models import Example
from src.app.auth import get_current_active_user, User
from src.app.config import settings

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    dataset_type: str = Form(...),  # 'train' or 'validation'
    current_user: User = Depends(get_current_active_user)
):
    """Upload a dataset (CSV or JSON)"""
    try:
        # Ensure datasets directory exists
        dataset_dir = os.path.join("data", dataset_type)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Generate a unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{timestamp}"
        
        # Process file based on extension
        extension = file.filename.split('.')[-1].lower()
        examples = []
        
        # Read file content
        content = await file.read()
        
        if extension == 'csv':
            # Parse CSV
            csv_text = content.decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(csv_text))
            
            for row in csv_reader:
                # Adapt field names if needed
                example = {
                    "user_input": row.get("user_input") or row.get("input") or row.get("question") or "",
                    "ground_truth_output": row.get("ground_truth_output") or row.get("output") or row.get("answer") or "",
                }
                examples.append(example)
                
        elif extension == 'json':
            # Parse JSON
            json_data = json.loads(content)
            
            # Handle different JSON formats
            if isinstance(json_data, list):
                # Array of examples
                examples = json_data
            elif isinstance(json_data, dict) and "examples" in json_data:
                # Object with examples array
                examples = json_data["examples"]
            else:
                # Unknown format
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid JSON format. Expected array of examples or object with 'examples' array"
                )
                
            # Validate and normalize examples
            normalized_examples = []
            for example in examples:
                normalized = {
                    "user_input": example.get("user_input") or example.get("input") or example.get("question") or "",
                    "ground_truth_output": example.get("ground_truth_output") or example.get("output") or example.get("answer") or "",
                }
                normalized_examples.append(normalized)
            
            examples = normalized_examples
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {extension}")
        
        # Limit the number of examples if needed
        if len(examples) > settings.MAX_EXAMPLES:
            examples = examples[:settings.MAX_EXAMPLES]
            logger.warning(f"Dataset truncated to {settings.MAX_EXAMPLES} examples")
        
        # Save to file
        output_path = os.path.join(dataset_dir, f"{filename}.json")
        with open(output_path, 'w') as f:
            json.dump({"examples": examples}, f, indent=2)
        
        # If this is set as current, save to current_{dataset_type}.json
        current_path = os.path.join(dataset_dir, f"current_{dataset_type}.json")
        with open(current_path, 'w') as f:
            json.dump({"examples": examples}, f, indent=2)
        
        return {
            "success": True,
            "message": f"Dataset uploaded as {filename}.json",
            "example_count": len(examples),
            "dataset_type": dataset_type,
            "is_current": True
        }
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_id}/sample")
async def get_dataset_sample(
    dataset_id: str,
    limit: int = 10,
    current_user: User = Depends(get_current_active_user)
):
    """Get a sample of examples from a dataset"""
    # Determine dataset path
    if dataset_id == "current_train":
        dataset_path = os.path.join("data", "train", "current_train.json")
    elif dataset_id == "current_validation":
        dataset_path = os.path.join("data", "validation", "current_validation.json")
    else:
        # Search in both train and validation
        for dataset_type in ["train", "validation"]:
            dataset_dir = os.path.join("data", dataset_type)
            if os.path.exists(os.path.join(dataset_dir, f"{dataset_id}.json")):
                dataset_path = os.path.join(dataset_dir, f"{dataset_id}.json")
                break
        else:
            # Not found
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Extract examples
    if isinstance(data, dict) and "examples" in data:
        examples = data["examples"]
    elif isinstance(data, list):
        examples = data
    else:
        examples = []
    
    # Limit the number of examples
    sample = examples[:limit]
    
    return {
        "dataset_id": dataset_id,
        "total_examples": len(examples),
        "sample_size": len(sample),
        "examples": sample
    }

@router.get("/")
async def list_datasets(current_user: User = Depends(get_current_active_user)):
    """List all available datasets"""
    datasets = []
    
    # Process both train and validation directories
    for dataset_type in ["train", "validation"]:
        dataset_dir = os.path.join("data", dataset_type)
        
        if not os.path.exists(dataset_dir):
            continue
        
        # Get current dataset
        current_path = os.path.join(dataset_dir, f"current_{dataset_type}.json")
        current_dataset = None
        if os.path.exists(current_path):
            current_dataset = f"current_{dataset_type}"
        
        # Get all datasets
        for filename in os.listdir(dataset_dir):
            if filename.endswith('.json'):
                # Skip current dataset file
                if filename == f"current_{dataset_type}.json":
                    continue
                
                dataset_id = filename.replace('.json', '')
                dataset_path = os.path.join(dataset_dir, filename)
                
                # Get metadata
                created_at = os.path.getctime(dataset_path)
                from datetime import datetime
                created_at_str = datetime.fromtimestamp(created_at).isoformat()
                
                # Count examples
                try:
                    with open(dataset_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict) and "examples" in data:
                        example_count = len(data["examples"])
                    elif isinstance(data, list):
                        example_count = len(data)
                    else:
                        example_count = 0
                except:
                    example_count = 0
                
                # Add to list
                datasets.append({
                    "id": dataset_id,
                    "filename": filename,
                    "type": dataset_type,
                    "created_at": created_at_str,
                    "example_count": example_count,
                    "is_current": dataset_id == current_dataset
                })
    
    # Sort by created_at, newest first
    datasets.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"datasets": datasets}
