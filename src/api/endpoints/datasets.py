
from fastapi import APIRouter, HTTPException, Depends, Security, Query
from fastapi.security import SecurityScopes
from typing import List, Optional, Dict
import uuid
from datetime import datetime
import random
import json
import os

from src.api.models import (
    DatasetCreate, DatasetResponse, DatasetSample, DatasetItem
)
from src.app.auth import get_current_active_user

router = APIRouter(prefix="/datasets", tags=["Datasets"])

# Mock database for development - would be replaced with real database in production
DATASETS = {}

@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    dataset: DatasetCreate,
    current_user = Security(get_current_active_user, scopes=["datasets"])
):
    """
    Create a new dataset for prompt optimization
    """
    try:
        dataset_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create dataset record
        dataset_record = {
            "dataset_id": dataset_id,
            "name": dataset.name,
            "description": dataset.description,
            "item_count": len(dataset.items),
            "tags": dataset.tags,
            "created_at": now,
            "last_modified": now,
            "created_by": current_user.user_id,
            "metadata": dataset.metadata,
            "items": [item.dict() for item in dataset.items]
        }
        
        # In a real implementation, this would save to database
        DATASETS[dataset_id] = dataset_record
        
        # Also save to filesystem for development
        os.makedirs("data/datasets", exist_ok=True)
        with open(f"data/datasets/{dataset_id}.json", "w") as f:
            json.dump(dataset_record, f)
        
        # Return response without items
        return DatasetResponse(
            dataset_id=dataset_id,
            name=dataset.name,
            description=dataset.description,
            item_count=len(dataset.items),
            tags=dataset.tags,
            created_at=now,
            last_modified=now,
            metadata=dataset.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    current_user = Security(get_current_active_user, scopes=["datasets"])
):
    """
    Get dataset metadata by ID
    """
    # Try to load from filesystem if not in memory
    if dataset_id not in DATASETS:
        try:
            with open(f"data/datasets/{dataset_id}.json", "r") as f:
                DATASETS[dataset_id] = json.load(f)
        except:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    dataset = DATASETS[dataset_id]
    
    return DatasetResponse(
        dataset_id=dataset["dataset_id"],
        name=dataset["name"],
        description=dataset["description"],
        item_count=dataset["item_count"],
        tags=dataset["tags"],
        created_at=dataset["created_at"],
        last_modified=dataset["last_modified"],
        metadata=dataset["metadata"]
    )


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(
    tags: Optional[List[str]] = Query(None),
    current_user = Security(get_current_active_user, scopes=["datasets"])
):
    """
    List all datasets, optionally filtered by tags
    """
    # Load datasets from filesystem for development
    if not DATASETS:
        os.makedirs("data/datasets", exist_ok=True)
        for filename in os.listdir("data/datasets"):
            if filename.endswith(".json"):
                try:
                    with open(f"data/datasets/{filename}", "r") as f:
                        dataset = json.load(f)
                        DATASETS[dataset["dataset_id"]] = dataset
                except:
                    continue
    
    results = []
    
    for dataset_id, dataset in DATASETS.items():
        # Filter by tags if provided
        if tags and not all(tag in dataset["tags"] for tag in tags):
            continue
            
        results.append(DatasetResponse(
            dataset_id=dataset["dataset_id"],
            name=dataset["name"],
            description=dataset["description"],
            item_count=dataset["item_count"],
            tags=dataset["tags"],
            created_at=dataset["created_at"] if isinstance(dataset["created_at"], datetime) else datetime.fromisoformat(dataset["created_at"]),
            last_modified=dataset["last_modified"] if isinstance(dataset["last_modified"], datetime) else datetime.fromisoformat(dataset["last_modified"]),
            metadata=dataset["metadata"]
        ))
    
    return results


@router.get("/{dataset_id}/sample", response_model=DatasetSample)
async def sample_dataset(
    dataset_id: str,
    sample_size: int = Query(5, ge=1, le=100),
    random_seed: Optional[int] = None,
    current_user = Security(get_current_active_user, scopes=["datasets"])
):
    """
    Get a random sample of items from a dataset
    """
    # Try to load from filesystem if not in memory
    if dataset_id not in DATASETS:
        try:
            with open(f"data/datasets/{dataset_id}.json", "r") as f:
                DATASETS[dataset_id] = json.load(f)
        except:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    dataset = DATASETS[dataset_id]
    items = dataset["items"]
    
    # Get random sample
    if random_seed is not None:
        random.seed(random_seed)
    
    sample_size = min(sample_size, len(items))
    sampled_items = random.sample(items, sample_size)
    
    # Convert to DatasetItem objects
    dataset_items = [DatasetItem(**item) for item in sampled_items]
    
    return DatasetSample(
        items=dataset_items,
        dataset_id=dataset_id,
        sample_size=sample_size,
        total_items=len(items)
    )


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user = Security(get_current_active_user, scopes=["datasets"])
):
    """
    Delete a dataset by ID
    """
    # Check if dataset exists
    if dataset_id not in DATASETS:
        try:
            with open(f"data/datasets/{dataset_id}.json", "r") as f:
                DATASETS[dataset_id] = json.load(f)
        except:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    # Delete from memory
    del DATASETS[dataset_id]
    
    # Delete from filesystem
    try:
        os.remove(f"data/datasets/{dataset_id}.json")
    except:
        pass
    
    return {"message": f"Dataset {dataset_id} deleted successfully"}
