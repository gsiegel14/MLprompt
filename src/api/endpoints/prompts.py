
"""
API endpoints for managing prompts
"""
from fastapi import APIRouter, HTTPException, Depends, Body
from typing import List, Dict, Any, Optional
import uuid
import time
from datetime import datetime

from src.api.models import PromptData, PromptResponse
from src.app.config import settings

router = APIRouter(prefix="/prompts", tags=["Prompts"])

# In-memory storage for demo purposes
# In a real app, we would use a database
prompt_store = {}

@router.post("/", response_model=PromptResponse)
async def create_prompt(prompt_data: PromptData):
    """Create a new prompt"""
    prompt_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    prompt = {
        "id": prompt_id,
        "system_prompt": prompt_data.system_prompt,
        "output_prompt": prompt_data.output_prompt,
        "version": 1,
        "name": prompt_data.name or f"Prompt {prompt_id[:8]}",
        "description": prompt_data.description or "",
        "tags": prompt_data.tags,
        "created_at": timestamp,
        "updated_at": timestamp
    }
    
    prompt_store[prompt_id] = prompt
    return prompt

@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: str):
    """Get a prompt by ID"""
    if prompt_id not in prompt_store:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
    
    return prompt_store[prompt_id]

@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(prompt_id: str, prompt_data: PromptData):
    """Update an existing prompt"""
    if prompt_id not in prompt_store:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
    
    existing = prompt_store[prompt_id]
    timestamp = datetime.now().isoformat()
    
    updated = {
        **existing,
        "system_prompt": prompt_data.system_prompt,
        "output_prompt": prompt_data.output_prompt,
        "version": existing["version"] + 1,
        "name": prompt_data.name or existing["name"],
        "description": prompt_data.description or existing["description"],
        "tags": prompt_data.tags or existing["tags"],
        "updated_at": timestamp
    }
    
    prompt_store[prompt_id] = updated
    return updated

@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt"""
    if prompt_id not in prompt_store:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
    
    del prompt_store[prompt_id]
    return {"status": "success", "message": f"Prompt {prompt_id} deleted"}

@router.get("/", response_model=List[PromptResponse])
async def list_prompts():
    """List all prompts"""
    return list(prompt_store.values())
