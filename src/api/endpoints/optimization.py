
"""
API endpoints for prompt optimization workflows
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import uuid
import time
from datetime import datetime

from src.api.models import OptimizationRequest, OptimizationResponse
from src.app.config import settings

# For a complete implementation, we would import the flow
# from src.flows.prompt_optimization_flow import prompt_optimization_flow

router = APIRouter(prefix="/optimization", tags=["Optimization"])

# In-memory storage for demo purposes
flow_store = {}

async def run_optimization_flow(flow_id: str, request: OptimizationRequest):
    """Background task to run the optimization flow"""
    # Update flow status
    flow_store[flow_id]["status"] = "running"
    
    try:
        # In a real implementation, we would call the Prefect flow
        # result = await prompt_optimization_flow(...)
        
        # Simulate flow execution
        await asyncio.sleep(5)
        
        # Update with success
        flow_store[flow_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "current_iteration": flow_store[flow_id]["max_iterations"],
            "best_score": 0.85
        })
    except Exception as e:
        # Update with error
        flow_store[flow_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })

@router.post("/", response_model=OptimizationResponse)
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Start a new prompt optimization workflow"""
    flow_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    flow = {
        "flow_id": flow_id,
        "status": "queued",
        "prompt_id": request.prompt_id,
        "dataset_id": request.dataset_id,
        "started_at": timestamp,
        "completed_at": None,
        "current_iteration": 0,
        "max_iterations": request.max_iterations,
        "best_score": None,
        "target_metric": request.target_metric,
        "target_threshold": request.target_threshold,
        "patience": request.patience,
        "primary_model": request.primary_model or settings.PRIMARY_MODEL_NAME,
        "optimizer_model": request.optimizer_model or settings.OPTIMIZER_MODEL_NAME
    }
    
    flow_store[flow_id] = flow
    
    # Start the flow in the background
    background_tasks.add_task(run_optimization_flow, flow_id, request)
    
    return flow

@router.get("/{flow_id}", response_model=OptimizationResponse)
async def get_optimization_status(flow_id: str):
    """Get the status of an optimization workflow"""
    if flow_id not in flow_store:
        raise HTTPException(status_code=404, detail=f"Optimization flow {flow_id} not found")
    
    return flow_store[flow_id]

@router.delete("/{flow_id}")
async def cancel_optimization(flow_id: str):
    """Cancel an ongoing optimization workflow"""
    if flow_id not in flow_store:
        raise HTTPException(status_code=404, detail=f"Optimization flow {flow_id} not found")
    
    if flow_store[flow_id]["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail=f"Flow {flow_id} is already {flow_store[flow_id]['status']}")
    
    # In a real implementation, we would cancel the Prefect flow
    
    flow_store[flow_id]["status"] = "cancelled"
    flow_store[flow_id]["completed_at"] = datetime.now().isoformat()
    
    return {"status": "success", "message": f"Optimization flow {flow_id} cancelled"}

@router.get("/", response_model=List[OptimizationResponse])
async def list_optimizations():
    """List all optimization workflows"""
    return list(flow_store.values())
