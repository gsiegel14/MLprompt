
"""
API endpoints for experiment management
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import logging
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.api.models import ExperimentData, ExperimentList
from src.app.auth import get_current_active_user, User

router = APIRouter(prefix="/experiments", tags=["experiments"])
logger = logging.getLogger(__name__)

# In-memory cache of active experiments 
# In a production app, this would be stored in a database
active_experiments = {}

@router.post("/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Start an experiment by ID"""
    # Check if experiment exists
    experiments_dir = "experiments"
    experiment_dir = os.path.join(experiments_dir, experiment_id)
    
    if not os.path.exists(experiment_dir):
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    # Function to run the experiment
    def run_experiment():
        try:
            logger.info(f"Starting experiment {experiment_id}")
            # Here we would trigger the actual workflow
            # For now, we'll just mark it as running
            active_experiments[experiment_id] = {
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "started_by": current_user.username
            }
            
            # In a real implementation, this would call the workflow
            # flow = prompt_optimization_flow(...)
            
            # Mark as completed when done
            active_experiments[experiment_id]["status"] = "completed"
            active_experiments[experiment_id]["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"Experiment {experiment_id} completed")
        except Exception as e:
            logger.error(f"Error in experiment {experiment_id}: {str(e)}")
            active_experiments[experiment_id]["status"] = "failed"
            active_experiments[experiment_id]["error"] = str(e)
    
    # Add task to background tasks
    background_tasks.add_task(run_experiment)
    
    return {
        "experiment_id": experiment_id,
        "status": "started",
        "message": f"Experiment {experiment_id} started"
    }

@router.get("/{experiment_id}/metrics")
async def get_experiment_metrics(
    experiment_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get metrics for an experiment"""
    experiments_dir = "experiments"
    experiment_dir = os.path.join(experiments_dir, experiment_id)
    
    if not os.path.exists(experiment_dir):
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    # Check for metrics files
    metrics_files = []
    for f in os.listdir(experiment_dir):
        if f.startswith('metrics_') and f.endswith('.json'):
            metrics_files.append(f)
    
    if not metrics_files:
        # Check newer directory structure
        metrics_dir = os.path.join(experiment_dir, "validation")
        if os.path.exists(metrics_dir) and os.path.exists(os.path.join(metrics_dir, "metrics.json")):
            with open(os.path.join(metrics_dir, "metrics.json"), 'r') as f:
                metrics = json.load(f)
            return {
                "experiment_id": experiment_id,
                "metrics": metrics
            }
        else:
            raise HTTPException(status_code=404, detail=f"No metrics found for experiment {experiment_id}")
    
    # Load the metrics from all iterations
    all_metrics = []
    for metrics_file in sorted(metrics_files):
        with open(os.path.join(experiment_dir, metrics_file), 'r') as f:
            metrics = json.load(f)
            iteration = int(metrics_file.split('_')[1].split('.')[0])
            all_metrics.append({
                "iteration": iteration,
                "metrics": metrics
            })
    
    return {
        "experiment_id": experiment_id,
        "iterations": len(all_metrics),
        "metrics_by_iteration": all_metrics
    }

@router.get("/", response_model=ExperimentList)
async def list_experiments(current_user: User = Depends(get_current_active_user)):
    """List all experiments"""
    experiments_dir = "experiments"
    
    if not os.path.exists(experiments_dir):
        return {"experiments": []}
    
    experiments = []
    for experiment_id in os.listdir(experiments_dir):
        experiment_dir = os.path.join(experiments_dir, experiment_id)
        
        # Skip if not a directory
        if not os.path.isdir(experiment_dir):
            continue
        
        # Get metadata
        created_at = datetime.fromtimestamp(os.path.getctime(experiment_dir)).isoformat()
        
        # Look for best metrics
        best_metrics = {}
        best_prompt_state = {}
        
        # Check for metrics files
        for f in os.listdir(experiment_dir):
            if f.startswith('metrics_') and f.endswith('.json'):
                with open(os.path.join(experiment_dir, f), 'r') as file:
                    metrics = json.load(file)
                    if not best_metrics or metrics.get("overall_score", 0) > best_metrics.get("overall_score", 0):
                        best_metrics = metrics
                        
                        # Load corresponding prompt state
                        prompt_file = f.replace('metrics_', 'prompts_')
                        if os.path.exists(os.path.join(experiment_dir, prompt_file)):
                            with open(os.path.join(experiment_dir, prompt_file), 'r') as prompt_f:
                                best_prompt_state = json.load(prompt_f)
        
        # Check newer directory structure
        prompts_dir = os.path.join(experiment_dir, "prompts")
        validation_dir = os.path.join(experiment_dir, "validation")
        if os.path.exists(validation_dir) and os.path.exists(os.path.join(validation_dir, "metrics.json")):
            with open(os.path.join(validation_dir, "metrics.json"), 'r') as f:
                best_metrics = json.load(f)
            
            if os.path.exists(prompts_dir):
                best_prompt_state = {
                    "system_prompt": "",
                    "output_prompt": ""
                }
                
                if os.path.exists(os.path.join(prompts_dir, "optimized_system.txt")):
                    with open(os.path.join(prompts_dir, "optimized_system.txt"), 'r') as f:
                        best_prompt_state["system_prompt"] = f.read()
                
                if os.path.exists(os.path.join(prompts_dir, "optimized_output.txt")):
                    with open(os.path.join(prompts_dir, "optimized_output.txt"), 'r') as f:
                        best_prompt_state["output_prompt"] = f.read()
        
        # Get count of iterations
        iterations = len([f for f in os.listdir(experiment_dir) if f.startswith('metrics_') and f.endswith('.json')])
        
        experiments.append({
            "id": experiment_id,
            "created_at": created_at,
            "iterations": iterations,
            "best_metrics": best_metrics,
            "best_prompt_state": best_prompt_state,
            "status": active_experiments.get(experiment_id, {}).get("status", "completed")
        })
    
    # Sort by created_at, newest first
    experiments.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"experiments": experiments}
