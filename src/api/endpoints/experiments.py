
from fastapi import APIRouter, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import SecurityScopes
from typing import List, Optional, Dict
import uuid
from datetime import datetime
import json
import os
import asyncio

from src.api.models import (
    ExperimentCreate, ExperimentResponse, ExperimentMetrics,
    ExperimentStatus, OptimizationStrategy
)
from src.app.auth import get_current_active_user
from src.flows.prompt_optimization_flow import run_optimization_flow

router = APIRouter(prefix="/experiments", tags=["Experiments"])

# Mock database for development - would be replaced with real database in production
EXPERIMENTS = {}

@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    experiment: ExperimentCreate,
    current_user = Security(get_current_active_user, scopes=["experiments"])
):
    """
    Create a new prompt optimization experiment
    """
    try:
        experiment_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create experiment record
        experiment_record = {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "prompt_id": experiment.prompt_id,
            "dataset_id": experiment.dataset_id,
            "status": ExperimentStatus.PENDING,
            "optimization_strategy": experiment.optimization_strategy,
            "current_iteration": 0,
            "max_iterations": experiment.max_iterations,
            "evaluation_metrics": experiment.evaluation_metrics,
            "created_at": now,
            "last_updated": now,
            "created_by": current_user.user_id,
            "metadata": experiment.metadata
        }
        
        # In a real implementation, this would save to database
        EXPERIMENTS[experiment_id] = experiment_record
        
        # Also save to filesystem for development
        os.makedirs("data/experiments", exist_ok=True)
        with open(f"data/experiments/{experiment_id}.json", "w") as f:
            json.dump(experiment_record, f)
        
        # Return response
        return ExperimentResponse(
            experiment_id=experiment_id,
            name=experiment.name,
            description=experiment.description,
            prompt_id=experiment.prompt_id,
            dataset_id=experiment.dataset_id,
            status=ExperimentStatus.PENDING,
            current_iteration=0,
            max_iterations=experiment.max_iterations,
            created_at=now,
            last_updated=now,
            metadata=experiment.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/start", response_model=ExperimentResponse)
async def start_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    current_user = Security(get_current_active_user, scopes=["experiments"])
):
    """
    Start a pending experiment
    """
    # Try to load from filesystem if not in memory
    if experiment_id not in EXPERIMENTS:
        try:
            with open(f"data/experiments/{experiment_id}.json", "r") as f:
                EXPERIMENTS[experiment_id] = json.load(f)
        except:
            raise HTTPException(status_code=404, detail=f"Experiment with ID {experiment_id} not found")
    
    experiment = EXPERIMENTS[experiment_id]
    
    # Check if experiment can be started
    if experiment["status"] not in [ExperimentStatus.PENDING, ExperimentStatus.FAILED]:
        raise HTTPException(
            status_code=400, 
            detail=f"Experiment has status {experiment['status']} and cannot be started"
        )
    
    # Update status
    experiment["status"] = ExperimentStatus.RUNNING
    experiment["last_updated"] = datetime.now()
    
    # Save updated state
    with open(f"data/experiments/{experiment_id}.json", "w") as f:
        json.dump(experiment, f)
    
    # Start experiment in background
    background_tasks.add_task(
        _run_experiment_flow,
        experiment_id=experiment_id,
        prompt_id=experiment["prompt_id"],
        dataset_id=experiment["dataset_id"],
        max_iterations=experiment["max_iterations"],
        optimization_strategy=experiment["optimization_strategy"],
        metrics=experiment["evaluation_metrics"]
    )
    
    # Return updated experiment
    return ExperimentResponse(
        experiment_id=experiment["experiment_id"],
        name=experiment["name"],
        description=experiment["description"],
        prompt_id=experiment["prompt_id"],
        dataset_id=experiment["dataset_id"],
        status=ExperimentStatus.RUNNING,
        current_iteration=experiment["current_iteration"],
        max_iterations=experiment["max_iterations"],
        created_at=experiment["created_at"] if isinstance(experiment["created_at"], datetime) else datetime.fromisoformat(experiment["created_at"]),
        last_updated=experiment["last_updated"] if isinstance(experiment["last_updated"], datetime) else datetime.fromisoformat(experiment["last_updated"]),
        metadata=experiment["metadata"]
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    current_user = Security(get_current_active_user, scopes=["experiments"])
):
    """
    Get experiment details by ID
    """
    # Try to load from filesystem if not in memory
    if experiment_id not in EXPERIMENTS:
        try:
            with open(f"data/experiments/{experiment_id}.json", "r") as f:
                EXPERIMENTS[experiment_id] = json.load(f)
        except:
            raise HTTPException(status_code=404, detail=f"Experiment with ID {experiment_id} not found")
    
    experiment = EXPERIMENTS[experiment_id]
    
    return ExperimentResponse(
        experiment_id=experiment["experiment_id"],
        name=experiment["name"],
        description=experiment["description"],
        prompt_id=experiment["prompt_id"],
        dataset_id=experiment["dataset_id"],
        status=experiment["status"],
        current_iteration=experiment["current_iteration"],
        max_iterations=experiment["max_iterations"],
        created_at=experiment["created_at"] if isinstance(experiment["created_at"], datetime) else datetime.fromisoformat(experiment["created_at"]),
        last_updated=experiment["last_updated"] if isinstance(experiment["last_updated"], datetime) else datetime.fromisoformat(experiment["last_updated"]),
        metadata=experiment["metadata"]
    )


@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[ExperimentStatus] = None,
    current_user = Security(get_current_active_user, scopes=["experiments"])
):
    """
    List all experiments, optionally filtered by status
    """
    # Load experiments from filesystem for development
    if not EXPERIMENTS:
        os.makedirs("data/experiments", exist_ok=True)
        for filename in os.listdir("data/experiments"):
            if filename.endswith(".json"):
                try:
                    with open(f"data/experiments/{filename}", "r") as f:
                        experiment = json.load(f)
                        EXPERIMENTS[experiment["experiment_id"]] = experiment
                except:
                    continue
    
    results = []
    
    for experiment_id, experiment in EXPERIMENTS.items():
        # Filter by status if provided
        if status and experiment["status"] != status:
            continue
            
        results.append(ExperimentResponse(
            experiment_id=experiment["experiment_id"],
            name=experiment["name"],
            description=experiment["description"],
            prompt_id=experiment["prompt_id"],
            dataset_id=experiment["dataset_id"],
            status=experiment["status"],
            current_iteration=experiment["current_iteration"],
            max_iterations=experiment["max_iterations"],
            created_at=experiment["created_at"] if isinstance(experiment["created_at"], datetime) else datetime.fromisoformat(experiment["created_at"]),
            last_updated=experiment["last_updated"] if isinstance(experiment["last_updated"], datetime) else datetime.fromisoformat(experiment["last_updated"]),
            metadata=experiment["metadata"]
        ))
    
    return results


@router.get("/{experiment_id}/metrics", response_model=ExperimentMetrics)
async def get_experiment_metrics(
    experiment_id: str,
    current_user = Security(get_current_active_user, scopes=["experiments"])
):
    """
    Get metrics for an experiment
    """
    # Try to load from filesystem if not in memory
    if experiment_id not in EXPERIMENTS:
        try:
            with open(f"data/experiments/{experiment_id}.json", "r") as f:
                EXPERIMENTS[experiment_id] = json.load(f)
        except:
            raise HTTPException(status_code=404, detail=f"Experiment with ID {experiment_id} not found")
    
    # Load metrics
    try:
        with open(f"data/experiments/{experiment_id}/metrics.json", "r") as f:
            metrics_data = json.load(f)
    except:
        # Return empty metrics if not found
        return ExperimentMetrics(
            experiment_id=experiment_id,
            iterations=[],
            best_iteration=0,
            best_metrics={},
            comparison={}
        )
    
    # Find best iteration
    best_iteration = 0
    best_score = 0
    primary_metric = "accuracy"  # Default to accuracy
    
    for i, iteration in enumerate(metrics_data["iterations"]):
        if iteration["metrics"].get(primary_metric, 0) > best_score:
            best_score = iteration["metrics"].get(primary_metric, 0)
            best_iteration = i
    
    # Compare first and best iteration
    comparison = {}
    if len(metrics_data["iterations"]) > 0:
        first_metrics = metrics_data["iterations"][0]["metrics"]
        best_metrics = metrics_data["iterations"][best_iteration]["metrics"]
        
        comparison = {
            "initial": first_metrics,
            "best": best_metrics,
            "improvement": {
                k: best_metrics.get(k, 0) - first_metrics.get(k, 0)
                for k in first_metrics.keys()
            }
        }
    
    return ExperimentMetrics(
        experiment_id=experiment_id,
        iterations=metrics_data["iterations"],
        best_iteration=best_iteration,
        best_metrics=metrics_data["iterations"][best_iteration]["metrics"] if metrics_data["iterations"] else {},
        comparison=comparison
    )


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    current_user = Security(get_current_active_user, scopes=["experiments"])
):
    """
    Delete an experiment by ID
    """
    # Check if experiment exists
    if experiment_id not in EXPERIMENTS:
        try:
            with open(f"data/experiments/{experiment_id}.json", "r") as f:
                EXPERIMENTS[experiment_id] = json.load(f)
        except:
            raise HTTPException(status_code=404, detail=f"Experiment with ID {experiment_id} not found")
    
    # Cancel if running
    experiment = EXPERIMENTS[experiment_id]
    if experiment["status"] == ExperimentStatus.RUNNING:
        experiment["status"] = ExperimentStatus.CANCELLED
        with open(f"data/experiments/{experiment_id}.json", "w") as f:
            json.dump(experiment, f)
    
    # Delete from memory
    del EXPERIMENTS[experiment_id]
    
    # In a production system, you might want to keep the data or mark as deleted
    # rather than physically deleting it
    
    return {"message": f"Experiment {experiment_id} deleted successfully"}


async def _run_experiment_flow(
    experiment_id: str,
    prompt_id: str,
    dataset_id: str,
    max_iterations: int,
    optimization_strategy: str,
    metrics: List[str]
):
    """
    Background task to run the optimization flow
    """
    try:
        # Get experiment record
        experiment = EXPERIMENTS[experiment_id]
        
        # Create experiment directory
        os.makedirs(f"data/experiments/{experiment_id}", exist_ok=True)
        
        # Initialize metrics tracking
        metrics_data = {
            "experiment_id": experiment_id,
            "iterations": []
        }
        
        # Run optimization flow for each iteration
        for iteration in range(max_iterations):
            # Update experiment status
            experiment["current_iteration"] = iteration + 1
            experiment["last_updated"] = datetime.now()
            with open(f"data/experiments/{experiment_id}.json", "w") as f:
                json.dump(experiment, f)
            
            # Run optimization flow
            iteration_results = await run_optimization_flow(
                prompt_id=prompt_id,
                dataset_id=dataset_id,
                iteration=iteration,
                strategy=optimization_strategy
            )
            
            # Save iteration results
            with open(f"data/experiments/{experiment_id}/iteration_{iteration}.json", "w") as f:
                json.dump(iteration_results, f)
            
            # Update metrics
            metrics_data["iterations"].append({
                "iteration": iteration + 1,
                "metrics": iteration_results["metrics"],
                "timestamp": datetime.now().isoformat()
            })
            
            with open(f"data/experiments/{experiment_id}/metrics.json", "w") as f:
                json.dump(metrics_data, f)
        
        # Mark experiment as completed
        experiment["status"] = ExperimentStatus.COMPLETED
        experiment["last_updated"] = datetime.now()
        with open(f"data/experiments/{experiment_id}.json", "w") as f:
            json.dump(experiment, f)
            
    except Exception as e:
        # Handle errors
        print(f"Error in experiment {experiment_id}: {str(e)}")
        
        # Mark experiment as failed
        experiment = EXPERIMENTS.get(experiment_id)
        if experiment:
            experiment["status"] = ExperimentStatus.FAILED
            experiment["last_updated"] = datetime.now()
            experiment["metadata"]["error"] = str(e)
            
            with open(f"data/experiments/{experiment_id}.json", "w") as f:
                json.dump(experiment, f)
