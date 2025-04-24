"""
Integration module for Prefect workflow orchestration in the ATLAS platform.

This module provides:
1. Function to register flows with the Prefect API
2. Utility to run flows from HTTP requests
3. Integration with the ML service layer
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import json

from prefect import get_client
from prefect.client.schemas import FlowRun
from prefect.deployments import Deployment
from prefect.infrastructure import Process

from app.ml.prefect_workflows import (
    five_api_workflow,
    train_lightgbm_model,
    train_rl_model
)

logger = logging.getLogger(__name__)

# Default deployment configuration
DEFAULT_ENTRYPOINT = "app/ml/prefect_workflows.py:five_api_workflow"

async def get_prefect_api_status() -> Dict[str, Any]:
    """Check the status of the Prefect API.
    
    Returns:
        Dictionary with API status
    """
    try:
        async with get_client() as client:
            # Check if we can connect to the API
            # This will raise an exception if the API is not available
            await client.hello()
            
            # Get the API URL
            api_url = os.environ.get("PREFECT_API_URL", "Unknown")
            
            # Get the number of flows
            flows = await client.read_flows()
            
            # Get the number of flow runs
            flow_runs = await client.read_flow_runs()
            
            return {
                "status": "ok",
                "api_url": api_url,
                "flows_count": len(flows),
                "flow_runs_count": len(flow_runs),
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.exception("Error checking Prefect API status")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def create_workflow_deployment(
    flow_name: str,
    deployment_name: str,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Create a new workflow deployment.
    
    Args:
        flow_name: Name of the flow
        deployment_name: Name of the deployment
        entrypoint: Path to the flow function
        parameters: Parameters for the flow
        
    Returns:
        Dictionary with deployment information
    """
    try:
        # Create the deployment
        deployment = await Deployment.build_from_flow(
            flow=eval(flow_name),  # This is safe because we control the input
            name=deployment_name,
            version=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            tags=["atlas", "ml", flow_name],
            description=f"ATLAS ML {flow_name} deployment",
            infrastructure=Process(),
            work_queue_name="atlas",
            parameters=parameters or {}
        )
        
        # Save the deployment
        deployment_id = await deployment.apply()
        
        return {
            "status": "success",
            "deployment_id": deployment_id,
            "flow_name": flow_name,
            "deployment_name": deployment_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.exception(f"Error creating deployment for {flow_name}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def run_flow_deployment(deployment_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a flow deployment.
    
    Args:
        deployment_id: ID of the deployment to run
        parameters: Parameters for the flow run
        
    Returns:
        Dictionary with flow run information
    """
    try:
        async with get_client() as client:
            # Create the flow run
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id=deployment_id,
                parameters=parameters or {}
            )
            
            return {
                "status": "success",
                "flow_run_id": flow_run.id,
                "flow_run_name": flow_run.name,
                "deployment_id": deployment_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.exception(f"Error running deployment {deployment_id}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def run_five_api_workflow(
    experiment_id: str,
    system_prompt: str,
    output_prompt: str,
    examples: List[Dict[str, Any]],
    max_iterations: int = 5,
    target_threshold: float = 0.9,
    early_stopping_patience: int = 2
) -> Dict[str, Any]:
    """Run the 5-API workflow directly.
    
    Args:
        experiment_id: ID of the experiment
        system_prompt: Initial system prompt
        output_prompt: Initial output prompt
        examples: List of examples for evaluation
        max_iterations: Maximum number of iterations
        target_threshold: Target threshold for early stopping
        early_stopping_patience: Number of iterations without improvement before stopping
        
    Returns:
        Dictionary with workflow results
    """
    # Create deployment parameters
    parameters = {
        "experiment_id": experiment_id,
        "system_prompt": system_prompt,
        "output_prompt": output_prompt,
        "examples": examples,
        "max_iterations": max_iterations,
        "target_threshold": target_threshold,
        "early_stopping_patience": early_stopping_patience
    }
    
    # Create the deployment
    deployment_info = await create_workflow_deployment(
        flow_name="five_api_workflow",
        deployment_name=f"5api-workflow-{experiment_id[:8]}",
        parameters=parameters
    )
    
    if deployment_info["status"] != "success":
        return deployment_info
    
    # Run the deployment
    return await run_flow_deployment(deployment_info["deployment_id"])

async def run_train_lightgbm(
    model_id: str,
    experiment_ids: List[str],
    hyperparameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Run the LightGBM training workflow.
    
    Args:
        model_id: ID of the meta-learning model
        experiment_ids: List of experiment IDs to use for training
        hyperparameters: Dictionary of hyperparameters
        
    Returns:
        Dictionary with training results
    """
    # Create deployment parameters
    parameters = {
        "model_id": model_id,
        "experiment_ids": experiment_ids,
        "hyperparameters": hyperparameters or {}
    }
    
    # Create the deployment
    deployment_info = await create_workflow_deployment(
        flow_name="train_lightgbm_model",
        deployment_name=f"lgbm-train-{model_id[:8]}",
        parameters=parameters
    )
    
    if deployment_info["status"] != "success":
        return deployment_info
    
    # Run the deployment
    return await run_flow_deployment(deployment_info["deployment_id"])

async def run_train_rl_model(
    model_id: str,
    examples: List[Dict[str, Any]],
    base_system_prompt: str,
    base_output_prompt: str,
    hyperparameters: Dict[str, Any] = None,
    total_timesteps: int = 10000
) -> Dict[str, Any]:
    """Run the RL model training workflow.
    
    Args:
        model_id: ID of the RL model
        examples: List of examples to use for training
        base_system_prompt: Initial system prompt
        base_output_prompt: Initial output prompt
        hyperparameters: Dictionary of hyperparameters
        total_timesteps: Total number of training timesteps
        
    Returns:
        Dictionary with training results
    """
    # Create deployment parameters
    parameters = {
        "model_id": model_id,
        "examples": examples,
        "base_system_prompt": base_system_prompt,
        "base_output_prompt": base_output_prompt,
        "hyperparameters": hyperparameters or {},
        "total_timesteps": total_timesteps
    }
    
    # Create the deployment
    deployment_info = await create_workflow_deployment(
        flow_name="train_rl_model",
        deployment_name=f"rl-train-{model_id[:8]}",
        parameters=parameters
    )
    
    if deployment_info["status"] != "success":
        return deployment_info
    
    # Run the deployment
    return await run_flow_deployment(deployment_info["deployment_id"])

async def get_flow_run_status(flow_run_id: str) -> Dict[str, Any]:
    """Get the status of a flow run.
    
    Args:
        flow_run_id: ID of the flow run
        
    Returns:
        Dictionary with flow run status
    """
    try:
        async with get_client() as client:
            # Get the flow run
            flow_run = await client.read_flow_run(flow_run_id)
            
            return {
                "status": "success",
                "flow_run_id": flow_run.id,
                "flow_run_name": flow_run.name,
                "state_type": flow_run.state_type,
                "state_name": flow_run.state_name,
                "state_message": flow_run.state_message,
                "start_time": flow_run.start_time.isoformat() if flow_run.start_time else None,
                "end_time": flow_run.end_time.isoformat() if flow_run.end_time else None,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.exception(f"Error getting flow run status for {flow_run_id}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def get_flow_runs() -> Dict[str, Any]:
    """Get all flow runs.
    
    Returns:
        Dictionary with flow runs
    """
    try:
        async with get_client() as client:
            # Get the flow runs
            flow_runs = await client.read_flow_runs()
            
            # Format the response
            formatted_runs = [
                {
                    "id": run.id,
                    "name": run.name,
                    "state_type": run.state_type,
                    "state_name": run.state_name,
                    "state_message": run.state_message,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None
                }
                for run in flow_runs
            ]
            
            return {
                "status": "success",
                "flow_runs": formatted_runs,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.exception("Error getting flow runs")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }