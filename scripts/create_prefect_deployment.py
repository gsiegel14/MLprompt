
#!/usr/bin/env python3
"""
Create a Prefect deployment for the prompt optimization flow
"""
import os
import sys
import logging
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from src.flows.prompt_optimization_flow import prompt_optimization_flow
from src.app.config import settings
from src.app.config.prefect_config import DEFAULT_QUEUE, DEFAULT_POOL

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_deployment():
    """Create a deployment for the prompt optimization flow"""
    logger.info("Creating prompt optimization flow deployment...")

    # Set up deployment parameters
    params = {
        "system_prompt_path": "prompts/system/medical_diagnosis.txt",
        "output_prompt_path": "prompts/output/medical_diagnosis.txt",
        "dataset_path": "data/train/examples.json",
        "metric_names": ["exact_match", "bleu"],
        "target_metric": "avg_score",
        "target_threshold": 0.85,
        "patience": 3,
        "max_iterations": 5,
        "batch_size": 5,
        "sample_k": 3,
        "optimizer_strategy": "reasoning_first"
    }
    
    # Create a deployment
    try:
        deployment = Deployment.build_from_flow(
            flow=prompt_optimization_flow,
            name="prompt-optimizer",
            version=os.environ.get("APP_VERSION", "0.1.0"),
            work_queue_name=DEFAULT_QUEUE,
            # Optional: schedule for regular runs
            # schedule=CronSchedule(cron="0 0 * * *"),  # Daily at midnight
            parameters=params
        )
        
        # Apply the deployment
        deployment_id = deployment.apply()
        logger.info(f"Deployment created with ID: {deployment_id}")
        return True
    except Exception as e:
        logger.error(f"Error creating deployment: {str(e)}")
        return False

if __name__ == "__main__":
    if not create_deployment():
        sys.exit(1)
