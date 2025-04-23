
#!/usr/bin/env python3
"""
Start a Prefect agent for processing prompt optimization workflows
"""
import os
import sys
import time
from prefect.agent import PrefectAgent
from prefect.client import get_client
from src.app.config.prefect_config import DEFAULT_QUEUE
from src.app.utils.logger import setup_logging

# Setup logging
logger = setup_logging("prefect_agent")

def create_work_queue_if_not_exists(queue_name):
    """Create the work queue if it doesn't already exist"""
    try:
        with get_client() as client:
            work_queues = client.read_work_queues()
            queue_exists = any(wq.name == queue_name for wq in work_queues)
            
            if not queue_exists:
                logger.info(f"Creating work queue: {queue_name}")
                client.create_work_queue(name=queue_name)
                return True
            else:
                logger.info(f"Work queue already exists: {queue_name}")
                return True
    except Exception as e:
        logger.error(f"Error creating work queue: {str(e)}")
        return False

def start_agent(queue_name):
    """Start the Prefect agent"""
    logger.info(f"Starting Prefect agent for queue: {queue_name}")
    
    try:
        # Create the queue if it doesn't exist
        if not create_work_queue_if_not_exists(queue_name):
            logger.error("Failed to create work queue, exiting")
            sys.exit(1)
        
        # Start the agent
        agent = PrefectAgent(
            work_queue_name=queue_name,
            prefetch_seconds=60
        )
        
        # Start serving
        agent.start()
        
    except KeyboardInterrupt:
        logger.info("Agent interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    queue_name = os.environ.get("PREFECT_WORK_QUEUE", DEFAULT_QUEUE)
    start_agent(queue_name)
