
#!/usr/bin/env python3
"""
Start a Prefect agent for processing prompt optimization workflows
"""
import os
import sys
import time
import logging
import subprocess
from prefect.agent import PrefectAgent
from prefect.client import get_client
from src.app.config.prefect_config import DEFAULT_QUEUE, DEFAULT_POOL
from src.app.utils.logger import setup_logging

# Setup logging
logger = setup_logging("prefect_agent")

def check_prefect_server_running():
    """Check if the Prefect server is running and accessible"""
    try:
        logger.info("Checking if Prefect server is running...")
        with get_client() as client:
            healthcheck = client.api_healthcheck()
            logger.info(f"Prefect API health check: {healthcheck}")
            return True
    except Exception as e:
        logger.error(f"Error connecting to Prefect server: {str(e)}")
        return False

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

def start_agent(queue_name, pool_name=None):
    """Start the Prefect agent"""
    logger.info(f"Starting Prefect agent for queue: {queue_name}")
    
    try:
        # Create the queue if it doesn't exist
        if not create_work_queue_if_not_exists(queue_name):
            logger.error("Failed to create work queue, exiting")
            sys.exit(1)
        
        # Agent configuration
        agent_kwargs = {
            "work_queue_name": queue_name,
            "prefetch_seconds": 60,
        }
        
        # Add pool name if provided
        if pool_name:
            agent_kwargs["work_pool_name"] = pool_name
        
        # Start the agent
        agent = PrefectAgent(**agent_kwargs)
        
        # Start serving
        logger.info(f"Agent started successfully for queue: {queue_name}")
        agent.start()
        
    except KeyboardInterrupt:
        logger.info("Agent interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if Prefect server is running
    if not check_prefect_server_running():
        logger.error("Prefect server is not running or not accessible")
        logger.info("Starting Prefect server...")
        try:
            # Start the server if it's not running
            subprocess.Popen(
                ["prefect", "server", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Give it time to start
            time.sleep(10)
            if not check_prefect_server_running():
                logger.error("Failed to start Prefect server, exiting")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error starting Prefect server: {str(e)}")
            sys.exit(1)
    
    # Get queue and pool names from environment or use defaults
    queue_name = os.environ.get("PREFECT_WORK_QUEUE", DEFAULT_QUEUE)
    pool_name = os.environ.get("PREFECT_WORK_POOL", DEFAULT_POOL)
    
    # Start the agent
    start_agent(queue_name, pool_name)
