
#!/bin/bash
# Entry point script for different execution modes

# Default to API mode
MODE=${MODE:-api}

# Configure environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

if [ "$MODE" = "api" ]; then
    echo "Starting Flask server..."
    gunicorn -b 0.0.0.0:5000 main:app
elif [ "$MODE" = "agent" ]; then
    echo "Starting Prefect agent..."
    python scripts/start_prefect_agent.py
elif [ "$MODE" = "deploy" ]; then
    echo "Creating Prefect deployment..."
    python scripts/create_prefect_deployment.py
else
    echo "Unknown mode: $MODE"
    echo "Valid modes: api, agent, deploy"
    exit 1
fi
