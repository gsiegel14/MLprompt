
#!/bin/bash

echo "Running startup script for deployment..."

# Set environment for production
export DEPLOYMENT_MODE=production
export DEBUG=false
export MAX_CACHE_SIZE_MB=50
export CACHE_TTL_HOURS=12

# Create required directories with placeholders
mkdir -p data/train data/validation data/test data/nejm cache
touch data/train/.gitkeep data/validation/.gitkeep data/test/.gitkeep data/nejm/.gitkeep

# Clean any existing cache
find cache -type f -delete

# Initialize cache directory
python -c "from src.utils.cache_manager import initialize_cache; initialize_cache()"

echo "Startup complete, launching application..."
exec gunicorn --bind 0.0.0.0:5000 main:app
