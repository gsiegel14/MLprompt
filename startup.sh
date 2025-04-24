
#!/bin/bash

echo "Running startup script for deployment..."

# Set environment for production
export DEPLOYMENT_MODE=production
export DEBUG=false
export LOG_LEVEL=INFO
export MAX_CACHE_SIZE_MB=50
export CACHE_TTL_HOURS=12

# Create required directories
mkdir -p data/train data/validation data/test data/nejm cache logs
touch data/train/.gitkeep data/validation/.gitkeep data/test/.gitkeep data/nejm/.gitkeep

# Initialize cache directory if needed
python -c "
import os
os.makedirs('cache', exist_ok=True)
try:
    from src.utils.cache_manager import initialize_cache
    initialize_cache()
    print('Cache initialized successfully')
except Exception as e:
    print(f'Cache initialization skipped: {str(e)}')
"

echo "Startup complete, launching application..."
exec gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 main:app
