
#!/bin/bash

# Pre-deployment cleanup script
echo "Starting pre-deployment cleanup..."

# Create required directories with placeholders if they don't exist
mkdir -p data/train data/validation data/test data/nejm cache
touch data/train/.gitkeep data/validation/.gitkeep data/test/.gitkeep data/nejm/.gitkeep

# Clean up cache files
echo "Cleaning cache directories..."
find cache -type f -delete 2>/dev/null || true
mkdir -p cache

# Clean up Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "*.pyd" -delete 2>/dev/null || true

# Keep only essential experiment data
echo "Cleaning up experiment data..."
rm -rf experiments/*/examples 2>/dev/null || true
find experiments -type f -name "examples_*.json" -delete 2>/dev/null || true
find experiments -type f -size +1M -delete 2>/dev/null || true

# Remove debug experiments
echo "Removing debug experiments..."
rm -rf debug_experiments/* 2>/dev/null || true

# Remove large datasets but keep structure
echo "Trimming dataset files..."
rm -rf data/train/* data/validation/* data/test/* data/nejm/* 2>/dev/null || true
mkdir -p data/train data/validation data/test data/nejm
touch data/train/.gitkeep data/validation/.gitkeep data/test/.gitkeep data/nejm/.gitkeep

# Remove database files - they should be recreated at startup
echo "Removing database files..."
rm -f data/db/*.db 2>/dev/null || true
rm -f *.db 2>/dev/null || true

# Remove log files
echo "Removing log files..."
rm -rf logs/* 2>/dev/null || true
mkdir -p logs

# Remove test outputs and cross validation data
echo "Removing test outputs and tuning data..."
rm -rf test_outputs/* 2>/dev/null || true
rm -rf hyperparameter_tuning/* 2>/dev/null || true
rm -rf cross_validation/* 2>/dev/null || true

# Remove any downloaded datasets
echo "Removing downloaded datasets..."
rm -rf download/* 2>/dev/null || true
rm -rf clean_repo/* 2>/dev/null || true
rm -rf attached_assets/* 2>/dev/null || true

# Remove any large model files
echo "Removing large model files..."
find . -name "*.model" -delete 2>/dev/null || true
find . -name "*.bin" -delete 2>/dev/null || true
find . -name "*.pkl" -delete 2>/dev/null || true
find . -name "*.h5" -delete 2>/dev/null || true
find . -type f -size +100M -delete 2>/dev/null || true

echo "Cleanup completed!"
echo "Run this script before deployment to reduce image size."
