
#!/bin/bash

# Pre-deployment cleanup script
echo "Starting pre-deployment cleanup..."

# Create required directories with placeholders
mkdir -p data/train data/validation data/test data/nejm
touch data/train/.gitkeep data/validation/.gitkeep data/test/.gitkeep data/nejm/.gitkeep

# Clean up cache files
echo "Cleaning cache directories..."
find cache -type f -name "*.pkl" -delete
find cache -type f -name "*.meta" -delete

# Keep only essential experiment data
echo "Cleaning up experiment data..."
find experiments -type f -name "examples_*.json" -delete
find experiments -type d -name "examples" -exec rm -rf {} +

# Remove large datasets but keep structure
echo "Trimming dataset files..."
mv data/train/current_train.json data/train/current_train.json.bak
mv data/validation/current_validation.json data/validation/current_validation.json.bak
rm -rf data/train/* data/validation/* data/test/* data/nejm/*
mkdir -p data/train data/validation data/test data/nejm
touch data/train/.gitkeep data/validation/.gitkeep data/test/.gitkeep data/nejm/.gitkeep
[ -f data/train/current_train.json.bak ] && mv data/train/current_train.json.bak data/train/current_train.json
[ -f data/validation/current_validation.json.bak ] && mv data/validation/current_validation.json.bak data/validation/current_validation.json

# Remove database files - they should be recreated at startup
echo "Removing database files..."
rm -f data/db/promptopt.db

# Clean Python cache
echo "Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Remove log files
echo "Removing log files..."
rm -rf logs/*
rm -f *.log

echo "Cleanup completed!"
echo "Run this script before deployment to reduce image size."
