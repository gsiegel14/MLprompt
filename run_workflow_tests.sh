#!/bin/bash

# Run Workflow Tests Script
# This script runs comprehensive workflow tests and saves the output to a log file

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="workflow_test_${TIMESTAMP}.log"

echo "Running 5-API workflow tests at $(date)"
echo "Results will be saved to ${LOG_FILE}"

echo "========================="
echo "Checking if server is running..."

# Check if the server is running
curl -s http://localhost:5000/ > /dev/null
SERVER_STATUS=$?

if [ $SERVER_STATUS -ne 0 ]; then
    echo "Server not running at http://localhost:5000/"
    echo "Please start the server first using: gunicorn --bind 0.0.0.0:5000 main:app"
    exit 1
fi

echo "Server is running."
echo "========================="
echo "Starting tests..."

# Run the tests and capture output to log file
python test_five_api_workflow.py | tee ${LOG_FILE}
TEST_STATUS=${PIPESTATUS[0]}

echo "========================="
if [ $TEST_STATUS -eq 0 ]; then
    echo "Tests completed successfully!"
else 
    echo "Tests completed with errors! Check ${LOG_FILE} for details."
fi
echo "Results saved to ${LOG_FILE}"
echo "========================="

exit $TEST_STATUS