#!/bin/bash

# Log file
LOG_FILE="workflow_test_$(date +%Y%m%d_%H%M%S).log"

# Terminal colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored headers
print_header() {
    echo -e "\n${BLUE}==========================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}==========================================================================${NC}\n"
}

# Function to print colored result
print_result() {
    if [ "$1" -eq 0 ]; then
        echo -e "\n${GREEN}✓ $2${NC}\n"
    else
        echo -e "\n${RED}✗ $2${NC}\n"
    fi
}

# Make the test script executable
chmod +x test_five_api_workflow.py

print_header "Running 5-API Workflow Comprehensive Tests"
echo "Test results will be saved to: $LOG_FILE"

# Check if Flask server is running
if ! curl -s http://localhost:5000/ > /dev/null; then
    echo -e "${YELLOW}Warning: Flask application does not appear to be running on port 5000.${NC}"
    echo -e "${YELLOW}Make sure the application is running before continuing.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the Python test script
print_header "Executing test_five_api_workflow.py"
python test_five_api_workflow.py | tee "$LOG_FILE"

# Check the test script's exit code
TEST_EXIT_CODE=${PIPESTATUS[0]}

# Report results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_result 0 "5-API Workflow test completed successfully"
else
    print_result 1 "5-API Workflow test failed"
fi

echo -e "${CYAN}Detailed test results are available in: ${YELLOW}$LOG_FILE${NC}"

exit $TEST_EXIT_CODE