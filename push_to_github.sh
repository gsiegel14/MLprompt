
#!/bin/bash

# GitHub Push Script
# This script safely pushes changes to GitHub

# Check if the user has provided a commit message
if [ "$#" -ne 1 ]; then
    echo "Usage: ./push_to_github.sh \"Your commit message\""
    exit 1
fi

COMMIT_MESSAGE="$1"

# Verify we're not pushing sensitive files
echo "Checking for sensitive files..."
if grep -q "GH_TOKEN" gh_login.sh 2>/dev/null; then
    echo "⚠️ Warning: Found sensitive token in gh_login.sh"
    echo "This file should not be committed. Please use gh_login.sh.template instead."
    exit 1
fi

# Add files to git
echo "Adding files to git..."
git add .

# Exclude sensitive files
echo "Excluding sensitive files..."
git reset -- gh_login.sh .config/gh/hosts.yml 2>/dev/null

# Commit changes
echo "Committing changes..."
git commit -m "$COMMIT_MESSAGE"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin main

echo "Push completed successfully! 🚀"
