#!/usr/bin/env python3
"""
Simple test to check if the HUGGING_FACE_TOKEN is accessible
"""

import os
import sys

def main():
    """Check if the HUGGING_FACE_TOKEN is set and accessible."""
    token = os.environ.get("HUGGING_FACE_TOKEN")
    
    if token:
        print(f"✓ HUGGING_FACE_TOKEN is set! Length: {len(token)}")
        print(f"  First 4 chars: {token[:4]}...")
        return 0
    else:
        print("✗ HUGGING_FACE_TOKEN is not set or not accessible!")
        return 1

if __name__ == "__main__":
    sys.exit(main())