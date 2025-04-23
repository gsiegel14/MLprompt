"""
API endpoints for the Prompt Optimization Platform

This package contains the implementation of the API endpoints.
"""
"""
API endpoint modules
"""

from src.api.endpoints import (
    optimization,
    experiments,
    prompts,
    datasets,
    cost_tracking,
    inference
)

# Ensure all endpoint modules are properly initialized
__all__ = [
    'optimization',
    'experiments',
    'prompts',
    'datasets',
    'cost_tracking',
    'inference'
]