
# Prompt Optimization Platform

A machine learning platform for iteratively refining LLM prompts through a 5-step ML-driven workflow.

## Architecture Overview

```
                   ┌───────────────────────────┐
                   │    FastAPI Application    │
                   │    (Main Entry Point)     │
                   └───────────┬───────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
    ┌───────────▼────────────┐  ┌─────────────▼─────────────┐
    │     Flask Dashboard    │  │      Prefect Flows        │
    │  (Monitoring & Admin)  │  │  (Workflow Orchestration) │
    └───────────┬────────────┘  └─────────────┬─────────────┘
                │                              │
                └──────────────┬──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Shared Services   │
                    │ (Auth, DB, Caching) │
                    └─────────────────────┘
```

## Features

- **5-Step ML-Driven Workflow**: A structured approach to prompt optimization
- **Prefect Flow Integration**: Orchestrated workflow management
- **FlaskUI Dashboard**: Monitoring and visualization for experiments and costs
- **FastAPI Backend**: High-performance REST API
- **Authentication**: API key validation for secure access
- **Cost Tracking**: Monitor token usage and associated costs
- **Caching**: Reduce API costs with response caching
- **Unified Logging**: Structured JSON logs across all components

## Getting Started

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Start the application:
```
uvicorn src.app.main:app --host 0.0.0.0 --port 5000 --reload
```

3. Access the application:
   - Dashboard: http://localhost:5000/dashboard
   - API Documentation: http://localhost:5000/api/docs

## Dashboard

The Flask dashboard provides visualization and monitoring for:

- Experiment tracking
- Cost monitoring
- Workflow status
- System health

## API Documentation

The API documentation is available at `/api/docs` and includes endpoints for:

- Prompt management
- Optimization workflows
- Experiment tracking
- Dataset management
- Cost reporting

## Configuration

Configure the application through environment variables or a `.env` file:

- `ENVIRONMENT`: The environment (development, production)
- `DEBUG`: Enable debug mode (1 or 0)
- `API_KEY`: API key for authentication
- `VERTEX_PROJECT_ID`: Google Cloud Vertex AI project ID
- `PRIMARY_MODEL`: Primary LLM model name
- `OPTIMIZER_MODEL`: Optimizer LLM model name
- `PREFECT_ENABLED`: Enable Prefect integration (1 or 0)
- `PREFECT_API_URL`: Prefect API URL
- `PREFECT_API_KEY`: Prefect API key

## License

[MIT License](LICENSE)

# MLprompt - Prompt Engineering ML Platform

## Overview
A web-based UI for iteratively testing and refining LLM prompts using a machine learning approach with Google's Gemini API. This platform allows users to experiment with different prompt variations, evaluate responses against expected outputs, track metrics over time, and visualize effectiveness of prompt refinements.

## Key Features
- **Interactive Training Interface**: ML-style training flow for prompt engineering
- **Three-LLM Architecture**: Primary LLM, Evaluation Engine, and Optimizer LLM
- **Multiple Optimization Strategies**: Full rewrite, targeted edit, example addition
- **Comprehensive Experiment Tracking**: History views and metrics visualization 
- **User-Friendly Design**: Modern styling with contextual help tooltips

## Technical Stack
- Flask backend with JavaScript frontend
- Google Gemini API Integration
- Experiment tracking and versioning
- Data visualization using Chart.js

## Getting Started
1. Clone this repository
2. Install dependencies with `pip install -r requirements.txt`
3. Set up your Google API key in environment variables
4. Run the application with `python main.py`

## Usage
- Input system prompt and output prompt for the LLM
- Add test cases with input-output pairs
- Run evaluation and view metrics
- Use the optimizer to refine prompts based on results
- Track experiment history and compare iterations

## Medical Case Study Feature
This application comes pre-loaded with NEJM (New England Journal of Medicine) case studies:
- 159 medical cases split 50/50 between training and validation sets
- Enhanced similarity evaluation that checks if the ground truth diagnosis appears in the LLM response
- Ideal for training diagnostic reasoning prompts
