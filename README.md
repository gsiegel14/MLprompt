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
