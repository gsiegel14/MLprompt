# Prompt Engineering ML Platform

A machine learning platform for iteratively refining LLM prompts through automated optimization.

## Overview

This platform enables prompt engineers to systematically improve system prompts and output prompts through a machine-learning-style interface. It uses a **three-LLM architecture operating in an autonomous feedback loop**:

1. **Primary LLM**: Processes user inputs with the current `system_prompt` and `output_prompt`.
2. **Evaluation Engine**: Measures response quality against ground truth.
3. **Optimizer LLM**: Analyzes evaluation results (especially failures) and **autonomously proposes corrected versions** of the `system_prompt` and `output_prompt` to improve performance.

The platform implements a complete ML training loop for prompt engineering, where prompts are automatically tested and refined based on their measured performance, with metrics tracking, version history, and visualization.

## System Architecture

```
┌───────────────────┐     ┌────────────────────┐     ┌───────────────────┐
│                   │     │                    │     │                   │
│  PRIMARY LLM      │────▶│  EVALUATION        │────▶│  OPTIMIZER LLM    │
│  (Google Gemini)  │     │                    │     │  (Google Gemini)  │
│                   │     │                    │     │                   │
└───────────────────┘     └────────────────────┘     └───────────────────┘
        ▲                          │                          │
        │                          │                          │
        │                          ▼                          │
        │                 ┌────────────────────┐             │
        │                 │                    │             │
        │                 │  METRICS DISPLAY   │             │
        │                 │                    │             │
        └─────────────────│                    │◀────────────┘
                          │                    │
                          └────────────────────┘
```

## Key Features

- **ML-style Training Interface**: Configure epochs, batch sizes, and optimization parameters
- **Automated Prompt Refinement Loop**: The Optimizer LLM **autonomously suggests and tests** prompt improvements in a closed loop
- **Metrics Tracking**: Visualize performance improvements across training iterations
- **Version Control**: Track prompt evolution with full history and comparisons
- **Train/Validation Split**: Properly validate prompt improvements to prevent overfitting
- **Google Gemini Integration**: Leverages Google's powerful LLM models

## Project Structure

```
prompt-ml-platform/
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── js/
│   │   │   ├── script.js
│   │   │   ├── training.js          # ML training loop UI logic
│   │   │   └── history.js           # Experiment history and visualization
│   │   └── img/
│   ├── templates/
│   │   ├── index.html               # Basic prompt refinement interface
│   │   ├── training.html            # ML training interface
│   │   └── history.html             # Experiment history tracking
│   ├── __init__.py
│   ├── main.py                      # Flask routes
│   ├── llm_client.py                # Google Gemini API interaction
│   ├── optimizer.py                 # Second LLM optimization logic
│   ├── evaluator.py                 # Metrics calculation
│   ├── data_module.py               # Dataset handling
│   └── experiment_tracker.py        # Tracking metrics/versions
│
├── data/
│   ├── train/                       # Training examples
│   └── validation/                  # Validation examples
│
├── prompts/
│   ├── system/                      # System prompts versions
│   ├── output/                      # Output prompts versions
│   └── optimizer/                   # Optimizer LLM prompts
│       └── default_optimizer.txt    # Default instructions for optimizer
│
├── experiments/                     # Saved experiment results
│   └── metrics/                     # Metrics over time
│
├── config.yaml                      # Configuration
├── requirements.txt
└── README.md
```

## Core Workflow

### 1. Initial Setup
- Configure Google Gemini API credentials
- Define initial system and output prompts
- Upload training/validation data (CSV with `user_input`/`ground_truth` pairs)
- Configure the initial instructions for the Optimizer LLM

### 2. Training Loop (Autonomous Refinement)
- The Primary LLM processes training examples with the current `system_prompt` and `output_prompt`
- The Evaluation Engine calculates performance metrics by comparing responses to the `ground_truth_output`
- The Optimizer LLM receives performance data (focusing on failures) and **autonomously proposes corrected versions** of the `system_prompt` and `output_prompt`
- **Crucially, these new prompts are then automatically tested**: The Primary LLM reruns queries (typically on a validation set) using the *corrected* prompts
- The Evaluation Engine assesses if the new prompts yield outputs closer to the `ground_truth_output`
- If performance improves (based on validation metrics), the corrected prompts are accepted and become the new baseline for the next iteration
- This feedback loop repeats for the specified number of epochs or until early stopping criteria are met

### 3. Analysis & Export
- Review metrics and prompt evolution over the training iterations
- Compare different prompt versions
- Export final optimized prompts

## Machine Learning Interface

The UI follows standard ML training conventions:

**Prompt Configuration Panel:**
- Input areas for system_prompt and output_prompt
- Version history dropdown

**Training Data Management:**
- CSV upload for input/output pairs
- Train/validation split options
- Dataset statistics

**Training Control Panel:**
- Start/stop training
- Epochs and batch size settings
- Early stopping options

**Optimizer LLM Configuration:**
- Configuration for optimizer prompt
- Strategy selection options

**Results Dashboard:**
- Metrics visualization
- Before/after examples table
- Prompt evolution display

**Logs Panel:**
- Real-time training progress
- Optimizer reasoning logs

## Usage Guide

### Preparing Your Dataset
Create a CSV file with the following columns:
- `user_input`: The input text to be sent to the LLM
- `ground_truth_output`: The expected output

Example:
```
user_input,ground_truth_output
"Translate to French: Hello world","Bonjour le monde"
"Translate to French: Good morning","Bonjour"
```

### Initial Prompts
Start with a basic system prompt and output prompt:

System Prompt Example:
```
You are a helpful, precise assistant that translates English to French.
```

Output Prompt Example:
```
Translate the above text to French. Return only the translation with no additional text.
```

### Optimization Strategies

The platform includes multiple optimization approaches:
1. **Full Rewrite**: Optimizer LLM completely rewrites prompts
2. **Targeted Edit**: Optimizer suggests specific changes to problematic sections
3. **Example Addition**: Dynamically adds few-shot examples to the prompts