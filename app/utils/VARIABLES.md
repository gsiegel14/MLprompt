# Prompt Variable Substitution System

This document explains the variable substitution system used in the ML Prompt Optimization Platform. Variables can be used in prompts and will be automatically replaced with their actual values when the prompt is processed.

## Available Variables

| Variable | Description | Example |
|----------|------------|---------|
| `$USER_INPUT` | The user input from CSV examples or manually entered input | "A 45-year-old male presents with fever and cough..." |
| `$EVAL_DATA_BASE` | Evaluation data from the Hugging Face API call #2 | "Evaluation Metrics:\n- exact_match: 0.75\n- semantic_similarity: 0.82..." |
| `$EVAL_DATA_OPTIMIZED` | Evaluation data from the Hugging Face API call #4 | "Evaluation Metrics:\n- exact_match: 0.88\n- semantic_similarity: 0.95..." |
| `$DATASET_ANSWERS_BASE` | Answers to the dataset for prompt optimizer | "Dataset Examples:\n\nExample 1:\nInput: ...\nExpected Output: ...\nModel Output: ..." |

## How to Use Variables

Variables can be included in any prompt template by adding the variable name with a dollar sign prefix. For example:

```
Please analyze the following input: $USER_INPUT

Based on the evaluation data: $EVAL_DATA_BASE 

Optimize the prompt to improve these metrics.
```

When this prompt is processed, the variables will be replaced with their actual values.

## Integration in Workflows

The variable substitution system is integrated into several workflows:

1. **Five-API Workflow**: Variables are substituted before sending prompts to the API
2. **Evaluation**: User inputs are substituted into prompts before evaluation
3. **Optimization**: Evaluation results are substituted into optimizer prompts

## Example Use Cases

### User Input Substitution
Use `$USER_INPUT` to include the test case or example in your prompt:

```
You are a medical diagnosis assistant.
Given the following case: $USER_INPUT
Provide a differential diagnosis based on the symptoms described.
```

### Evaluation Data Integration
Use `$EVAL_DATA_BASE` to include evaluation metrics in optimizer prompts:

```
You are a prompt optimization expert.
The current prompt has achieved the following metrics: $EVAL_DATA_BASE
Please improve the prompt to increase the exact match score.
```

### Dataset Answers for Optimization
Use `$DATASET_ANSWERS_BASE` to show example answers in optimizer prompts:

```
You are optimizing a prompt based on examples.
Here are the current model outputs: $DATASET_ANSWERS_BASE
Analyze these outputs and improve the prompt to increase accuracy.
```

## Adding New Variables

To add new variables to the system:

1. Update the `create_variables_dict` function in `app/utils/prompt_variables.py`
2. Add a formatting function if needed (like `format_dataset_answers`)
3. Update this documentation to include the new variable