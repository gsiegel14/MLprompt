# Prompt Optimization Variables

This document outlines the variables used in the prompt optimization workflow process.

## Variable Definitions

### Base Variables

- `$BASE_SYSTEM_MESSAGE` - The system message from base prompts (located at prompts/Base Prompts/Base_system_message.md)
- `$BASE_OUTPUT_PROMPT` - The output prompt from base prompts (located at prompts/Base Prompts/Base_output_prompt.md)
- `$BASE_PROMPTS` - Collection of base prompts to be incorporated into optimized prompts

### Evaluation Variables

- `$EVAL_PROMPT` - The prompt used by Hugging Face for evaluation (located at prompts/evaluator/evaluatorprompt.txt)
- `$DATASET_ANSWERS_BASE` - Ground truth answers from dataset used for comparison
- `$EVAL_DATA_BASE` - Evaluation data from testing base prompts with Hugging Face

### Optimized Variables

- `$OPTIMIZED_SYSTEM_MESSAGE` - The optimized system message (located at prompts/optimizer/Optimizer_systemmessage.md.txt)
- `$OPTIMIZED_OUTPUT_PROMPT` - The optimized output prompt (located at prompts/optimizer/optimizer_output_prompt.txt)
- `$OPTIMIZED_PROMPTS` - Combined optimized prompts for use in final API call
- `$EVAL_DATA_OPTIMIZED` - Evaluation data from testing optimized prompts with Hugging Face

## Workflow Process

The workflow follows these steps:

1. **API Call 1: Base Prompt Testing**
   - Inputs: `$BASE_SYSTEM_MESSAGE` and `$BASE_OUTPUT_PROMPT`
   - Process: Run user input with base prompts
   - Output: Initial model responses

2. **API Call 2: Base Prompt Evaluation**
   - Inputs: Model responses from API Call 1, `$EVAL_PROMPT`, and `$DATASET_ANSWERS_BASE`
   - Process: Hugging Face evaluates base prompt performance
   - Output: `$EVAL_DATA_BASE`

3. **API Call 3: Prompt Optimization**
   - Inputs: `$EVAL_DATA_BASE` and `$BASE_PROMPTS`
   - Process: Optimize prompts based on evaluation data
   - Output: `$OPTIMIZED_SYSTEM_MESSAGE` and `$OPTIMIZED_OUTPUT_PROMPT` (combined as `$OPTIMIZED_PROMPTS`)

4. **API Call 4: Optimized Prompt Testing**
   - Inputs: `$OPTIMIZED_SYSTEM_MESSAGE` and `$OPTIMIZED_OUTPUT_PROMPT`
   - Process: Run user input with optimized prompts
   - Output: Optimized model responses

5. **API Call 5: Optimized Prompt Evaluation**
   - Inputs: Model responses from API Call 4, `$EVAL_PROMPT`, and `$DATASET_ANSWERS_BASE`
   - Process: Hugging Face evaluates optimized prompt performance
   - Output: `$EVAL_DATA_OPTIMIZED`

This workflow enables continuous prompt improvement by:
1. Establishing baseline performance with initial prompts
2. Using evaluation data to guide optimization
3. Testing optimized prompts
4. Measuring improvement through comparative evaluation

## Directory Structure

```
prompts/
├── Base Prompts/
│   ├── Base_system_message.md
│   └── Base_output_prompt.md
├── evaluator/
│   └── evaluatorprompt.txt
├── optimizer/
│   ├── Optimizer_systemmessage.md.txt
│   └── optimizer_output_prompt.txt
└── output/
    └── [generated output files]
```