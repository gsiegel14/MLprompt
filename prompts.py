# This file contains the prompts used by the application
# These prompts and variables are used in the prompt optimization workflow

# Base system message template
base_system_message_template = """
$BASE_SYSTEM_MESSAGE
"""

# Base output prompt template
base_output_prompt_template = """
$BASE_OUTPUT_PROMPT
"""

# Base prompt collection
base_prompts_template = """
$BASE_PROMPTS
"""

# Evaluator prompt for Hugging Face evaluation
evaluator_prompt = """
$EVAL_PROMPT

Compare the model response with the ground truth answer:

Model Response:
{model_response}

Ground Truth:
{ground_truth}

Evaluate the model response on the following criteria:
1. Accuracy: How accurate is the response compared to ground truth?
2. Completeness: Does the response contain all necessary information?
3. Relevance: Is the response relevant to the question/task?

Score the response on a scale of 0-10 and provide brief justification.
"""

# Dataset answers template
dataset_answers_base_template = """
$DATASET_ANSWERS_BASE
"""

# The original optimizer output prompt with the added $BASE_PROMPTS variable
# This prompt template will be used when sending the prompt to the optimization service
optimizer_output_prompt = """
Your task is to optimize the following prompt to make it more effective, clear, and likely to produce high-quality outputs from AI systems.

First, consider these base prompts that should be incorporated or considered:

$BASE_PROMPTS

Now, please optimize the following prompt:

{original_prompt}

In your optimization:
1. Maintain the core intent and goals of the original prompt
2. Incorporate relevant elements from the base prompts provided above
3. Improve clarity, specificity, and structure
4. Add any necessary constraints or guidelines
5. Ensure the prompt is well-structured and free of ambiguity

Respond with the optimized version of the prompt only, without explanations or additional commentary.
"""

# Enhanced optimizer prompt that includes evaluation data from Hugging Face
optimizer_enhanced_prompt = """
Your task is to optimize the following prompt to make it more effective, clear, and likely to produce high-quality outputs from AI systems.

First, consider these base prompts that should be incorporated or considered:

$BASE_PROMPTS

Now, please review the evaluation data from initial testing:

$EVAL_DATA_BASE

Based on this evaluation data, please optimize the following prompt:

{original_prompt}

In your optimization:
1. Maintain the core intent and goals of the original prompt
2. Incorporate relevant elements from the base prompts provided above
3. Address the specific issues identified in the evaluation data
4. Improve clarity, specificity, and structure
5. Add any necessary constraints or guidelines
6. Ensure the prompt is well-structured and free of ambiguity

Your optimization should produce two distinct components:
1. A system message (instructions for the AI model)
2. An output prompt (instructions for generating the desired output)

Return your optimization in the following format:
===SYSTEM_MESSAGE===
[Your optimized system message here]
===OUTPUT_PROMPT===
[Your optimized output prompt here]
"""

# Variables for storing optimized prompts
optimized_system_message_template = """
$OPTIMIZED_SYSTEM_MESSAGE
"""

optimized_output_prompt_template = """
$OPTIMIZED_OUTPUT_PROMPT
"""

# Optimized prompts combined template
optimized_prompts_template = """
$OPTIMIZED_PROMPTS
"""

# Template for evaluation data from base prompt testing
evaluation_data_base_template = """
$EVAL_DATA_BASE
"""

# Template for evaluation data from optimized prompt testing
evaluation_data_optimized_template = """
$EVAL_DATA_OPTIMIZED
"""

# Template for final evaluation with optimized prompts
evaluation_template = """
Evaluation Results for Optimized Prompts:

$EVAL_DATA_OPTIMIZED

Summary:
- Initial performance score: {initial_score}
- Optimized performance score: {optimized_score}
- Improvement: {improvement}%

Recommendations for further optimization:
{recommendations}
"""
