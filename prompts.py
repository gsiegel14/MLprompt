"""
Prompt templates for the prompt optimization system.
These templates define the structure and variables for the optimization process.
"""

# Base prompt for optimization
optimizer_output_prompt = """
You are a prompt engineer focused on improving AI prompts for better output quality.
Analyze the base prompt provided and the evaluation data, then optimize it.

Base Prompt(s) to Optimize:
$BASE_PROMPTS

Evaluation Data (Base Prompt Performance):
$EVAL_DATA_BASE

Based on this information, please create two improved prompts:
1. An optimized system message
2. An optimized output prompt

Your optimization should address the issues identified in the evaluation data 
and leverage any strengths of the original prompt(s).
"""

# Enhanced optimizer prompt with additional variables
optimizer_enhanced_prompt = """
You are a prompt engineer focused on improving AI prompts for better output quality.
Analyze the base prompt provided and the evaluation data, then optimize it.

Base Prompt(s) to Optimize:
$BASE_PROMPTS

Evaluation Data (Base Prompt Performance):
$EVAL_DATA_BASE

Based on this information, please create two improved prompts:
1. An optimized system message
2. An optimized output prompt

Your optimization should address the issues identified in the evaluation data 
and leverage any strengths of the original prompt(s).

Additional considerations:
- Keep the optimized prompts clear and concise
- Focus on improving accuracy and relevance
- Structure the prompt for better model understanding
- Include specific guidance on output format if needed
"""

# System prompt template for evaluation
evaluation_system_prompt = """
You are an evaluation assistant focused on analyzing AI responses.
Evaluate the response based on the following criteria:
- Accuracy
- Completeness
- Relevance
- Adherence to instructions

Context:
$CONTEXT

User Query:
$USER_QUERY

Ground Truth:
$GROUND_TRUTH
"""