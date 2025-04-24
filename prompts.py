# This file contains the prompts used by the application

# The optimizer output prompt with the added $BASE_PROMPTS variable
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
