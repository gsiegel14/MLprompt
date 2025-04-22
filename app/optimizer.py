import os
import logging
import json
from app.llm_client import get_llm_response

logger = logging.getLogger(__name__)

# Default optimizer prompt if none is provided
DEFAULT_OPTIMIZER_PROMPT = """You are an expert prompt engineer tasked with optimizing system prompts and output prompts for a large language model.

Based on the examples provided and their evaluation metrics, your job is to analyze why the current prompts might not be producing optimal results and suggest improved versions.

You will receive:
1. The current system prompt
2. The current output prompt
3. A set of examples with:
   - User input
   - Expected ground truth output
   - Actual model response
   - Evaluation score (0-1, where 1 is perfect)

Analyze the patterns in failures and suggest refined prompts that will help the model produce responses closer to the ground truth.

For your response, please provide:
1. A new system prompt (with explanations of changes)
2. A new output prompt (with explanations of changes) 
3. A summary of your reasoning
"""

def load_optimizer_prompt():
    """Load the optimizer prompt from file or use default."""
    prompt_path = os.path.join('prompts', 'optimizer', 'default_optimizer.txt')
    try:
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                return f.read()
        else:
            # Create directory and file if it doesn't exist
            os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
            with open(prompt_path, 'w') as f:
                f.write(DEFAULT_OPTIMIZER_PROMPT)
            return DEFAULT_OPTIMIZER_PROMPT
    except Exception as e:
        logger.error(f"Error loading optimizer prompt: {e}")
        return DEFAULT_OPTIMIZER_PROMPT

def format_examples_for_optimizer(examples, limit=5):
    """Format examples in a way that's useful for the optimizer LLM."""
    # Sort examples by score (ascending) to prioritize the worst examples
    sorted_examples = sorted(examples, key=lambda x: x.get('score', 0))
    
    # Take the lowest performing examples up to the limit
    examples_to_include = sorted_examples[:limit]
    
    formatted_examples = []
    for i, example in enumerate(examples_to_include):
        formatted = f"""Example {i+1}:
User Input: {example.get('user_input', '')}
Ground Truth: {example.get('ground_truth_output', '')}
Model Response: {example.get('model_response', '')}
Score: {example.get('score', 0):.2f}
"""
        formatted_examples.append(formatted)
    
    return "\n".join(formatted_examples)

def calculate_aggregate_metrics(results):
    """Calculate aggregate metrics from results."""
    if not results:
        return {
            "avg_score": 0,
            "perfect_matches": 0,
            "total_examples": 0,
            "perfect_match_percent": 0
        }
    
    total_score = sum(r.get('score', 0) for r in results)
    perfect_matches = sum(1 for r in results if r.get('score', 0) >= 0.9)
    total = len(results)
    
    return {
        "avg_score": total_score / total if total > 0 else 0,
        "perfect_matches": perfect_matches,
        "total_examples": total,
        "perfect_match_percent": (perfect_matches / total * 100) if total > 0 else 0
    }

def optimize_prompts(current_system_prompt, current_output_prompt, examples, optimizer_system_prompt=None):
    """
    Use a second LLM to optimize the system and output prompts based on examples.
    
    Args:
        current_system_prompt (str): The current system prompt
        current_output_prompt (str): The current output prompt
        examples (list): List of examples with user input, ground truth, model response and score
        optimizer_system_prompt (str, optional): Custom instructions for the optimizer
        
    Returns:
        dict: New system prompt, output prompt, and optimizer reasoning
    """
    if optimizer_system_prompt is None:
        optimizer_system_prompt = load_optimizer_prompt()
    
    metrics = calculate_aggregate_metrics(examples)
    formatted_examples = format_examples_for_optimizer(examples)
    
    # Create input for the optimizer LLM
    optimizer_input = f"""I need you to help optimize my prompts based on these results.

Current System Prompt:
```
{current_system_prompt}
```

Current Output Prompt:
```
{current_output_prompt}
```

Performance Metrics:
- Average Score: {metrics['avg_score']:.2f}
- Perfect Matches: {metrics['perfect_matches']}/{metrics['total_examples']} ({metrics['perfect_match_percent']:.1f}%)

Examples (focus on improving these):
{formatted_examples}

Please provide:
1. An improved system prompt
2. An improved output prompt
3. Your reasoning for the changes

Format your response as:
[SYSTEM_PROMPT]
Your new system prompt here
[/SYSTEM_PROMPT]

[OUTPUT_PROMPT]
Your new output prompt here
[/OUTPUT_PROMPT]

[REASONING]
Your detailed reasoning for the changes made
[/REASONING]
"""
    
    try:
        # Call the optimizer LLM
        response = get_llm_response(
            optimizer_system_prompt,
            optimizer_input,
            "Please analyze the examples and provide improved prompts."
        )
        
        # Parse the response
        system_prompt = extract_section(response, "SYSTEM_PROMPT")
        output_prompt = extract_section(response, "OUTPUT_PROMPT")
        reasoning = extract_section(response, "REASONING")
        
        return {
            "system_prompt": system_prompt,
            "output_prompt": output_prompt,
            "reasoning": reasoning,
            "original_response": response
        }
    except Exception as e:
        logger.error(f"Error in optimizer: {e}")
        return {
            "system_prompt": current_system_prompt,
            "output_prompt": current_output_prompt,
            "reasoning": f"Optimization failed: {str(e)}",
            "original_response": ""
        }

def extract_section(text, section_name):
    """Extract a section from the response text."""
    start_tag = f"[{section_name}]"
    end_tag = f"[/{section_name}]"
    
    start_pos = text.find(start_tag)
    end_pos = text.find(end_tag)
    
    if start_pos != -1 and end_pos != -1:
        # Extract the content between the tags
        content = text[start_pos + len(start_tag):end_pos].strip()
        return content
    
    # If not found in the expected format, try to find the section another way
    lines = text.split('\n')
    section_content = []
    in_section = False
    
    for line in lines:
        if section_name.lower() in line.lower() and not in_section:
            in_section = True
            continue
        elif in_section and any(s.lower() in line.lower() for s in ["SYSTEM_PROMPT", "OUTPUT_PROMPT", "REASONING"]) and section_name.lower() not in line.lower():
            break
        elif in_section:
            section_content.append(line)
    
    return '\n'.join(section_content).strip()