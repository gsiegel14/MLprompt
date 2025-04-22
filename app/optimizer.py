import os
import logging
import json
import yaml
from typing import Dict, Any, List, Optional, Tuple
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

When analyzing examples, look for:
- Patterns in types of questions that perform poorly
- Missing context or instructions in the system prompt
- Format requirements that might be unclear
- Potential ambiguities or misunderstandings
- Specificity vs. generality issues

Your goal is to iteratively improve the prompts to get the model responses closer to the ground truth outputs.

Be precise and thorough in your analysis. Think like a machine learning engineer trying to optimize a model through better instructions.
"""

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {
            'optimizer': {
                'model_name': 'gemini-1.5-pro',
                'temperature': 0.7,
                'max_output_tokens': 2048,
                'strategies': ['full_rewrite']
            }
        }

def load_optimizer_prompt() -> str:
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

def select_examples_for_optimizer(examples: List[Dict[str, Any]], strategy: str = 'worst_performing', limit: int = 5) -> List[Dict[str, Any]]:
    """
    Select examples for optimizer based on strategy.
    
    Args:
        examples (list): List of examples with scores
        strategy (str): Strategy for selection ('worst_performing', 'diverse', 'random')
        limit (int): Maximum number of examples to include
        
    Returns:
        list: Selected examples
    """
    if not examples:
        return []
    
    if len(examples) <= limit:
        return examples
    
    if strategy == 'worst_performing':
        # Sort examples by score (ascending) to prioritize the worst examples
        sorted_examples = sorted(examples, key=lambda x: x.get('score', 0))
        return sorted_examples[:limit]
    
    elif strategy == 'diverse':
        # Try to get examples with diverse scores
        examples_by_score = {}
        for example in examples:
            score = round(example.get('score', 0), 1)
            if score not in examples_by_score:
                examples_by_score[score] = []
            examples_by_score[score].append(example)
        
        # Take examples from each score bucket
        selected = []
        scores = sorted(examples_by_score.keys())
        
        # First prioritize the worst scores
        for score in scores:
            if len(selected) >= limit:
                break
            if score < 0.5 and examples_by_score[score]:
                selected.append(examples_by_score[score][0])
                examples_by_score[score] = examples_by_score[score][1:]
        
        # Then add diverse scores
        while len(selected) < limit and any(examples_by_score.values()):
            for score in scores:
                if examples_by_score[score]:
                    selected.append(examples_by_score[score][0])
                    examples_by_score[score] = examples_by_score[score][1:]
                    if len(selected) >= limit:
                        break
        
        return selected
    
    elif strategy == 'random':
        # Randomly select examples
        import random
        return random.sample(examples, limit)
    
    else:
        # Default to worst performing
        sorted_examples = sorted(examples, key=lambda x: x.get('score', 0))
        return sorted_examples[:limit]

def format_examples_for_optimizer(examples: List[Dict[str, Any]], limit: int = 5) -> str:
    """Format examples in a way that's useful for the optimizer LLM."""
    # Select examples (default to worst performing)
    examples_to_include = select_examples_for_optimizer(examples, 'worst_performing', limit)
    
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

def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
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

def generate_optimization_prompt(current_system_prompt: str, current_output_prompt: str, 
                                examples: List[Dict[str, Any]], metrics: Dict[str, Any],
                                strategy: str = 'full_rewrite') -> str:
    """
    Generate the prompt for the optimizer LLM based on strategy.
    
    Args:
        current_system_prompt (str): The current system prompt
        current_output_prompt (str): The current output prompt
        examples (list): Examples with results
        metrics (dict): Performance metrics
        strategy (str): Optimization strategy
        
    Returns:
        str: Formatted prompt for the optimizer
    """
    formatted_examples = format_examples_for_optimizer(examples)
    
    if strategy == 'full_rewrite':
        return f"""I need you to help optimize my prompts based on these results.

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
    
    elif strategy == 'targeted_edit':
        return f"""I need you to help optimize my prompts by making targeted edits.

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

Instead of completely rewriting the prompts, identify specific problems and suggest targeted edits.
For each prompt, highlight exactly what parts should be changed, added, or removed.

Format your response as:
[SYSTEM_PROMPT_EDITS]
- Edit 1: Replace "X" with "Y" because...
- Edit 2: Add "Z" after "W" because...
[/SYSTEM_PROMPT_EDITS]

[MODIFIED_SYSTEM_PROMPT]
The complete edited system prompt
[/MODIFIED_SYSTEM_PROMPT]

[OUTPUT_PROMPT_EDITS]
- Edit 1: Replace "X" with "Y" because...
- Edit 2: Add "Z" after "W" because...
[/OUTPUT_PROMPT_EDITS]

[MODIFIED_OUTPUT_PROMPT]
The complete edited output prompt
[/MODIFIED_OUTPUT_PROMPT]

[REASONING]
Your detailed reasoning for the specific changes
[/REASONING]
"""
    
    elif strategy == 'example_addition':
        return f"""I need you to help optimize my prompts by adding few-shot examples.

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

Problem Examples (these aren't working well):
{formatted_examples}

Your task is to:
1. Analyze why the current prompts fail on these examples
2. Create 2-3 few-shot examples to add to the system prompt
3. Suggest any minor edits to the existing prompts

Format your response as:
[FEW_SHOT_EXAMPLES]
Example 1:
User Input: ...
Expected Output: ...

Example 2:
User Input: ...
Expected Output: ...
[/FEW_SHOT_EXAMPLES]

[MODIFIED_SYSTEM_PROMPT]
Your edited system prompt with few-shot examples integrated
[/MODIFIED_SYSTEM_PROMPT]

[MODIFIED_OUTPUT_PROMPT]
Your edited output prompt
[/MODIFIED_OUTPUT_PROMPT]

[REASONING]
Your detailed reasoning for adding these examples
[/REASONING]
"""
    
    else:
        # Default to full rewrite
        return generate_optimization_prompt(
            current_system_prompt, 
            current_output_prompt, 
            examples, 
            metrics, 
            'full_rewrite'
        )

def optimize_prompts(current_system_prompt: str, current_output_prompt: str, 
                    examples: List[Dict[str, Any]], optimizer_system_prompt: Optional[str] = None,
                    strategy: Optional[str] = None) -> Dict[str, Any]:
    """
    Use a second LLM to optimize the system and output prompts based on examples.
    
    Args:
        current_system_prompt (str): The current system prompt
        current_output_prompt (str): The current output prompt
        examples (list): List of examples with user input, ground truth, model response and score
        optimizer_system_prompt (str, optional): Custom instructions for the optimizer
        strategy (str, optional): Optimization strategy
        
    Returns:
        dict: New system prompt, output prompt, and optimizer reasoning
    """
    # Load configuration
    config = load_config()
    optimizer_config = config.get('optimizer', {})
    
    # Set default values if not provided
    if optimizer_system_prompt is None:
        optimizer_system_prompt = load_optimizer_prompt()
    
    if strategy is None:
        strategies = optimizer_config.get('strategies', ['full_rewrite'])
        strategy = strategies[0]
    
    # Calculate metrics
    metrics = calculate_aggregate_metrics(examples)
    
    # Generate input for the optimizer LLM based on strategy
    optimizer_input = generate_optimization_prompt(
        current_system_prompt,
        current_output_prompt,
        examples,
        metrics,
        strategy
    )
    
    try:
        # Get the optimizer LLM configuration
        llm_config = {
            'model_name': optimizer_config.get('model_name', 'gemini-1.0-pro'),  # Using free tier model
            'temperature': optimizer_config.get('temperature', 0.7),
            'max_output_tokens': optimizer_config.get('max_output_tokens', 2048)
        }
        
        # Call the optimizer LLM
        response = get_llm_response(
            optimizer_system_prompt,
            optimizer_input,
            "Please analyze the examples and provide improved prompts.",
            llm_config
        )
        
        # Parse the response based on strategy
        if strategy == 'full_rewrite':
            system_prompt = extract_section(response, "SYSTEM_PROMPT")
            output_prompt = extract_section(response, "OUTPUT_PROMPT")
            reasoning = extract_section(response, "REASONING")
        
        elif strategy == 'targeted_edit':
            system_prompt = extract_section(response, "MODIFIED_SYSTEM_PROMPT")
            output_prompt = extract_section(response, "MODIFIED_OUTPUT_PROMPT")
            reasoning = extract_section(response, "REASONING")
            # Also extract the edits for more detailed reasoning
            system_edits = extract_section(response, "SYSTEM_PROMPT_EDITS")
            output_edits = extract_section(response, "OUTPUT_PROMPT_EDITS")
            if system_edits and output_edits:
                reasoning = f"System Prompt Edits:\n{system_edits}\n\nOutput Prompt Edits:\n{output_edits}\n\n{reasoning}"
        
        elif strategy == 'example_addition':
            system_prompt = extract_section(response, "MODIFIED_SYSTEM_PROMPT")
            output_prompt = extract_section(response, "MODIFIED_OUTPUT_PROMPT")
            reasoning = extract_section(response, "REASONING")
            # Also extract the few-shot examples
            few_shot_examples = extract_section(response, "FEW_SHOT_EXAMPLES")
            if few_shot_examples:
                reasoning = f"Added Few-Shot Examples:\n{few_shot_examples}\n\n{reasoning}"
        
        else:
            # Default parsing
            system_prompt = extract_section(response, "SYSTEM_PROMPT") or extract_section(response, "MODIFIED_SYSTEM_PROMPT")
            output_prompt = extract_section(response, "OUTPUT_PROMPT") or extract_section(response, "MODIFIED_OUTPUT_PROMPT")
            reasoning = extract_section(response, "REASONING")
        
        # If we couldn't extract the sections properly, use the originals
        if not system_prompt:
            system_prompt = current_system_prompt
            reasoning = f"Failed to extract system prompt from response. Using original.\n{reasoning}"
        
        if not output_prompt:
            output_prompt = current_output_prompt
            reasoning = f"Failed to extract output prompt from response. Using original.\n{reasoning}"
        
        return {
            "system_prompt": system_prompt,
            "output_prompt": output_prompt,
            "reasoning": reasoning,
            "original_response": response,
            "strategy": strategy,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error in optimizer: {e}")
        return {
            "system_prompt": current_system_prompt,
            "output_prompt": current_output_prompt,
            "reasoning": f"Optimization failed: {str(e)}",
            "original_response": "",
            "strategy": strategy,
            "metrics": metrics,
            "error": str(e)
        }

def extract_section(text: str, section_name: str) -> str:
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
        elif in_section and any(s.lower() in line.lower() for s in ["SYSTEM_PROMPT", "OUTPUT_PROMPT", 
                                                                   "REASONING", "MODIFIED_SYSTEM_PROMPT", 
                                                                   "MODIFIED_OUTPUT_PROMPT", "SYSTEM_PROMPT_EDITS",
                                                                   "OUTPUT_PROMPT_EDITS", "FEW_SHOT_EXAMPLES"]) and section_name.lower() not in line.lower():
            break
        elif in_section:
            section_content.append(line)
    
    return '\n'.join(section_content).strip()

def get_optimization_strategies() -> List[Dict[str, str]]:
    """Get available optimization strategies with descriptions."""
    strategies = [
        {
            "id": "full_rewrite",
            "name": "Complete Rewrite",
            "description": "Completely rewrite both prompts based on examples"
        },
        {
            "id": "targeted_edit",
            "name": "Targeted Edits",
            "description": "Make specific changes to problematic sections"
        },
        {
            "id": "example_addition",
            "name": "Add Few-Shot Examples",
            "description": "Add few-shot examples to help the model learn patterns"
        }
    ]
    
    return strategies