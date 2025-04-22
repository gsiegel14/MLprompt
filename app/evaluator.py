def calculate_score(model_response, ground_truth_output):
    """
    Calculate a simple evaluation score between model response and ground truth.
    
    Args:
        model_response (str): The response from the LLM
        ground_truth_output (str): The expected output
        
    Returns:
        float: Score between 0 and 1
    """
    # Simple exact match scoring
    if not model_response or not ground_truth_output:
        return 0
    
    # Clean up and normalize text for comparison
    model_text = model_response.strip()
    ground_truth = ground_truth_output.strip()
    
    # Exact match
    if model_text == ground_truth:
        return 1.0
    
    # Substring match (partial credit)
    if model_text in ground_truth or ground_truth in model_text:
        # Calculate length of longer string and give proportional score
        longer_len = max(len(model_text), len(ground_truth))
        shorter_len = min(len(model_text), len(ground_truth))
        if longer_len > 0:
            return 0.5 * (shorter_len / longer_len)
    
    # No match
    return 0.0
