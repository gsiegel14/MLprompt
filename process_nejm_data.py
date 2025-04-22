#!/usr/bin/env python3
"""
Process NEJM case data into training and validation sets for the prompt engineering platform.
This script takes NEJM case studies, where:
- Second column (Case) = User input
- Third column (Final Diagnosis) = Ground truth output
"""

import os
import json
import pandas as pd
import random
from app.data_module import DataModule

# Create data module for handling the datasets
data_module = DataModule()

# Load and process the CSV file
def process_nejm_data(csv_path, train_ratio=0.5):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} cases from {csv_path}")
    
    # Convert to the format expected by the application
    examples = []
    for _, row in df.iterrows():
        try:
            case_number = row['Case Number']
            case_text = row['Case']
            diagnosis = row['Final Diagnosis']
            
            if pd.notna(case_text) and pd.notna(diagnosis):
                examples.append({
                    'user_input': case_text,
                    'ground_truth_output': diagnosis
                })
        except Exception as e:
            print(f"Error processing row: {e}")
    
    print(f"Processed {len(examples)} valid examples")
    
    # Shuffle the examples for random split
    random.shuffle(examples)
    
    # Split into train and validation sets
    split_index = int(len(examples) * train_ratio)
    train_examples = examples[:split_index]
    validation_examples = examples[split_index:]
    
    print(f"Split into {len(train_examples)} training and {len(validation_examples)} validation examples")
    
    # Save to the application's data directory
    data_module._save_examples(train_examples, 'data/train/current_train.json')
    data_module._save_examples(validation_examples, 'data/validation/current_validation.json')
    
    return train_examples, validation_examples

# Add a similarity check function for evaluation
def add_similarity_check():
    """Add or update the similarity check function in the evaluator.py file"""
    evaluator_path = 'app/evaluator.py'
    
    with open(evaluator_path, 'r') as f:
        content = f.read()
    
    # Check if function already exists
    if 'def calculate_score(model_response: str, ground_truth_output: str)' in content:
        # Update the existing function
        import re
        pattern = r'def calculate_score\(model_response: str, ground_truth_output: str\).*?return score'
        replacement = '''def calculate_score(model_response: str, ground_truth_output: str) -> float:
    """
    Calculate a similarity score between model response and ground truth.
    
    Args:
        model_response (str): The response from the LLM
        ground_truth_output (str): The expected output
        
    Returns:
        float: Score between 0 and 1
    """
    from difflib import SequenceMatcher
    
    # Clean and lowercase texts for comparison
    model_text = model_response.lower().strip()
    truth_text = ground_truth_output.lower().strip()
    
    # Check for exact match
    if model_text == truth_text:
        return 1.0
        
    # Check if ground truth is contained in the response
    if truth_text in model_text:
        return 0.9
    
    # Check if key elements of ground truth are in response
    # Split by common separators and check for presence of key terms
    truth_keywords = set([term.strip() for term in re.split(r'[,;.]', truth_text) if term.strip()])
    matches = sum(1 for keyword in truth_keywords if keyword in model_text)
    keyword_score = matches / len(truth_keywords) if truth_keywords else 0
    
    # Use sequence matcher for overall similarity
    similarity = SequenceMatcher(None, model_text, truth_text).ratio()
    
    # Combine scores with weights
    score = 0.6 * similarity + 0.4 * keyword_score
    
    return min(score, 0.95)  # Cap at 0.95 to reserve 1.0 for exact matches'''
        
        updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        with open(evaluator_path, 'w') as f:
            f.write(updated_content)
            
        print("Updated similarity check function in evaluator.py")
    else:
        print("Could not find calculate_score function to update")

if __name__ == "__main__":
    # Process the NEJM case data with a 50/50 train/validation split
    train_examples, validation_examples = process_nejm_data('data/nejm/cases.csv', train_ratio=0.5)
    
    # Add or update the similarity check function
    add_similarity_check()
    
    print("Data processing complete. The application has been updated with the NEJM case data.")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(validation_examples)}")