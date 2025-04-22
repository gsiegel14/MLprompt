import os
import csv
import json
import random
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

from app.utils import parse_text_examples, parse_csv_file

logger = logging.getLogger(__name__)

class DataModule:
    """
    Handles dataset management for prompt optimization, including loading,
    train/validation splitting, and batch creation.
    """
    
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, 'train')
        self.validation_dir = os.path.join(base_dir, 'validation')
        
        # Create directories if they don't exist
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)
        
        self.train_examples = []
        self.validation_examples = []
    
    def load_examples_from_text(self, text_content: str, train_ratio: float = 0.8) -> Tuple[List, List]:
        """
        Load examples from text content and split into train/validation sets.
        
        Args:
            text_content (str): Text with examples in CSV format
            train_ratio (float): Ratio of examples to use for training
            
        Returns:
            tuple: (train_examples, validation_examples)
        """
        examples = parse_text_examples(text_content)
        return self.split_examples(examples, train_ratio)
    
    def load_examples_from_csv(self, file_path: str, train_ratio: float = 0.8) -> Tuple[List, List]:
        """
        Load examples from a CSV file and split into train/validation sets.
        
        Args:
            file_path (str): Path to the CSV file
            train_ratio (float): Ratio of examples to use for training
            
        Returns:
            tuple: (train_examples, validation_examples)
        """
        with open(file_path, 'rb') as f:
            examples = parse_csv_file(f)
        
        return self.split_examples(examples, train_ratio)
    
    def split_examples(self, examples: List[Dict[str, str]], train_ratio: float = 0.8) -> Tuple[List, List]:
        """
        Split examples into training and validation sets.
        
        Args:
            examples (list): List of example dictionaries
            train_ratio (float): Ratio of examples to use for training
            
        Returns:
            tuple: (train_examples, validation_examples)
        """
        if not examples:
            return [], []
        
        # Shuffle examples to ensure random split
        random.shuffle(examples)
        
        # Calculate the split index
        split_idx = int(len(examples) * train_ratio)
        
        # Split the examples
        self.train_examples = examples[:split_idx]
        self.validation_examples = examples[split_idx:]
        
        # Save the split examples
        self._save_examples(self.train_examples, os.path.join(self.train_dir, 'current_train.json'))
        self._save_examples(self.validation_examples, os.path.join(self.validation_dir, 'current_validation.json'))
        
        logger.info(f"Split {len(examples)} examples into {len(self.train_examples)} train and {len(self.validation_examples)} validation examples")
        
        return self.train_examples, self.validation_examples
    
    def _save_examples(self, examples: List[Dict[str, str]], file_path: str) -> None:
        """Save examples to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(examples, f, indent=2)
    
    def get_train_examples(self) -> List[Dict[str, str]]:
        """Get the current training examples."""
        return self.train_examples
    
    def get_validation_examples(self) -> List[Dict[str, str]]:
        """Get the current validation examples."""
        return self.validation_examples
    
    def get_batch(self, batch_size: int = 0, validation: bool = False) -> List[Dict[str, str]]:
        """
        Get a batch of examples.
        
        Args:
            batch_size (int): Number of examples in batch (0 for all)
            validation (bool): Whether to use validation set
            
        Returns:
            list: List of example dictionaries
        """
        examples = self.validation_examples if validation else self.train_examples
        
        if not examples:
            return []
        
        if batch_size <= 0 or batch_size >= len(examples):
            return examples
        
        # Select random subset if batch_size is specified
        return random.sample(examples, batch_size)
    
    def save_dataset(self, examples: List[Dict[str, str]], name: str) -> str:
        """
        Save a dataset with a custom name.
        
        Args:
            examples (list): List of example dictionaries
            name (str): Dataset name
            
        Returns:
            str: Path to saved file
        """
        file_path = os.path.join(self.base_dir, f"{name}.json")
        self._save_examples(examples, file_path)
        return file_path
    
    def load_dataset(self, name: str) -> List[Dict[str, str]]:
        """
        Load a dataset by name.
        
        Args:
            name (str): Dataset name
            
        Returns:
            list: List of example dictionaries
        """
        file_path = os.path.join(self.base_dir, f"{name}.json")
        
        if not os.path.exists(file_path):
            logger.warning(f"Dataset {name} not found at {file_path}")
            return []
        
        try:
            with open(file_path, 'r') as f:
                examples = json.load(f)
            return examples
        except Exception as e:
            logger.error(f"Error loading dataset {name}: {e}")
            return []
    
    def export_to_csv(self, examples: List[Dict[str, str]], file_path: str) -> bool:
        """
        Export examples to a CSV file.
        
        Args:
            examples (list): List of example dictionaries
            file_path (str): Path to save the CSV file
            
        Returns:
            bool: Success status
        """
        try:
            df = pd.DataFrame(examples)
            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False