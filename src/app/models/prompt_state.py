
"""
Core data model for managing prompt versions
"""
from pydantic import BaseModel
import json
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PromptState(BaseModel):
    """
    Manages the state of system and output prompts with version tracking
    """
    system_prompt: str
    output_prompt: str
    version: int = 1
    metadata: Dict[str, Any] = {}
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "system_prompt": self.system_prompt,
            "output_prompt": self.output_prompt,
            "version": self.version,
            "metadata": self.metadata
        }
    
    @classmethod
    def load(cls, path: str) -> 'PromptState':
        """Load prompt state from file"""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded prompt state from {path}")
                return cls(**data)
            else:
                logger.warning(f"Prompt state file not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            logger.error(f"Error loading prompt state: {e}")
            raise
        
    def save(self, path: str) -> str:
        """Save prompt state to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.dict(), f, indent=2)
            logger.info(f"Saved prompt state to {path}")
            return path
        except Exception as e:
            logger.error(f"Error saving prompt state: {e}")
            raise
    
    def create_next_version(self, system_prompt: Optional[str] = None, output_prompt: Optional[str] = None) -> 'PromptState':
        """Create a new version with updated prompts"""
        return PromptState(
            system_prompt=system_prompt if system_prompt is not None else self.system_prompt,
            output_prompt=output_prompt if output_prompt is not None else self.output_prompt,
            version=self.version + 1,
            metadata=self.metadata.copy()
        )
