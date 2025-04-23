
"""
PromptState class for managing prompt versions
"""
from pydantic import BaseModel, Field
import json
import os
from typing import Optional, Dict, Any

class PromptState(BaseModel):
    """Class to manage prompt versions and their state"""
    
    system_prompt: str
    output_prompt: str
    version: int = 1
    id: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "system_prompt": self.system_prompt,
            "output_prompt": self.output_prompt,
            "version": self.version
        }
    
    @classmethod
    def load(cls, path: str):
        """
        Load prompt state from a file or GCS
        
        Args:
            path: Local file path or GCS path (gs://)
        
        Returns:
            PromptState object
        """
        if path.startswith("gs://"):
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                with fs.open(path, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                raise ValueError(f"Failed to load from GCS: {str(e)}")
        else:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                raise ValueError(f"Failed to load from local file: {str(e)}")
    
    def save(self, path: str):
        """
        Save prompt state to a file or GCS
        
        Args:
            path: Local file path or GCS path (gs://)
        """
        data = self.dict()
        
        if path.startswith("gs://"):
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                with fs.open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                raise ValueError(f"Failed to save to GCS: {str(e)}")
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            try:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                raise ValueError(f"Failed to save to local file: {str(e)}")
    
    def increment_version(self):
        """Create a new version by incrementing the version number"""
        return PromptState(
            system_prompt=self.system_prompt,
            output_prompt=self.output_prompt,
            version=self.version + 1
        )
