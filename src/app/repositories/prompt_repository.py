
"""
Repository for prompt-related database operations
"""
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from src.app.models.database_models import Prompt
from src.app.models.prompt_state import PromptState
from datetime import datetime

class PromptRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, system_prompt: str, output_prompt: str, parent_id: Optional[str] = None, 
               metadata: Optional[Dict[str, Any]] = None) -> Prompt:
        """Create a new prompt record"""
        # Get version if parent exists
        version = 1
        if parent_id:
            parent = self.db.query(Prompt).filter(Prompt.id == parent_id).first()
            if parent:
                version = parent.version + 1
        
        # Create new prompt
        prompt = Prompt(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            parent_id=parent_id,
            version=version,
            metadata=metadata or {}
        )
        
        self.db.add(prompt)
        self.db.commit()
        self.db.refresh(prompt)
        return prompt
    
    def get_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Get prompt by ID"""
        return self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
    
    def list_all(self, limit: int = 100, skip: int = 0) -> List[Prompt]:
        """List all prompts with pagination"""
        return self.db.query(Prompt).order_by(Prompt.created_at.desc()).offset(skip).limit(limit).all()
    
    def list_versions(self, parent_id: str) -> List[Prompt]:
        """List all versions of a prompt"""
        return self.db.query(Prompt).filter(Prompt.parent_id == parent_id).order_by(Prompt.version).all()
    
    def to_prompt_state(self, prompt: Prompt) -> PromptState:
        """Convert database model to domain model"""
        return PromptState(
            system_prompt=prompt.system_prompt,
            output_prompt=prompt.output_prompt,
            id=prompt.id,
            parent_id=prompt.parent_id,
            version=prompt.version,
            metadata=prompt.metadata,
            created_at=prompt.created_at
        )
    
    def update(self, prompt_id: str, system_prompt: Optional[str] = None, 
               output_prompt: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[Prompt]:
        """Update an existing prompt"""
        prompt = self.get_by_id(prompt_id)
        if not prompt:
            return None
            
        if system_prompt is not None:
            prompt.system_prompt = system_prompt
        if output_prompt is not None:
            prompt.output_prompt = output_prompt
        if metadata is not None:
            prompt.metadata = metadata
            
        self.db.commit()
        self.db.refresh(prompt)
        return prompt
    
    def delete(self, prompt_id: str) -> bool:
        """Delete a prompt"""
        prompt = self.get_by_id(prompt_id)
        if not prompt:
            return False
            
        self.db.delete(prompt)
        self.db.commit()
        return True
