
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional, Dict, Any
from src.app.models.database_models import Prompt
from src.app.models.prompt_state import PromptState
import uuid

class PromptRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, system_prompt: str, output_prompt: str, parent_id: Optional[str] = None, 
               metadata: Optional[Dict[str, Any]] = None) -> Prompt:
        """Create a new prompt record"""
        # Get version if parent exists
        version = 1
        if parent_id:
            parent = self.db.query(Prompt).filter(Prompt.id == uuid.UUID(parent_id)).first()
            if parent:
                version = parent.version + 1
        
        # Create new prompt
        prompt = Prompt(
            system_prompt=system_prompt,
            output_prompt=output_prompt,
            parent_id=uuid.UUID(parent_id) if parent_id else None,
            version=version,
            metadata=metadata or {}
        )
        
        self.db.add(prompt)
        self.db.commit()
        self.db.refresh(prompt)
        return prompt
    
    def get_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Get prompt by ID"""
        return self.db.query(Prompt).filter(Prompt.id == uuid.UUID(prompt_id)).first()
    
    def get_latest_version(self, parent_id: str = None) -> Optional[Prompt]:
        """Get the latest version of a prompt family"""
        query = self.db.query(Prompt)
        if parent_id:
            query = query.filter(Prompt.parent_id == uuid.UUID(parent_id))
        return query.order_by(desc(Prompt.version)).first()
    
    def list_versions(self, parent_id: str) -> List[Prompt]:
        """List all versions of a prompt family"""
        return self.db.query(Prompt).filter(
            (Prompt.id == uuid.UUID(parent_id)) | (Prompt.parent_id == uuid.UUID(parent_id))
        ).order_by(Prompt.version).all()
    
    def to_prompt_state(self, prompt: Prompt) -> PromptState:
        """Convert database model to domain model"""
        return PromptState(
            system_prompt=prompt.system_prompt,
            output_prompt=prompt.output_prompt,
            id=str(prompt.id),
            parent_id=str(prompt.parent_id) if prompt.parent_id else None,
            version=prompt.version,
            metadata=prompt.metadata,
            created_at=prompt.created_at
        )
