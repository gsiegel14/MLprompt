
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from src.app.models.database_models import Dataset
import uuid
import json
import os

class DatasetRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, name: str, file_path: str, row_count: Optional[int] = None, 
               columns: Optional[Dict] = None) -> Dataset:
        """Create a new dataset record"""
        # Auto-calculate row count and columns if file exists
        if os.path.exists(file_path) and not row_count:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        row_count = len(data)
                        if data and isinstance(data[0], dict):
                            columns = {k: type(v).__name__ for k, v in data[0].items()}
            except Exception:
                # If we can't auto-analyze, continue without it
                pass
        
        dataset = Dataset(
            name=name,
            file_path=file_path,
            row_count=row_count,
            columns=columns or {}
        )
        
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset
    
    def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        """Get dataset by ID"""
        return self.db.query(Dataset).filter(Dataset.id == uuid.UUID(dataset_id)).first()
    
    def get_by_name(self, name: str) -> Optional[Dataset]:
        """Get dataset by name"""
        return self.db.query(Dataset).filter(Dataset.name == name).first()
    
    def list_datasets(self, limit: int = 100, offset: int = 0) -> List[Dataset]:
        """List all datasets with pagination"""
        return self.db.query(Dataset).order_by(Dataset.created_at.desc()).limit(limit).offset(offset).all()
    
    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset by ID"""
        dataset = self.get_by_id(dataset_id)
        if dataset:
            self.db.delete(dataset)
            self.db.commit()
            return True
        return False
