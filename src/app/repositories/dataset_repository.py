
"""
Repository for dataset-related database operations
"""
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from src.app.models.database_models import Dataset
from datetime import datetime

class DatasetRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, name: str, description: Optional[str] = None, file_path: Optional[str] = None,
               example_count: int = 0, metadata: Optional[Dict[str, Any]] = None,
               user_id: Optional[str] = None) -> Dataset:
        """Create a new dataset record"""
        dataset = Dataset(
            name=name,
            description=description,
            file_path=file_path,
            example_count=example_count,
            metadata=metadata or {},
            user_id=user_id
        )
        
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset
    
    def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        """Get dataset by ID"""
        return self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    def list_all(self, limit: int = 100, skip: int = 0, user_id: Optional[str] = None) -> List[Dataset]:
        """List all datasets with pagination and optional user filter"""
        query = self.db.query(Dataset)
        if user_id:
            query = query.filter(Dataset.user_id == user_id)
        return query.order_by(Dataset.created_at.desc()).offset(skip).limit(limit).all()
    
    def update(self, dataset_id: str, name: Optional[str] = None, description: Optional[str] = None,
               file_path: Optional[str] = None, example_count: Optional[int] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Optional[Dataset]:
        """Update an existing dataset"""
        dataset = self.get_by_id(dataset_id)
        if not dataset:
            return None
            
        if name is not None:
            dataset.name = name
        if description is not None:
            dataset.description = description
        if file_path is not None:
            dataset.file_path = file_path
        if example_count is not None:
            dataset.example_count = example_count
        if metadata is not None:
            dataset.metadata = metadata
            
        self.db.commit()
        self.db.refresh(dataset)
        return dataset
    
    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset"""
        dataset = self.get_by_id(dataset_id)
        if not dataset:
            return False
            
        self.db.delete(dataset)
        self.db.commit()
        return True
