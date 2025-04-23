
"""
Repository for metrics-related database operations
"""
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from src.app.models.database_models import MetricsRecord, Experiment
from datetime import datetime

class MetricsRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, experiment_id: str, iteration: int, metrics: Dict[str, Any], 
               prompt_id: Optional[str] = None) -> MetricsRecord:
        """Create a new metrics record"""
        metrics_record = MetricsRecord(
            experiment_id=experiment_id,
            iteration=iteration,
            metrics=metrics,
            prompt_id=prompt_id
        )
        
        self.db.add(metrics_record)
        self.db.commit()
        self.db.refresh(metrics_record)
        return metrics_record
    
    def get_by_id(self, metrics_id: str) -> Optional[MetricsRecord]:
        """Get metrics record by ID"""
        return self.db.query(MetricsRecord).filter(MetricsRecord.id == metrics_id).first()
    
    def get_by_experiment(self, experiment_id: str) -> List[MetricsRecord]:
        """Get all metrics records for an experiment"""
        return self.db.query(MetricsRecord).filter(MetricsRecord.experiment_id == experiment_id).order_by(MetricsRecord.iteration).all()
    
    def get_by_experiment_iteration(self, experiment_id: str, iteration: int) -> Optional[MetricsRecord]:
        """Get metrics record for a specific experiment iteration"""
        return self.db.query(MetricsRecord).filter(
            MetricsRecord.experiment_id == experiment_id,
            MetricsRecord.iteration == iteration
        ).first()
    
    def delete(self, metrics_id: str) -> bool:
        """Delete a metrics record"""
        metrics = self.get_by_id(metrics_id)
        if not metrics:
            return False
            
        self.db.delete(metrics)
        self.db.commit()
        return True
