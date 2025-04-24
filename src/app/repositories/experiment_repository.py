
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from src.app.models.database_models import Experiment, MetricsRecord
import uuid

class ExperimentRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, name: str, initial_prompt_id: str, dataset_id: str,
               metrics: Optional[List[str]] = None, max_epochs: int = 10,
               target_threshold: float = 0.8) -> Experiment:
        """Create a new experiment record"""
        experiment = Experiment(
            name=name,
            initial_prompt_id=uuid.UUID(initial_prompt_id),
            dataset_id=uuid.UUID(dataset_id),
            metrics=metrics or [],
            max_epochs=max_epochs,
            target_threshold=target_threshold,
            status="created"
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def get_by_id(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.db.query(Experiment).filter(Experiment.id == uuid.UUID(experiment_id)).first()
    
    def update_status(self, experiment_id: str, status: str) -> Optional[Experiment]:
        """Update experiment status"""
        experiment = self.get_by_id(experiment_id)
        if experiment:
            experiment.status = status
            self.db.commit()
            self.db.refresh(experiment)
            return experiment
        return None
    
    def update_best_prompt(self, experiment_id: str, prompt_id: str) -> Optional[Experiment]:
        """Update the best prompt for an experiment"""
        experiment = self.get_by_id(experiment_id)
        if experiment:
            experiment.best_prompt_id = uuid.UUID(prompt_id)
            self.db.commit()
            self.db.refresh(experiment)
            return experiment
        return None
    
    def list_experiments(self, limit: int = 100, offset: int = 0) -> List[Experiment]:
        """List all experiments with pagination"""
        return self.db.query(Experiment).order_by(Experiment.created_at.desc()).limit(limit).offset(offset).all()
    
    def add_metrics_record(self, experiment_id: str, epoch: int, metrics: Dict[str, Any], 
                         prompt_id: Optional[str] = None) -> MetricsRecord:
        """Add a metrics record for an experiment epoch"""
        metrics_record = MetricsRecord(
            experiment_id=uuid.UUID(experiment_id),
            epoch=epoch,
            metrics=metrics,
            prompt_id=uuid.UUID(prompt_id) if prompt_id else None
        )
        
        self.db.add(metrics_record)
        self.db.commit()
        self.db.refresh(metrics_record)
        return metrics_record
    
    def get_metrics_history(self, experiment_id: str) -> List[MetricsRecord]:
        """Get metrics history for an experiment"""
        return (self.db.query(MetricsRecord)
                .filter(MetricsRecord.experiment_id == uuid.UUID(experiment_id))
                .order_by(MetricsRecord.epoch)
                .all())
