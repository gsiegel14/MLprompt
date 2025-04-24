
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from src.app.models.database_models import MetricsRecord
import uuid

class MetricsRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, metrics_id: str) -> Optional[MetricsRecord]:
        """Get metrics record by ID"""
        return self.db.query(MetricsRecord).filter(MetricsRecord.id == uuid.UUID(metrics_id)).first()
    
    def get_by_experiment_and_epoch(self, experiment_id: str, epoch: int) -> Optional[MetricsRecord]:
        """Get metrics record by experiment ID and epoch"""
        return (self.db.query(MetricsRecord)
                .filter(MetricsRecord.experiment_id == uuid.UUID(experiment_id), 
                       MetricsRecord.epoch == epoch)
                .first())
    
    def get_latest_for_experiment(self, experiment_id: str) -> Optional[MetricsRecord]:
        """Get the latest metrics record for an experiment"""
        return (self.db.query(MetricsRecord)
                .filter(MetricsRecord.experiment_id == uuid.UUID(experiment_id))
                .order_by(MetricsRecord.epoch.desc())
                .first())
    
    def list_by_experiment(self, experiment_id: str) -> List[MetricsRecord]:
        """List all metrics records for an experiment"""
        return (self.db.query(MetricsRecord)
                .filter(MetricsRecord.experiment_id == uuid.UUID(experiment_id))
                .order_by(MetricsRecord.epoch)
                .all())
