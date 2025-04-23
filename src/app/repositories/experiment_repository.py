
"""
Repository for experiment-related database operations
"""
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from src.app.models.database_models import Experiment
from datetime import datetime

class ExperimentRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, name: str, model_config_id: str, metric_config_id: str, dataset_id: str,
               initial_prompt_id: str, strategy: str, description: Optional[str] = None,
               max_iterations: int = 5, user_id: Optional[str] = None) -> Experiment:
        """Create a new experiment record"""
        experiment = Experiment(
            name=name,
            description=description,
            model_config_id=model_config_id,
            metric_config_id=metric_config_id,
            dataset_id=dataset_id,
            initial_prompt_id=initial_prompt_id,
            strategy=strategy,
            max_iterations=max_iterations,
            user_id=user_id
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def get_by_id(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    def list_all(self, limit: int = 100, skip: int = 0, user_id: Optional[str] = None) -> List[Experiment]:
        """List all experiments with pagination and optional user filter"""
        query = self.db.query(Experiment)
        if user_id:
            query = query.filter(Experiment.user_id == user_id)
        return query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()
    
    def update_status(self, experiment_id: str, status: str) -> Optional[Experiment]:
        """Update experiment status"""
        experiment = self.get_by_id(experiment_id)
        if not experiment:
            return None
            
        experiment.status = status
        
        if status == "running" and not experiment.started_at:
            experiment.started_at = datetime.utcnow()
            
        if status in ["completed", "failed"]:
            experiment.completed_at = datetime.utcnow()
            
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def update_best_prompt(self, experiment_id: str, prompt_id: str, score: float) -> Optional[Experiment]:
        """Update experiment best prompt and score"""
        experiment = self.get_by_id(experiment_id)
        if not experiment:
            return None
            
        experiment.best_prompt_id = prompt_id
        experiment.current_best_score = score
        
        # Calculate improvement if we have a previous score to compare against
        if experiment.current_best_score > 0:
            initial_score = experiment.metrics_records[0].metrics.get("average_score", 0) if experiment.metrics_records else 0
            if initial_score > 0:
                experiment.improvement_percentage = ((score - initial_score) / initial_score) * 100
            
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def update_iteration(self, experiment_id: str, iteration: int) -> Optional[Experiment]:
        """Update experiment iteration count"""
        experiment = self.get_by_id(experiment_id)
        if not experiment:
            return None
            
        experiment.iterations_completed = iteration
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment"""
        experiment = self.get_by_id(experiment_id)
        if not experiment:
            return False
            
        self.db.delete(experiment)
        self.db.commit()
        return True
