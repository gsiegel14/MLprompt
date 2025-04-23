
"""
Database models for ML settings
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, JSON, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()

# Association table for many-to-many relationships
model_metric_association = Table(
    'model_metric_association',
    Base.metadata,
    Column('model_id', String, ForeignKey('model_configurations.id')),
    Column('metric_id', String, ForeignKey('metric_configurations.id'))
)

class ModelConfiguration(Base):
    __tablename__ = "model_configurations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    primary_model = Column(String, nullable=False)  # e.g., "gemini-1.5-flash"
    optimizer_model = Column(String, nullable=False)  # e.g., "gemini-1.5-pro"
    temperature = Column(Float, default=0.0)
    max_tokens = Column(Integer, default=1024)
    top_p = Column(Float, default=1.0)
    top_k = Column(Integer, default=40)
    is_default = Column(Boolean, default=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    metrics = relationship("MetricConfiguration", 
                          secondary=model_metric_association,
                          back_populates="models")
    experiments = relationship("Experiment", back_populates="model_config")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "primary_model": self.primary_model,
            "optimizer_model": self.optimizer_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "is_default": self.is_default
        }

class MetricConfiguration(Base):
    __tablename__ = "metric_configurations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    metrics = Column(JSON, nullable=False)  # e.g., ["exact_match", "bleu"]
    metric_weights = Column(JSON, default=dict)  # e.g., {"exact_match": 0.7, "bleu": 0.3}
    target_threshold = Column(Float, default=0.8)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    models = relationship("ModelConfiguration", 
                        secondary=model_metric_association,
                        back_populates="metrics")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "metrics": self.metrics,
            "metric_weights": self.metric_weights,
            "target_threshold": self.target_threshold
        }

class MetaLearningConfiguration(Base):
    __tablename__ = "meta_learning_configurations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    model_type = Column(String, default="xgboost")  # "xgboost", "random_forest", etc.
    hyperparameters = Column(JSON, default=dict)
    feature_selection = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "feature_selection": self.feature_selection,
            "is_active": self.is_active
        }

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    model_config_id = Column(String, ForeignKey("model_configurations.id"), nullable=True)
    start_time = Column(String, nullable=True)
    end_time = Column(String, nullable=True)
    metrics = Column(JSON, default=dict)
    iterations = Column(Integer, default=0)
    best_iteration = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending, running, completed, failed
    results_path = Column(String, nullable=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    model_config = relationship("ModelConfiguration", back_populates="experiments")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "model_config_id": self.model_config_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.metrics,
            "iterations": self.iterations,
            "best_iteration": self.best_iteration,
            "status": self.status,
            "results_path": self.results_path
        }

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    api_key = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    model_configs = relationship("ModelConfiguration", backref="user")
    metric_configs = relationship("MetricConfiguration", backref="user")
    meta_learning_configs = relationship("MetaLearningConfiguration", backref="user")
    experiments = relationship("Experiment", backref="user")
