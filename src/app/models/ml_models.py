
from sqlalchemy import Column, String, Integer, Float, Boolean, JSON, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.app.models.base import Base
import uuid

class ModelConfiguration(Base):
    __tablename__ = "model_configurations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    primary_model = Column(String(50), nullable=False)  # e.g., "gemini-1.5-flash"
    optimizer_model = Column(String(50), nullable=False)  # e.g., "gemini-1.5-pro"
    temperature = Column(Float, default=0.0)
    max_tokens = Column(Integer, default=1024)
    top_p = Column(Float, default=1.0)
    top_k = Column(Integer, default=40)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    # Relationship with User model
    # user = relationship("User", back_populates="model_configs")

class MetricConfiguration(Base):
    __tablename__ = "metric_configurations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    metrics = Column(JSON, nullable=False)  # e.g., ["exact_match", "bleu"]
    metric_weights = Column(JSON, default=dict)  # e.g., {"exact_match": 0.7, "bleu": 0.3}
    target_threshold = Column(Float, default=0.8)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    # Relationship with User model
    # user = relationship("User", back_populates="metric_configs")

class MetaLearningConfiguration(Base):
    __tablename__ = "meta_learning_configurations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), default="xgboost")  # "xgboost", "random_forest", etc.
    hyperparameters = Column(JSON, default=dict)
    feature_selection = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    last_trained = Column(DateTime, nullable=True)
    performance = Column(Float, nullable=True)
    model_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    # Relationship with User model
    # user = relationship("User", back_populates="meta_learning_configs")
