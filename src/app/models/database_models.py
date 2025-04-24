
from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, Text, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from src.app.database.db import Base

class Prompt(Base):
    __tablename__ = "prompts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    system_prompt = Column(Text, nullable=False)
    output_prompt = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("prompts.id"), nullable=True)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    parent = relationship("Prompt", remote_side=[id], backref="versions")
    experiments = relationship("Experiment", foreign_keys="Experiment.initial_prompt_id", back_populates="initial_prompt")
    
    __table_args__ = (
        Index('idx_prompt_parent_id', parent_id),
        Index('idx_prompt_created_at', created_at),
    )
    
class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    row_count = Column(Integer)
    columns = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    experiments = relationship("Experiment", back_populates="dataset")
    
    __table_args__ = (
        Index('idx_dataset_name', name),
    )

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    initial_prompt_id = Column(UUID(as_uuid=True), ForeignKey("prompts.id"), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    metrics = Column(JSONB, default={})
    max_epochs = Column(Integer, default=10)
    target_threshold = Column(Float, default=0.8)
    status = Column(String(50), default="created")
    best_prompt_id = Column(UUID(as_uuid=True), ForeignKey("prompts.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    initial_prompt = relationship("Prompt", foreign_keys=[initial_prompt_id])
    best_prompt = relationship("Prompt", foreign_keys=[best_prompt_id])
    dataset = relationship("Dataset", back_populates="experiments")
    metrics_history = relationship("MetricsRecord", back_populates="experiment")
    
    __table_args__ = (
        Index('idx_experiment_dataset_id', dataset_id),
        Index('idx_experiment_status', status),
    )

class MetricsRecord(Base):
    __tablename__ = "metrics_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    epoch = Column(Integer, nullable=False)
    metrics = Column(JSONB, nullable=False)
    prompt_id = Column(UUID(as_uuid=True), ForeignKey("prompts.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    experiment = relationship("Experiment", back_populates="metrics_history")
    prompt = relationship("Prompt")
    
    __table_args__ = (
        Index('idx_metrics_experiment_id', experiment_id),
        Index('idx_metrics_epoch', experiment_id, epoch),
    )

class ModelConfiguration(Base):
    __tablename__ = "model_configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    primary_model = Column(String(255), nullable=False)  # e.g., "gemini-1.5-flash"
    optimizer_model = Column(String(255), nullable=False)  # e.g., "gemini-1.5-pro"
    temperature = Column(Float, default=0.0)
    max_tokens = Column(Integer, default=1024)
    top_p = Column(Float, default=1.0)
    top_k = Column(Integer, default=40)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_model_configuration_name', name),
        Index('idx_model_configuration_is_default', is_default),
    )
