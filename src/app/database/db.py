
"""
Database connection module using SQLAlchemy
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Environment-based configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"sqlite:///{BASE_DIR}/data/prompt_optimizer.db"
)

# Create engine - echo=True only in development
engine = create_engine(
    DATABASE_URL, 
    echo=os.getenv("ENV", "development") == "development",
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for routes
def get_db():
    """Database session dependency for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
