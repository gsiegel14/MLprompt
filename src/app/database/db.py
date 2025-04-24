
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

# Environment-based configuration
# Get Replit Database URL or fallback to local PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/promptopt"
)

# Check if we're on Replit and using the Replit Database
is_replit = "REPL_ID" in os.environ and "REPLIT_DB_URL" in os.environ

# Create PostgreSQL engine with specific configuration
engine = create_engine(
    DATABASE_URL, 
    echo=os.getenv("ENV", "development") == "development",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Reconnect after 30 minutes
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
