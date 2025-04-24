
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

# Environment-based configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///data/db/promptopt.db"
)

# Create engine with database-specific configuration
if DATABASE_URL.startswith('postgresql'):
    # PostgreSQL-specific configuration
    engine = create_engine(
        DATABASE_URL, 
        echo=os.getenv("ENV", "development") == "development",
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # Reconnect after 30 minutes
    )
else:
    # SQLite or other database
    engine = create_engine(
        DATABASE_URL,
        echo=os.getenv("ENV", "development") == "development",
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {}
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
