from fastapi import APIRouter, Depends, HTTPException
from src.app.auth import get_api_key
import sqlite3

from src.api.endpoints import (
    prompts,
    optimization,
    experiments,
    datasets,
    inference,
    cost_tracking,
    ml_settings
)

# Create main API router
api_router = APIRouter()

# Include each module's router with appropriate prefix
api_router.include_router(
    prompts.router,
    prefix="/prompts",
    tags=["Prompts"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    optimization.router, 
    prefix="/optimize", 
    tags=["Optimization"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    experiments.router,
    prefix="/experiments",
    tags=["Experiments"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["Datasets"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    inference.router,
    prefix="/inference",
    tags=["Inference"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    cost_tracking.router,
    prefix="/costs",
    tags=["Costs"],
    dependencies=[Depends(get_api_key)]
)

api_router.include_router(
    ml_settings.router,
    prefix="/ml-settings",
    tags=["ML Settings"],
    dependencies=[Depends(get_api_key)]
)


# --- Database Implementation (Simplified Example) ---
class Database:
    def __init__(self, db_path="prompts.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_prompt(self, prompt):
        self.cursor.execute("INSERT INTO prompts (prompt) VALUES (?)", (prompt,))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_prompts(self):
        self.cursor.execute("SELECT * FROM prompts")
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

# Example usage (would be integrated into relevant API endpoints)
db = Database()
#db.add_prompt("This is a test prompt.")
#prompts = db.get_prompts()
#print(prompts)
#db.close()