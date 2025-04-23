
"""
API endpoints for cost tracking and monitoring
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional

from src.app.utils.cost_tracker import CostTracker
from src.app.config import settings

router = APIRouter(
    prefix="/costs",
    tags=["costs"],
    responses={404: {"description": "Not found"}},
)

# Create a singleton cost tracker
cost_tracker = CostTracker(save_dir=settings.COST_REPORTS_DIR)

@router.get("/summary")
async def get_cost_summary() -> Dict[str, Any]:
    """Get a summary of current cost tracking"""
    return cost_tracker.get_cost_report()

@router.post("/reset")
async def reset_cost_tracking() -> Dict[str, str]:
    """Reset all cost tracking data"""
    cost_tracker.reset_tracking()
    return {"status": "Cost tracking data reset successfully"}

@router.post("/save")
async def save_cost_report(filename: Optional[str] = None) -> Dict[str, str]:
    """
    Save the current cost report to a file
    
    Args:
        filename: Optional filename for the report
    """
    filepath = cost_tracker.save_report(filename)
    return {"status": "Cost report saved successfully", "filepath": filepath}
