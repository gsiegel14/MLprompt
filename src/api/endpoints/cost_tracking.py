
"""
Cost Tracking API endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime

from src.app.utils.cost_tracker import CostTracker
from src.app.config import settings

router = APIRouter()
cost_tracker = CostTracker(save_dir=settings.COST_REPORTS_DIR)

@router.get("/", summary="Get current cost tracking report")
async def get_cost_report() -> Dict[str, Any]:
    """
    Get the current cost tracking report for the session.
    
    Returns:
        A dictionary with cost data for all components
    """
    try:
        return cost_tracker.get_cost_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cost report: {str(e)}")

@router.post("/reset", summary="Reset cost tracking")
async def reset_cost_tracking() -> Dict[str, str]:
    """
    Reset all cost tracking data for a new session.
    
    Returns:
        A confirmation message
    """
    try:
        cost_tracker.reset_tracking()
        return {"message": "Cost tracking data reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting cost tracking: {str(e)}")

@router.post("/save", summary="Save current cost report")
async def save_cost_report(filename: Optional[str] = None) -> Dict[str, str]:
    """
    Save the current cost report to a file.
    
    Args:
        filename: Optional custom filename, defaults to timestamp-based name
        
    Returns:
        Path to the saved report file
    """
    try:
        report_path = cost_tracker.save_report(filename)
        return {"file_path": report_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving cost report: {str(e)}")

@router.get("/reports", summary="List all saved cost reports")
async def list_cost_reports() -> List[Dict[str, Any]]:
    """
    List all saved cost reports.
    
    Returns:
        List of report metadata
    """
    try:
        reports = []
        if os.path.exists(settings.COST_REPORTS_DIR):
            for filename in os.listdir(settings.COST_REPORTS_DIR):
                if filename.endswith('.json'):
                    file_path = os.path.join(settings.COST_REPORTS_DIR, filename)
                    file_stats = os.stat(file_path)
                    
                    # Try to read the total cost from the file
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            total_cost = data.get('total_estimated_cost_usd', 'N/A')
                    except:
                        total_cost = 'N/A'
                    
                    reports.append({
                        "filename": filename,
                        "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                        "size_bytes": file_stats.st_size,
                        "total_cost": total_cost
                    })
        
        # Sort by creation time, newest first
        reports.sort(key=lambda x: x["created_at"], reverse=True)
        return reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing cost reports: {str(e)}")

@router.get("/reports/{filename}", summary="Get a specific cost report")
async def get_cost_report_by_name(filename: str) -> Dict[str, Any]:
    """
    Get a specific cost report by filename.
    
    Args:
        filename: Name of the cost report file
        
    Returns:
        The cost report data
    """
    try:
        file_path = os.path.join(settings.COST_REPORTS_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Cost report '{filename}' not found")
            
        with open(file_path, 'r') as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading cost report: {str(e)}")

@router.post("/vertex_client", summary="Update cost data from Vertex client")
async def update_vertex_client_costs(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Update cost tracking data from the Vertex AI client.
    
    Args:
        data: Cost data from the Vertex AI client
        
    Returns:
        A confirmation message
    """
    try:
        cost_tracker.update_component_costs("vertex_ai", data, should_save=True)
        return {"message": "Vertex AI cost data updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating cost data: {str(e)}")
