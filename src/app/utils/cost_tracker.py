
"""
Cost tracking utility for monitoring and reporting API usage costs
"""
import json
import logging
import time
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CostTracker:
    """
    Utility for tracking and reporting API costs across different components
    """
    
    def __init__(self, save_dir: str = "cost_reports"):
        """
        Initialize the cost tracker
        
        Args:
            save_dir: Directory to save cost reports
        """
        self.save_dir = save_dir
        self.tracking_data = {
            "session_start": time.time(),
            "components": {},
            "total_estimated_cost": 0.0,
            "last_updated": time.time()
        }
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def update_component_costs(self, 
                              component_name: str, 
                              data: Dict[str, Any],
                              should_save: bool = False):
        """
        Update cost data for a specific component
        
        Args:
            component_name: Name of the component (e.g., "vertex_ai", "hf_evaluator")
            data: Cost data from the component
            should_save: Whether to save the report after updating
        """
        self.tracking_data["components"][component_name] = data
        
        # Recalculate total cost
        total_cost = 0.0
        for component, component_data in self.tracking_data["components"].items():
            if isinstance(component_data, dict) and "total_estimated_cost_usd" in component_data:
                total_cost += component_data["total_estimated_cost_usd"]
        
        self.tracking_data["total_estimated_cost"] = round(total_cost, 4)
        self.tracking_data["last_updated"] = time.time()
        
        if should_save:
            self.save_report()
    
    def get_cost_report(self) -> Dict[str, Any]:
        """
        Get the current cost report
        
        Returns:
            Dictionary with cost data for all components
        """
        # Calculate session duration
        session_duration = time.time() - self.tracking_data["session_start"]
        
        return {
            "session_start": datetime.fromtimestamp(self.tracking_data["session_start"]).isoformat(),
            "session_duration_minutes": round(session_duration / 60, 2),
            "total_estimated_cost_usd": self.tracking_data["total_estimated_cost"],
            "components": self.tracking_data["components"],
            "last_updated": datetime.fromtimestamp(self.tracking_data["last_updated"]).isoformat()
        }
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """
        Save the current cost report to a file
        
        Args:
            filename: Optional custom filename, defaults to timestamp-based name
            
        Returns:
            Path to the saved report file
        """
        if filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cost_report_{current_time}.json"
        
        file_path = os.path.join(self.save_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(self.get_cost_report(), f, indent=2)
        
        logger.info(f"Saved cost report to {file_path}")
        return file_path
    
    def reset_tracking(self):
        """Reset all tracking data for a new session"""
        self.tracking_data = {
            "session_start": time.time(),
            "components": {},
            "total_estimated_cost": 0.0,
            "last_updated": time.time()
        }
