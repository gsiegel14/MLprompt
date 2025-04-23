
import time
import logging
from functools import wraps
from flask import request, jsonify, g
from collections import defaultdict
from typing import Dict, List, Callable, Any, Optional
import threading

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter for API endpoints.
    Limits requests based on IP address and optionally API keys.
    """
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests  # Maximum requests per window
        self.window_seconds = window_seconds  # Time window in seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Start background cleanup
        self._start_cleanup()
    
    def _get_key(self, request_obj) -> str:
        """
        Get unique key for the request (IP address or API key)
        """
        # Use API key if available
        api_key = request_obj.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key}"
        
        # Fall back to IP address
        return f"ip:{request_obj.remote_addr}"
    
    def check_rate_limit(self, request_obj) -> bool:
        """
        Check if request is within rate limits
        """
        key = self._get_key(request_obj)
        current_time = time.time()
        
        with self.lock:
            # Remove timestamps outside the current window
            self.requests[key] = [ts for ts in self.requests[key] 
                                if current_time - ts < self.window_seconds]
            
            # Check if we're at the limit
            if len(self.requests[key]) >= self.max_requests:
                logger.warning(f"Rate limit exceeded for {key}")
                return False
            
            # Add current request timestamp
            self.requests[key].append(current_time)
            return True
    
    def _cleanup_old_data(self):
        """
        Remove data for expired windows
        """
        current_time = time.time()
        with self.lock:
            expired_keys = []
            for key, timestamps in self.requests.items():
                # Keep only timestamps within the window
                valid_timestamps = [ts for ts in timestamps 
                                   if current_time - ts < self.window_seconds]
                if valid_timestamps:
                    self.requests[key] = valid_timestamps
                else:
                    expired_keys.append(key)
            
            # Remove empty entries
            for key in expired_keys:
                del self.requests[key]
    
    def _start_cleanup(self):
        """
        Start background thread for periodic cleanup
        """
        def cleanup_job():
            while True:
                time.sleep(self.window_seconds)
                self._cleanup_old_data()
        
        cleanup_thread = threading.Thread(target=cleanup_job, daemon=True)
        cleanup_thread.start()

# Create global rate limiter instances for different endpoint types
standard_limiter = RateLimiter(max_requests=60, window_seconds=60)  # 60 requests per minute
model_call_limiter = RateLimiter(max_requests=20, window_seconds=60)  # 20 model calls per minute
admin_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 admin requests per minute

def rate_limit(limiter_type: str = "standard"):
    """
    Rate limiting decorator for Flask routes
    
    Args:
        limiter_type: Type of limiter to use ("standard", "model_call", "admin")
    """
    limiter = {
        "standard": standard_limiter,
        "model_call": model_call_limiter,
        "admin": admin_limiter
    }.get(limiter_type, standard_limiter)
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not limiter.check_rate_limit(request):
                response = {
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later."
                }
                return jsonify(response), 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator
