
"""
Advanced logging configuration with structured logs and performance tracking.
"""

import os
import logging
import json
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import threading
from functools import wraps

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Initialize request context using thread-local storage
request_context = threading.local()

class StructuredLogFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    """
    def format(self, record):
        # Get standard record info
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add context from thread-local storage if available
        if hasattr(request_context, "data"):
            log_data.update(request_context.data)
        
        # Serialize to JSON
        return json.dumps(log_data)

def configure_logging(level=logging.INFO):
    """
    Configure logging with both console and file handlers
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Create file handler for JSON structured logs
    json_file_handler = logging.FileHandler("logs/app.json.log")
    json_file_handler.setLevel(level)
    json_file_handler.setFormatter(StructuredLogFormatter())
    
    # Create file handler for standard logs
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(json_file_handler)
    root_logger.addHandler(file_handler)
    
    # Configure flask logger separately
    flask_logger = logging.getLogger('flask')
    flask_logger.setLevel(level)
    
    # Configure werkzeug logger
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING)  # Reduce werkzeug noise

def set_request_context(data: Dict[str, Any]):
    """
    Set request context data for the current thread
    """
    if not hasattr(request_context, "data"):
        request_context.data = {}
    request_context.data.update(data)

def clear_request_context():
    """
    Clear request context for the current thread
    """
    if hasattr(request_context, "data"):
        del request_context.data

def generate_request_id() -> str:
    """
    Generate a unique ID for a request
    """
    return str(uuid.uuid4())

def log_request(logger):
    """
    Decorator to log requests and responses
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            request_id = generate_request_id()
            start_time = time.time()
            
            # Set context for structured logging
            set_request_context({
                "request_id": request_id,
                "start_time": datetime.utcnow().isoformat(),
            })
            
            logger.info(f"Request {request_id} started")
            
            try:
                # Call the original handler
                response = f(*args, **kwargs)
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                # Log successful completion
                logger.info(f"Request {request_id} completed in {duration_ms:.2f}ms")
                return response
            except Exception as e:
                # Log exception
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                logger.error(f"Request {request_id} failed after {duration_ms:.2f}ms: {str(e)}", exc_info=True)
                raise
            finally:
                # Clear request context
                clear_request_context()
                
        return decorated_function
    return decorator

def log_model_call(logger):
    """
    Decorator to log model API calls
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            call_id = generate_request_id()
            start_time = time.time()
            
            # Get model name from kwargs or use default
            model_name = kwargs.get('model_name', 'unknown')
            
            logger.info(f"Model call {call_id} to {model_name} started")
            
            try:
                # Call the original function
                response = f(*args, **kwargs)
                
                # Calculate and log metrics
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                # Get response length if possible
                response_length = len(response) if isinstance(response, str) else 0
                
                logger.info(f"Model call {call_id} completed in {duration_ms:.2f}ms, response length: {response_length}")
                return response
            except Exception as e:
                # Log failed model call
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                logger.error(f"Model call {call_id} failed after {duration_ms:.2f}ms: {str(e)}", exc_info=True)
                raise
                
        return decorated_function
    return decorator

# Initialize logging when module is imported
configure_logging()
