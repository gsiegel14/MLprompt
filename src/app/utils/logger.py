
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
    Custom formatter for structured JSON logs
    """
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add request context if available
        if hasattr(request_context, "request_id"):
            log_data["request_id"] = request_context.request_id
            
        if hasattr(request_context, "user_id"):
            log_data["user_id"] = request_context.user_id
            
        if hasattr(request_context, "extra") and isinstance(request_context.extra, dict):
            log_data.update(request_context.extra)
        
        return json.dumps(log_data)

def setup_unified_logging():
    """
    Configure unified logging across all components
    """
    # Create a JSON formatter
    json_formatter = StructuredLogFormatter()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    
    # Create file handler for all logs
    file_handler = logging.FileHandler('logs/application.log')
    file_handler.setFormatter(json_formatter)
    
    # Create file handler for errors only
    error_handler = logging.FileHandler('logs/errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Set specific log levels for different components
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("gunicorn").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    
    # Log initialization
    logging.getLogger(__name__).info("Unified logging configured")

def set_request_context(request_id: Optional[str] = None, 
                        user_id: Optional[str] = None, 
                        **kwargs):
    """
    Set context for the current request
    
    Args:
        request_id: Unique ID for the request
        user_id: User ID associated with the request
        **kwargs: Additional context values
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    request_context.request_id = request_id
    
    if user_id:
        request_context.user_id = user_id
    
    # Store extra context
    if not hasattr(request_context, "extra"):
        request_context.extra = {}
    
    request_context.extra.update(kwargs)

def clear_request_context():
    """
    Clear the current request context
    """
    if hasattr(request_context, "request_id"):
        delattr(request_context, "request_id")
    
    if hasattr(request_context, "user_id"):
        delattr(request_context, "user_id")
    
    if hasattr(request_context, "extra"):
        delattr(request_context, "extra")

def log_execution_time(function_name=None):
    """
    Decorator to log execution time of functions
    
    Args:
        function_name: Optional name to use in logs
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = function_name or func.__name__
            logger = logging.getLogger(func.__module__)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Function {name} executed in {execution_time:.4f}s")
                
                # Add execution time to request context if it exists
                if hasattr(request_context, "extra"):
                    if "execution_times" not in request_context.extra:
                        request_context.extra["execution_times"] = {}
                    
                    request_context.extra["execution_times"][name] = execution_time
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {name} failed after {execution_time:.4f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator

def log_api_call(logger=None):
    """
    Decorator to log API calls with request and response details
    
    Args:
        logger: Optional logger to use
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create logger
            log = logger or logging.getLogger(func.__module__)
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Set request context
            set_request_context(request_id=request_id, endpoint=func.__name__)
            
            try:
                # Log request
                log.info(f"API call started: {func.__name__}")
                
                # Execute the endpoint
                start_time = time.time()
                response = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful response
                log.info(f"API call completed: {func.__name__} in {execution_time:.4f}s")
                
                return response
            except Exception as e:
                # Log error response
                log.error(f"API call failed: {func.__name__} - {str(e)}")
                # Re-raise the exception
                raise
            finally:
                # Clear request context
                clear_request_context()
                
        return wrapper
    return decorator
"""
Centralized logging configuration for the prompt optimization platform.
Provides structured JSON logging for production and readable logs for development.
"""
import os
import sys
import logging
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_FORMAT_DEV = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = "logs"

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production"""
    
    def format(self, record):
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_object["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if available
        if hasattr(record, "extra"):
            log_object["extra"] = record.extra
            
        return json.dumps(log_object)

def configure_logging(app_name="prompt_optimizer", log_level=None, enable_json=None):
    """
    Configure application logging
    
    Args:
        app_name: Name of the application for log identification
        log_level: Override default log level
        enable_json: Override default JSON formatting setting
    """
    # Determine environment
    env = os.environ.get("ENVIRONMENT", "development")
    
    # Determine log level
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO")
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Determine if we should use JSON formatting
    if enable_json is None:
        enable_json = env == "production"
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to prevent duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        f"{LOG_DIR}/{app_name}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    
    # Configure formatters based on environment
    if enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT_DEV)
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Create application logger
    logger = logging.getLogger(app_name)
    logger.info(f"Logging configured: level={log_level}, json_format={enable_json}, env={env}")
    
    return logger
