from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from functools import wraps

from fastapi import Depends, HTTPException, status, Request, APIRouter
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import ValidationError, BaseModel

from src.api.models import TokenData, UserResponse  # Assuming these models exist
from src.app.config import get_settings  # Assuming this function exists


settings = get_settings()

# Password hashing utility
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def get_user(username: str):
    # Replace with actual database query
    if username == "admin":
        return {
            "user_id": "admin-user-id",
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "Admin User",
            "hashed_password": get_password_hash("adminpassword"),
            "is_active": True,
            "scopes": ["prompts", "experiments", "inference", "datasets", "admin"],
            "created_at": datetime.now() - timedelta(days=30)
        }
    return None


async def authenticate_user(username: str, password: str):
    user = await get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "scopes": data.get("scopes", [])})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
        token_data = TokenData(username=username, scopes=payload.get("scopes", []))
    except (JWTError, ValidationError):
        raise credentials_exception
    user = await get_user(username=token_data.username)
    if not user:
        raise credentials_exception
    return UserResponse(**user)


async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_admin_user(current_user: UserResponse = Depends(get_current_user)):
    if "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user

# Example of a route requiring authentication
router = APIRouter()

@router.get("/protected", dependencies=[Depends(get_current_active_user)])
async def protected_route(current_user: UserResponse = Depends(get_current_active_user)):
    return {"message": f"Hello, {current_user.username}!"}

@router.get("/admin", dependencies=[Depends(get_current_admin_user)])
async def admin_route(current_user: UserResponse = Depends(get_current_admin_user)):
    return {"message": "Admin route accessed!"}
"""
Authentication utilities for the API
"""
import os
from fastapi import Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader
from src.app.config import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validate API key from request header
    
    Args:
        api_key_header: The API key from the request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    # In development mode, allow no API key
    if settings.ENVIRONMENT == "development" and not settings.ENFORCE_API_KEY:
        return "development_key"
    
    if api_key_header == settings.API_KEY:
        return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key"
    )
"""
Authentication and security for the API
"""
import os
import time
import logging
from typing import Dict, Optional, List
from fastapi import Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

logger = logging.getLogger(__name__)

# API key configuration
API_KEY_NAME = "X-API-Key"
API_KEY = os.environ.get("API_KEY", "")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Rate limiting configuration
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "1") == "1"
# Default rate limits (can be overridden with environment variables)
DEFAULT_RATE_LIMIT = int(os.environ.get("DEFAULT_RATE_LIMIT", "100"))  # requests per window
DEFAULT_RATE_WINDOW = int(os.environ.get("DEFAULT_RATE_LIMIT_WINDOW", "3600"))  # window in seconds (1 hour)

# In-memory rate limit store (would be replaced with Redis in production)
rate_limit_store: Dict[str, Dict[str, int]] = {}

def get_api_key(request: Request, api_key: str = Security(api_key_header)) -> str:
    """
    Validate API key and enforce rate limiting
    
    Args:
        request: The incoming request
        api_key: The API key from the header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If the API key is invalid or rate limit is exceeded
    """
    if not API_KEY:
        logger.warning("API_KEY environment variable not set - running in unsecured mode")
        if not api_key:
            # If no API key is configured, we'll still validate that the header is present
            # This ensures clients are configured correctly for when API keys are enabled
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, 
                detail="API key is required"
            )
        return api_key
    
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key[:5]}...")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, 
            detail="Invalid API key"
        )
    
    # Rate limiting check
    if RATE_LIMIT_ENABLED:
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        
        if not check_rate_limit(api_key, client_ip, endpoint):
            logger.warning(f"Rate limit exceeded for key {api_key[:5]}... from {client_ip}")
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later."
            )
    
    return api_key

def check_rate_limit(api_key: str, client_ip: str, endpoint: str) -> bool:
    """
    Check if the current request exceeds rate limits
    
    Args:
        api_key: The API key being used
        client_ip: The client IP address
        endpoint: The API endpoint being accessed
        
    Returns:
        bool: True if within rate limits, False if exceeded
    """
    # Get current timestamp
    now = int(time.time())
    
    # Create a unique identifier for this client
    client_id = f"{api_key}:{client_ip}"
    
    # Initialize or clean up rate limit data for this client
    if client_id not in rate_limit_store:
        rate_limit_store[client_id] = {}
        
    # Clean up expired entries
    clean_expired_entries(client_id, now)
    
    # Get the client's request history
    client_data = rate_limit_store[client_id]
    
    # Increment request count for this timestamp
    if now not in client_data:
        client_data[now] = 0
    client_data[now] += 1
    
    # Calculate total requests in current window
    window_start = now - DEFAULT_RATE_WINDOW
    total_requests = sum(count for timestamp, count in client_data.items() 
                          if timestamp > window_start)
    
    # Check if rate limit is exceeded
    return total_requests <= DEFAULT_RATE_LIMIT

def clean_expired_entries(client_id: str, current_time: int) -> None:
    """
    Remove expired entries from the rate limit store
    
    Args:
        client_id: The client identifier
        current_time: The current timestamp
    """
    if client_id in rate_limit_store:
        # Remove entries older than the rate limit window
        window_start = current_time - DEFAULT_RATE_WINDOW
        rate_limit_store[client_id] = {
            timestamp: count 
            for timestamp, count in rate_limit_store[client_id].items()
            if timestamp > window_start
        }
