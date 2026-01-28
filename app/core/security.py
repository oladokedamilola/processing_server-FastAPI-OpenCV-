"""
Security middleware and authentication
"""
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from typing import Optional
import hashlib
import time
from .config import settings
from ..utils.logger import logger

# API Key security schemes
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self):
        self.requests = {}
        
    def is_rate_limited(self, client_id: str) -> bool:
        current_time = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < settings.RATE_LIMIT_PERIOD
        ]
        
        # Check if rate limited
        if len(self.requests[client_id]) >= settings.RATE_LIMIT_REQUESTS:
            return True
        
        # Add current request
        self.requests[client_id].append(current_time)
        return False

rate_limiter = RateLimiter()

def get_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> str:
    """Validate API key from header or query parameter"""
    
    # First check header, then query parameter
    api_key = api_key_header or api_key_query
    
    if not api_key:
        logger.warning("API key missing from request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
        )
    
    # Simple validation - in production, you'd check against a database
    if not validate_api_key(api_key):
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return api_key

def validate_api_key(api_key: str) -> bool:
    """Validate the API key"""
    # In development, accept the configured API key
    if settings.ENVIRONMENT == "development":
        return api_key == settings.API_KEY
    
    # In production, you might:
    # 1. Check against database of valid keys
    # 2. Validate key format
    # 3. Check expiration
    # For now, simple comparison
    return api_key == settings.API_KEY

def get_client_identifier(request) -> str:
    """Get unique identifier for client (for rate limiting)"""
    # Use X-Forwarded-For if behind proxy, otherwise client host
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0]
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    # Add API key for per-key rate limiting
    api_key = request.headers.get(settings.API_KEY_HEADER, "")
    return f"{client_ip}:{hashlib.md5(api_key.encode()).hexdigest()[:8]}"

async def rate_limit_middleware(request, call_next):
    """Middleware for rate limiting"""
    if not settings.RATE_LIMIT_ENABLED:
        response = await call_next(request)
        return response
    
    client_id = get_client_identifier(request)
    
    if rate_limiter.is_rate_limited(client_id):
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(settings.RATE_LIMIT_PERIOD)},
        )
    
    response = await call_next(request)
    return response