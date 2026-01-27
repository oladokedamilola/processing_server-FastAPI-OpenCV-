"""
API v1 endpoints
"""
from fastapi import APIRouter
from .endpoints import health

router = APIRouter(prefix="/api/v1")

# Include routers
router.include_router(health.router, tags=["health"])