"""
API v1 endpoints with authentication
"""
from fastapi import APIRouter
from .endpoints import health, process, jobs, advanced

# Create main router
router = APIRouter(prefix="/api/v1")

# Include routers
router.include_router(health.router, tags=["health"])
router.include_router(process.router, tags=["processing"])
router.include_router(jobs.router, tags=["jobs"])
router.include_router(advanced.router, tags=["advanced"])