"""
Health check endpoints
"""
from fastapi import APIRouter, Depends
from typing import Dict
import psutil
import platform
from datetime import datetime

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check() -> Dict:
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FastAPI Processing Server"
    }

@router.get("/system")
async def system_info() -> Dict:
    """System information endpoint"""
    return {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }