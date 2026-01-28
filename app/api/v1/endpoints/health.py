"""
Health check endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import psutil
import platform
import time
from datetime import datetime

from ....core.config import settings
from ....core.security import get_api_key
from ....models.schemas import HealthResponse, SystemInfoResponse
from ....utils.logger import logger

router = APIRouter(prefix="/health", tags=["health"])

# Track server start time
SERVER_START_TIME = time.time()

@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Public health check endpoint - no authentication required"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        uptime=time.time() - SERVER_START_TIME,
    )

@router.get("/system", response_model=SystemInfoResponse)
async def system_info(
    api_key: str = Depends(get_api_key)  # Requires API key
) -> SystemInfoResponse:
    """System information endpoint - requires authentication"""
    return SystemInfoResponse(
        system=platform.system(),
        python_version=platform.python_version(),
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        disk_usage=psutil.disk_usage('/').percent,
        active_jobs=0,  # Will be updated when job system is implemented
        total_processed=0,  # Will be updated when processing is implemented
        server_uptime=time.time() - SERVER_START_TIME,
    )

@router.get("/detailed")
async def detailed_health_check(
    api_key: str = Depends(get_api_key)  # Requires API key
) -> Dict[str, Any]:
    """Detailed health check with all system metrics - requires authentication"""
    try:
        # Get system info
        cpu_times = psutil.cpu_times()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Get process info
        process = psutil.Process()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu": {
                    "percent": psutil.cpu_percent(),
                    "cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                    "times": {
                        "user": cpu_times.user,
                        "system": cpu_times.system,
                        "idle": cpu_times.idle,
                    }
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                }
            },
            "process": {
                "pid": process.pid,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads(),
                "connections": len(process.connections()),
            },
            "application": {
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
                "uptime": time.time() - SERVER_START_TIME,
            }
        }
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}")
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }