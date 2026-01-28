"""
FastAPI Processing Server - Main Application
"""
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
import traceback
from typing import Dict, Any
from fastapi.middleware.gzip import GZipMiddleware
from .core.config import settings
from .core.exceptions import ProcessingException, create_http_exception
from .core.security import rate_limit_middleware
from .api.v1 import router as api_v1_router
from .utils.logger import logger
from .models.schemas import ErrorResponse
from .core.processor import init_processor, get_processor

# Import job management components
from .jobs.manager import JobQueueManager
from .jobs.processors import JobProcessors
from .jobs.models import JobType

# Global instances
processor = None
job_manager = None

def init_job_manager():
    """Initialize job manager"""
    global job_manager
    
    if job_manager is None:
        try:
            job_manager = JobQueueManager()
            
            # Initialize processors
            job_processors = JobProcessors()
            
            # Register job processors
            job_manager.register_processor(
                JobType.IMAGE_PROCESSING,
                job_processors.get_image_processor
            )
            job_manager.register_processor(
                JobType.VIDEO_PROCESSING,
                job_processors.get_video_processor
            )
            job_manager.register_processor(
                JobType.BATCH_PROCESSING,
                job_processors.get_batch_processor
            )
            
            # Start workers
            job_manager.start_workers()
            
            logger.info("Job manager initialized and workers started")
            
        except Exception as e:
            logger.error(f"Failed to initialize job manager: {str(e)}")
            job_manager = None

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events"""
    global processor, job_manager
    
    # Startup
    logger.info("Starting FastAPI Processing Server", extra={
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "host": settings.HOST,
        "port": settings.PORT,
    })
    
    # Log directory creation
    logger.info("Application directories created", extra={
        "upload_dir": str(settings.UPLOAD_PATH),
        "models_dir": str(settings.MODELS_PATH),
        "processed_dir": str(settings.PROCESSED_PATH),
    })
    
    # Initialize image processor
    try:
        if init_processor():
            logger.info("Image processor initialized successfully")
        else:
            logger.error("Failed to initialize image processor")
    except Exception as e:
        logger.error(f"Failed to initialize image processor: {str(e)}")
    
    # Initialize job manager
    init_job_manager()
    
    yield
    
    # Shutdown
    if job_manager:
        job_manager.shutdown()
        logger.info("Job manager shutdown complete")
    
    logger.info("Shutting down FastAPI Processing Server")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.PROJECT_DESCRIPTION,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Processing-Time", "X-Job-ID"],
)

if settings.ENABLE_GZIP:
    app.add_middleware(
        GZipMiddleware, 
        minimum_size=500,  # Only compress responses larger than 500 bytes
        compresslevel=6    # Balanced compression level
    )

# Custom middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Incoming request: {request.method} {request.url.path}", extra={
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", ""),
        "content_type": request.headers.get("content-type", ""),
    })
    
    try:
        response = await call_next(request)
    except Exception as e:
        # Log unhandled exceptions
        logger.error(f"Unhandled exception in request: {str(e)}", extra={
            "method": request.method,
            "path": request.url.path,
            "traceback": traceback.format_exc(),
        })
        raise
    
    # Add processing time header
    process_time = time.time() - start_time
    response.headers["X-Processing-Time"] = str(process_time)
    
    # Log completed request
    logger.api_log(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        processing_time=process_time,
        client_ip=request.client.host if request.client else "unknown",
    )
    
    return response

# Add rate limiting middleware
@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    return await rate_limit_middleware(request, call_next)

# Include API routers
app.include_router(api_v1_router)

# Exception handlers
@app.exception_handler(ProcessingException)
async def processing_exception_handler(request: Request, exc: ProcessingException):
    """Handle custom processing exceptions"""
    logger.error(f"Processing error: {exc.message}", extra={
        "error_code": exc.code,
        "details": exc.details,
        "path": request.url.path,
    })
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=True,
            message=exc.message,
            error_code=exc.code,
            details=exc.details,
        ).dict(),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    logger.warning(f"Validation error: {errors}", extra={
        "path": request.url.path,
        "errors": errors,
    })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=True,
            message="Validation error",
            error_code="VALIDATION_ERROR",
            details={"errors": errors},
        ).dict(),
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP error: {exc.detail}", extra={
        "status_code": exc.status_code,
        "path": request.url.path,
    })
    
    # If detail is already our ErrorResponse format, return as-is
    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=True,
            message=str(exc.detail),
            error_code=f"HTTP_{exc.status_code}",
        ).dict(),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", extra={
        "path": request.url.path,
        "method": request.method,
        "traceback": traceback.format_exc(),
    })
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=True,
            message="Internal server error",
            error_code="INTERNAL_SERVER_ERROR",
            details={
                "exception_type": exc.__class__.__name__,
                "exception_message": str(exc),
            } if settings.DEBUG else {},
        ).dict(),
    )

# Health check endpoint
@app.get("/", tags=["health"])
async def root():
    """Root endpoint with basic info"""
    global processor, job_manager
    
    models_status = "not_initialized"
    if processor:
        models = processor.get_available_models()
        loaded_models = [m for m in models if m.get('loaded')]
        models_status = f"{len(loaded_models)}/{len(models)} models loaded"
    
    job_status = "not_initialized"
    if job_manager:
        stats = job_manager.get_job_stats()
        job_status = f"{stats.active_jobs} active, {stats.completed_jobs} completed"
    
    processing_stats = "not_available"
    if processor:
        try:
            stats = processor._get_processing_statistics()
            processing_stats = f"{stats['total_images_processed']} images, {stats['total_detections']} detections"
        except:
            pass
    
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "status": "operational",
        "environment": settings.ENVIRONMENT,
        "models_status": models_status,
        "job_status": job_status,
        "processing_stats": processing_stats,
        "documentation": "/docs" if settings.ENVIRONMENT != "production" else None,
        "health_check": "/health",
        "api_v1": "/api/v1",
        "advanced_features": {
            "crowd_detection": "/api/v1/advanced/crowd-detection",
            "vehicle_counting": "/api/v1/advanced/vehicle-counting",
            "statistics": "/api/v1/advanced/processing-statistics"
        }
    }

@app.get("/health", tags=["health"], response_model=Dict[str, Any])
async def health_check():
    """Basic health check"""
    global processor, job_manager
    
    processor_status = "initialized" if processor else "not_initialized"
    job_manager_status = "initialized" if job_manager else "not_initialized"
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "processor_status": processor_status,
        "job_manager_status": job_manager_status,
    }