"""
FastAPI Processing Server - Main Application
Simplified version without database/job queue for Django integration
"""
# APPLY PYTORCH PATCH FIRST - BEFORE ANY OTHER IMPORTS!
import app.core.pytorch_patch  # Ensure PyTorch patch is applied BEFORE anything else
from fastapi import FastAPI, Request, status, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
import traceback
from typing import Dict, Any
from fastapi.staticfiles import StaticFiles
import os

from .core.config import settings
from .core.exceptions import ProcessingException
from .core.security import rate_limit_middleware
from .api.v1 import router as api_v1_router
from .utils.logger import logger
from .models.schemas import ErrorResponse
from .core.processor import init_processor, get_processor

# Global processor instance
processor = None

# Lifespan context manager - SIMPLIFIED
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events - Simplified version"""
    global processor
    
    # Startup
    logger.info("Starting FastAPI Processing Server - Simplified for Django Integration", extra={
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "host": settings.HOST,
        "port": settings.PORT,
        "save_processed_images": settings.SAVE_PROCESSED_IMAGES,
        "return_base64_images": settings.RETURN_BASE64_IMAGES,
    })
    
    # Log directory creation
    logger.info("Application directories created", extra={
        "models_dir": str(settings.MODELS_PATH),
        "temp_dir": str(settings.TEMP_PATH),
    })
    
    # Initialize image processor only (no job queue)
    try:
        if init_processor():
            logger.info("Image processor initialized successfully")
            processor = get_processor()
        else:
            logger.error("Failed to initialize image processor")
    except Exception as e:
        logger.error(f"Failed to initialize image processor: {str(e)}")
    
    # Log available models
    if processor:
        try:
            models = processor.get_available_models()
            logger.info(f"Available models: {len(models)}", extra={
                "models": [m.get('name', 'unknown') for m in models]
            })
        except Exception as e:
            logger.warning(f"Could not get model list: {str(e)}")
    
    yield
    
    # Shutdown - Simplified
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
    logger.debug(f"Request completed: {request.method} {request.url.path} - {response.status_code} in {process_time:.3f}s")
    
    return response

# Add rate limiting middleware
@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    return await rate_limit_middleware(request, call_next)

# Include API routers
app.include_router(api_v1_router)

# Exception handlers (keep these)
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
    
    # Create a JSON-serializable response
    response_content = {
        "error": True,
        "message": "Validation error",
        "error_code": "VALIDATION_ERROR",
        "details": {"errors": errors},
    }
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_content,
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP error: {exc.detail}", extra={
        "status_code": exc.status_code,
        "path": request.url.path,
    })
    
    # Create a JSON-serializable response
    response_content = {
        "error": True,
        "message": str(exc.detail),
        "error_code": f"HTTP_{exc.status_code}",
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content,
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", extra={
        "path": request.url.path,
        "method": request.method,
        "traceback": traceback.format_exc(),
    })
    
    # Create a JSON-serializable response
    response_content = {
        "error": True,
        "message": "Internal server error",
        "error_code": "INTERNAL_SERVER_ERROR",
    }
    
    # Add debug info only if enabled
    if settings.DEBUG:
        response_content["details"] = {
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc),
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_content,
    )

# Root endpoint - SIMPLIFIED
@app.get("/", tags=["health"])
async def root():
    """Root endpoint with basic info - Simplified for Django architecture"""
    global processor
    
    # Get processor status
    processor_status = "not_initialized"
    models_status = "unknown"
    
    if processor:
        processor_status = "initialized"
        try:
            models = processor.get_available_models()
            loaded_models = [m for m in models if m.get('loaded')]
            models_status = f"{len(loaded_models)}/{len(models)} models loaded"
        except:
            models_status = "error"
    
    return {
        "message": f"Welcome to {settings.APP_NAME} - Processing Server",
        "version": settings.APP_VERSION,
        "status": "operational",
        "environment": settings.ENVIRONMENT,
        "architecture": "Stateless Processing Server for Django",
        "database": "None (Django handles storage)",
        "processor_status": processor_status,
        "models_status": models_status,
        "configuration": {
            "save_processed_images": settings.SAVE_PROCESSED_IMAGES,
            "return_base64_images": settings.RETURN_BASE64_IMAGES,
            "max_image_size_mb": settings.MAX_IMAGE_SIZE / (1024 * 1024),
            "max_video_size_mb": settings.MAX_VIDEO_SIZE / (1024 * 1024),
        },
        "endpoints": {
            "image_processing": "/api/v1/process/image",
            "video_processing": "/api/v1/process/video",
            "health_check": "/health",
            "models": "/api/v1/process/models",
            "docs": "/docs" if settings.ENVIRONMENT != "production" else None,
        },
        "integration": {
            "django_compatible": True,
            "data_format": "JSON with optional base64 images",
            "authentication": "API Key required for processing endpoints",
        }
    }

# Health check endpoint - SIMPLIFIED
@app.get("/health", tags=["health"], response_model=Dict[str, Any])
async def health_check():
    """Basic health check - Simplified"""
    global processor
    
    processor_status = "initialized" if processor else "not_initialized"
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "processor_status": processor_status,
        "database": "none",  # No database in FastAPI
        "architecture": "stateless_processing",
        "django_integration": True,
    }

# Test endpoints (optional - keep for debugging)
@app.post("/api/v1/test/image")
async def test_process_image(image: UploadFile = File(...)):
    """Test endpoint to debug image processing"""
    try:
        # Read the image
        contents = await image.read()
        
        # Log some info
        logger.info(f"Test endpoint: Received image {image.filename}, size: {len(contents)} bytes")
        
        # Return a simple response
        return {
            "success": True,
            "filename": image.filename,
            "size_bytes": len(contents),
            "content_type": image.content_type,
            "message": "Test endpoint working",
            "base64_support": settings.RETURN_BASE64_IMAGES,
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        raise

@app.post("/api/v1/simple/process/video")
async def simple_process_video(file: UploadFile = File(...)):
    """Simple video processing endpoint for testing"""
    try:
        # Read the video
        contents = await file.read()
        
        # Log some info
        logger.info(f"Simple video endpoint: Received {file.filename}, size: {len(contents)} bytes")
        
        # Generate a job ID
        import uuid
        job_id = f"video_{uuid.uuid4().hex[:8]}"
        
        # Return mock response
        return {
            "job_id": job_id,
            "status": "submitted",
            "message": "Video submitted for processing",
            "estimated_completion": "60 seconds",
            "base64_key_frames": settings.RETURN_BASE64_IMAGES,
        }
    except Exception as e:
        logger.error(f"Simple video endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve processed images for debugging only (if enabled)
if settings.SAVE_PROCESSED_IMAGES and settings.PROCESSED_IMAGES_PATH.exists():
    app.mount("/media/processed", StaticFiles(directory=str(settings.PROCESSED_IMAGES_PATH)), name="processed-media")
    logger.info(f"Serving processed images from: {settings.PROCESSED_IMAGES_PATH}")
    logger.warning("WARNING: Image serving is enabled for debugging only. Disable SAVE_PROCESSED_IMAGES in production.")
else:
    logger.info("Processed image serving is disabled (SAVE_PROCESSED_IMAGES=False)")