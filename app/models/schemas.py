# app/models/schemas.py
"""
Pydantic models for request/response schemas
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Import DetectionType from detection module
from .detection import DetectionType

# Enums (keep only JobStatus here since DetectionType is imported)
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"

# Request Models
class ImageProcessingRequest(BaseModel):
    """Request model for image processing"""
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    detection_types: Optional[List[DetectionType]] = Field(
        default=[DetectionType.PERSON, DetectionType.VEHICLE]
    )
    return_image: Optional[bool] = Field(False, description="Return processed image")
    image_format: Optional[str] = Field("jpeg", description="Format for returned image")
    enable_advanced_features: Optional[bool] = Field(True, description="Enable advanced features like crowd detection")

class VideoProcessingRequest(BaseModel):
    """Request model for video processing"""
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    detection_types: Optional[List[DetectionType]] = Field(
        default=[DetectionType.PERSON, DetectionType.VEHICLE, DetectionType.MOTION]
    )
    frame_sample_rate: Optional[int] = Field(5, ge=1, le=30, description="Process every Nth frame")
    analyze_motion: Optional[bool] = Field(True)
    return_summary_only: Optional[bool] = Field(False, description="Return only summary, not per-frame results")
    enable_advanced_features: Optional[bool] = Field(True, description="Enable advanced features")

# Response Models
class DetectionResult(BaseModel):
    """Single detection result"""
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[int] = Field(..., min_items=4, max_items=4)  # [x1, y1, x2, y2]
    type: DetectionType
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ProcessingResponse(BaseModel):
    """Base response for processing operations"""
    success: bool
    job_id: Optional[str] = None
    processing_time: Optional[float] = None
    detections: Optional[List[DetectionResult]] = None
    detection_count: Optional[int] = None
    image_size: Optional[str] = None
    message: Optional[str] = None
    warnings: Optional[List[str]] = Field(default_factory=list)



class ImageProcessingResponse(ProcessingResponse):
    """Response for image processing"""
    processed_image_url: Optional[str] = None
    processed_image_base64: Optional[str] = None  # base64 encoded image
    has_processed_image: Optional[bool] = Field(False, description="Whether processed image with bounding boxes is available")
    image_format: Optional[str] = None
    models_used: Optional[List[str]] = Field(default_factory=list, description="Models used for processing")
    statistics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing statistics")
    advanced_results: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Advanced processing results")
    django_media_id: Optional[int] = None  # Pass Django ID back
    django_user_id: Optional[int] = None   # Pass Django user ID back
    
    
class VideoProcessingResponse(BaseModel):
    """Response for video processing initiation"""
    success: bool
    job_id: str
    status: JobStatus
    message: str
    estimated_time: Optional[float] = None
    results_url: Optional[str] = None

class JobStatusResponse(BaseModel):
    """Response for job status check"""
    job_id: str
    status: JobStatus
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Response for health check"""
    status: str
    timestamp: datetime
    service: str
    version: str
    environment: str
    uptime: Optional[float] = None

class SystemInfoResponse(BaseModel):
    """Response for system information"""
    system: str
    python_version: str
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    active_jobs: int
    total_processed: int
    server_uptime: float

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: bool = True
    message: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    

class VideoProcessingResponse(BaseModel):
    """Response for video processing"""
    success: bool
    job_id: str
    video_info: Dict[str, Any]
    summary: Dict[str, Any]
    processing_time: float
    key_frames_base64: Optional[List[Dict[str, Any]]] = None  # Base64 key frames
    has_key_frames: Optional[bool] = False
    key_frames_count: Optional[int] = 0
    frame_results: Optional[List[Dict[str, Any]]] = None
    optimizations: Optional[Dict[str, Any]] = None
    django_media_id: Optional[int] = None
    django_user_id: Optional[int] = None