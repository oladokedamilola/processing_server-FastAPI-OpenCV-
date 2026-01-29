# app/jobs/models.py
"""
Job data models and schemas
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class JobType(str, Enum):
    IMAGE_PROCESSING = "image_processing"
    VIDEO_PROCESSING = "video_processing"
    BATCH_PROCESSING = "batch_processing"

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class JobCreate(BaseModel):
    """Schema for creating a new job"""
    job_type: JobType
    parameters: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class JobUpdate(BaseModel):
    """Schema for updating a job"""
    status: Optional[JobStatus] = None
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class JobResponse(BaseModel):
    """Schema for job response"""
    job_id: str
    job_type: JobType
    status: JobStatus
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    priority: JobPriority
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    
    @property
    def is_completed(self) -> bool:
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    @property
    def is_active(self) -> bool:
        return self.status in [JobStatus.PENDING, JobStatus.PROCESSING]

class JobStats(BaseModel):
    """Schema for job statistics"""
    total_jobs: Optional[int] = 0
    pending_jobs: Optional[int] = 0
    processing_jobs: Optional[int] = 0
    completed_jobs: Optional[int] = 0
    failed_jobs: Optional[int] = 0
    cancelled_jobs: Optional[int] = 0
    average_processing_time: Optional[float] = None
    success_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Add a computed property for active_jobs
    @property
    def active_jobs(self) -> int:
        """Compute active jobs (pending + processing)"""
        pending = self.pending_jobs or 0
        processing = self.processing_jobs or 0
        return pending + processing

class JobFilter(BaseModel):
    """Schema for filtering jobs"""
    status: Optional[JobStatus] = None
    job_type: Optional[JobType] = None
    min_priority: Optional[JobPriority] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = 100
    offset: int = 0