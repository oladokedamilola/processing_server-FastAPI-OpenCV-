"""
Job management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Form, File, UploadFile
from typing import List, Optional
from datetime import datetime, timedelta

from ....core.security import get_api_key
from ....core.config import settings
from ....utils.logger import logger
from ....utils.file_handling import save_upload_file
from ....jobs.manager import JobQueueManager
from ....jobs.processors import JobProcessors
from ....jobs.models import (
    JobType, JobStatus, JobPriority, JobResponse, 
    JobStats, JobFilter, JobCreate
)
from ....models.schemas import (
    ImageProcessingRequest, VideoProcessingRequest,
    VideoProcessingResponse, JobStatusResponse
)

# Create global instances
job_manager = None
job_processors = None

def get_job_manager():
    """Get or create job manager instance"""
    global job_manager
    if job_manager is None:
        job_manager = JobQueueManager()
        
        # Register processors
        global job_processors
        if job_processors is None:
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
    
    return job_manager

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.get("", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    job_type: Optional[JobType] = Query(None, description="Filter by job type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    api_key: str = Depends(get_api_key)
):
    """List jobs with optional filtering"""
    try:
        manager = get_job_manager()
        jobs = manager.list_jobs(
            status=status,
            job_type=job_type,
            limit=limit,
            offset=offset
        )
        return jobs
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=JobStats)
async def get_job_stats(api_key: str = Depends(get_api_key)):
    """Get job statistics"""
    try:
        manager = get_job_manager()
        stats = manager.get_job_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting job stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, api_key: str = Depends(get_api_key)):
    """Get job details by ID"""
    try:
        manager = get_job_manager()
        job = manager.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str, api_key: str = Depends(get_api_key)):
    """Get job status by ID"""
    try:
        manager = get_job_manager()
        job = manager.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            message=f"Job is {job.status.value}",
            result=job.result,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{job_id}/cancel")
async def cancel_job(job_id: str, api_key: str = Depends(get_api_key)):
    """Cancel a pending or processing job"""
    try:
        manager = get_job_manager()
        success = manager.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel job {job_id}. It may already be completed or not cancellable."
            )
        
        return {"success": True, "message": f"Job {job_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_old_jobs(
    max_age_hours: int = Query(24, ge=1, le=720, description="Maximum age of jobs to keep (hours)"),
    api_key: str = Depends(get_api_key)
):
    """Clean up old completed jobs"""
    try:
        manager = get_job_manager()
        manager.job_db.delete_old_jobs(max_age_hours)
        
        return {
            "success": True,
            "message": f"Cleaned up jobs older than {max_age_hours} hours"
        }
    except Exception as e:
        logger.error(f"Error cleaning up jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
# Add new endpoint to jobs.py
@router.get("/video/formats")
async def get_supported_video_formats(api_key: str = Depends(get_api_key)):
    """Get supported video formats and codecs"""
    try:
        from ....processing.video_processor import VideoProcessor
        from ....core.processor import get_processor
        
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        video_processor = VideoProcessor(processor)
        formats = video_processor.get_supported_formats()
        
        return {
            "success": True,
            "formats": formats,
            "ffmpeg_available": formats.get("conversion_available", False)
        }
        
    except Exception as e:
        logger.error(f"Error getting supported formats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/video/compatibility-check")
async def check_video_compatibility(
    file: UploadFile = File(..., description="Video file to check"),
    api_key: str = Depends(get_api_key)
):
    """Check video file compatibility"""
    try:
        from ....processing.video_processor import VideoProcessor
        from ....core.processor import get_processor
        
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        # Save uploaded file temporarily
        video_path = save_upload_file(file, settings.TEMP_PATH / "compatibility")
        
        # Check compatibility
        video_processor = VideoProcessor(processor)
        compatibility = video_processor.check_video_compatibility(video_path)
        
        # Clean up temp file
        try:
            if video_path.exists():
                video_path.unlink()
        except:
            pass
        
        return {
            "success": True,
            "compatibility": compatibility,
            "filename": file.filename,
            "content_type": file.content_type
        }
        
    except Exception as e:
        logger.error(f"Error checking video compatibility: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Updated video processing endpoint with enhanced format support
@router.post("/process/video", response_model=VideoProcessingResponse)
async def process_video_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to process"),
    confidence_threshold: Optional[float] = Form(None),
    detection_types: Optional[str] = Form(None),
    frame_sample_rate: Optional[int] = Form(None),
    analyze_motion: Optional[bool] = Form(True),
    return_summary_only: Optional[bool] = Form(False),
    enable_advanced_features: Optional[bool] = Form(True),
    force_conversion: Optional[bool] = Form(False),
    priority: Optional[int] = Form(1),
    api_key: str = Depends(get_api_key)
):
    """
    Process a video file in the background with enhanced format support
    
    - **file**: Video file (multiple formats supported)
    - **force_conversion**: Force video conversion if needed
    - **enable_advanced_features**: Enable crowd detection, vehicle counting, etc.
    """
    try:
        # Get job manager
        manager = get_job_manager()
        
        # Save uploaded file
        video_path = save_upload_file(file, settings.TEMP_PATH / "videos/jobs")
        
        # Parse detection types
        parsed_detection_types = None
        if detection_types:
            from ....models.detection import DetectionType
            try:
                types = detection_types.split(",")
                parsed_detection_types = [DetectionType(t.strip().lower()) for t in types]
            except ValueError:
                parsed_detection_types = None
        
        # Prepare job parameters
        job_params = {
            "file_path": str(video_path),
            "confidence_threshold": confidence_threshold,
            "detection_types": parsed_detection_types,
            "frame_sample_rate": frame_sample_rate,
            "analyze_motion": analyze_motion,
            "return_summary_only": return_summary_only,
            "enable_advanced_features": enable_advanced_features,
            "force_conversion": force_conversion,
        }
        
        # Convert priority
        job_priority = JobPriority(min(max(priority, 0), 3))
        
        # Submit job
        job_id = manager.submit_job(
            job_type=JobType.VIDEO_PROCESSING,
            parameters=job_params,
            priority=job_priority,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "original_filename": file.filename,
                "enable_advanced_features": enable_advanced_features,
            }
        )
        
        logger.info(f"Video processing job submitted: {job_id} (priority: {job_priority.value})")
        
        return VideoProcessingResponse(
            success=True,
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Video processing job submitted successfully",
            estimated_time=None,
            results_url=f"/api/v1/jobs/{job_id}/status"
        )
        
    except Exception as e:
        logger.error(f"Error submitting video job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))