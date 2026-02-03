# app/api/v1/endpoints/video.py
"""
Video processing endpoints for Django integration
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid
import json
from pathlib import Path
import time
from ....core.config import settings
from ....core.security import get_api_key
from ....core.exceptions import ProcessingException
from ....core.processor import get_processor
from ....processing.video_processor import VideoProcessor
from ....processing.image_processor import ImageProcessor
from ....models.schemas import DetectionType
from ....utils.logger import logger

router = APIRouter(prefix="/process", tags=["video"])

# In-memory job storage (for simple implementation)
# In production, use Redis or database
video_jobs = {}

@router.post("/video", response_model=Dict[str, Any])
async def process_video(
    file: UploadFile = File(..., description="Video file to process"),
    confidence_threshold: Optional[float] = Form(None),
    detection_types: Optional[str] = Form(None),
    frame_sample_rate: Optional[int] = Form(5),
    return_summary_only: Optional[bool] = Form(True),
    enable_advanced_features: Optional[bool] = Form(True),
    return_key_frames_base64: Optional[bool] = Form(True),  # Return base64 key frames
    num_key_frames: Optional[int] = Form(5),  # Number of key frames to extract
    django_media_id: Optional[int] = Form(None),  # Django media ID
    django_user_id: Optional[int] = Form(None),   # Django user ID
    api_key: str = Depends(get_api_key)
):
    """
    Process a video file with base64 output support for Django
    
    - **file**: Video file (MP4, AVI, MOV, MKV)
    - **confidence_threshold**: Minimum confidence for detections (0.0 to 1.0)
    - **detection_types**: Comma-separated list of detection types
    - **frame_sample_rate**: Process every Nth frame
    - **return_summary_only**: Return only summary, not per-frame results
    - **enable_advanced_features**: Enable advanced features like crowd detection
    - **return_key_frames_base64**: Return base64 encoded key frames (default: True)
    - **num_key_frames**: Number of key frames to extract as base64 (default: 5)
    - **django_media_id**: Optional ID of the MediaUpload in Django
    - **django_user_id**: Optional ID of the User in Django
    """
    try:
        # Create job ID
        job_id = f"video_{uuid.uuid4().hex[:8]}"
        
        # Parse detection types
        parsed_detection_types = None
        if detection_types:
            try:
                types = detection_types.split(",")
                parsed_detection_types = [DetectionType(t.strip().lower()) for t in types]
            except ValueError as e:
                raise ProcessingException(
                    message="Invalid detection type",
                    details={"valid_types": [dt.value for dt in DetectionType]}
                )
        
        # Save file temporarily
        import tempfile
        import os
        
        suffix = Path(file.filename).suffix if file.filename else '.mp4'
        temp_filepath = None
        
        try:
            # Read file content
            file_content = await file.read()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_content)
                temp_filepath = tmp.name
            
            # Create video path
            video_path = Path(temp_filepath)
            
            # Initialize video processor
            image_processor = ImageProcessor()
            video_processor = VideoProcessor(image_processor=image_processor)
            
            # Process video
            result = video_processor.process_video(
                video_path=video_path,
                confidence_threshold=confidence_threshold,
                detection_types=parsed_detection_types,
                frame_sample_rate=frame_sample_rate,
                return_summary_only=return_summary_only,
                enable_advanced_features=enable_advanced_features,
                return_key_frames_base64=return_key_frames_base64,
                num_key_frames=num_key_frames,
                key_frames_quality=settings.BASE64_VIDEO_QUALITY
            )
            
            # Add job ID and Django references
            result["job_id"] = job_id
            if django_media_id:
                result["django_media_id"] = django_media_id
            if django_user_id:
                result["django_user_id"] = django_user_id
            
            # Store job result
            video_jobs[job_id] = {
                "status": "completed",
                "result": result,
                "django_media_id": django_media_id,
                "created_at": time.time()
            }
            
            logger.info(f"Video processing completed: {job_id} for Django media ID: {django_media_id or 'N/A'}")
            
            # Log base64 key frames info
            if return_key_frames_base64 and "key_frames_base64" in result:
                logger.info(f"Returned {len(result['key_frames_base64'])} base64 key frames")
            
            return result
            
        finally:
            # Clean up temp file
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.unlink(temp_filepath)
                except:
                    pass
                
    except ProcessingException as e:
        logger.error(f"Processing error: {e.message}", extra=e.details)
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Unexpected error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/video/jobs/{job_id}")
async def get_video_job_status(
    job_id: str,
    api_key: str = Depends(get_api_key)
):
    """Get video processing job status"""
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = video_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "django_media_id": job.get("django_media_id"),
        "created_at": job.get("created_at"),
        "result": job.get("result") if job["status"] == "completed" else None
    }

@router.delete("/video/jobs/{job_id}")
async def delete_video_job(
    job_id: str,
    api_key: str = Depends(get_api_key)
):
    """Delete video processing job"""
    if job_id in video_jobs:
        del video_jobs[job_id]
        return {"success": True, "message": f"Job {job_id} deleted"}
    
    raise HTTPException(status_code=404, detail="Job not found")

@router.get("/video/supported-formats")
async def get_supported_video_formats(
    api_key: str = Depends(get_api_key)
):
    """Get supported video formats"""
    try:
        video_processor = VideoProcessor()
        formats = video_processor.get_supported_formats()
        
        return {
            "success": True,
            "formats": formats,
            "max_size_mb": settings.MAX_VIDEO_SIZE / (1024 * 1024),
            "max_duration_seconds": settings.MAX_VIDEO_DURATION
        }
        
    except Exception as e:
        logger.error(f"Error getting supported formats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting formats: {str(e)}"
        )