"""
Advanced processing endpoints for crowd detection, vehicle counting, etc.
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from typing import List, Optional
from pathlib import Path

from ....core.security import get_api_key
from ....core.config import settings
from ....utils.logger import logger
from ....utils.file_handling import save_upload_file
from ....core.processor import get_processor
from ....models.schemas import DetectionType, ProcessingResponse
from ....models.detection import DetectionType as DT

router = APIRouter(prefix="/advanced", tags=["advanced"])

@router.post("/crowd-detection", response_model=ProcessingResponse)
async def detect_crowd(
    file: UploadFile = File(..., description="Image or video file for crowd detection"),
    confidence_threshold: Optional[float] = Form(0.5),
    min_people_count: Optional[int] = Form(3, ge=1, le=50),
    density_threshold: Optional[float] = Form(0.3, ge=0.1, le=1.0),
    return_image: Optional[bool] = Form(False),
    api_key: str = Depends(get_api_key)
):
    """
    Detect crowds in an image or video
    
    - **file**: Image or video file
    - **confidence_threshold**: Minimum confidence for person detections
    - **min_people_count**: Minimum number of people to consider as crowd
    - **density_threshold**: Minimum density to consider as crowd (0.1-1.0)
    - **return_image**: Whether to return processed image/video with detections
    """
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Check file type
        is_video = any(video_type in file.content_type for video_type in 
                      ["video/mp4", "video/avi", "video/mov"])
        
        if is_video:
            # Video processing for crowd detection
            from ....processing.video_processor import VideoProcessor
            video_processor = VideoProcessor(processor)
            
            result = video_processor.process_video(
                video_path=file_path,
                confidence_threshold=confidence_threshold,
                detection_types=[DT.PERSON, DT.CROWD],
                enable_advanced_features=True,
                return_summary_only=not return_image
            )
            
            # Extract crowd statistics from result
            crowd_stats = result.get("summary", {}).get("crowd_statistics", {})
            
            response_data = {
                "success": True,
                "processing_time": result["processing_time"],
                "detection_count": result["summary"].get("total_detections", 0),
                "image_size": result["video_info"]["resolution"],
                "crowd_statistics": crowd_stats,
                "video_info": result["video_info"],
                "message": "Crowd detection completed on video"
            }
            
            if return_image and "processed_image_url" in result:
                response_data["processed_image_url"] = result["processed_image_url"]
            
        else:
            # Image processing for crowd detection
            result = processor.process_image(
                image_path=file_path,
                detection_types=[DT.PERSON, DT.CROWD],
                confidence_threshold=confidence_threshold,
                return_image=return_image,
                enable_advanced_features=True
            )
            
            # Extract crowd statistics
            crowd_stats = result.get("advanced_results", {}).get("crowd_statistics", {})
            
            response_data = {
                "success": True,
                "processing_time": result["processing_time"],
                "detections": result["detections"],
                "detection_count": result["detection_count"],
                "image_size": result["image_size"],
                "crowd_statistics": crowd_stats,
                "message": "Crowd detection completed on image"
            }
        
        return ProcessingResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in crowd detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vehicle-counting", response_model=ProcessingResponse)
async def count_vehicles(
    file: UploadFile = File(..., description="Video file for vehicle counting"),
    confidence_threshold: Optional[float] = Form(0.5),
    counting_line_position: Optional[float] = Form(0.5, ge=0.1, le=0.9),
    frame_sample_rate: Optional[int] = Form(5, ge=1, le=30),
    api_key: str = Depends(get_api_key)
):
    """
    Count vehicles in a video
    
    - **file**: Video file
    - **confidence_threshold**: Minimum confidence for vehicle detections
    - **counting_line_position**: Position of virtual counting line (0.1-0.9)
    - **frame_sample_rate**: Process every Nth frame
    """
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        # Save uploaded file
        video_path = await save_upload_file(file, subdirectory="videos/counting")
        
        # Process video with vehicle counting
        from ....processing.video_processor import VideoProcessor
        video_processor = VideoProcessor(processor)
        
        result = video_processor.process_video(
            video_path=video_path,
            confidence_threshold=confidence_threshold,
            detection_types=[DT.VEHICLE],
            frame_sample_rate=frame_sample_rate,
            enable_advanced_features=True,
            return_summary_only=True
        )
        
        # Extract vehicle counting statistics
        vehicle_stats = result.get("summary", {}).get("vehicle_counting", {})
        
        return ProcessingResponse(
            success=True,
            processing_time=result["processing_time"],
            detection_count=result["summary"].get("total_detections", 0),
            image_size=result["video_info"]["resolution"],
            message="Vehicle counting completed",
            advanced_results={
                "vehicle_counting": vehicle_stats,
                "video_info": result["video_info"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error in vehicle counting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crowd-statistics")
async def get_crowd_statistics(api_key: str = Depends(get_api_key)):
    """Get current crowd detection statistics"""
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        stats = processor.get_crowd_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": "current_time"
        }
        
    except Exception as e:
        logger.error(f"Error getting crowd statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vehicle-statistics")
async def get_vehicle_statistics(api_key: str = Depends(get_api_key)):
    """Get current vehicle counting statistics"""
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        stats = processor.get_vehicle_counting_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": "current_time"
        }
        
    except Exception as e:
        logger.error(f"Error getting vehicle statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vehicle-counts/reset")
async def reset_vehicle_counts(api_key: str = Depends(get_api_key)):
    """Reset vehicle counts"""
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        processor.reset_vehicle_counts()
        
        return {
            "success": True,
            "message": "Vehicle counts reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting vehicle counts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processing-statistics")
async def get_processing_statistics(api_key: str = Depends(get_api_key)):
    """Get overall processing statistics"""
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        stats = processor._get_processing_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": "current_time"
        }
        
    except Exception as e:
        logger.error(f"Error getting processing statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))