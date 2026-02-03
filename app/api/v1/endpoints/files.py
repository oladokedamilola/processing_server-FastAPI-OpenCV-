# app/api/v1/endpoints/files.py
"""
Endpoints for serving processed files (images with bounding boxes)
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pathlib import Path
import os

from ....core.config import settings
from ....core.security import get_api_key
from ....utils.logger import logger

router = APIRouter(prefix="/files", tags=["files"])

@router.get("/processed/images/{filename}")
async def get_processed_image(filename: str, api_key: str = Depends(get_api_key)):
    """
    Serve a processed image with bounding boxes.
    
    - **filename**: Name of the processed image file
    """
    try:
        # Security check: ensure filename doesn't contain path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Build the full path
        file_path = settings.PROCESSED_IMAGES_PATH / filename
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Processed image not found: {filename}")
            raise HTTPException(status_code=404, detail="Processed image not found")
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file")
        
        # Determine content type
        content_type = "image/jpeg"  # Default
        
        if filename.lower().endswith('.png'):
            content_type = "image/png"
        elif filename.lower().endswith('.gif'):
            content_type = "image/gif"
        elif filename.lower().endswith('.bmp'):
            content_type = "image/bmp"
        elif filename.lower().endswith('.webp'):
            content_type = "image/webp"
        
        logger.info(f"Serving processed image: {filename}")
        
        # Return file
        return FileResponse(
            path=file_path,
            media_type=content_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving processed image {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processed/videos/{filename}")
async def get_processed_video(filename: str, api_key: str = Depends(get_api_key)):
    """
    Serve a processed video with annotations.
    
    - **filename**: Name of the processed video file
    """
    try:
        # Security check
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Build the full path
        file_path = settings.PROCESSED_VIDEOS_PATH / filename
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Processed video not found: {filename}")
            raise HTTPException(status_code=404, detail="Processed video not found")
        
        # Check if it's a file
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file")
        
        # Determine content type
        content_type = "video/mp4"  # Default
        
        if filename.lower().endswith('.avi'):
            content_type = "video/x-msvideo"
        elif filename.lower().endswith('.mov'):
            content_type = "video/quicktime"
        elif filename.lower().endswith('.mkv'):
            content_type = "video/x-matroska"
        elif filename.lower().endswith('.webm'):
            content_type = "video/webm"
        
        logger.info(f"Serving processed video: {filename}")
        
        # Return file
        return FileResponse(
            path=file_path,
            media_type=content_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving processed video {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processed/list")
async def list_processed_files(api_key: str = Depends(get_api_key)):
    """List all processed files"""
    try:
        processed_files = {
            "images": [],
            "videos": []
        }
        
        # List processed images
        if settings.PROCESSED_IMAGES_PATH.exists():
            for file_path in settings.PROCESSED_IMAGES_PATH.iterdir():
                if file_path.is_file():
                    processed_files["images"].append({
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                        "url": f"/api/v1/files/processed/images/{file_path.name}"
                    })
        
        # List processed videos
        if settings.PROCESSED_VIDEOS_PATH.exists():
            for file_path in settings.PROCESSED_VIDEOS_PATH.iterdir():
                if file_path.is_file():
                    processed_files["videos"].append({
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                        "url": f"/api/v1/files/processed/videos/{file_path.name}"
                    })
        
        # Sort by modification time (newest first)
        processed_files["images"].sort(key=lambda x: x["modified"], reverse=True)
        processed_files["videos"].sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "success": True,
            "count": {
                "images": len(processed_files["images"]),
                "videos": len(processed_files["videos"])
            },
            "files": processed_files
        }
        
    except Exception as e:
        logger.error(f"Error listing processed files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    
    
@router.get("/processed/images/absolute/{filename}")
async def get_processed_image_absolute(filename: str, api_key: str = Depends(get_api_key)):
    """
    Serve processed image from absolute path (for debugging).
    """
    try:
        # Use the exact absolute path where you know the file exists
        absolute_path = r"C:\Users\PC\Desktop\processing_server (FastAPI + OpenCV)\processing_server\processed\images" / filename
        
        logger.info(f"Trying absolute path: {absolute_path}")
        
        if not absolute_path.exists():
            raise HTTPException(status_code=404, detail="Processed image not found at absolute path")
        
        # Return file
        return FileResponse(
            path=absolute_path,
            media_type="image/jpeg",
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Error serving absolute path image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")