"""
Public endpoints for serving processed files (no authentication required)
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from ....core.config import settings
from ....utils.logger import logger

router = APIRouter(prefix="/public", tags=["public"])

@router.get("/processed/images/{filename}")
async def get_processed_image_public(filename: str):
    """
    Serve a processed image with bounding boxes - NO AUTHENTICATION REQUIRED.
    
    - **filename**: Name of the processed image file
    """
    try:
        # Security check: ensure filename doesn't contain path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Use the configured path from settings
        file_path = settings.PROCESSED_IMAGES_PATH / filename
        
        logger.info(f"Public access - Looking for processed image at: {file_path}")
        logger.info(f"PROCESSED_IMAGES_PATH: {settings.PROCESSED_IMAGES_PATH}")
        logger.info(f"Directory exists: {settings.PROCESSED_IMAGES_PATH.exists()}")
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Processed image not found: {file_path}")
            raise HTTPException(status_code=404, detail="Processed image not found")
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file")
        
        # Determine content type based on extension
        ext = file_path.suffix.lower()
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
        }
        content_type = content_type_map.get(ext, 'image/jpeg')
        
        logger.info(f"Serving processed image (public): {filename}")
        
        # Return file - NO AUTHENTICATION REQUIRED
        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            filename=filename,
            headers={
                'Cache-Control': 'public, max-age=3600'  # Cache for 1 hour
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving processed image {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")