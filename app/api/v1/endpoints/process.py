# app/api/v1/endpoints/process.py
"""
Image processing endpoints - NO DATABASE VERSION
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import List, Optional

from ....core.config import settings
from ....core.security import get_api_key
from ....core.exceptions import ProcessingException
from ....core.processor import get_processor
from ....models.schemas import (
    ImageProcessingResponse,
    DetectionType,
)
from ....utils.logger import logger

import uuid

router = APIRouter(prefix="/process", tags=["processing"])



@router.post("/image", response_model=ImageProcessingResponse)
async def process_image(
    file: UploadFile = File(..., description="Image file to process"),
    confidence_threshold: Optional[float] = Form(None),
    detection_types: Optional[str] = Form(None),
    return_image: Optional[bool] = Form(False),
    image_format: Optional[str] = Form("jpeg"),
    return_base64: Optional[bool] = Form(True),  # Return base64 by default
    django_media_id: Optional[int] = Form(None),  # For Django reference only
    django_user_id: Optional[int] = Form(None),   # For Django reference only
    api_key: str = Depends(get_api_key)
):
    """
    Process a single image for object detection
    
    - **file**: Image file (JPEG, PNG)
    - **confidence_threshold**: Minimum confidence for detections (0.0 to 1.0)
    - **detection_types**: Comma-separated list of detection types (person, vehicle, face, etc.)
    - **return_image**: Whether to return processed image with detections drawn (legacy)
    - **image_format**: Format for returned image (jpeg, png)
    - **return_base64**: Whether to return base64 encoded processed image (default: True)
    - **django_media_id**: Optional ID of the MediaUpload in Django (passed through)
    - **django_user_id**: Optional ID of the User in Django (passed through)
    """
    try:
        # Get processor instance
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
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
        
        # Read file into memory
        file_content = await file.read()
        
        # Process image directly from bytes with return_base64 parameter
        result = processor.process_image_from_bytes(
            image_bytes=file_content,
            filename=file.filename or "uploaded_image.jpg",
            detection_types=parsed_detection_types,
            confidence_threshold=confidence_threshold,
            return_image=return_image,
            image_format=image_format,
            enable_advanced_features=True,
            return_base64=return_base64  # Pass the parameter
        )
        
        # Generate job ID for tracking
        job_id = f"img_{uuid.uuid4().hex[:8]}"
        result["job_id"] = job_id
        
        # Add Django references if provided
        if django_media_id:
            result["django_media_id"] = django_media_id
        if django_user_id:
            result["django_user_id"] = django_user_id
        
        # Log processing
        logger.info(
            f"Image processing completed: {result['detection_count']} detections "
            f"in {result['processing_time']}s for Django media ID: {django_media_id or 'N/A'}"
        )
        
        # Log if base64 image was returned
        if 'processed_image_base64' in result:
            logger.info(f"Base64 processed image returned: {len(result['processed_image_base64'])} chars")
        elif 'processed_image_url' in result:
            logger.info(f"Processed image URL included: {result['processed_image_url']}")
        else:
            logger.warning("No processed image returned")
            
        return ImageProcessingResponse(**result)
            
    except ProcessingException as e:
        logger.error(f"Processing error: {e.message}", extra=e.details)
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Unexpected error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Remove all database-related endpoints since Django handles that
@router.get("/models")
async def list_models(api_key: str = Depends(get_api_key)):
    """List available detection models"""
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        models = processor.get_available_models()
        
        return {
            "success": True,
            "models": models,
            "count": len(models),
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )

@router.post("/models/{model_name}/load")
async def load_model(model_name: str, api_key: str = Depends(get_api_key)):
    """Load a specific detection model"""
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        success = processor.load_model(model_name)
        
        if success:
            return {
                "success": True,
                "message": f"Model {model_name} loaded successfully",
                "model": model_name,
                "loaded": True,
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model {model_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )


# Remove video endpoint from this file - move it to separate video processing file
# Video processing should be handled separately with job queuing