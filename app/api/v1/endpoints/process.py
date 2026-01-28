"""
Image processing endpoints
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path

from ....core.config import settings
from ....core.security import get_api_key
from ....core.exceptions import ProcessingException
from ....core.processor import get_processor
from ....models.schemas import (
    ImageProcessingRequest, 
    ImageProcessingResponse,
    DetectionType,
    ProcessingResponse,
    ErrorResponse
)
from ....utils.file_handling import save_upload_file
from ....utils.logger import logger

router = APIRouter(prefix="/process", tags=["processing"])

@router.post("/image", response_model=ImageProcessingResponse)
async def process_image(
    file: UploadFile = File(..., description="Image file to process"),
    confidence_threshold: Optional[float] = Form(None),
    detection_types: Optional[str] = Form(None),
    return_image: Optional[bool] = Form(False),
    image_format: Optional[str] = Form("jpeg"),
    api_key: str = Depends(get_api_key)
):
    """
    Process a single image for object detection
    
    - **file**: Image file (JPEG, PNG)
    - **confidence_threshold**: Minimum confidence for detections (0.0 to 1.0)
    - **detection_types**: Comma-separated list of detection types (person, vehicle, face, etc.)
    - **return_image**: Whether to return processed image with detections drawn
    - **image_format**: Format for returned image (jpeg, png)
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
        
        # Save uploaded file
        image_path = await save_upload_file(file)
        
        # Process image
        result = processor.process_image(
            image_path=image_path,
            detection_types=parsed_detection_types,
            confidence_threshold=confidence_threshold,
            return_image=return_image,
            image_format=image_format
        )
        
        # Add job ID (for consistency with video processing)
        result["job_id"] = f"img_{image_path.stem}"
        
        logger.processing_log(
            job_id=result["job_id"],
            job_type="image",
            status="completed",
            details={
                "detection_count": result["detection_count"],
                "processing_time": result["processing_time"],
                "file_size": image_path.stat().st_size,
            }
        )
        
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

@router.post("/models/{model_name}/unload")
async def unload_model(model_name: str, api_key: str = Depends(get_api_key)):
    """Unload a specific detection model"""
    try:
        processor = get_processor()
        if processor is None:
            raise HTTPException(
                status_code=503,
                detail="Image processor not initialized"
            )
        
        processor.unload_model(model_name)
        
        return {
            "success": True,
            "message": f"Model {model_name} unloaded successfully",
            "model": model_name,
            "loaded": False,
        }
        
    except Exception as e:
        logger.error(f"Error unloading model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error unloading model: {str(e)}"
        )