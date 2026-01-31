# processing_server/app/api/v1/endpoints/process.py
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

def process_image_from_bytes(self, image_bytes: bytes, filename: str = None, **kwargs):
    """
    Process image from bytes instead of file path.
    """
    import tempfile
    from pathlib import Path
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix if filename else '.jpg', delete=False) as tmp:
        # Write bytes to temp file
        tmp.write(image_bytes)
        tmp_path = tmp.name
    
    try:
        # Process using existing method
        result = self.process_image(image_path=tmp_path, **kwargs)
        return result
    finally:
        # Clean up temp file
        import os
        try:
            os.unlink(tmp_path)
        except:
            pass


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
    import tempfile
    import os
    import uuid
    from pathlib import Path
    
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
        
        # Create temporary file path
        temp_filepath = None
        
        try:
            # Create temp file
            suffix = Path(file.filename).suffix if file.filename else '.jpg'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_content)
                temp_filepath = tmp.name
            
            # Convert to Path object - THIS IS THE FIX!
            image_path_obj = Path(temp_filepath)
            
            # Process image using existing method
            result = processor.process_image_from_bytes(
                image_bytes=file_content,
                filename=file.filename,
                detection_types=parsed_detection_types,
                confidence_threshold=confidence_threshold,
                return_image=return_image,
                image_format=image_format
            )
            
            # Add job ID
            result["job_id"] = f"img_{uuid.uuid4().hex[:8]}"
            
            logger.processing_log(
                job_id=result["job_id"],
                job_type="image",
                status="completed",
                details={
                    "detection_count": result["detection_count"],
                    "processing_time": result["processing_time"],
                    "file_size": len(file_content),
                }
            )
            
            return ImageProcessingResponse(**result)
            
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