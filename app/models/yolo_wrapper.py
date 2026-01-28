"""
YOLOv8 model wrapper for object detection
"""
import os
from typing import List, Optional, Dict, Any
import numpy as np
from ultralytics import YOLO
import torch
import cv2

from ..core.config import settings
from ..core.exceptions import ModelLoadingError
from ..utils.logger import logger
from .detection import Detection, DetectionModel, DetectionType

class YOLODetector(DetectionModel):
    """YOLOv8 object detector"""
    
    # YOLO class IDs to our detection types
    CLASS_MAPPING = {
        0: DetectionType.PERSON,      # person
        1: DetectionType.OBJECT,      # bicycle
        2: DetectionType.OBJECT,      # car
        3: DetectionType.OBJECT,      # motorcycle
        5: DetectionType.OBJECT,      # bus
        7: DetectionType.OBJECT,      # truck
        # Add more mappings as needed
    }
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        super().__init__(name="yolov8", model_type="object_detection")
        self.model_name = model_name
        self.model_path = settings.MODELS_PATH / model_name
        self.model: Optional[YOLO] = None
        self.offline_mode = False
        
        # Default classes to detect (person, car, truck, bus, motorcycle, bicycle)
        self.classes = [0, 1, 2, 3, 5, 7]
        
    def load(self):
        """Load YOLO model - try online download first, fallback to offline if fails"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            
            # Check if model already exists locally
            if self.model_path.exists():
                logger.info(f"Loading YOLO model from local file: {self.model_path}")
                self.model = YOLO(str(self.model_path))
                self.offline_mode = False
            else:
                logger.info(f"Model not found locally, attempting download: {self.model_name}")
                try:
                    # Try to download the model
                    self.model = YOLO(self.model_name)
                    # Save for future use
                    self.model.save(self.model_path)
                    self.offline_mode = False
                    logger.info(f"YOLO model downloaded and saved: {self.model_path}")
                except Exception as download_error:
                    logger.warning(f"Online download failed: {download_error}")
                    logger.info("Falling back to offline mode - YOLO will be unavailable")
                    self.offline_mode = True
                    self.loaded = False
                    return
            
            # Move to CPU (since we're on free tier without GPU)
            if torch.cuda.is_available():
                logger.info("CUDA available, moving model to GPU")
                self.model.to("cuda")
            else:
                logger.info("Using CPU for inference")
                self.model.to("cpu")
            
            # Warm up the model with a dummy inference
            if not self.offline_mode:
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy_image, verbose=False)
            
            self.loaded = True
            logger.info(f"YOLO model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            self.offline_mode = True
            self.loaded = False
            raise ModelLoadingError(
                message=f"Failed to load YOLO model: {str(e)}",
                details={
                    "model_name": self.model_name,
                    "offline_mode": True,
                    "suggestion": "Download the model manually or use traditional CV methods"
                }
            )
    
    def unload(self):
        """Unload model from memory"""
        if self.model and not self.offline_mode:
            # Clear model from memory
            del self.model
            self.model = None
        self.loaded = False
        self.offline_mode = False
        logger.info("YOLO model unloaded")
    
    def detect(
        self, 
        image: np.ndarray, 
        confidence_threshold: float = None,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """Run object detection on image"""
        if not self.loaded or self.offline_mode:
            logger.warning("YOLO model not available (offline mode). Returning empty detections.")
            return []
        
        try:
            # Use provided threshold or default
            conf = confidence_threshold or settings.CONFIDENCE_THRESHOLD
            
            # Use provided classes or default
            detect_classes = classes or self.classes
            
            # Run inference
            results = self.model(
                image, 
                conf=conf,
                classes=detect_classes,
                verbose=False,
                device="cpu"  # Force CPU for free tier compatibility
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                    
                    for box, confidence, class_id in zip(boxes, confidences, class_ids):
                        # Convert box coordinates to integers
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Get label from class ID
                        label = self.model.names[class_id]
                        
                        # Map to our detection type
                        detection_type = self.CLASS_MAPPING.get(
                            class_id, 
                            DetectionType.OBJECT
                        )
                        
                        # Create detection object
                        detection = Detection(
                            label=label,
                            confidence=float(confidence),
                            bbox=[x1, y1, x2, y2],
                            detection_type=detection_type,
                            attributes={
                                "class_id": int(class_id),
                                "model": "yolov8",
                                "model_version": self.model_name,
                            }
                        )
                        
                        detections.append(detection)
            
            logger.debug(f"YOLO detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO detection: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model or self.offline_mode:
            return {
                "loaded": False,
                "offline_mode": True,
                "available": False,
                "message": "YOLO model not available. Download manually or use traditional CV methods."
            }
        
        return {
            "loaded": True,
            "offline_mode": False,
            "available": True,
            "name": self.model_name,
            "classes": len(self.model.names),
            "class_names": self.model.names,
            "device": next(self.model.model.parameters()).device,
        }
    
    def is_available(self) -> bool:
        """Check if YOLO is available (not in offline mode)"""
        return self.loaded and not self.offline_mode