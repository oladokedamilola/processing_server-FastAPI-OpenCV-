# app/models/yolo_wrapper.py
"""
YOLOv8 model wrapper for object detection
"""
# ============================================================================
# PATCH PYTORCH 2.6 FOR YOLO COMPATIBILITY - MUST BE FIRST!
# ============================================================================
import sys
import os

# Apply patch before ANY other imports
try:
    import torch
    
    # Monkey-patch torch.load globally
    original_torch_load = torch.load
    
    def patched_torch_load(f, *args, **kwargs):
        """Patch torch.load to use weights_only=False for YOLO models"""
        # Check if it's likely a YOLO model file
        if isinstance(f, (str, os.PathLike)) and str(f).endswith('.pt'):
            kwargs['weights_only'] = False
        elif 'weights_only' not in kwargs:
            # Default to False for safety with YOLO
            kwargs['weights_only'] = False
        return original_torch_load(f, *args, **kwargs)
    
    torch.load = patched_torch_load
    print("Applied global torch.load patch for PyTorch 2.6 YOLO compatibility")
    
    # Also try to add safe globals
    try:
        # Add common torch modules
        torch.serialization.add_safe_globals([
            torch.nn.modules.container.Sequential,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.dropout.Dropout,
        ])
        
        # Try to add Ultralytics modules
        try:
            from ultralytics.nn.modules import Conv, Bottleneck, C2f, SPPF, Concat, Detect
            from ultralytics.nn.tasks import DetectionModel
            
            torch.serialization.add_safe_globals([
                Conv, Bottleneck, C2f, SPPF, Concat, Detect, DetectionModel
            ])
            print("Added Ultralytics modules to safe globals")
        except ImportError:
            # Try dynamic import
            try:
                import importlib
                modules_to_try = ['Conv', 'Bottleneck', 'C2f', 'SPPF', 'Concat', 'Detect']
                for module_name in modules_to_try:
                    try:
                        ultralytics_modules = importlib.import_module('ultralytics.nn.modules')
                        module_class = getattr(ultralytics_modules, module_name, None)
                        if module_class:
                            torch.serialization.add_safe_globals([module_class])
                    except:
                        pass
                
                # Try to add DetectionModel
                try:
                    ultralytics_tasks = importlib.import_module('ultralytics.nn.tasks')
                    detection_model = getattr(ultralytics_tasks, 'DetectionModel', None)
                    if detection_model:
                        torch.serialization.add_safe_globals([detection_model])
                except:
                    pass
            except:
                pass
        
    except Exception as safe_globals_error:
        print(f"Safe globals setup failed (non-critical): {safe_globals_error}")
        
except Exception as patch_error:
    print(f"CRITICAL: Failed to apply PyTorch patch: {patch_error}")
    print("YOLO model loading will likely fail with PyTorch 2.6")
    sys.exit(1)

# ============================================================================
# NOW IMPORT OTHER MODULES
# ============================================================================
from typing import List, Optional, Dict, Any
import numpy as np
from ultralytics import YOLO
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
            
            # Create safe globals context with all known classes
            safe_classes = []
            
            # Try to collect all known Ultralytics modules
            try:
                # Import common ultralytics modules
                from ultralytics.nn.modules import Conv, Bottleneck, C2f, SPPF, Concat, Detect
                safe_classes.extend([Conv, Bottleneck, C2f, SPPF, Concat, Detect])
            except ImportError as e:
                logger.warning(f"Could not import some Ultralytics modules: {e}")
            
            try:
                from ultralytics.nn.tasks import DetectionModel
                safe_classes.append(DetectionModel)
            except ImportError:
                pass
            
            # Add torch modules
            safe_classes.extend([
                torch.nn.modules.container.Sequential,
                torch.nn.modules.conv.Conv2d,
                torch.nn.modules.batchnorm.BatchNorm2d,
                torch.nn.modules.activation.SiLU,
                torch.nn.modules.activation.ReLU,
                torch.nn.modules.pooling.MaxPool2d,
                torch.nn.modules.linear.Linear,
                torch.nn.modules.dropout.Dropout,
            ])
            
            # Load with safe globals
            with torch.serialization.safe_globals(safe_classes):
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