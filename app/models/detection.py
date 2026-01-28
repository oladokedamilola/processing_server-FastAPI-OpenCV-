"""
Detection models and result handling
"""
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

# Define DetectionType locally to avoid circular import
class DetectionType(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    FACE = "face"
    MOTION = "motion"
    CROWD = "crowd"
    OBJECT = "object"

@dataclass
class Detection:
    """Detection result container"""
    label: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    detection_type: DetectionType
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        
        # Ensure bbox is valid
        if len(self.bbox) != 4:
            raise ValueError(f"Bounding box must have 4 values, got {len(self.bbox)}")
        
        # Ensure coordinates are integers
        self.bbox = [int(coord) for coord in self.bbox]
        
        # Add area calculation
        x1, y1, x2, y2 = self.bbox
        self.attributes["area"] = (x2 - x1) * (y2 - y1)
        
        # Add center point
        self.attributes["center"] = {
            "x": (x1 + x2) // 2,
            "y": (y1 + y2) // 2
        }

class DetectionModel:
    """Base class for all detection models"""
    
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self.loaded = False
        
    def load(self):
        """Load model into memory"""
        raise NotImplementedError
        
    def unload(self):
        """Unload model from memory"""
        raise NotImplementedError
        
    def detect(self, image: np.ndarray, **kwargs) -> List[Detection]:
        """Run detection on image"""
        raise NotImplementedError
        
    def is_loaded(self) -> bool:
        return self.loaded

class ModelManager:
    """Manager for loading and using detection models"""
    
    def __init__(self):
        self.models: Dict[str, DetectionModel] = {}
        self.active_model: Optional[str] = None
        
    def register_model(self, model: DetectionModel):
        """Register a new model"""
        self.models[model.name] = model
        
    def load_model(self, model_name: str):
        """Load a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if self.active_model and self.active_model != model_name:
            self.unload_model(self.active_model)
        
        self.models[model_name].load()
        self.active_model = model_name
        
    def unload_model(self, model_name: str):
        """Unload a specific model"""
        if model_name in self.models:
            self.models[model_name].unload()
            if self.active_model == model_name:
                self.active_model = None
                
    def get_active_model(self) -> Optional[DetectionModel]:
        """Get currently active model"""
        if self.active_model:
            return self.models[self.active_model]
        return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        return [
            {
                "name": name,
                "type": model.model_type,
                "loaded": model.is_loaded(),
                "active": name == self.active_model
            }
            for name, model in self.models.items()
        ]