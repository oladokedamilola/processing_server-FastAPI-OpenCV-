"""
Traditional computer vision methods for detection
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
import os

from ..core.config import settings
from ..utils.logger import logger
from .detection import Detection, DetectionModel, DetectionType

class HaarCascadeDetector(DetectionModel):
    """Face detection using Haar Cascade"""
    
    def __init__(self):
        super().__init__(name="haar_cascade", model_type="face_detection")
        self.face_cascade = None
        self.cascade_path = None
        
    def load(self):
        """Load Haar Cascade classifier"""
        try:
            # Try to find cascade file
            cascade_files = [
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
                cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
                cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
            ]
            
            for cascade_file in cascade_files:
                if os.path.exists(cascade_file):
                    self.cascade_path = cascade_file
                    break
            
            if not self.cascade_path:
                raise FileNotFoundError("No Haar Cascade file found")
            
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            
            # Test loading
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load Haar Cascade classifier")
            
            self.loaded = True
            logger.info(f"Haar Cascade loaded: {self.cascade_path}")
            
        except Exception as e:
            logger.error(f"Error loading Haar Cascade: {str(e)}")
            self.loaded = False
            raise
    
    def unload(self):
        """Unload classifier"""
        self.face_cascade = None
        self.loaded = False
        logger.info("Haar Cascade unloaded")
    
    def detect(
        self, 
        image: np.ndarray, 
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[Detection]:
        """Detect faces in image"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            
            detections = []
            
            for (x, y, w, h) in faces:
                detection = Detection(
                    label="face",
                    confidence=0.9,  # Haar Cascade doesn't provide confidence
                    bbox=[x, y, x + w, y + h],
                    detection_type=DetectionType.FACE,
                    attributes={
                        "width": w,
                        "height": h,
                        "method": "haar_cascade",
                        "scale_factor": scale_factor,
                        "min_neighbors": min_neighbors,
                    }
                )
                detections.append(detection)
            
            logger.debug(f"Haar Cascade detected {len(detections)} faces")
            return detections
            
        except Exception as e:
            logger.error(f"Error during Haar Cascade detection: {str(e)}")
            raise

class HOGDetector(DetectionModel):
    """People detection using HOG descriptor"""
    
    def __init__(self):
        super().__init__(name="hog", model_type="people_detection")
        self.hog = None
        
    def load(self):
        """Initialize HOG descriptor for people detection"""
        try:
            # Initialize HOG descriptor for people detection
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            self.loaded = True
            logger.info("HOG descriptor loaded for people detection")
            
        except Exception as e:
            logger.error(f"Error loading HOG descriptor: {str(e)}")
            self.loaded = False
            raise
    
    def unload(self):
        """Unload HOG descriptor"""
        self.hog = None
        self.loaded = False
        logger.info("HOG descriptor unloaded")
    
    def detect(
        self, 
        image: np.ndarray,
        win_stride: Tuple[int, int] = (8, 8),
        padding: Tuple[int, int] = (8, 8),
        scale: float = 1.05
    ) -> List[Detection]:
        """Detect people in image using HOG"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Detect people
            (rects, weights) = self.hog.detectMultiScale(
                image,
                winStride=win_stride,
                padding=padding,
                scale=scale
            )
            
            detections = []
            
            for i, (x, y, w, h) in enumerate(rects):
                confidence = float(weights[i][0]) if len(weights) > i else 0.5
                
                detection = Detection(
                    label="person",
                    confidence=confidence,
                    bbox=[x, y, x + w, y + h],
                    detection_type=DetectionType.PERSON,
                    attributes={
                        "width": w,
                        "height": h,
                        "method": "hog",
                        "win_stride": win_stride,
                        "padding": padding,
                        "scale": scale,
                    }
                )
                detections.append(detection)
            
            logger.debug(f"HOG detected {len(detections)} people")
            return detections
            
        except Exception as e:
            logger.error(f"Error during HOG detection: {str(e)}")
            raise