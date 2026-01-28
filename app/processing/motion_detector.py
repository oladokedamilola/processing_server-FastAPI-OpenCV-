"""
Motion detection using frame differencing and background subtraction
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ..models.detection import Detection, DetectionType

@dataclass
class MotionConfig:
    """Configuration for motion detection"""
    min_area: int = 500  # Minimum contour area to consider as motion
    threshold_value: int = 25  # Threshold for binary image
    blur_size: Tuple[int, int] = (21, 21)  # Gaussian blur kernel size
    history: int = 500  # Background subtractor history
    var_threshold: int = 16  # Background subtractor variance threshold
    detect_shadows: bool = True  # Whether to detect shadows

class MotionDetector:
    """Motion detection using multiple methods"""
    
    def __init__(self, config: MotionConfig = None):
        self.config = config or MotionConfig()
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows
        )
        
        # Store previous frame for frame differencing
        self.previous_frame = None
        
    def detect_with_background_subtraction(
        self, 
        frame: np.ndarray
    ) -> List[Detection]:
        """Detect motion using background subtraction"""
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Remove shadows (gray pixels in mask)
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            detections = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > self.config.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate motion intensity (normalized area)
                    motion_intensity = min(area / (frame.shape[0] * frame.shape[1]), 1.0)
                    
                    detection = Detection(
                        label="motion",
                        confidence=motion_intensity,
                        bbox=[x, y, x + w, y + h],
                        detection_type=DetectionType.MOTION,
                        attributes={
                            "area": int(area),
                            "method": "background_subtraction",
                            "contour_points": len(contour),
                            "motion_intensity": motion_intensity,
                        }
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            raise RuntimeError(f"Error in background subtraction: {str(e)}")
    
    def detect_with_frame_differencing(
        self, 
        frame: np.ndarray
    ) -> List[Detection]:
        """Detect motion using frame differencing"""
        try:
            if self.previous_frame is None:
                self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            frame_diff = cv2.absdiff(self.previous_frame, gray)
            
            # Apply threshold
            _, thresh = cv2.threshold(
                frame_diff, 
                self.config.threshold_value, 
                255, 
                cv2.THRESH_BINARY
            )
            
            # Apply blur to smooth the image
            thresh = cv2.GaussianBlur(thresh, self.config.blur_size, 0)
            
            # Apply threshold again
            _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            detections = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > self.config.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate motion intensity
                    motion_intensity = min(area / (frame.shape[0] * frame.shape[1]), 1.0)
                    
                    detection = Detection(
                        label="motion",
                        confidence=motion_intensity,
                        bbox=[x, y, x + w, y + h],
                        detection_type=DetectionType.MOTION,
                        attributes={
                            "area": int(area),
                            "method": "frame_differencing",
                            "contour_points": len(contour),
                            "motion_intensity": motion_intensity,
                        }
                    )
                    detections.append(detection)
            
            # Update previous frame
            self.previous_frame = gray.copy()
            
            return detections
            
        except Exception as e:
            raise RuntimeError(f"Error in frame differencing: {str(e)}")
    
    def detect(
        self, 
        frame: np.ndarray, 
        method: str = "background_subtraction"
    ) -> List[Detection]:
        """Detect motion using specified method"""
        if method == "background_subtraction":
            return self.detect_with_background_subtraction(frame)
        elif method == "frame_differencing":
            return self.detect_with_frame_differencing(frame)
        else:
            raise ValueError(f"Unknown motion detection method: {method}")
    
    def reset_background(self):
        """Reset background model"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows
        )
        self.previous_frame = None
    
    def get_background_image(self) -> Optional[np.ndarray]:
        """Get current background image from subtractor"""
        try:
            return self.bg_subtractor.getBackgroundImage()
        except:
            return None