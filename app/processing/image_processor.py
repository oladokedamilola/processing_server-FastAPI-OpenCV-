"""
Image processing pipeline for detection tasks
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from ..core.config import settings
from ..core.exceptions import ProcessingException
from ..utils.logger import logger
from ..models.detection import Detection, DetectionModel, ModelManager, DetectionType
from ..models.yolo_wrapper import YOLODetector
from ..models.traditional_cv import HaarCascadeDetector, HOGDetector
from ..models.advanced_detection import CrowdDetector, VehicleCounter
from ..processing.motion_detector import MotionDetector

class ImageProcessor:
    """Main image processing class"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.motion_detector = MotionDetector()
        self.crowd_detector = CrowdDetector()
        self.vehicle_counter = VehicleCounter()
        self.yolo_available = True
        
        # Statistics
        self.processing_stats = {
            "total_images_processed": 0,
            "total_detections": 0,
            "processing_times": []
        }
        
        # Register available models
        self._register_models()
        
        # Load default model
        self._load_default_model()
    
    def _register_models(self):
        """Register all available detection models"""
        try:
            # YOLO for general object detection
            yolo_detector = YOLODetector(settings.YOLO_MODEL)
            self.model_manager.register_model(yolo_detector)
            logger.info("YOLO model registered")
        except Exception as e:
            logger.warning(f"Failed to register YOLO model: {str(e)}")
            self.yolo_available = False
        
        # Traditional CV methods (always available)
        try:
            haar_detector = HaarCascadeDetector()
            hog_detector = HOGDetector()
            
            self.model_manager.register_model(haar_detector)
            self.model_manager.register_model(hog_detector)
            
            logger.info("Traditional CV models registered")
        except Exception as e:
            logger.error(f"Failed to register traditional CV models: {str(e)}")
        
        logger.info(f"Registered {len(self.model_manager.models)} detection models")
    
    def _load_default_model(self):
        """Load the default detection model"""
        try:
            # Try to load YOLO first
            if "yolov8" in self.model_manager.models:
                self.model_manager.load_model("yolov8")
                yolo_model = self.model_manager.models["yolov8"]
                if hasattr(yolo_model, 'is_available') and yolo_model.is_available():
                    logger.info("Default model (YOLOv8) loaded successfully")
                else:
                    logger.warning("YOLO model registered but not available (offline mode)")
                    # Load HOG as fallback
                    if "hog" in self.model_manager.models:
                        self.model_manager.load_model("hog")
                        logger.info("Fallback model (HOG) loaded for people detection")
            else:
                # Load HOG if YOLO not available
                if "hog" in self.model_manager.models:
                    self.model_manager.load_model("hog")
                    logger.info("Default model (HOG) loaded for people detection")
                    
        except Exception as e:
            logger.error(f"Failed to load default model: {str(e)}")
            # Try to load HOG as fallback
            try:
                if "hog" in self.model_manager.models:
                    self.model_manager.load_model("hog")
                    logger.info("Fallback model (HOG) loaded after error")
            except:
                logger.error("No detection models available")
    
    def process_image(
        self,
        image_path: Path,
        detection_types: List[DetectionType] = None,
        confidence_threshold: float = None,
        return_image: bool = False,
        image_format: str = "jpeg",
        enable_advanced_features: bool = True
    ) -> Dict[str, Any]:
        """Process a single image with multiple detection methods"""
        start_time = time.time()
        
        try:
            # Validate image path
            if not image_path.exists():
                raise ProcessingException(
                    message="Image file not found",
                    details={"image_path": str(image_path)}
                )
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ProcessingException(
                    message="Unable to load image file",
                    details={"image_path": str(image_path)}
                )
            
            # Get image info
            height, width = image.shape[:2]
            image_size = f"{width}x{height}"
            
            # Default detection types
            if detection_types is None:
                detection_types = [DetectionType.PERSON, DetectionType.VEHICLE]
            
            # Run detections
            all_detections = []
            warnings = []
            advanced_results = {}
            
            # Check if YOLO is available
            yolo_model = self.model_manager.models.get("yolov8")
            yolo_available = yolo_model and yolo_model.loaded and hasattr(yolo_model, 'is_available') and yolo_model.is_available()
            
            # Get person and vehicle detections (for advanced features)
            person_detections = []
            vehicle_detections = []
            
            # Use YOLO for object detection if available
            if yolo_available and any(dt in detection_types for dt in [DetectionType.PERSON, DetectionType.VEHICLE, DetectionType.OBJECT]):
                yolo_detections = self._detect_with_yolo(
                    image, 
                    confidence_threshold,
                    detection_types
                )
                all_detections.extend(yolo_detections)
                
                # Separate person and vehicle detections for advanced processing
                person_detections = [d for d in yolo_detections if d.detection_type == DetectionType.PERSON]
                vehicle_detections = [d for d in yolo_detections if d.detection_type == DetectionType.VEHICLE]
            elif any(dt in detection_types for dt in [DetectionType.PERSON, DetectionType.VEHICLE, DetectionType.OBJECT]):
                warnings.append("YOLO model not available for object detection. Using traditional methods only.")
            
            # Use Haar Cascade for face detection
            if DetectionType.FACE in detection_types:
                face_detections = self._detect_faces(image)
                all_detections.extend(face_detections)
            
            # Use HOG for people detection (alternative to YOLO)
            if DetectionType.PERSON in detection_types and not yolo_available:
                hog_detections = self._detect_people_hog(image)
                all_detections.extend(hog_detections)
                person_detections.extend(hog_detections)
            
            # Advanced features
            if enable_advanced_features:
                # Crowd detection
                if DetectionType.CROWD in detection_types and person_detections:
                    crowd_detections = self.crowd_detector.detect_crowd(
                        person_detections=person_detections,
                        image_shape=image.shape[:2]
                    )
                    all_detections.extend(crowd_detections)
                    
                    # Add crowd statistics to advanced results
                    if crowd_detections:
                        crowd_stats = self.crowd_detector.get_crowd_statistics()
                        advanced_results["crowd_statistics"] = crowd_stats
                
                # Vehicle counting (for single images, just count vehicles)
                if DetectionType.VEHICLE in detection_types:
                    vehicle_count = len(vehicle_detections)
                    advanced_results["vehicle_count"] = vehicle_count
            
            # Motion detection (for static images, use frame differencing with previous)
            if DetectionType.MOTION in detection_types:
                motion_detections = self.motion_detector.detect(image, method="frame_differencing")
                all_detections.extend(motion_detections)
            
            # Process image with detections drawn (if requested)
            processed_image_path = None
            if return_image:
                processed_image = self._draw_detections(image, all_detections)
                processed_image_path = self._save_processed_image(
                    processed_image, image_path, image_format
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_statistics(processing_time, len(all_detections))
            
            # Prepare response
            response = {
                "success": True,
                "processing_time": round(processing_time, 3),
                "detections": [self._detection_to_dict(d) for d in all_detections],
                "detection_count": len(all_detections),
                "image_size": image_size,
                "detection_summary": self._get_detection_summary(all_detections),
                "advanced_results": advanced_results if advanced_results else None,
                "warnings": warnings,
                "models_used": self._get_models_used(yolo_available, detection_types, enable_advanced_features),
                "statistics": self._get_processing_statistics()
            }
            
            if processed_image_path:
                response["processed_image_url"] = f"/processed/{processed_image_path.name}"
            
            logger.info(f"Image processed successfully: {len(all_detections)} detections in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def process_video_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        detection_types: List[DetectionType] = None,
        confidence_threshold: float = None,
        enable_advanced_features: bool = True
    ) -> Dict[str, Any]:
        """Process a single video frame with tracking and counting"""
        try:
            # Default detection types
            if detection_types is None:
                detection_types = [DetectionType.PERSON, DetectionType.VEHICLE, DetectionType.MOTION]
            
            # Run basic detections
            all_detections = []
            person_detections = []
            vehicle_detections = []
            
            # Check if YOLO is available
            yolo_model = self.model_manager.models.get("yolov8")
            yolo_available = yolo_model and yolo_model.loaded and hasattr(yolo_model, 'is_available') and yolo_model.is_available()
            
            # Use YOLO for object detection if available
            if yolo_available:
                yolo_detections = self._detect_with_yolo(
                    frame, 
                    confidence_threshold,
                    detection_types
                )
                all_detections.extend(yolo_detections)
                
                # Separate person and vehicle detections
                person_detections = [d for d in yolo_detections if d.detection_type == DetectionType.PERSON]
                vehicle_detections = [d for d in yolo_detections if d.detection_type == DetectionType.VEHICLE]
            
            # Motion detection
            if DetectionType.MOTION in detection_types:
                motion_detections = self.motion_detector.detect(frame, method="background_subtraction")
                all_detections.extend(motion_detections)
            
            # Advanced features
            advanced_results = {}
            
            if enable_advanced_features:
                # Crowd detection
                if DetectionType.CROWD in detection_types and person_detections:
                    crowd_detections = self.crowd_detector.detect_crowd(
                        person_detections=person_detections,
                        image_shape=frame.shape[:2]
                    )
                    all_detections.extend(crowd_detections)
                
                # Vehicle counting (with tracking)
                if DetectionType.VEHICLE in detection_types and vehicle_detections:
                    counting_results = self.vehicle_counter.count_vehicles(
                        vehicle_detections=vehicle_detections,
                        frame_number=frame_number,
                        image_width=frame.shape[1]
                    )
                    advanced_results["vehicle_counting"] = counting_results
            
            return {
                "frame_number": frame_number,
                "detections": [self._detection_to_dict(d) for d in all_detections],
                "detection_count": len(all_detections),
                "person_count": len(person_detections),
                "vehicle_count": len(vehicle_detections),
                "advanced_results": advanced_results if advanced_results else None
            }
            
        except Exception as e:
            logger.error(f"Error processing video frame {frame_number}: {str(e)}")
            return {
                "frame_number": frame_number,
                "detections": [],
                "detection_count": 0,
                "error": str(e)
            }
    
    def _detect_with_yolo(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        detection_types: List[DetectionType]
    ) -> List[Detection]:
        """Detect objects using YOLO"""
        try:
            model = self.model_manager.get_active_model()
            if not model or model.name != "yolov8":
                # Load YOLO if not active
                self.model_manager.load_model("yolov8")
                model = self.model_manager.get_active_model()
            
            # Check if YOLO is actually available
            if not hasattr(model, 'is_available') or not model.is_available():
                return []
            
            # Map detection types to YOLO class IDs
            class_mapping = {
                DetectionType.PERSON: [0],
                DetectionType.VEHICLE: [1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
                DetectionType.OBJECT: list(range(0, 80)),  # All COCO classes
            }
            
            # Get class IDs for requested detection types
            class_ids = []
            for dt in detection_types:
                if dt in class_mapping:
                    class_ids.extend(class_mapping[dt])
            
            # Remove duplicates
            class_ids = list(set(class_ids))
            
            # Run detection
            return model.detect(
                image, 
                confidence_threshold=confidence_threshold,
                classes=class_ids if class_ids else None
            )
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}")
            return []
    
    def _detect_faces(self, image: np.ndarray) -> List[Detection]:
        """Detect faces using Haar Cascade"""
        try:
            model = self.model_manager.models.get("haar_cascade")
            if not model:
                return []
            
            if not model.loaded:
                model.load()
            
            return model.detect(image)
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def _detect_people_hog(self, image: np.ndarray) -> List[Detection]:
        """Detect people using HOG"""
        try:
            model = self.model_manager.models.get("hog")
            if not model:
                return []
            
            if not model.loaded:
                model.load()
            
            return model.detect(image)
            
        except Exception as e:
            logger.error(f"HOG detection failed: {str(e)}")
            return []
    
    def _draw_detections(
        self, 
        image: np.ndarray, 
        detections: List[Detection]
    ) -> np.ndarray:
        """Draw detection boxes on image"""
        # Create a copy of the image
        result_image = image.copy()
        
        # Colors for different detection types
        colors = {
            DetectionType.PERSON: (0, 255, 0),  # Green
            DetectionType.VEHICLE: (255, 0, 0),  # Blue
            DetectionType.FACE: (0, 255, 255),  # Yellow
            DetectionType.MOTION: (0, 0, 255),  # Red
            DetectionType.OBJECT: (255, 255, 0),  # Cyan
            DetectionType.CROWD: (255, 0, 255),  # Magenta
        }
        
        for detection in detections:
            # Get color for this detection type
            color = colors.get(detection.detection_type, (255, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{detection.label}: {detection.confidence:.2f}"
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                result_image, 
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1
            )
        
        return result_image
    
    def _save_processed_image(
        self, 
        image: np.ndarray, 
        original_path: Path,
        image_format: str = "jpeg"
    ) -> Path:
        """Save processed image to disk"""
        try:
            # Generate unique filename
            import uuid
            filename = f"processed_{uuid.uuid4()}.{image_format}"
            output_path = settings.PROCESSED_PATH / "images" / filename
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            if image_format.lower() in ["jpg", "jpeg"]:
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            elif image_format.lower() == "png":
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            else:
                # Default to JPEG
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            logger.info(f"Processed image saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save processed image: {str(e)}")
            raise
    
    def _detection_to_dict(self, detection: Detection) -> Dict[str, Any]:
        """Convert Detection object to dictionary"""
        return {
            "label": detection.label,
            "confidence": round(detection.confidence, 3),
            "bbox": detection.bbox,
            "type": detection.detection_type.value,
            "attributes": detection.attributes,
        }
    
    def _get_detection_summary(self, detections: List[Detection]) -> Dict[str, int]:
        """Get summary of detections by type"""
        summary = {}
        for detection in detections:
            det_type = detection.detection_type.value
            summary[det_type] = summary.get(det_type, 0) + 1
        return summary
    
    def _get_models_used(self, yolo_available: bool, detection_types: List[DetectionType], enable_advanced_features: bool = True) -> List[str]:
        """Get list of models used for processing"""
        models_used = []
        
        if yolo_available and any(dt in detection_types for dt in [DetectionType.PERSON, DetectionType.VEHICLE, DetectionType.OBJECT]):
            models_used.append("yolov8")
        
        if DetectionType.FACE in detection_types:
            models_used.append("haar_cascade")
        
        if DetectionType.PERSON in detection_types and not yolo_available:
            models_used.append("hog")
        
        if enable_advanced_features and DetectionType.CROWD in detection_types:
            models_used.append("crowd_detector")
        
        return models_used
    
    def _update_statistics(self, processing_time: float, detection_count: int):
        """Update processing statistics"""
        self.processing_stats["total_images_processed"] += 1
        self.processing_stats["total_detections"] += detection_count
        self.processing_stats["processing_times"].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.processing_stats["processing_times"]) > 100:
            self.processing_stats["processing_times"] = self.processing_stats["processing_times"][-100:]
    
    def _get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        processing_times = self.processing_stats["processing_times"]
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
        else:
            avg_time = max_time = min_time = 0
        
        return {
            "total_images_processed": self.processing_stats["total_images_processed"],
            "total_detections": self.processing_stats["total_detections"],
            "average_processing_time": round(avg_time, 3),
            "max_processing_time": round(max_time, 3),
            "min_processing_time": round(min_time, 3),
            "detections_per_image": round(
                self.processing_stats["total_detections"] / 
                max(1, self.processing_stats["total_images_processed"]), 
                2
            )
        }
    
    def get_crowd_statistics(self) -> Dict[str, Any]:
        """Get crowd detection statistics"""
        return self.crowd_detector.get_crowd_statistics()
    
    def get_vehicle_counting_statistics(self) -> Dict[str, Any]:
        """Get vehicle counting statistics"""
        return self.vehicle_counter.get_statistics()
    
    def reset_vehicle_counts(self):
        """Reset vehicle counts"""
        self.vehicle_counter.reset_counts()
        logger.info("Vehicle counts reset")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return self.model_manager.list_models()
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        try:
            self.model_manager.load_model(model_name)
            logger.info(f"Model loaded: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def unload_model(self, model_name: str):
        """Unload a specific model"""
        try:
            self.model_manager.unload_model(model_name)
            logger.info(f"Model unloaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {str(e)}")
            raise