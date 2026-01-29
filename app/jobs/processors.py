# app/jobs/processors.py
"""
Job processor functions for different job types
"""
import asyncio
from typing import Dict, Any, Callable
from pathlib import Path
import tempfile

from ..core.config import settings
from ..utils.logger import logger
from ..processing.image_processor import ImageProcessor
from ..processing.video_processor import VideoProcessor
from ..utils.file_handling import save_upload_file
from .models import JobType

class JobProcessors:
    """Collection of job processor functions"""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor(self.image_processor)
    
    def get_image_processor(self, parameters: Dict[str, Any], progress_callback: Callable) -> Dict[str, Any]:
        """Process an image processing job"""
        try:
            # Extract parameters
            file_path = Path(parameters["file_path"])
            detection_types = parameters.get("detection_types")
            confidence_threshold = parameters.get("confidence_threshold")
            return_image = parameters.get("return_image", False)
            image_format = parameters.get("image_format", "jpeg")
            
            # Update progress
            progress_callback(10, {"stage": "loading_image"})
            
            # Process image
            result = self.image_processor.process_image(
                image_path=file_path,
                detection_types=detection_types,
                confidence_threshold=confidence_threshold,
                return_image=return_image,
                image_format=image_format
            )
            
            # Update progress
            progress_callback(100, {"stage": "completed"})
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing job failed: {str(e)}")
            raise
    
    def get_video_processor(self, parameters: Dict[str, Any], progress_callback: Callable) -> Dict[str, Any]:
        """Process a video processing job with enhanced format handling"""
        try:
            # Extract parameters
            file_path = Path(parameters["file_path"])
            detection_types = parameters.get("detection_types")
            confidence_threshold = parameters.get("confidence_threshold")
            frame_sample_rate = parameters.get("frame_sample_rate", settings.VIDEO_FRAME_SAMPLE_RATE)
            analyze_motion = parameters.get("analyze_motion", True)
            return_summary_only = parameters.get("return_summary_only", False)
            enable_advanced_features = parameters.get("enable_advanced_features", True)
            force_conversion = parameters.get("force_conversion", False)
            
            # Update progress
            progress_callback(5, {"stage": "validating_video"})
            
            # Process video with enhanced format handling
            result = self.video_processor.process_video_with_format_handling(
                video_path=file_path,
                progress_callback=lambda prog, extra: progress_callback(
                    5 + (prog * 0.9),  # 5-95% for video processing
                    {"stage": "processing_video", **extra}
                ),
                confidence_threshold=confidence_threshold,
                detection_types=detection_types,
                frame_sample_rate=frame_sample_rate,
                analyze_motion=analyze_motion,
                return_summary_only=return_summary_only,
                enable_advanced_features=enable_advanced_features,
                force_conversion=force_conversion
            )
            
            # Update progress
            progress_callback(100, {"stage": "completed"})
            
            return result
            
        except Exception as e:
            logger.error(f"Video processing job failed: {str(e)}")
            raise
    
    def get_batch_processor(self, parameters: Dict[str, Any], progress_callback: Callable) -> Dict[str, Any]:
        """Process a batch of images/videos"""
        try:
            # Extract parameters
            file_paths = [Path(p) for p in parameters["file_paths"]]
            job_type = parameters.get("batch_type", "images")  # "images" or "videos"
            detection_types = parameters.get("detection_types")
            confidence_threshold = parameters.get("confidence_threshold")
            
            total_files = len(file_paths)
            results = []
            
            for i, file_path in enumerate(file_paths):
                # Calculate progress
                file_progress = (i / total_files) * 100
                progress_callback(
                    file_progress,
                    {
                        "stage": f"processing_{job_type[:-1]}",  # "image" or "video"
                        "current_file": i + 1,
                        "total_files": total_files,
                        "filename": file_path.name
                    }
                )
                
                try:
                    if job_type == "images":
                        result = self.image_processor.process_image(
                            image_path=file_path,
                            detection_types=detection_types,
                            confidence_threshold=confidence_threshold,
                            return_image=False
                        )
                    else:  # videos
                        result = self.video_processor.process_video(
                            video_path=file_path,
                            confidence_threshold=confidence_threshold,
                            detection_types=detection_types,
                            return_summary_only=True
                        )
                    
                    result["filename"] = file_path.name
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    results.append({
                        "filename": file_path.name,
                        "success": False,
                        "error": str(e)
                    })
            
            # Generate batch summary
            successful = sum(1 for r in results if r.get("success", False))
            failed = total_files - successful
            
            summary = {
                "total_files": total_files,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total_files * 100) if total_files > 0 else 0,
                "results": results
            }
            
            progress_callback(100, {"stage": "completed"})
            
            return {
                "success": True,
                "summary": summary,
                "individual_results": results
            }
            
        except Exception as e:
            logger.error(f"Batch processing job failed: {str(e)}")
            raise
    
    def get_processor(self, job_type: JobType) -> Callable:
        """Get processor function for job type"""
        processors = {
            JobType.IMAGE_PROCESSING: self.get_image_processor,
            JobType.VIDEO_PROCESSING: self.get_video_processor,
            JobType.BATCH_PROCESSING: self.get_batch_processor,
        }
        
        return processors.get(job_type)