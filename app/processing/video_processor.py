# app/processing/video_processor.py
"""
Enhanced video processing with format compatibility, advanced features, and optimizations
"""
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
import time
import threading
import gc
import base64
import tempfile
import subprocess
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.config import settings
from ..core.exceptions import ProcessingException
from ..core.memory_manager import memory_manager
from ..utils.logger import logger
from ..utils.file_handling import (
    validate_video_file,
    get_video_info,
    convert_video_format,
    preprocess_video_for_analysis,
    get_supported_video_formats,
    is_ffmpeg_available
)
from ..models.detection import DetectionType
from ..processing.image_processor import ImageProcessor
from ..processing.motion_detector import MotionDetector

class VideoProcessor:
    """Process video files with enhanced format compatibility, advanced features, and optimizations"""
    
    def __init__(self, image_processor: ImageProcessor = None):
        self.image_processor = image_processor or ImageProcessor()
        self.motion_detector = MotionDetector()
        self.supported_formats = get_supported_video_formats()
        
        # Thread pool for parallel frame processing (optimization)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=2,  # Conservative for free tier
            thread_name_prefix="VideoProcessor"
        )
        
        # Cache for frame processing results (optimization)
        self.frame_cache = {}
        self.cache_lock = threading.Lock()
        
        logger.info("Video processor initialized with optimizations")
    
    def _frame_to_base64(self, frame: np.ndarray, quality: int = 85) -> str:
        """Convert OpenCV frame to base64 string"""
        try:
            # Convert BGR to RGB
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Encode to JPEG
            success, encoded_image = cv2.imencode('.jpg', frame_rgb, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if not success:
                raise ProcessingException(message="Failed to encode frame to JPEG")
            
            # Convert to base64
            base64_string = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            
            # Return data URL format
            return f"data:image/jpeg;base64,{base64_string}"
            
        except Exception as e:
            logger.error(f"Error converting frame to base64: {str(e)}")
            return None
    
    def _extract_key_frames_base64(
        self,
        cap,
        total_frames: int,
        num_key_frames: int = 5,
        quality: int = 70
    ) -> List[Dict[str, Any]]:
        """Extract key frames as base64 images"""
        try:
            if total_frames <= 0 or num_key_frames <= 0:
                return []
            
            # Calculate frame indices for key frames
            if num_key_frames >= total_frames:
                # If we want more key frames than total frames, use all frames
                frame_indices = list(range(total_frames))
            else:
                # Distribute key frames evenly
                frame_indices = [int(i * total_frames / num_key_frames) for i in range(num_key_frames)]
            
            key_frames = []
            
            for frame_idx in frame_indices:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert to base64
                    frame_base64 = self._frame_to_base64(frame, quality)
                    
                    if frame_base64:
                        # Get timestamp
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        timestamp = frame_idx / fps if fps > 0 else 0
                        
                        key_frames.append({
                            "frame_number": frame_idx,
                            "timestamp": round(timestamp, 2),
                            "frame_base64": frame_base64,
                            "frame_size": f"{frame.shape[1]}x{frame.shape[0]}"
                        })
                
                # Reset to beginning for next operations
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            return key_frames
            
        except Exception as e:
            logger.error(f"Error extracting key frames: {str(e)}")
            return []
    
    def validate_and_prepare_video(
        self,
        video_path: Path
    ) -> Tuple[Path, Dict[str, Any], Optional[str]]:
        """Validate video and prepare for processing"""
        try:
            # Get video info
            video_info = get_video_info(video_path)
            
            if not video_info["valid"]:
                raise ProcessingException(
                    message="Invalid video file",
                    details={"video_info": video_info}
                )
            
            # Check duration
            if video_info["duration"] > settings.MAX_VIDEO_DURATION:
                raise ProcessingException(
                    message=f"Video duration ({video_info['duration']:.1f}s) exceeds maximum ({settings.MAX_VIDEO_DURATION}s)",
                    details={
                        "duration": video_info["duration"],
                        "max_duration": settings.MAX_VIDEO_DURATION
                    }
                )
            
            # Check if preprocessing is needed
            processed_path = video_path
            preprocessing_message = None
            
            if settings.VIDEO_PREPROCESSING_ENABLED:
                processed_path, preprocessing_message = preprocess_video_for_analysis(
                    video_path=video_path,
                    target_fps=settings.VIDEO_TARGET_FPS,
                    target_width=settings.VIDEO_TARGET_WIDTH,
                    target_height=settings.VIDEO_TARGET_HEIGHT,
                    max_duration=settings.MAX_VIDEO_DURATION
                )
            
            # Update video info for processed file
            if processed_path != video_path:
                processed_info = get_video_info(processed_path)
                if processed_info["valid"]:
                    video_info.update(processed_info)
                    video_info["preprocessed"] = True
                    video_info["preprocessing_message"] = preprocessing_message
            
            return processed_path, video_info, preprocessing_message
            
        except ProcessingException:
            raise
        except Exception as e:
            logger.error(f"Error preparing video: {str(e)}")
            raise ProcessingException(
                message="Error preparing video for processing",
                details={"error": str(e)}
            )
    
    def process_video(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None,
        confidence_threshold: float = None,
        detection_types: List[DetectionType] = None,
        frame_sample_rate: int = None,
        analyze_motion: bool = True,
        return_summary_only: bool = False,
        enable_advanced_features: bool = True,
        return_key_frames_base64: bool = True,  # NEW: parameter for base64 key frames
        num_key_frames: int = 5,  # NEW: number of key frames to extract
        key_frames_quality: int = 70,  # NEW: quality for key frames
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a video file with enhanced format handling, advanced features, and optimizations"""
        start_time = time.time()
        
        try:
            # Check memory before starting (optimization)
            memory_status = memory_manager.check_memory_usage()
            if memory_status["status"] == "critical":
                raise ProcessingException(
                    message="Memory critical, cannot process video",
                    details={"memory_status": memory_status}
                )
            
            # Validate and prepare video
            processed_path, video_info, prep_message = self.validate_and_prepare_video(video_path)
            
            # Log video information with optimization info
            logger.info(f"Processing video: {video_path.name}")
            logger.info(f"  Format: {video_info.get('codec', 'unknown')}")
            logger.info(f"  Resolution: {video_info.get('resolution', 'unknown')}")
            logger.info(f"  FPS: {video_info.get('fps', 0):.1f}")
            logger.info(f"  Duration: {video_info.get('duration', 0):.1f}s")
            logger.info(f"  Frames: {video_info.get('frames', 0)}")
            
            if prep_message:
                logger.info(f"  Preprocessing: {prep_message}")
            
            # Open video file with conversion fallback
            cap = cv2.VideoCapture(str(processed_path))
            
            if not cap.isOpened():
                # Try conversion if enabled
                if settings.VIDEO_CONVERSION_ENABLED and is_ffmpeg_available():
                    logger.info(f"Video cannot be opened, attempting conversion...")
                    
                    converted_path = video_path.with_suffix('.converted.mp4')
                    success, error = convert_video_format(
                        video_path,
                        converted_path,
                        target_format='mp4',
                        target_codec='h264',
                        crf=settings.VIDEO_CONVERSION_CRF,
                        preset=settings.VIDEO_CONVERSION_PRESET
                    )
                    
                    if success:
                        processed_path = converted_path
                        cap = cv2.VideoCapture(str(processed_path))
                        video_info = get_video_info(processed_path)
                        video_info["converted"] = True
                    else:
                        raise ProcessingException(
                            message="Cannot open video file and conversion failed",
                            details={"error": error}
                        )
                else:
                    raise ProcessingException(
                        message="Cannot open video file",
                        details={"path": str(processed_path)}
                    )
            
            if not cap.isOpened():
                raise ProcessingException(
                    message="Cannot open video file after conversion attempt",
                    details={"path": str(processed_path)}
                )
            
            # Extract key frames as base64 if requested
            key_frames_base64 = []
            if return_key_frames_base64:
                logger.info(f"Extracting {num_key_frames} key frames as base64...")
                key_frames_base64 = self._extract_key_frames_base64(
                    cap=cap,
                    total_frames=video_info["frames"],
                    num_key_frames=num_key_frames,
                    quality=key_frames_quality
                )
                logger.info(f"Extracted {len(key_frames_base64)} key frames as base64")
            
            # Get video properties from OpenCV (may differ from ffprobe)
            opencv_fps = cap.get(cv2.CAP_PROP_FPS)
            opencv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Use OpenCV values if ffprobe failed
            if video_info["fps"] <= 0:
                video_info["fps"] = opencv_fps
            if video_info["frames"] <= 0:
                video_info["frames"] = opencv_frames
            if video_info["resolution"] == "0x0":
                video_info["resolution"] = f"{width}x{height}"
            
            total_frames = video_info["frames"]
            fps = video_info["fps"]
            duration = video_info["duration"]
            
            # Use optimized parameters
            sample_rate = frame_sample_rate or self._calculate_optimal_sample_rate(fps, total_frames)
            conf_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
            det_types = detection_types or [DetectionType.PERSON, DetectionType.VEHICLE, DetectionType.MOTION]
            
            if enable_advanced_features:
                det_types.append(DetectionType.CROWD)
            
            logger.info(f"  Using frame sample rate: {sample_rate}")
            logger.info(f"  Estimated frames to process: {total_frames // sample_rate}")
            
            # Reset motion detector for new video
            self.motion_detector.reset_background()
            
            # Reset vehicle counts if advanced features enabled
            if enable_advanced_features:
                self.image_processor.reset_vehicle_counts()
            
            # Process frames with optimization
            frame_results, processed_frames = self._process_frames_optimized(
                cap=cap,
                total_frames=total_frames,
                sample_rate=sample_rate,
                fps=fps,
                detection_types=det_types,
                confidence_threshold=conf_threshold,
                analyze_motion=analyze_motion,
                enable_advanced_features=enable_advanced_features,
                progress_callback=progress_callback,
                max_workers=max_workers or 2
            )
            
            # Release video capture
            cap.release()
            
            # Clean up converted file if it was created
            if processed_path != video_path and processed_path.exists():
                try:
                    if video_info.get("converted", False):
                        processed_path.unlink()
                        logger.info(f"Cleaned up converted file: {processed_path.name}")
                except Exception as e:
                    logger.warning(f"Could not clean up converted file: {str(e)}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate summary
            summary = self._generate_advanced_summary(
                frame_results=frame_results,
                all_detections=[det for result in frame_results for det in result["detections"]],
                video_info={
                    "original_filename": video_path.name,
                    "processed_filename": processed_path.name if processed_path != video_path else video_path.name,
                    "total_frames": total_frames,
                    "processed_frames": processed_frames,
                    "fps": fps,
                    "duration": duration,
                    "resolution": video_info["resolution"],
                    "codec": video_info.get("codec", "unknown"),
                    "preprocessed": video_info.get("preprocessed", False),
                    "converted": video_info.get("converted", False),
                    "file_size_bytes": video_info.get("size_bytes", 0),
                    "preprocessing_message": prep_message,
                    "sample_rate": sample_rate
                },
                processing_time=processing_time,
                enable_advanced_features=enable_advanced_features
            )
            
            logger.info(f"Video processing complete: {processed_frames} frames processed in {processing_time:.1f}s")
            logger.info(f"  Processing speed: {processed_frames/processing_time:.1f} FPS")
            
            # Clear caches to free memory
            self._clear_caches()
            
            # Prepare response with base64 support
            response = {
                "success": True,
                "processing_time": round(processing_time, 3),
                "video_info": summary["video_info"],
                "summary": summary,
                "optimizations": {
                    "sample_rate": sample_rate,
                    "processing_speed_fps": round(processed_frames / processing_time, 1)
                },
                "key_frames_base64": key_frames_base64 if return_key_frames_base64 else [],  # NEW
                "has_key_frames": len(key_frames_base64) > 0 if return_key_frames_base64 else False,  # NEW
                "key_frames_count": len(key_frames_base64) if return_key_frames_base64 else 0  # NEW
            }
            
            if not return_summary_only:
                # Limit frame results to avoid huge responses
                limited_frame_results = frame_results[:50]  # First 50 frames
                
                # Optionally add base64 for first few frames
                if return_key_frames_base64 and len(frame_results) > 0:
                    # Add base64 for first frame with detections
                    first_detection_frame = next((fr for fr in frame_results if fr["detection_count"] > 0), None)
                    if first_detection_frame:
                        # We would need to extract this frame, but for now just mark it
                        response["first_detection_frame"] = first_detection_frame["frame_number"]
                
                response["frame_results"] = limited_frame_results
            
            return response
            
        except ProcessingException:
            raise
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise ProcessingException(
                message="Error processing video",
                details={"error": str(e)}
            )
    
    def process_video_in_chunks(
        self,
        video_path: Path,
        chunk_duration: int = 60,  # 60-second chunks
        **kwargs
    ) -> Dict[str, Any]:
        """Process video in chunks for better memory management"""
        try:
            # Get video duration
            video_info = get_video_info(video_path)
            duration = video_info["duration"]
            
            # If video is short, use normal processing
            if duration <= chunk_duration * 2:  # Less than 2 chunks
                return self.process_video(video_path, **kwargs)
            
            logger.info(f"Processing video in chunks: {duration:.1f}s total, {chunk_duration}s chunks")
            
            # Calculate number of chunks
            num_chunks = max(1, int(duration / chunk_duration) + 1)
            
            chunk_results = []
            
            for chunk_idx in range(num_chunks):
                start_time = chunk_idx * chunk_duration
                end_time = min((chunk_idx + 1) * chunk_duration, duration)
                chunk_length = end_time - start_time
                
                logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}: {start_time:.1f}s - {end_time:.1f}s")
                
                # Create temp file for chunk
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    chunk_path = Path(tmp.name)
                
                # Extract chunk using FFmpeg
                cmd = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-ss', str(start_time),
                    '-t', str(chunk_length),
                    '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    str(chunk_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    logger.warning(f"Failed to extract chunk {chunk_idx}: {result.stderr}")
                    continue
                
                # Process chunk
                chunk_result = self.process_video(
                    chunk_path,
                    return_summary_only=True,
                    **{k: v for k, v in kwargs.items() if k != 'return_summary_only'}
                )
                
                # Add chunk timing info
                chunk_result["chunk_info"] = {
                    "chunk_index": chunk_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": chunk_length
                }
                
                chunk_results.append(chunk_result)
                
                # Clean up chunk file
                if chunk_path.exists():
                    try:
                        chunk_path.unlink()
                    except:
                        pass
            
            # Combine results
            combined_result = self._combine_chunk_results(chunk_results, video_info)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error processing video in chunks: {str(e)}")
            raise ProcessingException(
                message="Error processing video in chunks",
                details={"error": str(e)}
            )
    
    def _combine_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]],
        original_video_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine results from multiple chunks"""
        try:
            if not chunk_results:
                raise ProcessingException(message="No chunk results to combine")
            
            # Combine all detections
            all_detections = []
            all_frame_results = []
            
            for chunk_result in chunk_results:
                chunk_info = chunk_result.get("chunk_info", {})
                start_time = chunk_info.get("start_time", 0)
                
                # Adjust timestamps for chunk detections
                if "summary" in chunk_result and "all_detections" in chunk_result["summary"]:
                    for detection in chunk_result["summary"]["all_detections"]:
                        # Adjust timestamps to absolute video time
                        if "timestamp" in detection:
                            detection["timestamp"] += start_time
                        all_detections.append(detection)
                
                # Adjust frame results timestamps
                if "frame_results" in chunk_result:
                    for frame_result in chunk_result["frame_results"]:
                        frame_result["timestamp"] += start_time
                        all_frame_results.append(frame_result)
            
            # Create combined summary
            combined_summary = {
                "video_info": original_video_info,
                "total_detections": len(all_detections),
                "detection_counts": self._count_detections_by_type(all_detections),
                "chunks_processed": len(chunk_results),
                "processing_method": "chunked"
            }
            
            # Calculate overall statistics
            total_processing_time = sum(chunk.get("processing_time", 0) for chunk in chunk_results)
            total_processed_frames = sum(chunk.get("summary", {}).get("frames_processed", 0) for chunk in chunk_results)
            
            return {
                "success": True,
                "processing_time": round(total_processing_time, 3),
                "video_info": original_video_info,
                "summary": combined_summary,
                "frame_results": all_frame_results[:100],  # Limit to first 100 frames
                "chunk_count": len(chunk_results),
                "processing_method": "chunked"
            }
            
        except Exception as e:
            logger.error(f"Error combining chunk results: {str(e)}")
            raise
    
    def _count_detections_by_type(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count detections by type"""
        counts = {}
        for detection in detections:
            det_type = detection.get("type", "unknown")
            counts[det_type] = counts.get(det_type, 0) + 1
        return counts
    
    def _process_frames_optimized(
        self,
        cap,
        total_frames: int,
        sample_rate: int,
        fps: float,
        detection_types: List[DetectionType],
        confidence_threshold: float,
        analyze_motion: bool,
        enable_advanced_features: bool,
        progress_callback: Optional[Callable],
        max_workers: int = 2
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Process frames with memory and performance optimizations"""
        frame_results = []
        processed_frames = 0
        frame_count = 0
        last_progress_update = 0
        
        # Use smaller batch size for memory efficiency - IMPORTANT
        batch_size = 1  # Process one frame at a time to save memory
        
        while True:
            # Read batch of frames
            frames_batch = []
            frame_indices = []
            
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames based on sample rate
                if frame_count % sample_rate != 0:
                    continue
                
                frames_batch.append(frame)
                frame_indices.append(frame_count)
            
            if not frames_batch:
                break
            
            # Process batch in parallel
            batch_results = self._process_frame_batch(
                frames=frames_batch,
                frame_indices=frame_indices,
                fps=fps,
                detection_types=detection_types,
                confidence_threshold=confidence_threshold,
                analyze_motion=analyze_motion,
                enable_advanced_features=enable_advanced_features,
                max_workers=1  # Reduced to 1 worker for memory
            )
            
            frame_results.extend(batch_results)
            processed_frames += len(frames_batch)
            
            # Update progress (throttle updates - from original)
            if progress_callback:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                
                if progress - last_progress_update >= 1 or processed_frames == len(frames_batch):
                    progress_callback(
                        progress,
                        {
                            "frame": frame_count,
                            "processed_frames": processed_frames,
                            "total_frames": total_frames,
                            "current_time": frame_count / fps if fps > 0 else 0,
                            "batch_size": len(frames_batch),
                            "memory_mb": memory_manager.get_memory_stats().process_rss / 1024 / 1024,
                            "video_info": {
                                "fps": fps,
                                "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                                "codec": "unknown"
                            }
                        }
                    )
                    last_progress_update = progress
            
            # Check memory periodically and clear aggressively
            if processed_frames % 5 == 0:
                # Force garbage collection
                gc.collect()
                
                # Clear the frames batch to free memory
                frames_batch.clear()
                
                # Check memory status
                memory_status = memory_manager.check_memory_usage()
                if memory_status["status"] == "critical":
                    logger.warning("Memory critical during video processing")
                    memory_manager.optimize_memory()
        
        return frame_results, processed_frames
    
    def _process_frame_batch(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        fps: float,
        detection_types: List[DetectionType],
        confidence_threshold: float,
        analyze_motion: bool,
        enable_advanced_features: bool,
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """Process a batch of frames in parallel"""
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=1) as executor:  # Only 1 worker for memory
            # Submit all frames for processing
            future_to_index = {}
            for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
                future = executor.submit(
                    self._process_single_frame,
                    frame=frame,
                    frame_number=frame_idx,
                    timestamp=frame_idx / fps if fps > 0 else 0,
                    detection_types=detection_types,
                    confidence_threshold=confidence_threshold,
                    analyze_motion=analyze_motion,
                    enable_advanced_features=enable_advanced_features
                )
                future_to_index[future] = i
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    result = future.result(timeout=15.0)  # Increased timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_indices[i]}: {str(e)}")
                    # Add empty result for failed frame
                    results.append({
                        "frame_number": frame_indices[i],
                        "timestamp": frame_indices[i] / fps if fps > 0 else 0,
                        "detections": [],
                        "detection_count": 0,
                        "motion_detections": [],
                        "motion_count": 0,
                        "error": str(e)
                    })
        
        # Sort by frame number
        results.sort(key=lambda x: x["frame_number"])
        return results
    
    def _process_single_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        detection_types: List[DetectionType],
        confidence_threshold: float,
        analyze_motion: bool,
        enable_advanced_features: bool
    ) -> Dict[str, Any]:
        """Process a single frame with caching"""
        # Generate cache key
        cache_key = self._generate_frame_cache_key(
            frame, frame_number, detection_types, confidence_threshold
        )
        
        # Check cache
        with self.cache_lock:
            if cache_key in self.frame_cache:
                result = self.frame_cache[cache_key]
                result["frame_number"] = frame_number
                result["timestamp"] = timestamp
                result["cached"] = True
                return result
        
        # Process frame with advanced features
        frame_result = self.image_processor.process_video_frame(
            frame=frame,
            frame_number=frame_number,
            detection_types=detection_types,
            confidence_threshold=confidence_threshold,
            enable_advanced_features=enable_advanced_features
        )
        
        # Add timestamp and cache info
        frame_result["timestamp"] = round(timestamp, 2)
        frame_result["cached"] = False
        
        # Add motion detection if enabled
        if analyze_motion and DetectionType.MOTION in detection_types:
            motion_detections = self.motion_detector.detect(frame, method="background_subtraction")
            
            if motion_detections:
                frame_result["motion_detections"] = [
                    {
                        "bbox": det.bbox,
                        "confidence": det.confidence,
                        "area": det.attributes.get("area", 0),
                        "center": det.attributes.get("center", {})
                    }
                    for det in motion_detections
                ]
                frame_result["motion_count"] = len(motion_detections)
            else:
                frame_result["motion_detections"] = []
                frame_result["motion_count"] = 0
        
        # Cache result (limit cache size)
        with self.cache_lock:
            if len(self.frame_cache) < 100:  # Keep only 100 frames in cache
                self.frame_cache[cache_key] = frame_result
        
        return frame_result
    
    def _generate_frame_cache_key(
        self,
        frame: np.ndarray,
        frame_number: int,
        detection_types: List[DetectionType],
        confidence_threshold: float
    ) -> str:
        """Generate cache key for frame"""
        # Use frame hash and processing parameters
        # Create simplified frame signature (average color + dimensions)
        frame_info = f"{frame.shape}_{frame.mean():.2f}_{frame.std():.2f}"
        params = f"{sorted(detection_types)}_{confidence_threshold}"
        
        key_data = f"{frame_info}_{params}_{frame_number // 10}"  # Group every 10 frames
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_optimal_sample_rate(self, fps: float, total_frames: int) -> int:
        """Calculate optimal frame sample rate based on video characteristics"""
        if fps <= 0:
            return 15  # Increased for memory efficiency
        
        # Process fewer frames for memory efficiency
        duration = total_frames / fps
        
        if duration > 300:  # Very long video (>5 min)
            return max(20, int(fps / 5))  # Process at most 0.2 FPS
        elif duration > 120:  # Long video (2-5 min)
            return max(15, int(fps / 7))  # Process at most ~0.14 FPS
        elif duration > 60:  # Medium video (1-2 min)
            return max(10, int(fps / 10))  # Process at most 0.1 FPS
        else:  # Short video (<1 min)
            return max(5, int(fps / 15))  # Process at most ~0.07 FPS
    
    # Keep all the summary and utility methods from the original
    def process_video_legacy(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None,
        confidence_threshold: float = None,
        detection_types: List[DetectionType] = None,
        frame_sample_rate: int = None,
        analyze_motion: bool = True,
        return_summary_only: bool = False
    ) -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        return self.process_video(
            video_path=video_path,
            progress_callback=progress_callback,
            confidence_threshold=confidence_threshold,
            detection_types=detection_types,
            frame_sample_rate=frame_sample_rate,
            analyze_motion=analyze_motion,
            return_summary_only=return_summary_only,
            enable_advanced_features=False
        )
    
    def _process_frame_legacy(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        confidence_threshold: float,
        detection_types: List[DetectionType],
        analyze_motion: bool
    ) -> Dict[str, Any]:
        """Legacy frame processing method for backward compatibility"""
        frame_result = {
            "frame_number": frame_number,
            "timestamp": round(timestamp, 2),
            "detections": [],
            "detection_count": 0,
            "motion_detections": [],
            "motion_count": 0
        }
        
        try:
            # Object detection
            if any(dt in detection_types for dt in [DetectionType.PERSON, DetectionType.VEHICLE, DetectionType.OBJECT, DetectionType.FACE]):
                # Convert frame for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame temporarily for processing
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    cv2.imwrite(str(tmp_path), frame)
                    
                    # Process frame using image processor
                    result = self.image_processor.process_image(
                        image_path=tmp_path,
                        detection_types=[dt for dt in detection_types if dt != DetectionType.MOTION],
                        confidence_threshold=confidence_threshold,
                        return_image=False,
                        enable_advanced_features=False
                    )
                    
                    # Add detections to frame result
                    if result["success"] and result["detections"]:
                        frame_result["detections"] = result["detections"]
                        frame_result["detection_count"] = result["detection_count"]
                    
                    # Clean up temp file
                    tmp_path.unlink()
            
            # Motion detection
            if analyze_motion and DetectionType.MOTION in detection_types:
                motion_detections = self.motion_detector.detect(frame, method="background_subtraction")
                
                if motion_detections:
                    frame_result["motion_detections"] = [
                        {
                            "bbox": det.bbox,
                            "confidence": det.confidence,
                            "area": det.attributes.get("area", 0),
                            "center": det.attributes.get("center", {})
                        }
                        for det in motion_detections
                    ]
                    frame_result["motion_count"] = len(motion_detections)
            
            return frame_result
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {str(e)}")
            return frame_result
    
    def _generate_advanced_summary(
        self,
        frame_results: List[Dict[str, Any]],
        all_detections: List[Dict[str, Any]],
        video_info: Dict[str, Any],
        processing_time: float,
        enable_advanced_features: bool
    ) -> Dict[str, Any]:
        """Generate summary statistics for the video with advanced features"""
        
        # Basic statistics
        detection_counts = {}
        for detection in all_detections:
            det_type = detection.get("type", "unknown")
            detection_counts[det_type] = detection_counts.get(det_type, 0) + 1
        
        # Calculate detection rate
        total_processed_frames = len(frame_results)
        frames_with_detections = sum(1 for result in frame_results if result["detection_count"] > 0)
        detection_rate = (frames_with_detections / total_processed_frames * 100) if total_processed_frames > 0 else 0
        
        # Initialize summary
        summary = {
            "video_info": video_info,
            "total_detections": len(all_detections),
            "detection_counts": detection_counts,
            "detection_rate": round(detection_rate, 1),
            "processing_speed": round(video_info["processed_frames"] / processing_time, 2) if processing_time > 0 else 0,
            "frames_processed": total_processed_frames,
            "frames_with_activity": frames_with_detections,
        }
        
        # Add advanced statistics if enabled
        if enable_advanced_features:
            # Crowd statistics
            try:
                crowd_stats = self.image_processor.get_crowd_statistics()
                summary["crowd_statistics"] = crowd_stats
            except Exception as e:
                logger.warning(f"Could not get crowd statistics: {str(e)}")
            
            # Vehicle counting statistics
            try:
                vehicle_stats = self.image_processor.get_vehicle_counting_statistics()
                summary["vehicle_counting"] = vehicle_stats
            except Exception as e:
                logger.warning(f"Could not get vehicle statistics: {str(e)}")
            
            # Calculate peak activity periods
            try:
                activity_over_time = self._calculate_activity_over_time(frame_results, video_info["fps"])
                summary["activity_timeline"] = activity_over_time
            except Exception as e:
                logger.warning(f"Could not calculate activity timeline: {str(e)}")
            
            # Find key events
            try:
                key_events = self._identify_key_events(frame_results, video_info["fps"])
                summary["key_events"] = key_events
            except Exception as e:
                logger.warning(f"Could not identify key events: {str(e)}")
            
            # Calculate density heatmap (simplified)
            try:
                density_map = self._calculate_density_map(frame_results, video_info["resolution"])
                summary["density_analysis"] = density_map
            except Exception as e:
                logger.warning(f"Could not calculate density map: {str(e)}")
        
        return summary
    
    def _calculate_activity_over_time(
        self, 
        frame_results: List[Dict[str, Any]], 
        fps: float
    ) -> List[Dict[str, Any]]:
        """Calculate activity levels over time"""
        if not frame_results or fps <= 0:
            return []
        
        # Group by time windows (e.g., 10-second intervals)
        time_window = 10  # seconds
        frames_per_window = int(time_window * fps)
        
        activity_windows = []
        
        for i in range(0, len(frame_results), frames_per_window):
            window_frames = frame_results[i:i + frames_per_window]
            
            if not window_frames:
                continue
            
            # Calculate average activity in this window
            total_detections = sum(frame["detection_count"] for frame in window_frames)
            avg_detections = total_detections / len(window_frames)
            
            # Calculate timestamp for this window
            start_frame = window_frames[0]["frame_number"]
            start_time = start_frame / fps if fps > 0 else 0
            
            activity_windows.append({
                "start_time": round(start_time, 1),
                "duration": time_window,
                "average_activity": round(avg_detections, 2),
                "peak_activity": max(frame["detection_count"] for frame in window_frames),
                "frames_in_window": len(window_frames)
            })
        
        return activity_windows
    
    def _identify_key_events(
        self, 
        frame_results: List[Dict[str, Any]], 
        fps: float
    ) -> List[Dict[str, Any]]:
        """Identify key events in the video"""
        if not frame_results or fps <= 0:
            return []
        
        key_events = []
        
        # Look for spikes in activity
        activity_threshold = 5  # Minimum detections to be considered an event
        
        for i, frame in enumerate(frame_results):
            if frame["detection_count"] >= activity_threshold:
                # Check if this is part of an existing event
                is_new_event = True
                
                for event in key_events:
                    last_frame_in_event = event["end_frame"]
                    frames_between = frame["frame_number"] - last_frame_in_event
                    time_between = frames_between / fps if fps > 0 else 0
                    
                    if time_between <= 5:  # Events within 5 seconds are merged
                        # Update existing event
                        event["end_frame"] = frame["frame_number"]
                        event["end_time"] = frame["frame_number"] / fps if fps > 0 else 0
                        event["peak_activity"] = max(event["peak_activity"], frame["detection_count"])
                        event["total_detections"] += frame["detection_count"]
                        is_new_event = False
                        break
                
                if is_new_event:
                    # Create new event
                    key_events.append({
                        "start_frame": frame["frame_number"],
                        "end_frame": frame["frame_number"],
                        "start_time": frame["frame_number"] / fps if fps > 0 else 0,
                        "end_time": frame["frame_number"] / fps if fps > 0 else 0,
                        "peak_activity": frame["detection_count"],
                        "total_detections": frame["detection_count"],
                        "event_type": "activity_spike"
                    })
        
        # Calculate event durations and add more details
        for event in key_events:
            event_duration = event["end_time"] - event["start_time"]
            event["duration"] = round(event_duration, 1)
            event["average_activity"] = round(event["total_detections"] / (event["end_frame"] - event["start_frame"] + 1), 2)
        
        # Sort by peak activity (most active first)
        key_events.sort(key=lambda x: x["peak_activity"], reverse=True)
        
        return key_events[:10]  # Return top 10 events
    
    def _calculate_density_map(
        self, 
        frame_results: List[Dict[str, Any]], 
        resolution: str
    ) -> Dict[str, Any]:
        """Calculate simplified density map"""
        try:
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            # Initialize density grid (4x4 grid for simplicity)
            grid_rows, grid_cols = 4, 4
            cell_width = width // grid_cols
            cell_height = height // grid_rows
            
            density_grid = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
            
            # Count detections in each grid cell
            for frame in frame_results:
                for detection in frame.get("detections", []):
                    bbox = detection.get("bbox", [0, 0, 0, 0])
                    if len(bbox) == 4:
                        # Calculate center of detection
                        center_x = (bbox[0] + bbox[2]) // 2
                        center_y = (bbox[1] + bbox[3]) // 2
                        
                        # Determine grid cell
                        cell_col = min(center_x // cell_width, grid_cols - 1)
                        cell_row = min(center_y // cell_height, grid_rows - 1)
                        
                        density_grid[cell_row][cell_col] += 1
            
            # Normalize to 0-1 range
            max_density = max(max(row) for row in density_grid) if any(any(row) for row in density_grid) else 1
            
            normalized_grid = []
            for row in density_grid:
                normalized_row = [round(count / max_density, 2) for count in row]
                normalized_grid.append(normalized_row)
            
            # Find hotspots (cells with density > 0.5)
            hotspots = []
            for row_idx, row in enumerate(normalized_grid):
                for col_idx, density in enumerate(row):
                    if density > 0.5:
                        hotspots.append({
                            "row": row_idx,
                            "col": col_idx,
                            "density": density,
                            "bbox": [
                                col_idx * cell_width,
                                row_idx * cell_height,
                                (col_idx + 1) * cell_width,
                                (row_idx + 1) * cell_height
                            ]
                        })
            
            return {
                "grid_size": f"{grid_rows}x{grid_cols}",
                "cell_size": f"{cell_width}x{cell_height}",
                "density_grid": normalized_grid,
                "hotspots": hotspots,
                "max_density": max_density
            }
            
        except Exception as e:
            logger.error(f"Error calculating density map: {str(e)}")
            return {
                "grid_size": "0x0",
                "density_grid": [],
                "hotspots": [],
                "error": str(e)
            }
    
    def extract_key_frames(
        self,
        video_path: Path,
        num_frames: int = 10,
        method: str = "activity"  # "activity", "uniform", or "motion"
    ) -> List[Dict[str, Any]]:
        """Extract key frames from video based on activity"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if method == "uniform":
                # Extract frames uniformly spaced
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                key_frames = []
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        key_frames.append({
                            "frame_number": idx,
                            "timestamp": idx / fps if fps > 0 else 0,
                            "frame": frame
                        })
                
                cap.release()
                return key_frames
            
            else:
                # For activity-based extraction, we need to process the video
                # This is a simplified version - in production you'd want to optimize
                cap.release()
                
                # Process video to get activity scores
                result = self.process_video(
                    video_path=video_path,
                    frame_sample_rate=5,  # Sample every 5th frame for speed
                    return_summary_only=False,
                    enable_advanced_features=True
                )
                
                if not result["success"]:
                    return []
                
                # Get frames with highest activity
                frame_activities = []
                for frame_result in result.get("frame_results", []):
                    activity_score = (
                        frame_result["detection_count"] * 1.0 +
                        frame_result.get("person_count", 0) * 0.5 +
                        frame_result.get("vehicle_count", 0) * 0.3
                    )
                    
                    frame_activities.append({
                        "frame_number": frame_result["frame_number"],
                        "timestamp": frame_result["timestamp"],
                        "activity_score": activity_score
                    })
                
                # Sort by activity and take top N
                frame_activities.sort(key=lambda x: x["activity_score"], reverse=True)
                top_frames = frame_activities[:num_frames]
                
                # Extract the actual frames
                cap = cv2.VideoCapture(str(video_path))
                key_frames = []
                
                for frame_info in top_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_info["frame_number"])
                    ret, frame = cap.read()
                    
                    if ret:
                        key_frames.append({
                            "frame_number": frame_info["frame_number"],
                            "timestamp": frame_info["timestamp"],
                            "activity_score": frame_info["activity_score"],
                            "frame": frame
                        })
                
                cap.release()
                return key_frames
                
        except Exception as e:
            logger.error(f"Error extracting key frames: {str(e)}")
            return []
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get information about supported video formats"""
        return self.supported_formats
    
    def check_video_compatibility(self, video_path: Path) -> Dict[str, Any]:
        """Check if video is compatible with the processing system"""
        try:
            video_info = get_video_info(video_path)
            
            compatibility = {
                "file_exists": video_info["exists"],
                "is_valid": video_info["valid"],
                "video_info": video_info,
                "can_open_with_opencv": False,
                "ffmpeg_available": is_ffmpeg_available(),
                "recommended_action": None,
                "issues": []
            }
            
            if video_info["valid"]:
                # Try to open with OpenCV
                cap = cv2.VideoCapture(str(video_path))
                compatibility["can_open_with_opencv"] = cap.isOpened()
                if cap.isOpened():
                    cap.release()
                
                # Check for issues
                if not compatibility["can_open_with_opencv"]:
                    compatibility["issues"].append("Cannot open with OpenCV")
                    if is_ffmpeg_available():
                        compatibility["recommended_action"] = "Convert to MP4/H.264"
                    else:
                        compatibility["recommended_action"] = "Install FFmpeg for conversion"
                
                if video_info["duration"] > settings.MAX_VIDEO_DURATION:
                    compatibility["issues"].append(f"Duration exceeds limit ({settings.MAX_VIDEO_DURATION}s)")
                    compatibility["recommended_action"] = "Trim video or increase MAX_VIDEO_DURATION setting"
                
                if video_info["fps"] > 60:
                    compatibility["issues"].append("High FPS may cause slow processing")
                    compatibility["recommended_action"] = "Consider reducing FPS during preprocessing"
            
            return compatibility
            
        except Exception as e:
            return {
                "file_exists": video_path.exists(),
                "is_valid": False,
                "error": str(e),
                "recommended_action": "Check file format and codec"
            }
    
    # Optimization methods
    def _clear_frame_cache(self):
        """Clear frame cache to free memory"""
        with self.cache_lock:
            self.frame_cache.clear()
    
    def _clear_caches(self):
        """Clear all caches"""
        self._clear_frame_cache()
        gc.collect()