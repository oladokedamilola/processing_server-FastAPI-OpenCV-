# app/utils/file_handling.py
"""
Enhanced file handling utilities with video format support
"""
import os
import uuid
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, BinaryIO, List
from fastapi import UploadFile, HTTPException
import magic
from PIL import Image
import cv2
import numpy as np
import subprocess
import tempfile
from io import BytesIO
import datetime
import random


from ..core.config import settings
from ..core.exceptions import FileValidationError
from ..utils.logger import logger

# Enhanced video format support
VIDEO_EXTENSIONS = {
    '.mp4': 'video/mp4',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
    '.mkv': 'video/x-matroska',
    '.flv': 'video/x-flv',
    '.wmv': 'video/x-ms-wmv',
    '.webm': 'video/webm',
    '.m4v': 'video/x-m4v',
    '.3gp': 'video/3gpp',
}

# Codec support mapping
SUPPORTED_CODECS = {
    'h264': 'libx264',
    'h265': 'libx265',
    'vp8': 'libvpx',
    'vp9': 'libvpx-vp9',
    'mpeg4': 'mpeg4',
}

def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get detailed video information using OpenCV and FFmpeg"""
    info = {
        "path": str(video_path),
        "exists": video_path.exists(),
        "valid": False,
        "codec": "unknown",
        "fps": 0,
        "frames": 0,
        "duration": 0,
        "resolution": "0x0",
        "bitrate": 0,
        "size_bytes": 0,
    }
    
    if not video_path.exists():
        return info
    
    try:
        # Get file size
        info["size_bytes"] = video_path.stat().st_size
        
        # Try OpenCV first
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            info["fps"] = cap.get(cv2.CAP_PROP_FPS)
            info["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["resolution"] = f"{width}x{height}"
            
            # Try to get codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            info["codec"] = decode_fourcc(fourcc)
            
            # Calculate duration
            if info["fps"] > 0:
                info["duration"] = info["frames"] / info["fps"]
            
            cap.release()
            info["valid"] = True
        
        # If OpenCV failed or we need more info, try FFmpeg
        if not info["valid"] or info["codec"] == "unknown":
            ffmpeg_info = get_video_info_ffmpeg(video_path)
            if ffmpeg_info:
                info.update(ffmpeg_info)
                info["valid"] = True
        
        logger.debug(f"Video info retrieved: {video_path.name}")
        return info
        
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {str(e)}")
        return info

def get_video_info_ffmpeg(video_path: Path) -> Optional[Dict[str, Any]]:
    """Get video information using FFmpeg"""
    try:
        import json
        
        # Run ffprobe to get video info
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        
        info = {}
        
        # Extract format info
        if 'format' in data:
            format_info = data['format']
            info["size_bytes"] = int(format_info.get('size', 0))
            info["bitrate"] = int(format_info.get('bit_rate', 0))
            info["duration"] = float(format_info.get('duration', 0))
        
        # Extract video stream info
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if video_stream:
            info["codec"] = video_stream.get('codec_name', 'unknown')
            info["width"] = int(video_stream.get('width', 0))
            info["height"] = int(video_stream.get('height', 0))
            info["resolution"] = f"{info['width']}x{info['height']}"
            info["fps"] = eval_fps(video_stream.get('avg_frame_rate', '0/0'))
            info["frames"] = int(info["duration"] * info["fps"]) if info["duration"] > 0 else 0
        
        return info
        
    except Exception as e:
        logger.error(f"FFmpeg error for {video_path}: {str(e)}")
        return None

def eval_fps(fps_str: str) -> float:
    """Evaluate FPS string like '30000/1001'"""
    try:
        if '/' in fps_str:
            num, den = fps_str.split('/')
            return float(num) / float(den) if float(den) != 0 else 0
        return float(fps_str)
    except:
        return 0

def decode_fourcc(fourcc: int) -> str:
    """Decode FourCC code to string"""
    try:
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    except:
        return "unknown"

def validate_video_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
    """Validate video file format and codec"""
    try:
        # Read first 4096 bytes for magic number detection
        content = file.file.read(4096)
        file.file.seek(0)
        
        # Use python-magic to detect MIME type
        mime_type = magic.from_buffer(content, mime=True)
        
        # Check MIME type
        if not mime_type.startswith('video/'):
            return False, f"File is not a video (detected: {mime_type})"
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in VIDEO_EXTENSIONS:
            return False, f"Unsupported video extension: {file_ext}"
        
        # Check if we can read the video
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            
            cap = cv2.VideoCapture(tmp.name)
            can_read = cap.isOpened()
            cap.release()
            
            os.unlink(tmp.name)
            
            if not can_read:
                return False, "Cannot read video file (unsupported codec or corrupt file)"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Video validation error: {str(e)}")
        return False, f"Video validation error: {str(e)}"

def convert_video_format(
    input_path: Path,
    output_path: Path,
    target_format: str = 'mp4',
    target_codec: str = 'h264',
    crf: int = 23,
    preset: str = 'medium'
) -> Tuple[bool, Optional[str]]:
    """Convert video to supported format using FFmpeg"""
    try:
        # Check if FFmpeg is available
        if not is_ffmpeg_available():
            return False, "FFmpeg is not available for video conversion"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        video_info = get_video_info(input_path)
        
        if not video_info["valid"]:
            return False, "Invalid video file"
        
        # Check if conversion is needed
        input_ext = input_path.suffix.lower()
        if (input_ext == f'.{target_format}' and 
            video_info.get('codec', '').lower() == target_codec.lower()):
            # No conversion needed
            return True, "Video already in target format"
        
        # Build FFmpeg command
        codec = SUPPORTED_CODECS.get(target_codec.lower(), 'libx264')
        
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', codec,
            '-crf', str(crf),
            '-preset', preset,
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',  # Overwrite output file
            str(output_path)
        ]
        
        # Run conversion
        logger.info(f"Converting video: {input_path.name} -> {output_path.name}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return False, f"Video conversion failed: {result.stderr[:200]}"
        
        # Verify output
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Video converted successfully: {output_path}")
            return True, None
        else:
            return False, "Conversion produced empty file"
        
    except subprocess.TimeoutExpired:
        logger.error(f"Video conversion timeout: {input_path}")
        return False, "Video conversion timeout"
    except Exception as e:
        logger.error(f"Video conversion error: {str(e)}")
        return False, f"Video conversion error: {str(e)}"

def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    frame_sample_rate: int = 1,
    max_frames: int = 1000,
    image_format: str = 'jpg',
    quality: int = 85
) -> Tuple[List[Path], Optional[str]]:
    """Extract frames from video at specified sample rate"""
    frames = []
    
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], "Cannot open video file"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Extracting frames from {video_path.name}: {total_frames} frames at {fps:.1f} FPS")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame based on sample rate
            if frame_count % frame_sample_rate == 0:
                # Generate frame filename
                timestamp = frame_count / fps if fps > 0 else frame_count
                frame_filename = f"frame_{frame_count:06d}_{timestamp:.2f}s.{image_format}"
                frame_path = output_dir / frame_filename
                
                # Save frame
                if image_format.lower() in ['jpg', 'jpeg']:
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                elif image_format.lower() == 'png':
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 10)])
                else:
                    cv2.imwrite(str(frame_path), frame)
                
                frames.append(frame_path)
                extracted_count += 1
                
                if extracted_count >= max_frames:
                    logger.info(f"Reached maximum frames limit: {max_frames}")
                    break
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames, None
        
    except Exception as e:
        logger.error(f"Frame extraction error: {str(e)}")
        return [], f"Frame extraction error: {str(e)}"

def create_video_from_frames(
    frames_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    codec: str = 'h264',
    crf: int = 23
) -> Tuple[bool, Optional[str]]:
    """Create video from sequence of frames"""
    try:
        # Get sorted list of frame files
        frame_files = sorted(frames_dir.glob("*.jpg") + frames_dir.glob("*.png") +
                           frames_dir.glob("*.jpeg"))
        
        if not frame_files:
            return False, "No frame files found"
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            return False, "Cannot read first frame"
        
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not out.isOpened():
            return False, "Cannot create video writer"
        
        # Write frames
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)
        
        out.release()
        
        # Convert to better codec if FFmpeg is available
        if is_ffmpeg_available() and codec != 'mp4v':
            temp_path = output_path.with_suffix('.temp.mp4')
            output_path.rename(temp_path)
            
            success, error = convert_video_format(
                temp_path,
                output_path,
                target_codec=codec,
                crf=crf
            )
            
            if success:
                temp_path.unlink()
            else:
                # Restore original
                temp_path.rename(output_path)
                logger.warning(f"Could not re-encode video: {error}")
        
        return True, None
        
    except Exception as e:
        logger.error(f"Video creation error: {str(e)}")
        return False, f"Video creation error: {str(e)}"

def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_supported_video_formats() -> Dict[str, List[str]]:
    """Get list of supported video formats and codecs"""
    formats = {
        "extensions": list(VIDEO_EXTENSIONS.keys()),
        "mime_types": list(set(VIDEO_EXTENSIONS.values())),
        "codecs": list(SUPPORTED_CODECS.keys()),
        "conversion_available": is_ffmpeg_available(),
    }
    
    if is_ffmpeg_available():
        formats["conversion_formats"] = ["mp4", "avi", "mov", "mkv", "webm"]
    else:
        formats["conversion_formats"] = []
        formats["warning"] = "FFmpeg not available - limited video support"
    
    return formats

def preprocess_video_for_analysis(
    video_path: Path,
    target_fps: float = 10.0,
    target_width: int = 640,
    target_height: int = 480,
    max_duration: int = 300  # 5 minutes
) -> Tuple[Optional[Path], Optional[str]]:
    """Preprocess video for efficient analysis"""
    try:
        # Get video info
        video_info = get_video_info(video_path)
        
        if not video_info["valid"]:
            return None, "Invalid video file"
        
        # Check duration limit
        if video_info["duration"] > max_duration:
            return None, f"Video duration ({video_info['duration']:.1f}s) exceeds limit ({max_duration}s)"
        
        # Create temp directory for processed video
        temp_dir = settings.TEMP_PATH / "processed_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output path
        output_path = temp_dir / f"processed_{video_path.stem}.mp4"
        
        # Check if FFmpeg is available for preprocessing
        if is_ffmpeg_available():
            # Use FFmpeg for efficient preprocessing
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'fps={target_fps},scale={target_width}:{target_height}',
                '-c:v', 'libx264',
                '-crf', '28',  # Higher CRF for smaller file size
                '-preset', 'fast',
                '-c:a', 'aac',
                '-b:a', '64k',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout
            )
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Video preprocessed: {output_path}")
                return output_path, None
            else:
                logger.warning(f"FFmpeg preprocessing failed: {result.stderr[:200]}")
                # Fall back to original
                return video_path, "Using original video (preprocessing failed)"
        
        # Fallback: return original video
        return video_path, "FFmpeg not available, using original video"
        
    except Exception as e:
        logger.error(f"Video preprocessing error: {str(e)}")
        return video_path, f"Preprocessing error: {str(e)}"
    
    
    
def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    """
    Save an uploaded file to the specified destination.
    
    Args:
        upload_file: FastAPI UploadFile object
        destination: Path where the file should be saved
    
    Returns:
        str: Path where the file was saved
    
    Raises:
        FileValidationError: If the file cannot be saved
    """
    try:
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique filename to avoid collisions
        file_extension = Path(upload_file.filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = destination / unique_filename
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = upload_file.file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded file saved: {upload_file.filename} -> {file_path}")
        
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving uploaded file {upload_file.filename}: {str(e)}")
        raise FileValidationError(f"Could not save file: {str(e)}")
    
    
    
def draw_bounding_boxes(image: np.ndarray, detections: list, 
                       confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Draw bounding boxes on image based on detections.
    
    Args:
        image: Original image as numpy array (BGR format from OpenCV)
        detections: List of detection dictionaries
        confidence_threshold: Minimum confidence to draw box
    
    Returns:
        Image with bounding boxes drawn
    """
    # Create a copy of the image
    annotated_image = image.copy()
    
    # Define colors for different object types
    color_map = {
        'person': (0, 255, 0),      # Green
        'car': (255, 0, 0),         # Blue
        'truck': (0, 0, 255),       # Red
        'bus': (255, 255, 0),       # Cyan
        'motorcycle': (0, 255, 255), # Yellow
        'vehicle': (128, 0, 128),   # Purple
        'default': (255, 165, 0)    # Orange for other objects
    }
    
    for detection in detections:
        # Skip if confidence is below threshold
        confidence = detection.get('confidence', 0)
        if confidence < confidence_threshold:
            continue
        
        # Get object type/label
        obj_type = detection.get('label', detection.get('type', detection.get('class', 'unknown')))
        obj_type_lower = obj_type.lower()
        
        # Get bounding box coordinates
        bbox = detection.get('bbox')
        if not bbox:
            continue
        
        # Parse bounding box (could be list or dict)
        if isinstance(bbox, dict):
            x1 = int(bbox.get('x1', 0))
            y1 = int(bbox.get('y1', 0))
            x2 = int(bbox.get('x2', 0))
            y2 = int(bbox.get('y2', 0))
        elif isinstance(bbox, list) and len(bbox) >= 4:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
        else:
            continue  # Skip invalid bbox
        
        # Ensure coordinates are within image bounds
        height, width = annotated_image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        if x1 >= x2 or y1 >= y2:
            continue  # Skip invalid box
        
        # Get color for this object type
        color = None
        for key in color_map:
            if key in obj_type_lower:
                color = color_map[key]
                break
        
        if color is None:
            color = color_map['default']
        
        # Draw bounding box
        thickness = 2
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with confidence
        label = f"{obj_type}: {confidence:.2f}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        
        # Draw filled rectangle for text background
        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            text_thickness,
            cv2.LINE_AA
        )
    
    return annotated_image


def save_processed_image(original_image: np.ndarray, detections: list, 
                        output_path: Path, image_format: str = 'jpg',
                        quality: int = 85) -> Tuple[bool, Optional[str]]:
    """
    Save processed image with bounding boxes drawn.
    
    Args:
        original_image: Original image as numpy array (BGR format)
        detections: List of detection dictionaries
        output_path: Path where to save the processed image
        image_format: Output format ('jpg', 'png', etc.)
        quality: Quality for JPEG (1-100)
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Draw bounding boxes on image
        annotated_image = draw_bounding_boxes(original_image, detections)
        
        # Save the image
        if image_format.lower() in ['jpg', 'jpeg']:
            cv2.imwrite(
                str(output_path), 
                annotated_image, 
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
        elif image_format.lower() == 'png':
            cv2.imwrite(
                str(output_path), 
                annotated_image, 
                [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 10)]
            )
        else:
            cv2.imwrite(str(output_path), annotated_image)
        
        logger.info(f"Processed image saved: {output_path}")
        return True, None
        
    except Exception as e:
        logger.error(f"Error saving processed image: {str(e)}")
        return False, str(e)


def generate_processed_filename(original_filename: str, suffix: str = "processed") -> str:
    """
    Generate a unique filename for processed images.
    
    Args:
        original_filename: Original uploaded filename
        suffix: Suffix to add (e.g., 'processed', 'annotated')
    
    Returns:
        Generated filename
    """
    # Get original file extension
    original_path = Path(original_filename)
    extension = original_path.suffix.lower()
    
    if not extension:
        extension = '.jpg'  # Default
    
    # Generate unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
    
    # Clean original filename (remove extension)
    original_stem = original_path.stem
    original_stem = ''.join(c for c in original_stem if c.isalnum() or c in '._- ')
    
    # Build new filename
    new_filename = f"{original_stem}_{suffix}_{timestamp}_{random_str}{extension}"
    
    return new_filename


# app/utils/file_handling.py
def get_processed_image_url(filename: str) -> str:
    """
    Generate PUBLIC URL for processed images.
    """
    # Use simple, predictable URL format
    return f"http://localhost:8001/media/processed/{filename}"



def save_and_get_processed_image(original_image: np.ndarray, detections: list, 
                               original_filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Save processed image and return its URL - OPTIONAL FOR DEBUGGING
    """
    from ..core.config import settings
    
    # Check if we should save images
    if not settings.SAVE_PROCESSED_IMAGES:
        logger.debug("Image saving disabled in config (SAVE_PROCESSED_IMAGES=False)")
        return None, "Image saving disabled in config"
    
    try:
        # Generate filename for processed image
        processed_filename = generate_processed_filename(original_filename)
        processed_path = settings.PROCESSED_IMAGES_PATH / processed_filename
        
        # DEBUG: Log where we're saving
        logger.debug(f"Saving processed image to: {processed_path}")
        logger.debug(f"PROCESSED_IMAGES_PATH: {settings.PROCESSED_IMAGES_PATH}")
        logger.debug(f"Directory exists: {settings.PROCESSED_IMAGES_PATH.exists()}")
        
        # Ensure directory exists
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed image
        success, error = save_processed_image(
            original_image, 
            detections, 
            processed_path,
            image_format='jpg',
            quality=90
        )
        
        if not success:
            return None, error
        
        # Generate URL for the processed image (for debugging only)
        image_url = f"{settings.SERVER_BASE_URL}/media/processed/{processed_filename}"
        
        logger.info(f"Processed image saved (debug mode): {processed_path}")
        logger.debug(f"Debug URL generated: {image_url}")
        return image_url, None
        
    except Exception as e:
        logger.error(f"Error in save_and_get_processed_image: {str(e)}")
        return None, str(e)


def cleanup_old_processed_files(max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up old processed files to save disk space.
    
    Args:
        max_age_hours: Maximum age of files to keep (hours)
    
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        import time
        from datetime import datetime, timedelta
        
        stats = {
            "processed_images": {"total": 0, "deleted": 0, "errors": 0},
            "processed_videos": {"total": 0, "deleted": 0, "errors": 0},
            "total_deleted_size_mb": 0
        }
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean processed images
        if settings.PROCESSED_IMAGES_PATH.exists():
            for file_path in settings.PROCESSED_IMAGES_PATH.glob("*"):
                if file_path.is_file():
                    stats["processed_images"]["total"] += 1
                    
                    # Check file age
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        try:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            stats["processed_images"]["deleted"] += 1
                            stats["total_deleted_size_mb"] += file_size / (1024 * 1024)
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {str(e)}")
                            stats["processed_images"]["errors"] += 1
        
        # Clean processed videos
        if settings.PROCESSED_VIDEOS_PATH.exists():
            for file_path in settings.PROCESSED_VIDEOS_PATH.glob("*"):
                if file_path.is_file():
                    stats["processed_videos"]["total"] += 1
                    
                    # Check file age
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        try:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            stats["processed_videos"]["deleted"] += 1
                            stats["total_deleted_size_mb"] += file_size / (1024 * 1024)
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {str(e)}")
                            stats["processed_videos"]["errors"] += 1
        
        logger.info(f"Cleanup completed: {stats['processed_images']['deleted']} images and "
                   f"{stats['processed_videos']['deleted']} videos deleted")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_processed_files: {str(e)}")
        return {"error": str(e)}
    
    
    
