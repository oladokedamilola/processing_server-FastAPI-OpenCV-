# app/core/config.py
"""
Configuration settings for the application - Simplified version
"""
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Settings:
    # Server Configuration
    APP_NAME: str = os.getenv("APP_NAME", "FastAPI Processing Server")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # API Configuration
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")
    API_V1_PREFIX: str = os.getenv("API_V1_PREFIX", "/api/v1")
    PROJECT_DESCRIPTION: str = os.getenv(
        "PROJECT_DESCRIPTION", 
        "FastAPI server for computer vision processing in smart surveillance systems"
    )
    
    # Security
    API_KEY: str = os.getenv("API_KEY", "dev-key-change-in-production")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-32-characters-min")
    API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_PERIOD: int = int(os.getenv("RATE_LIMIT_PERIOD", "900"))
    
    # File Upload Limits (in bytes)
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
    MAX_VIDEO_SIZE: int = int(os.getenv("MAX_VIDEO_SIZE", "52428800"))  # 50MB
    
    # Processing Configuration
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    DETECTION_TIMEOUT: int = int(os.getenv("DETECTION_TIMEOUT", "30"))
    VIDEO_FRAME_SAMPLE_RATE: int = int(os.getenv("VIDEO_FRAME_SAMPLE_RATE", "5"))
    MAX_VIDEO_DURATION: int = int(os.getenv("MAX_VIDEO_DURATION", "300"))
    
    # Job Queue Configuration
    JOB_TIMEOUT: int = int(os.getenv("JOB_TIMEOUT", "3600"))
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
    JOB_CLEANUP_INTERVAL: int = int(os.getenv("JOB_CLEANUP_INTERVAL", "300"))
    
    # Storage Configuration - Simplified for Django architecture
    # Set SAVE_PROCESSED_IMAGES to False since Django handles storage
    SAVE_PROCESSED_IMAGES: bool = os.getenv("SAVE_PROCESSED_IMAGES", "false").lower() == "true"
    SAVE_PROCESSED_VIDEOS: bool = os.getenv("SAVE_PROCESSED_VIDEOS", "false").lower() == "true"
    
    # Only create minimal temp directories
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    
    # NEW: Return base64 images by default for Django integration
    RETURN_BASE64_IMAGES: bool = os.getenv("RETURN_BASE64_IMAGES", "true").lower() == "true"
    RETURN_BASE64_VIDEOS: bool = os.getenv("RETURN_BASE64_VIDEOS", "false").lower() == "true"
    
    # Video compression for base64 return (if enabled)
    BASE64_IMAGE_QUALITY: int = int(os.getenv("BASE64_IMAGE_QUALITY", "85"))
    BASE64_VIDEO_QUALITY: int = int(os.getenv("BASE64_VIDEO_QUALITY", "70"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")
    ACCESS_LOG: bool = os.getenv("ACCESS_LOG", "true").lower() == "true"
    
    # Integration with Django
    DJANGO_WEBHOOK_URL: Optional[str] = os.getenv("DJANGO_WEBHOOK_URL")
    WEBHOOK_ENABLED: bool = os.getenv("WEBHOOK_ENABLED", "false").lower() == "true"
    DJANGO_BASE_URL: str = os.getenv("DJANGO_BASE_URL", "http://localhost:8000")
    
    # Performance
    ENABLE_GZIP: bool = os.getenv("ENABLE_GZIP", "true").lower() == "true"
    MAX_UPLOAD_WORKERS: int = int(os.getenv("MAX_UPLOAD_WORKERS", "4"))
    
    # Video Processing
    VIDEO_PREPROCESSING_ENABLED: bool = True
    VIDEO_TARGET_FPS: float = 10.0
    VIDEO_TARGET_WIDTH: int = 640
    VIDEO_TARGET_HEIGHT: int = 480
    VIDEO_MAX_DURATION: int = 300  # 5 minutes
    VIDEO_CONVERSION_ENABLED: bool = True
    VIDEO_CONVERSION_CRF: int = 23
    VIDEO_CONVERSION_PRESET: str = "medium"
    
    # Frame Extraction
    MAX_EXTRACTED_FRAMES: int = 1000
    DEFAULT_FRAME_SAMPLE_RATE: int = 5
    
    # FFmpeg Path (optional)
    FFMPEG_PATH: Optional[str] = None
    
    # Timeout Settings
    REQUEST_TIMEOUT: int = 30  # Seconds for synchronous requests
    VIDEO_PROCESSING_TIMEOUT: int = 1800  # 30 minutes for video processing
    MODEL_LOAD_TIMEOUT: int = 60  # Seconds for model loading
    FILE_UPLOAD_TIMEOUT: int = 300  # 5 minutes for file uploads
    
    # Concurrency Limits
    MAX_CONCURRENT_REQUESTS: int = 10
    MAX_CONCURRENT_UPLOADS: int = 3
    MAX_CONCURRENT_VIDEO_JOBS: int = 2
    
    
    @property
    def BASE_DIR(self) -> Path:
        """Get the correct project root directory"""
        try:
            # Method 1: Try environment variable first (for deployment)
            env_base = os.getenv("PROJECT_ROOT")
            if env_base:
                base_path = Path(env_base).resolve()
                if base_path.exists():
                    logger.info(f"Using PROJECT_ROOT from env: {base_path}")
                    return base_path
            
            # Method 2: Get the directory containing the 'app' folder
            # Find where this config.py file is located
            config_file_path = Path(__file__).resolve().absolute()
            
            # Start from config.py and go up until we find 'app' directory
            current_path = config_file_path.parent  # Start at core/
            max_levels = 10  # Safety limit
            
            for _ in range(max_levels):
                # Check if 'app' directory exists here
                app_dir = current_path / "app"
                if app_dir.exists() and app_dir.is_dir():
                    # Found it! current_path is the project root
                    logger.info(f"Found project root at: {current_path}")
                    return current_path
                
                # Go up one level
                parent_path = current_path.parent
                if parent_path == current_path:  # Reached root
                    break
                current_path = parent_path
            
            # Method 3: Fall back to current working directory
            cwd = Path.cwd()
            if (cwd / "app").exists():
                logger.info(f"Using current directory as project root: {cwd}")
                return cwd
            
            # Method 4: Last resort - hardcoded for your local dev
            # Remove this in production!
            local_path = Path(r"C:\Users\PC\Desktop\processing_server (FastAPI + OpenCV)\processing_server")
            if local_path.exists():
                logger.warning(f"Using hardcoded local path: {local_path}")
                return local_path
            
            # Ultimate fallback
            logger.error("Could not determine project root!")
            return Path.cwd()
            
        except Exception as e:
            logger.error(f"Error determining BASE_DIR: {e}")
            return Path.cwd()
    
    @property
    def MODELS_PATH(self) -> Path:
        return self.BASE_DIR / self.MODELS_DIR
    
    @property
    def TEMP_PATH(self) -> Path:
        return self.BASE_DIR / self.TEMP_DIR
    
    # Optional: Keep processed directories for debugging only
    @property
    def PROCESSED_IMAGES_PATH(self) -> Path:
        """Path for processed images with bounding boxes (debugging only)"""
        if self.SAVE_PROCESSED_IMAGES:
            return self.BASE_DIR / "processed" / "images"
        return self.TEMP_PATH / "processed_debug" / "images"
    
    @property
    def PROCESSED_VIDEOS_PATH(self) -> Path:
        """Path for processed videos with annotations (debugging only)"""
        if self.SAVE_PROCESSED_VIDEOS:
            return self.BASE_DIR / "processed" / "videos"
        return self.TEMP_PATH / "processed_debug" / "videos"
    
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Parse ALLOWED_ORIGINS from environment variable"""
        origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
        return [origin.strip() for origin in origins_str.split(',') if origin.strip()]
    
    @property
    def ALLOWED_IMAGE_TYPES(self) -> List[str]:
        """Parse ALLOWED_IMAGE_TYPES from environment variable"""
        types_str = os.getenv("ALLOWED_IMAGE_TYPES", "image/jpeg,image/png,image/jpg")
        return [item.strip() for item in types_str.split(',') if item.strip()]
    
    @property
    def ALLOWED_VIDEO_TYPES(self) -> List[str]:
        """Parse ALLOWED_VIDEO_TYPES from environment variable"""
        types_str = os.getenv("ALLOWED_VIDEO_TYPES", "video/mp4,video/avi,video/mov")
        return [item.strip() for item in types_str.split(',') if item.strip()]
    
    def create_directories(self):
        """Create required directories if they don't exist"""
        # Always create these
        required_dirs = [
            self.MODELS_PATH,
            self.TEMP_PATH,
            self.TEMP_PATH / "uploads",
            self.TEMP_PATH / "frames",
        ]
        
        # Only create processed directories if enabled
        if self.SAVE_PROCESSED_IMAGES:
            required_dirs.append(self.PROCESSED_IMAGES_PATH)
        if self.SAVE_PROCESSED_VIDEOS:
            required_dirs.append(self.PROCESSED_VIDEOS_PATH)
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

# Create settings instance
settings = Settings()
settings.create_directories()

# Log the configuration
logger.info(f"Application: {settings.APP_NAME} v{settings.APP_VERSION}")
logger.info(f"Environment: {settings.ENVIRONMENT}")
logger.info(f"Save processed images: {settings.SAVE_PROCESSED_IMAGES}")
logger.info(f"Return base64 images: {settings.RETURN_BASE64_IMAGES}")
logger.info(f"Base directory: {settings.BASE_DIR}")