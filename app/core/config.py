"""
Configuration settings for the application - Simplified version
"""
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

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
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))
    MAX_VIDEO_SIZE: int = int(os.getenv("MAX_VIDEO_SIZE", "52428800"))
    
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
    
    # Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", "processed")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")
    ACCESS_LOG: bool = os.getenv("ACCESS_LOG", "true").lower() == "true"
    
    # Integration
    DJANGO_WEBHOOK_URL: Optional[str] = os.getenv("DJANGO_WEBHOOK_URL")
    WEBHOOK_ENABLED: bool = os.getenv("WEBHOOK_ENABLED", "false").lower() == "true"
    
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
        return Path(__file__).parent.parent.parent
    
    @property
    def UPLOAD_PATH(self) -> Path:
        return self.BASE_DIR / self.UPLOAD_DIR
    
    @property
    def PROCESSED_PATH(self) -> Path:
        return self.BASE_DIR / self.PROCESSED_DIR
    
    @property
    def MODELS_PATH(self) -> Path:
        return self.BASE_DIR / self.MODELS_DIR
    
    @property
    def TEMP_PATH(self) -> Path:
        return self.BASE_DIR / self.TEMP_DIR
    
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
        directories = [
            self.UPLOAD_PATH,
            self.PROCESSED_PATH,
            self.MODELS_PATH,
            self.TEMP_PATH,
            self.UPLOAD_PATH / "images",
            self.UPLOAD_PATH / "videos",
            self.PROCESSED_PATH / "images",
            self.PROCESSED_PATH / "videos",
            self.TEMP_PATH / "frames",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Create settings instance
settings = Settings()
settings.create_directories()