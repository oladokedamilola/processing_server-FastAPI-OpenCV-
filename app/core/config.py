"""
Configuration settings for the application
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Server Configuration
    APP_NAME: str = "FastAPI Processing Server"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = True
    
    # Security
    API_KEY: str = "dev-key-change-in-production"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    
    # File Upload Limits (in bytes)
    MAX_IMAGE_SIZE: int = 10485760  # 10MB
    MAX_VIDEO_SIZE: int = 52428800  # 50MB
    
    # Processing Configuration
    YOLO_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    DETECTION_TIMEOUT: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()