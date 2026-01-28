"""
Logging configuration with structured logging
"""
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import traceback
import os

# Try to import settings, but handle if it fails
try:
    from ..core.config import settings
    HAS_SETTINGS = True
except ImportError:
    HAS_SETTINGS = False
    # Create default settings for logging initialization
    class DefaultSettings:
        LOG_LEVEL = "INFO"
        LOG_FILE = "app.log"
        ENVIRONMENT = "development"
    
    settings = DefaultSettings()

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_object["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_object.update(record.extra)
        
        return json.dumps(log_object, default=str)

class CustomLogger(logging.Logger):
    """Custom logger with additional methods"""
    
    def api_log(self, endpoint: str, method: str, status_code: int, 
                processing_time: float = None, client_ip: str = None, 
                extra: Dict[str, Any] = None):
        """Log API request details"""
        log_data = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "processing_time": processing_time,
            "client_ip": client_ip,
            "type": "api_request",
        }
        
        if extra:
            log_data.update(extra)
        
        self.info(f"API Request: {method} {endpoint} - {status_code}", 
                  extra=log_data)
    
    def processing_log(self, job_id: str, job_type: str, status: str, 
                      details: Dict[str, Any] = None):
        """Log processing job details"""
        log_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": status,
            "type": "processing_job",
        }
        
        if details:
            log_data.update(details)
        
        self.info(f"Processing Job: {job_type} - {status} ({job_id})", 
                  extra=log_data)

def setup_logger() -> CustomLogger:
    """Configure and return application logger"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler (structured JSON in production, pretty in development)
    console_handler = logging.StreamHandler(sys.stdout)
    if HAS_SETTINGS and settings.ENVIRONMENT == "production":
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    console_handler.setFormatter(console_formatter)
    
    # File handler (always JSON)
    file_handler = logging.FileHandler(log_dir / settings.LOG_FILE)
    file_formatter = JSONFormatter()
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Create application logger
    logger = CustomLogger("processing_server")
    
    # Log initialization
    logger.info("Logger initialized", extra={
        "environment": settings.ENVIRONMENT if HAS_SETTINGS else "unknown",
        "log_level": settings.LOG_LEVEL,
        "log_file": str(log_dir / settings.LOG_FILE),
    })
    
    return logger

# Create global logger instance
logger = setup_logger()