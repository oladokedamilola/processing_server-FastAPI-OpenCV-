"""
Logging configuration
"""
import logging
import sys
from ..core.config import settings

def setup_logger():
    """Configure application logger"""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, settings.LOG_FILE)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logger initialized with level: {settings.LOG_LEVEL}")
    
    return logger

logger = setup_logger()