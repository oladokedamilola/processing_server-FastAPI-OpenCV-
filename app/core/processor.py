# app/core/processor.py
"""
Global processor instance to avoid circular imports
"""

# Global processor instance
processor = None

def init_processor():
    """Initialize the global processor instance"""
    global processor
    if processor is None:
        try:
            from ..processing.image_processor import ImageProcessor
            processor = ImageProcessor()
            return True
        except Exception as e:
            print(f"Failed to initialize processor: {e}")
            return False
    return True

def get_processor():
    """Get the global processor instance"""
    global processor
    if processor is None:
        init_processor()
    return processor