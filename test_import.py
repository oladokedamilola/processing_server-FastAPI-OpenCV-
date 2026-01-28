"""
Test script to check if imports work correctly
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing imports...")

try:
    from app.core.config import settings
    print("✓ Config module imported")
    
    from app.utils.logger import logger
    print("✓ Logger module imported")
    
    from app.models.detection import Detection, DetectionType
    print("✓ Detection module imported")
    
    from app.core.processor import init_processor, get_processor
    print("✓ Processor module imported")
    
    # Try to initialize processor
    if init_processor():
        print("✓ Processor initialized successfully")
        processor = get_processor()
        print(f"✓ Processor retrieved: {type(processor)}")
    else:
        print("✗ Failed to initialize processor")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"Python path: {sys.path}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nAll imports tested!")