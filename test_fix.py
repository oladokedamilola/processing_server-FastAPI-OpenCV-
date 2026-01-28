"""
Test the configuration fix
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.core.config import settings
    print("✅ Configuration loaded successfully!")
    print(f"App Name: {settings.APP_NAME}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Allowed Origins: {settings.ALLOWED_ORIGINS}")
    print(f"API Key Header: {settings.API_KEY_HEADER}")
    print("\n✅ Directories created:")
    print(f"  Upload Path: {settings.UPLOAD_PATH}")
    print(f"  Models Path: {settings.MODELS_PATH}")
    print(f"  Processed Path: {settings.PROCESSED_PATH}")
    
    # Test directory creation
    if settings.UPLOAD_PATH.exists():
        print("✅ Upload directory exists")
    else:
        print("❌ Upload directory doesn't exist")
        
except Exception as e:
    print(f"❌ Error loading configuration: {e}")
    import traceback
    traceback.print_exc()