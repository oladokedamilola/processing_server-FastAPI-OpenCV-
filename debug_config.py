"""
Debug configuration loading
"""
import os
from pathlib import Path

# First, let's check what's in the .env file
env_path = Path(".env")
print(f"Checking .env file at: {env_path.absolute()}")
print(f".env exists: {env_path.exists()}")

if env_path.exists():
    print("\nContents of .env file:")
    print("=" * 50)
    with open(env_path, 'r') as f:
        print(f.read())
    print("=" * 50)

# Try to load environment variables manually
print("\nTrying to load environment variables...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Successfully loaded .env file")
    
    # Check specific variables
    print("\nChecking ALLOWED_ORIGINS:")
    allowed_origins = os.getenv("ALLOWED_ORIGINS")
    print(f"  Raw value: {allowed_origins}")
    print(f"  Type: {type(allowed_origins)}")
    
    if allowed_origins:
        parsed = [origin.strip() for origin in allowed_origins.split(',') if origin.strip()]
        print(f"  Parsed: {parsed}")
    
except Exception as e:
    print(f"❌ Error loading .env: {e}")

# Test the simplified settings
print("\n\nTesting simplified settings...")
try:
    # Import after loading dotenv
    from app.core.config import Settings
    
    settings = Settings()
    print("✅ Settings created successfully!")
    print(f"  APP_NAME: {settings.APP_NAME}")
    print(f"  ENVIRONMENT: {settings.ENVIRONMENT}")
    print(f"  ALLOWED_ORIGINS: {settings.ALLOWED_ORIGINS}")
    print(f"  ALLOWED_IMAGE_TYPES: {settings.ALLOWED_IMAGE_TYPES}")
    print(f"  ALLOWED_VIDEO_TYPES: {settings.ALLOWED_VIDEO_TYPES}")
    
    # Check directories
    print(f"\n  UPLOAD_PATH: {settings.UPLOAD_PATH}")
    print(f"  UPLOAD_PATH exists: {settings.UPLOAD_PATH.exists()}")
    
except Exception as e:
    print(f"❌ Error creating settings: {e}")
    import traceback
    traceback.print_exc()