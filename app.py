"""
Hugging Face Spaces entry point
"""
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    # Import your main FastAPI application
    from app.main import app
    
    # HF Spaces looks for a variable named 'app' or 'application'
    application = app
    
    print("✓ FastAPI app imported successfully")
    print(f"✓ App title: {app.title}")
    print(f"✓ App version: {app.version}")
    
except ImportError as e:
    print(f"✗ Error importing FastAPI app: {e}")
    print("✗ Make sure 'app/main.py' exists and contains a FastAPI instance")
    
    # Create a minimal app for debugging
    from fastapi import FastAPI
    application = FastAPI(title="Fallback App", version="1.0.0")
    
    @application.get("/")
    async def root():
        return {"error": "Main app failed to load", "message": str(e)}
    
    @application.get("/health")
    async def health():
        return {"status": "degraded", "reason": "import_failed"}

# HF Spaces will look for 'app' variable
app = application