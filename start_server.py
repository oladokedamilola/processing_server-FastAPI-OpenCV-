"""
Start server with proper Python path configuration
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run uvicorn
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable (HF Spaces uses 7860)
    port = int(os.getenv("PORT", 7860))
    
    print(f"Starting FastAPI server on port {port}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    
    # Check if models directory exists
    models_dir = os.path.join(project_root, "models")
    if os.path.exists(models_dir):
        print(f"Models directory found: {models_dir}")
        print(f"Files in models: {os.listdir(models_dir)}")
    else:
        print("WARNING: Models directory not found!")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production (HF Spaces)
        log_level="info",
        workers=1  # Single worker for memory efficiency
    )