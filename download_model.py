"""
Script to manually download the YOLOv8 model
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def download_yolo_model():
    """Download YOLOv8 model manually"""
    print("Downloading YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        from app.core.config import settings
        
        # Create models directory if it doesn't exist
        models_dir = settings.MODELS_PATH
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "yolov8n.pt"
        
        if model_path.exists():
            print(f"Model already exists at: {model_path}")
            print(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True
        
        print("Downloading model (this may take a moment)...")
        
        # Download the model
        model = YOLO("yolov8n.pt")
        
        # Save it locally
        model.save(str(model_path))
        
        print(f"Model downloaded successfully!")
        print(f"Location: {model_path}")
        print(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nAlternative: Download manually from:")
        print("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        print(f"\nThen place it in: {models_dir}")
        return False

def check_models():
    """Check what models are available"""
    print("\nChecking available models...")
    
    try:
        from app.core.config import settings
        
        models_dir = settings.MODELS_PATH
        print(f"Models directory: {models_dir}")
        
        if not models_dir.exists():
            print("Models directory does not exist.")
            return
        
        model_files = list(models_dir.glob("*.pt"))
        
        if model_files:
            print(f"Found {len(model_files)} model(s):")
            for model_file in model_files:
                size_mb = model_file.stat().st_size / 1024 / 1024
                print(f"  - {model_file.name} ({size_mb:.2f} MB)")
        else:
            print("No model files found.")
            
    except Exception as e:
        print(f"Error checking models: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO Model Download Utility")
    print("=" * 60)
    
    # Check current models
    check_models()
    
    # Ask if user wants to download
    response = input("\nDo you want to download the YOLOv8n model? (y/n): ").strip().lower()
    
    if response == 'y':
        success = download_yolo_model()
        if success:
            print("\n✓ Model download completed!")
        else:
            print("\n✗ Model download failed.")
    else:
        print("\nSkipping download.")
    
    print("\nYou can still use the server with traditional CV methods:")
    print("- Haar Cascade for face detection")
    print("- HOG for people detection")
    print("\nTo enable YOLO later, download the model and restart the server.")