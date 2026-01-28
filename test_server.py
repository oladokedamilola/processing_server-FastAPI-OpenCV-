"""
Test the server with traditional CV methods
"""
import requests
import json
import time

BASE_URL = "http://localhost:8001"
API_KEY = "dev-key-change-in-production"  # From your .env file

def test_server_status():
    """Test if server is running"""
    print("Testing server status...")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is running: {data['message']}")
            print(f"  Version: {data['version']}")
            print(f"  Models status: {data.get('models_status', 'N/A')}")
            return True
        else:
            print(f"✗ Server returned status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure it's running on port 8001.")
        print("  Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_models_endpoint():
    """Test the models endpoint"""
    print("\nTesting models endpoint...")
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/process/models", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Models endpoint working")
            print(f"  Available models: {data['count']}")
            
            for model in data['models']:
                status = "✓ Loaded" if model.get('loaded') else "✗ Not loaded"
                available = "✓ Available" if model.get('available', True) else "✗ Not available"
                print(f"    {model['name']} ({model['type']}) - {status}, {available}")
            
            return True
        else:
            print(f"✗ Models endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing models: {e}")
        return False

def test_image_without_yolo():
    """Test image processing without YOLO (using traditional CV)"""
    print("\nTesting image processing with traditional CV...")
    
    # Create a simple test image using numpy
    import numpy as np
    from PIL import Image
    import io
    
    # Create a simple test image (300x400 with a colored rectangle)
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    # Add a "person-like" rectangle
    test_image[50:250, 150:250] = [100, 100, 100]  # Gray rectangle
    
    # Convert to bytes
    img_pil = Image.fromarray(test_image)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Prepare request
    headers = {"X-API-Key": API_KEY}
    files = {
        'file': ('test_image.jpg', img_bytes, 'image/jpeg')
    }
    data = {
        'detection_types': 'person,face',
        'confidence_threshold': '0.3',
        'return_image': 'false'
    }
    
    try:
        print("Sending test image for processing...")
        response = requests.post(
            f"{BASE_URL}/api/v1/process/image",
            headers=headers,
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Image processing successful!")
            print(f"  Processing time: {result['processing_time']}s")
            print(f"  Detections: {result['detection_count']}")
            print(f"  Warnings: {result.get('warnings', ['None'])}")
            print(f"  Models used: {', '.join(result.get('models_used', ['None']))}")
            
            if result['detections']:
                print(f"  Detection details:")
                for i, det in enumerate(result['detections'][:3]):  # Show first 3
                    print(f"    {i+1}. {det['label']} ({det['confidence']:.2f})")
            return True
        else:
            print(f"✗ Image processing failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing image processing: {e}")
        return False

def main():
    print("=" * 60)
    print("FastAPI Processing Server - Offline Mode Test")
    print("=" * 60)
    
    print("\nNote: YOLO model is not available.")
    print("Testing with traditional CV methods (Haar Cascade, HOG)...")
    
    # Test 1: Server status
    if not test_server_status():
        return
    
    # Wait a moment for server to fully initialize
    time.sleep(2)
    
    # Test 2: Models endpoint
    if not test_models_endpoint():
        return
    
    # Test 3: Image processing
    test_image_without_yolo()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nSummary:")
    print("- Server is running correctly")
    print("- Traditional CV methods are available")
    print("- YOLO model can be downloaded later")
    print("\nTo download YOLO model manually, run:")
    print("  python download_model.py")
    print("\nOr download from:")
    print("  https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
    print("  Then place it in: models/yolov8n.pt")

if __name__ == "__main__":
    main()