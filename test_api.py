"""
Test script for API endpoints
"""
import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8001"
API_KEY = "dev-key-change-in-production"  # From your .env file

def test_health_endpoints():
    """Test health check endpoints"""
    print("Testing health endpoints...")
    
    # Test root endpoint
    response = requests.get(f"{BASE_URL}/")
    print(f"GET / - Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Test public health endpoint
    response = requests.get(f"{BASE_URL}/health")
    print(f"\nGET /health - Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Test system info with valid API key
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/api/v1/health/system", headers=headers)
    print(f"\nGET /api/v1/health/system (with valid API key) - Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))

def test_models_endpoints():
    """Test model management endpoints"""
    print("\n" + "=" * 50)
    print("Testing model endpoints...")
    
    headers = {"X-API-Key": API_KEY}
    
    # List models
    response = requests.get(f"{BASE_URL}/api/v1/process/models", headers=headers)
    print(f"GET /api/v1/process/models - Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))

def test_image_processing():
    """Test image processing endpoint"""
    print("\n" + "=" * 50)
    print("Testing image processing...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Look for test images
    test_image_paths = []
    
    # Check common test image locations
    possible_paths = [
        Path("test_image.jpg"),
        Path("test_images/test_image.jpg"),
        Path("app/test_image.jpg"),
        Path("uploads/images/test_image.jpg"),
    ]
    
    for path in possible_paths:
        if path.exists():
            test_image_paths.append(path)
    
    if not test_image_paths:
        print("No test images found. Please add a test image to continue.")
        print("You can download a sample image or use any JPEG/PNG image.")
        return
    
    print(f"Found {len(test_image_paths)} test images")
    
    # Test with first found image
    image_path = test_image_paths[0]
    
    # Test basic image processing
    files = {
        'file': (image_path.name, open(image_path, 'rb'), 'image/jpeg')
    }
    data = {
        'confidence_threshold': '0.5',
        'return_image': 'false'
    }
    
    print(f"\nProcessing image: {image_path.name}")
    response = requests.post(
        f"{BASE_URL}/api/v1/process/image",
        headers=headers,
        files=files,
        data=data
    )
    
    print(f"POST /api/v1/process/image - Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Processing time: {result['processing_time']}s")
        print(f"Detections: {result['detection_count']}")
        print(f"Image size: {result['image_size']}")
        
        if result['detections']:
            print("\nTop 5 detections:")
            for i, detection in enumerate(result['detections'][:5]):
                print(f"  {i+1}. {detection['label']} ({detection['confidence']:.2f})")
    elif response.status_code == 400:
        print(f"Error: {response.json()}")
    else:
        print(f"Unexpected status: {response.status_code}")
        print(response.text)

def test_authentication():
    """Test authentication requirements"""
    print("\n" + "=" * 50)
    print("Testing authentication requirements...")
    
    # Test without API key (should fail with 401)
    print("\nTesting without API key...")
    response = requests.get(f"{BASE_URL}/api/v1/process/models")
    print(f"GET /api/v1/process/models (no API key) - Status: {response.status_code}")
    if response.status_code != 200:
        print(json.dumps(response.json(), indent=2))
    
    # Test with invalid API key
    print("\nTesting with invalid API key...")
    headers = {"X-API-Key": "invalid-key"}
    response = requests.get(f"{BASE_URL}/api/v1/process/models", headers=headers)
    print(f"GET /api/v1/process/models (invalid API key) - Status: {response.status_code}")
    if response.status_code != 200:
        print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("=" * 60)
    print("FastAPI Processing Server - Comprehensive API Tests")
    print("=" * 60)
    
    try:
        test_health_endpoints()
        test_models_endpoints()
        test_image_processing()
        test_authentication()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to server. Make sure the server is running on port 8001.")
        print("Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")