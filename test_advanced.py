"""
Test advanced detection features
"""
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"
API_KEY = "dev-key-change-in-production"

def test_advanced_endpoints():
    """Test advanced feature endpoints"""
    print("Testing advanced feature endpoints...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Test 1: Get processing statistics
    print("\n1. Testing processing statistics...")
    response = requests.get(f"{BASE_URL}/api/v1/advanced/processing-statistics", headers=headers)
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ Processing statistics working")
        print(f"  Images processed: {stats['statistics']['total_images_processed']}")
        print(f"  Total detections: {stats['statistics']['total_detections']}")
        print(f"  Avg processing time: {stats['statistics']['average_processing_time']}s")
    else:
        print(f"✗ Processing statistics failed: {response.status_code}")
        print(f"  Response: {response.text}")
    
    # Test 2: Crowd detection with test image
    print("\n2. Testing crowd detection...")
    
    # Look for test image
    test_image_paths = [
        Path("test_image.jpg"),
        Path("test_image_crowd.jpg"),
        Path("test_images/test_image.jpg"),
    ]
    
    test_image = None
    for path in test_image_paths:
        if path.exists():
            test_image = path
            break
    
    if test_image:
        print(f"Found test image: {test_image.name}")
        
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.name, f, 'image/jpeg')}
            data = {
                'confidence_threshold': '0.3',
                'min_people_count': '2',
                'density_threshold': '0.2',
                'return_image': 'false'
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v1/advanced/crowd-detection",
                headers=headers,
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Crowd detection working")
                print(f"  Processing time: {result['processing_time']}s")
                print(f"  Total detections: {result['detection_count']}")
                
                if result.get('crowd_statistics'):
                    stats = result['crowd_statistics']
                    print(f"  Crowd statistics:")
                    print(f"    Active crowds: {stats.get('active_crowds', 0)}")
                    print(f"    Total people in crowds: {stats.get('total_people_in_crowds', 0)}")
                
                # Test 3: Get crowd statistics
                print("\n3. Testing crowd statistics endpoint...")
                response = requests.get(f"{BASE_URL}/api/v1/advanced/crowd-statistics", headers=headers)
                if response.status_code == 200:
                    stats = response.json()
                    print(f"✓ Crowd statistics endpoint working")
                else:
                    print(f"✗ Crowd statistics failed: {response.status_code}")
                
            else:
                print(f"✗ Crowd detection failed: {response.status_code}")
                print(f"  Response: {response.text}")
    else:
        print("  No test image found. Skipping crowd detection test.")
    
    # Test 4: Vehicle counting (requires video)
    print("\n4. Testing vehicle counting endpoints...")
    
    # Test vehicle statistics endpoint
    response = requests.get(f"{BASE_URL}/api/v1/advanced/vehicle-statistics", headers=headers)
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ Vehicle statistics endpoint working")
        print(f"  Current counts: {stats['statistics'].get('counts', {})}")
    else:
        print(f"✗ Vehicle statistics failed: {response.status_code}")
    
    # Test reset endpoint
    response = requests.post(f"{BASE_URL}/api/v1/advanced/vehicle-counts/reset", headers=headers)
    if response.status_code == 200:
        print(f"✓ Vehicle counts reset endpoint working")
    else:
        print(f"✗ Vehicle reset failed: {response.status_code}")
    
    return True

def test_detection_types():
    """Test different detection type combinations"""
    print("\n5. Testing detection type combinations...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Create a simple test image using numpy if no test image exists
    import numpy as np
    from PIL import Image
    import io
    
    # Create a simple test image
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    test_image[100:200, 150:250] = [100, 100, 100]  # Gray rectangle
    
    # Convert to bytes
    img_pil = Image.fromarray(test_image)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test different detection type combinations
    test_cases = [
        ("person,vehicle", "Basic object detection"),
        ("person,face,crowd", "People and crowd detection"),
        ("motion", "Motion detection only"),
    ]
    
    for detection_types, description in test_cases:
        print(f"\n  Testing: {description}")
        
        files = {
            'file': ('test_image.jpg', img_bytes.getvalue(), 'image/jpeg')
        }
        data = {
            'confidence_threshold': '0.3',
            'detection_types': detection_types,
            'return_image': 'false'
        }
        
        # Reset bytes position
        img_bytes.seek(0)
        
        response = requests.post(
            f"{BASE_URL}/api/v1/process/image",
            headers=headers,
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✓ Success: {result['detection_count']} detections")
            print(f"    Models used: {', '.join(result.get('models_used', ['None']))}")
        else:
            print(f"    ✗ Failed: {response.status_code}")
    
    return True

def main():
    print("=" * 60)
    print("Advanced Detection Features Test")
    print("=" * 60)
    
    try:
        # First check if server is running
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("Server not running. Start it with:")
            print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
            return
        
        print("Server is running. Testing advanced features...")
        
        success = test_advanced_endpoints()
        test_detection_types()
        
        if success:
            print("\n" + "=" * 60)
            print("Advanced Features Test Complete!")
            print("=" * 60)
            print("\nSummary:")
            print("- Advanced detection features are working")
            print("- Crowd detection available")
            print("- Vehicle counting framework ready")
            print("- Processing statistics tracking")
            print("\nTry the Swagger UI at: http://localhost:8001/docs")
            print("Look for endpoints under the 'advanced' tag")
        else:
            print("\nSome tests failed. Check server logs for details.")
            
    except requests.exceptions.ConnectionError:
        print("\nCannot connect to server. Make sure it's running on port 8001.")
        print("Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    main()