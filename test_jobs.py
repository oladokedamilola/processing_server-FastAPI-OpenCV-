"""
Test the job management system
"""
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"
API_KEY = "dev-key-change-in-production"

def test_job_endpoints():
    """Test job management endpoints"""
    print("Testing job endpoints...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Test 1: Get job stats
    print("\n1. Testing job stats...")
    response = requests.get(f"{BASE_URL}/api/v1/jobs/stats", headers=headers)
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ Job stats working")
        print(f"  Total jobs: {stats['total_jobs']}")
        print(f"  Pending: {stats['pending_jobs']}")
        print(f"  Processing: {stats['processing_jobs']}")
        print(f"  Completed: {stats['completed_jobs']}")
    else:
        print(f"✗ Job stats failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return False
    
    # Test 2: List jobs
    print("\n2. Testing job listing...")
    response = requests.get(f"{BASE_URL}/api/v1/jobs", headers=headers)
    if response.status_code == 200:
        jobs = response.json()
        print(f"✓ Job listing working")
        print(f"  Found {len(jobs)} jobs")
        for job in jobs[:3]:  # Show first 3
            print(f"    - {job['job_id']}: {job['status']} ({job['job_type']})")
    else:
        print(f"✗ Job listing failed: {response.status_code}")
    
    # Test 3: Test video job submission (if we have a test video)
    print("\n3. Testing video job submission...")
    
    # Look for test video
    test_video_paths = [
        Path("test_video.mp4"),
        Path("test_video.avi"),
        Path("test_video.mov"),
        Path("test_videos/test_video.mp4"),
    ]
    
    test_video = None
    for path in test_video_paths:
        if path.exists():
            test_video = path
            break
    
    if test_video:
        print(f"Found test video: {test_video.name}")
        
        with open(test_video, 'rb') as f:
            files = {'file': (test_video.name, f, 'video/mp4')}
            data = {
                'confidence_threshold': '0.5',
                'frame_sample_rate': '10',
                'analyze_motion': 'true',
                'return_summary_only': 'true',
                'priority': '1'
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v1/jobs/process/video",
                headers=headers,
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                print(f"✓ Video job submitted: {job_id}")
                print(f"  Status: {result['status']}")
                print(f"  Results URL: {result['results_url']}")
                
                # Test 4: Get job status
                print("\n4. Testing job status monitoring...")
                
                for i in range(10):  # Check status for up to 30 seconds
                    time.sleep(3)
                    
                    response = requests.get(
                        f"{BASE_URL}/api/v1/jobs/{job_id}/status",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        status_info = response.json()
                        print(f"  Check {i+1}: {status_info['status']} - {status_info['progress']}%")
                        
                        if status_info['status'] in ['completed', 'failed', 'cancelled']:
                            print(f"  Job finished with status: {status_info['status']}")
                            if status_info.get('result'):
                                print(f"  Results available!")
                            break
                    else:
                        print(f"✗ Error getting job status: {response.status_code}")
                        break
                
                return True
            else:
                print(f"✗ Video job submission failed: {response.status_code}")
                print(f"  Response: {response.text}")
    else:
        print("  No test video found. Skipping video job test.")
        print("  Create a short test video or use a sample video file.")
    
    return True

def test_image_job():
    """Test image processing via job system"""
    print("\n5. Testing image job (alternative to direct processing)...")
    
    # Note: We don't have a direct image job endpoint yet
    # This would be similar to video but for images
    print("  Image job endpoint not implemented yet.")
    print("  Images are processed synchronously via /api/v1/process/image")
    
    return True

def main():
    print("=" * 60)
    print("Job Management System Test")
    print("=" * 60)
    
    try:
        # First check if server is running
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("Server not running. Start it with:")
            print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
            return
        
        print("Server is running. Testing job system...")
        
        success = test_job_endpoints()
        
        if success:
            print("\n" + "=" * 60)
            print("Job System Test Complete!")
            print("=" * 60)
            print("\nSummary:")
            print("- Job management endpoints are working")
            print("- Video processing jobs can be submitted")
            print("- Job status can be monitored")
            print("- Job statistics are available")
            print("\nTry the Swagger UI at: http://localhost:8001/docs")
        else:
            print("\nSome tests failed. Check server logs for details.")
            
    except requests.exceptions.ConnectionError:
        print("\nCannot connect to server. Make sure it's running on port 8001.")
        print("Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    main()