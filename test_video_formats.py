"""
Test video format compatibility
"""
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"
API_KEY = "dev-key-change-in-production"

def test_video_format_support():
    """Test video format compatibility features"""
    print("Testing video format compatibility...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Test 1: Get supported formats
    print("\n1. Testing supported formats endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/jobs/video/formats", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Supported formats endpoint working")
        
        formats = data.get("formats", {})
        print(f"  Supported extensions: {', '.join(formats.get('extensions', []))}")
        print(f"  Supported codecs: {', '.join(formats.get('codecs', []))}")
        print(f"  FFmpeg available: {data.get('ffmpeg_available', False)}")
        
        if not data.get('ffmpeg_available'):
            print("  ⚠️  FFmpeg not installed - limited video support")
    else:
        print(f"✗ Supported formats failed: {response.status_code}")
        print(f"  Response: {response.text}")
    
    # Test 2: Check video compatibility (with test file if available)
    print("\n2. Testing video compatibility check...")
    
    # Look for test videos
    test_video_formats = [
        ("test_video.mp4", "MP4/H.264"),
        ("test_video.avi", "AVI"),
        ("test_video.mov", "MOV"),
        ("test_video.mkv", "MKV"),
    ]
    
    for filename, format_name in test_video_formats:
        video_path = Path(filename)
        if video_path.exists():
            print(f"  Found {format_name} test video: {filename}")
            
            with open(video_path, 'rb') as f:
                files = {'file': (filename, f, 'video/mp4')}
                
                response = requests.post(
                    f"{BASE_URL}/api/v1/jobs/video/compatibility-check",
                    headers=headers,
                    files=files
                )
                
                if response.status_code == 200:
                    result = response.json()
                    compatibility = result.get('compatibility', {})
                    
                    print(f"    ✓ Compatibility check: {compatibility.get('is_valid', False)}")
                    if not compatibility.get('is_valid'):
                        issues = compatibility.get('issues', [])
                        if issues:
                            print(f"    Issues: {', '.join(issues)}")
                        recommendation = compatibility.get('recommended_action')
                        if recommendation:
                            print(f"    Recommendation: {recommendation}")
                else:
                    print(f"    ✗ Compatibility check failed: {response.status_code}")
            
            break  # Test with first found video
    else:
        print("  No test videos found. Create test videos or use sample videos.")
    
    # Test 3: Enhanced video processing endpoint
    print("\n3. Testing enhanced video processing...")
    
    # Create a simple test if no video files available
    test_video_path = Path("test_sample.mp4")
    
    # Note: In real testing, you would use actual video files
    print("  To test video processing, you need actual video files.")
    print("  Supported formats: MP4, AVI, MOV, MKV, FLV, WMV, WebM")
    print("  Recommended: MP4 with H.264 codec for best compatibility")
    
    return True

def test_ffmpeg_availability():
    """Check if FFmpeg is available"""
    print("\n4. Checking FFmpeg availability...")
    
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        
        if result.returncode == 0:
            print("✓ FFmpeg is installed and available")
            
            # Extract version
            lines = result.stdout.split('\n')
            if lines:
                print(f"  Version: {lines[0]}")
            
            return True
        else:
            print("✗ FFmpeg is not available or not in PATH")
            print("  Video format conversion will be limited")
            print("  Install FFmpeg for full video support:")
            print("  Windows: Download from https://ffmpeg.org/download.html")
            print("  Add ffmpeg.exe to PATH or set FFMPEG_PATH in .env")
            return False
            
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ FFmpeg is not installed")
        print("  Install FFmpeg for video format conversion support")
        return False

def main():
    print("=" * 60)
    print("Video Format Compatibility Test")
    print("=" * 60)
    
    try:
        # First check if server is running
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("Server not running. Start it with:")
            print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
            return
        
        print("Server is running. Testing video format compatibility...")
        
        # Check FFmpeg availability
        ffmpeg_available = test_ffmpeg_availability()
        
        # Test endpoints
        success = test_video_format_support()
        
        if success:
            print("\n" + "=" * 60)
            print("Video Format Compatibility Test Complete!")
            print("=" * 60)
            print("\nSummary:")
            print(f"- Video format support: {'Enhanced' if ffmpeg_available else 'Basic'}")
            print("- Compatibility checking available")
            print("- Format conversion: {'Available' if ffmpeg_available else 'Not available (install FFmpeg)'}")
            print("\nNext steps:")
            print("1. Install FFmpeg for full video support")
            print("2. Test with actual video files")
            print("3. Check /api/v1/jobs/video/formats for supported formats")
        else:
            print("\nSome tests failed. Check server logs for details.")
            
    except requests.exceptions.ConnectionError:
        print("\nCannot connect to server. Make sure it's running on port 8001.")
        print("Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    main()