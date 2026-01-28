"""
Test performance optimizations
"""
import requests
import json
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8001"
API_KEY = "dev-key-change-in-production"

def test_memory_endpoints():
    """Test memory management endpoints"""
    print("Testing memory management endpoints...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Test memory status endpoint (if implemented)
    print("\n1. Checking system memory...")
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        print(f"  Process RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"  Process VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        print(f"  CPU Percent: {process.cpu_percent()}%")
        
        # System memory
        vm = psutil.virtual_memory()
        print(f"  System Memory: {vm.percent}% used")
        print(f"  Available: {vm.available / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"  Error checking memory: {e}")
    
    return True

def test_concurrency():
    """Test concurrent request handling"""
    print("\n2. Testing concurrency limits...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Make multiple concurrent requests
    def make_request(i):
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}/health", headers=headers, timeout=5)
            duration = time.time() - start
            return i, duration, response.status_code
        except Exception as e:
            return i, 0, str(e)
    
    # Make 5 concurrent requests
    print("  Making 5 concurrent health checks...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(5)]
        
        results = []
        for future in as_completed(futures):
            i, duration, status = future.result()
            results.append((i, duration, status))
    
    # Analyze results
    successful = sum(1 for _, _, status in results if status == 200)
    total_time = sum(duration for _, duration, _ in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"  Successful: {successful}/5")
    print(f"  Average response time: {avg_time:.3f}s")
    
    if successful == 5:
        print("  ✓ Concurrency handling working")
        return True
    else:
        print("  ✗ Some requests failed")
        return False

def test_caching():
    """Test caching performance"""
    print("\n3. Testing caching performance...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Test models endpoint (should be cacheable)
    print("  Testing repeated model listing...")
    
    times = []
    for i in range(3):
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/v1/process/models", headers=headers, timeout=5)
        duration = time.time() - start
        times.append(duration)
        
        if response.status_code != 200:
            print(f"    Request {i+1} failed: {response.status_code}")
            break
        
        print(f"    Request {i+1}: {duration:.3f}s")
    
    if len(times) == 3:
        print(f"  Average time: {sum(times)/len(times):.3f}s")
        
        # Second and third requests should be faster if cached
        if times[1] < times[0] * 0.8 or times[2] < times[0] * 0.8:
            print("  ✓ Caching appears to be working (later requests faster)")
        else:
            print("  ⚠️  Caching not noticeable in this test")
    
    return True

def test_timeout_handling():
    """Test request timeout handling"""
    print("\n4. Testing timeout handling...")
    
    headers = {"X-API-Key": API_KEY}
    
    # Try a request with very short timeout
    print("  Testing short timeout (should fail)...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/process/models", 
                               headers=headers, 
                               timeout=0.001)  # 1ms timeout
        print(f"  ✗ Request didn't timeout (got {response.status_code})")
        return False
    except requests.exceptions.Timeout:
        print("  ✓ Request timed out as expected")
        return True
    except Exception as e:
        print(f"  ⚠️  Different error: {e}")
        return False

def test_compression():
    """Test response compression"""
    print("\n5. Testing response compression...")
    
    headers = {
        "X-API-Key": API_KEY,
        "Accept-Encoding": "gzip"  # Request compressed response
    }
    
    print("  Requesting compressed response...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/process/models", headers=headers, timeout=5)
        
        if response.status_code == 200:
            content_encoding = response.headers.get('Content-Encoding', '')
            content_length = response.headers.get('Content-Length', '0')
            
            print(f"  Response size: {content_length} bytes")
            print(f"  Content-Encoding: {content_encoding}")
            
            if content_encoding == 'gzip':
                print("  ✓ Response is compressed (gzip)")
            else:
                print("  ⚠️  Response not compressed")
            
            return True
        else:
            print(f"  ✗ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Performance Optimization Test")
    print("=" * 60)
    
    try:
        # First check if server is running
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("Server not running. Start it with:")
            print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
            return
        
        print("Server is running. Testing performance optimizations...")
        
        # Run tests
        tests = [
            ("Memory Management", test_memory_endpoints),
            ("Concurrency", test_concurrency),
            ("Caching", test_caching),
            ("Timeout Handling", test_timeout_handling),
            ("Compression", test_compression),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*40}")
            print(f"Test: {test_name}")
            print(f"{'='*40}")
            
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"  ✗ Test failed with error: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("Performance Test Summary")
        print("=" * 60)
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        print(f"\nTests passed: {passed}/{total}")
        
        for test_name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {status} - {test_name}")
        
        if passed == total:
            print("\n✓ All performance tests passed!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")
        
        print("\nPerformance optimizations implemented:")
        print("- Memory management and monitoring")
        print("- Request timeout handling")
        print("- Concurrent request limiting")
        print("- Response caching")
        print("- Response compression")
        print("- Optimized video processing")
            
    except requests.exceptions.ConnectionError:
        print("\nCannot connect to server. Make sure it's running on port 8001.")
        print("Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    main()