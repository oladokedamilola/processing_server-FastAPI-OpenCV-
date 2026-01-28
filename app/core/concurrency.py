"""
Concurrency control for resource-constrained environments
"""
import asyncio
import threading
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict

from ..core.config import settings
from ..utils.logger import logger

@dataclass
class ConcurrencyStats:
    """Concurrency statistics"""
    active_requests: int
    active_uploads: int
    active_video_jobs: int
    total_requests: int
    total_uploads: int
    total_video_jobs: int
    queue_sizes: Dict[str, int]

class ConcurrencyLimiter:
    """Limit concurrent operations to prevent resource exhaustion"""
    
    def __init__(self):
        # Semaphores for different operation types
        self.request_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        self.upload_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_UPLOADS)
        self.video_job_semaphore = threading.Semaphore(settings.MAX_CONCURRENT_VIDEO_JOBS)
        
        # Counters
        self.active_counts = {
            "requests": 0,
            "uploads": 0,
            "video_jobs": 0
        }
        
        self.total_counts = {
            "requests": 0,
            "uploads": 0,
            "video_jobs": 0
        }
        
        self.queue_sizes = {
            "requests": 0,
            "uploads": 0,
            "video_jobs": 0
        }
        
        self.lock = threading.Lock()
        
        logger.info(f"Concurrency limiter initialized: "
                   f"{settings.MAX_CONCURRENT_REQUESTS} requests, "
                   f"{settings.MAX_CONCURRENT_UPLOADS} uploads, "
                   f"{settings.MAX_CONCURRENT_VIDEO_JOBS} video jobs")
    
    @asynccontextmanager
    async def limit_requests(self):
        """Limit concurrent API requests"""
        self.queue_sizes["requests"] += 1
        
        async with self.request_semaphore:
            with self.lock:
                self.queue_sizes["requests"] -= 1
                self.active_counts["requests"] += 1
                self.total_counts["requests"] += 1
            
            start_time = time.time()
            
            try:
                yield
            finally:
                with self.lock:
                    self.active_counts["requests"] -= 1
                
                duration = time.time() - start_time
                if duration > 5.0:  # Log slow requests
                    logger.debug(f"Request completed in {duration:.1f}s")
    
    @asynccontextmanager
    async def limit_uploads(self):
        """Limit concurrent file uploads"""
        self.queue_sizes["uploads"] += 1
        
        async with self.upload_semaphore:
            with self.lock:
                self.queue_sizes["uploads"] -= 1
                self.active_counts["uploads"] += 1
                self.total_counts["uploads"] += 1
            
            start_time = time.time()
            
            try:
                yield
            finally:
                with self.lock:
                    self.active_counts["uploads"] -= 1
                
                duration = time.time() - start_time
                logger.debug(f"Upload completed in {duration:.1f}s")
    
    @contextmanager
    def limit_video_jobs(self):
        """Limit concurrent video processing jobs"""
        self.queue_sizes["video_jobs"] += 1
        
        acquired = self.video_job_semaphore.acquire(timeout=60)  # Wait up to 60 seconds
        
        if not acquired:
            self.queue_sizes["video_jobs"] -= 1
            raise Exception("Timeout waiting for video job slot")
        
        with self.lock:
            self.queue_sizes["video_jobs"] -= 1
            self.active_counts["video_jobs"] += 1
            self.total_counts["video_jobs"] += 1
        
        start_time = time.time()
        
        try:
            yield
        finally:
            with self.lock:
                self.active_counts["video_jobs"] -= 1
            self.video_job_semaphore.release()
            
            duration = time.time() - start_time
            logger.info(f"Video job completed in {duration:.1f}s")
    
    def get_stats(self) -> ConcurrencyStats:
        """Get current concurrency statistics"""
        with self.lock:
            return ConcurrencyStats(
                active_requests=self.active_counts["requests"],
                active_uploads=self.active_counts["uploads"],
                active_video_jobs=self.active_counts["video_jobs"],
                total_requests=self.total_counts["requests"],
                total_uploads=self.total_counts["uploads"],
                total_video_jobs=self.total_counts["video_jobs"],
                queue_sizes=self.queue_sizes.copy()
            )
    
    def can_accept_request(self, request_type: str = "request") -> bool:
        """Check if we can accept a new request"""
        with self.lock:
            if request_type == "upload":
                return self.active_counts["uploads"] < settings.MAX_CONCURRENT_UPLOADS
            elif request_type == "video_job":
                return self.active_counts["video_jobs"] < settings.MAX_CONCURRENT_VIDEO_JOBS
            else:  # general request
                return self.active_counts["requests"] < settings.MAX_CONCURRENT_REQUESTS
    
    def get_wait_time_estimate(self, request_type: str = "request") -> float:
        """Estimate wait time for a request"""
        with self.lock:
            if request_type == "upload":
                queue_size = self.queue_sizes["uploads"]
                active = self.active_counts["uploads"]
                max_concurrent = settings.MAX_CONCURRENT_UPLOADS
            elif request_type == "video_job":
                queue_size = self.queue_sizes["video_jobs"]
                active = self.active_counts["video_jobs"]
                max_concurrent = settings.MAX_CONCURRENT_VIDEO_JOBS
            else:
                queue_size = self.queue_sizes["requests"]
                active = self.active_counts["requests"]
                max_concurrent = settings.MAX_CONCURRENT_REQUESTS
            
            # Simple estimation: assume average processing time
            if active >= max_concurrent:
                # All slots full, wait for one to finish
                estimated_wait = 5.0 * (queue_size + 1)  # 5 seconds per queued item
            else:
                estimated_wait = 0.0
            
            return estimated_wait

# Global concurrency limiter
concurrency_limiter = ConcurrencyLimiter()