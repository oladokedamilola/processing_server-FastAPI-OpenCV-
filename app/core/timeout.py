"""
Timeout handling for long-running operations
"""
import asyncio
import threading
import signal
import time
from typing import Optional, Callable, Any
from contextlib import contextmanager
from functools import wraps

from ..core.config import settings
from ..utils.logger import logger

class TimeoutException(Exception):
    """Exception raised when a timeout occurs"""
    pass

class TimeoutManager:
    """Manage timeouts for different operations"""
    
    def __init__(self):
        self.active_timeouts = {}
        self.timeout_lock = threading.Lock()
        
    def timeout_decorator(self, timeout_seconds: int):
        """Decorator for timeout handling"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.run_with_timeout(func, timeout_seconds, *args, **kwargs)
            return wrapper
        return decorator
    
    def async_timeout_decorator(self, timeout_seconds: int):
        """Async decorator for timeout handling"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.run_async_with_timeout(func, timeout_seconds, *args, **kwargs)
            return wrapper
        return decorator
    
    def run_with_timeout(self, func: Callable, timeout_seconds: int, *args, **kwargs) -> Any:
        """Run function with timeout"""
        result = None
        exception = None
        
        def worker():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            raise TimeoutException(f"Operation timed out after {timeout_seconds} seconds")
        
        if exception:
            raise exception
        
        return result
    
    async def run_async_with_timeout(self, func: Callable, timeout_seconds: int, *args, **kwargs) -> Any:
        """Run async function with timeout"""
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutException(f"Async operation timed out after {timeout_seconds} seconds")
    
    @contextmanager
    def timeout_context(self, timeout_seconds: int, operation_name: str = "operation"):
        """Context manager for timeout handling"""
        start_time = time.time()
        
        def check_timeout():
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutException(f"{operation_name} timed out after {timeout_seconds} seconds")
        
        # Store timeout info
        timeout_id = id(threading.current_thread())
        with self.timeout_lock:
            self.active_timeouts[timeout_id] = {
                "start_time": start_time,
                "timeout": timeout_seconds,
                "operation": operation_name
            }
        
        try:
            yield check_timeout
        finally:
            with self.timeout_lock:
                self.active_timeouts.pop(timeout_id, None)
    
    def get_active_timeouts(self) -> dict:
        """Get information about active timeouts"""
        with self.timeout_lock:
            active = {}
            current_time = time.time()
            
            for timeout_id, info in self.active_timeouts.items():
                elapsed = current_time - info["start_time"]
                remaining = max(0, info["timeout"] - elapsed)
                
                active[timeout_id] = {
                    "operation": info["operation"],
                    "elapsed": round(elapsed, 1),
                    "remaining": round(remaining, 1),
                    "progress_percent": round((elapsed / info["timeout"]) * 100, 1)
                }
            
            return active

# Global timeout manager
timeout_manager = TimeoutManager()

# Pre-defined timeout decorators
request_timeout = timeout_manager.timeout_decorator(settings.REQUEST_TIMEOUT)
video_timeout = timeout_manager.timeout_decorator(settings.VIDEO_PROCESSING_TIMEOUT)
model_load_timeout = timeout_manager.timeout_decorator(settings.MODEL_LOAD_TIMEOUT)
upload_timeout = timeout_manager.timeout_decorator(settings.FILE_UPLOAD_TIMEOUT)

# Async versions
async_request_timeout = timeout_manager.async_timeout_decorator(settings.REQUEST_TIMEOUT)