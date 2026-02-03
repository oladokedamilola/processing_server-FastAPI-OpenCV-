# app/core/memory_manager.py
"""
Memory management and optimization for resource constraints
"""
import os
import psutil
import gc
import threading
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..core.config import settings
from ..utils.logger import logger

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total: int
    available: int
    used: int
    percent: float
    process_rss: int  # Resident Set Size
    process_vms: int  # Virtual Memory Size
    process_percent: float

class MemoryManager:
    """Manage memory usage and prevent out-of-memory conditions"""
    
    def __init__(self, memory_limit_mb: int = None):
        if memory_limit_mb is None:
            # Get from environment or use default
            memory_limit_mb = int(os.getenv("MAX_PROCESS_MEMORY_MB", "500"))
        
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.memory_warning_threshold = 0.8  # 80% usage warning
        self.memory_critical_threshold = 0.9  # 90% usage critical
        
        # Tracking
        self.high_memory_warnings = 0
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 60  # seconds
        
        # Start monitoring thread
        self._start_monitoring_thread()
        
        logger.info(f"Memory manager initialized: {memory_limit_mb}MB limit")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        vm = psutil.virtual_memory()
        process = psutil.Process()
        
        process_memory = process.memory_info()
        
        return MemoryStats(
            total=vm.total,
            available=vm.available,
            used=vm.used,
            percent=vm.percent,
            process_rss=process_memory.rss,
            process_vms=process_memory.vms,
            process_percent=process.memory_percent()
        )
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage and return status"""
        stats = self.get_memory_stats()
        
        status = "normal"
        warnings = []
        
        # Check system memory
        if stats.percent > self.memory_critical_threshold * 100:
            status = "critical"
            warnings.append(f"System memory critical: {stats.percent:.1f}% used")
        elif stats.percent > self.memory_warning_threshold * 100:
            status = "warning"
            warnings.append(f"System memory high: {stats.percent:.1f}% used")
        
        # Check process memory against limit
        if stats.process_rss > self.memory_limit_bytes:
            status = "critical"
            warnings.append(f"Process memory exceeds limit: {stats.process_rss / 1024 / 1024:.1f}MB > {self.memory_limit_bytes / 1024 / 1024:.1f}MB")
        elif stats.process_rss > self.memory_limit_bytes * 0.8:
            if status != "critical":
                status = "warning"
            warnings.append(f"Process memory approaching limit: {stats.process_rss / 1024 / 1024:.1f}MB")
        
        return {
            "status": status,
            "warnings": warnings,
            "stats": {
                "system_percent": round(stats.percent, 1),
                "system_used_mb": round(stats.used / 1024 / 1024, 1),
                "system_available_mb": round(stats.available / 1024 / 1024, 1),
                "process_rss_mb": round(stats.process_rss / 1024 / 1024, 1),
                "process_percent": round(stats.process_percent, 1),
                "memory_limit_mb": self.memory_limit_bytes / 1024 / 1024,
            }
        }
    
    def optimize_memory(self, force_gc: bool = True) -> Dict[str, Any]:
        """Optimize memory usage"""
        initial_stats = self.get_memory_stats()
        
        actions = []
        
        # Force garbage collection
        if force_gc:
            gc.collect()
            actions.append("Garbage collection")
        
        # Clear internal caches if they exist
        try:
            from ..core.cache import cache_manager
            cache_stats_before = cache_manager.get_stats()
            
            # Cleanup caches
            cache_manager.memory_cache.cleanup()
            cache_manager.disk_cache.cleanup()
            
            cache_stats_after = cache_manager.get_stats()
            memory_freed = cache_stats_before["memory"]["size_mb"] - cache_stats_after["memory"]["size_mb"]
            
            if memory_freed > 0.1:  # Only log if significant memory freed
                actions.append(f"Cache cleanup freed {memory_freed:.1f}MB")
        except:
            pass
        
        final_stats = self.get_memory_stats()
        
        memory_freed = (initial_stats.process_rss - final_stats.process_rss) / 1024 / 1024
        
        if memory_freed > 0.1:
            logger.info(f"Memory optimization freed {memory_freed:.1f}MB")
        
        return {
            "actions": actions,
            "memory_freed_mb": round(memory_freed, 1),
            "before_mb": round(initial_stats.process_rss / 1024 / 1024, 1),
            "after_mb": round(final_stats.process_rss / 1024 / 1024, 1),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def should_accept_request(self, estimated_memory_mb: float = 10.0) -> Tuple[bool, str]:
        """Check if we should accept a new request based on memory"""
        status = self.check_memory_usage()
        
        if status["status"] == "critical":
            return False, f"Memory critical: {status['warnings'][0] if status['warnings'] else 'Unknown'}"
        
        # Check if accepting this request would push us over the limit
        stats = status["stats"]
        estimated_bytes = estimated_memory_mb * 1024 * 1024
        
        if stats["process_rss_mb"] + estimated_memory_mb > stats["memory_limit_mb"] * 0.9:
            return False, f"Estimated memory ({estimated_memory_mb}MB) would exceed safe limit"
        
        if status["status"] == "warning" and estimated_memory_mb > 50:
            return False, f"High memory request ({estimated_memory_mb}MB) during warning state"
        
        return True, "OK"
    
    def _start_monitoring_thread(self):
        """Start background memory monitoring thread"""
        def monitor_worker():
            while True:
                try:
                    time.sleep(30)  # Check every 30 seconds
                    
                    status = self.check_memory_usage()
                    
                    if status["status"] == "critical":
                        logger.warning(f"Memory critical: {status['warnings']}")
                        self.high_memory_warnings += 1
                        
                        # Auto-optimize on critical
                        result = self.optimize_memory()
                        logger.info(f"Auto-optimized memory: freed {result['memory_freed_mb']}MB")
                        
                        # If still critical after optimization, log error
                        status_after = self.check_memory_usage()
                        if status_after["status"] == "critical":
                            logger.error("Memory still critical after optimization")
                    
                    elif status["status"] == "warning":
                        # Periodic cleanup on warning
                        current_time = time.time()
                        if current_time - self.last_cleanup_time > self.cleanup_interval:
                            result = self.optimize_memory(force_gc=True)
                            self.last_cleanup_time = current_time
                            
                            if result["memory_freed_mb"] > 5.0:
                                logger.info(f"Periodic memory optimization: freed {result['memory_freed_mb']}MB")
                    
                except Exception as e:
                    logger.error(f"Memory monitor error: {str(e)}")
        
        thread = threading.Thread(target=monitor_worker, daemon=True, name="MemoryMonitor")
        thread.start()
        
        logger.info("Memory monitoring thread started")

# Global memory manager instance
memory_manager = MemoryManager()