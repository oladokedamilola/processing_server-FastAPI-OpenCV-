"""
Caching system for models and processed results
"""
import time
import threading
from typing import Dict, Any, Optional, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import pickle
import gzip
from pathlib import Path

from ..core.config import settings
from ..utils.logger import logger

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    data: T
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0

class MemoryCache(Generic[T]):
    """In-memory cache with LRU eviction policy"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry[T]] = {}
        self.current_size = 0
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Memory cache initialized: {max_size_mb}MB max, {default_ttl}s TTL")
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if self._is_expired(entry):
                    self._remove(key)
                    self.misses += 1
                    return None
                
                # Update access info
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                self.hits += 1
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                return entry.data
            else:
                self.misses += 1
                return None
    
    def set(self, key: str, data: T, ttl: Optional[int] = None) -> bool:
        """Set item in cache"""
        with self.lock:
            # Calculate data size (approximate)
            try:
                size = len(pickle.dumps(data))
            except:
                size = 1024  # Default size if cannot pickle
            
            # Check if we need to make space
            if size > self.max_size_bytes:
                logger.warning(f"Item too large for cache: {size} bytes > {self.max_size_bytes} bytes")
                return False
            
            self._make_space(size)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                size_bytes=size
            )
            
            self.cache[key] = entry
            self.current_size += size
            
            logger.debug(f"Cached item: {key} ({size} bytes)")
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self.lock:
            return self._remove(key)
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            logger.info("Cache cleared")
    
    def cleanup(self):
        """Clean up expired entries and enforce size limits"""
        with self.lock:
            removed = 0
            
            # Remove expired entries
            for key in list(self.cache.keys()):
                if self._is_expired(self.cache[key]):
                    self._remove(key)
                    removed += 1
            
            # Remove least recently used if still over limit
            while self.current_size > self.max_size_bytes and self.cache:
                key = next(iter(self.cache))
                self._remove(key)
                removed += 1
            
            if removed > 0:
                logger.debug(f"Cache cleanup removed {removed} entries")
            
            return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_hits = self.hits + self.misses
            hit_rate = (self.hits / total_hits * 100) if total_hits > 0 else 0
            
            return {
                "entries": len(self.cache),
                "size_bytes": self.current_size,
                "size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 1),
                "avg_entry_size": self.current_size / len(self.cache) if self.cache else 0,
            }
    
    def _make_space(self, required_size: int):
        """Make space for new item using LRU policy"""
        while self.current_size + required_size > self.max_size_bytes and self.cache:
            # Remove least recently used
            key = next(iter(self.cache))
            self._remove(key)
    
    def _remove(self, key: str) -> bool:
        """Remove item from cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes
            return True
        return False
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        age = datetime.utcnow() - entry.created_at
        return age.total_seconds() > self.default_ttl

class DiskCache:
    """Disk-based cache for larger items"""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 500):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = cache_dir / "metadata.json"
        self.metadata: Dict[str, Dict[str, Any]] = self._load_metadata()
        self.lock = threading.RLock()
        
        logger.info(f"Disk cache initialized: {cache_dir}, {max_size_mb}MB max")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache"""
        with self.lock:
            if key not in self.metadata:
                return None
            
            entry = self.metadata[key]
            file_path = self.cache_dir / f"{key}.cache"
            
            # Check if expired
            if self._is_expired(entry):
                self.delete(key)
                return None
            
            if not file_path.exists():
                self.delete(key)
                return None
            
            try:
                # Read and decompress
                with open(file_path, 'rb') as f:
                    compressed = f.read()
                
                # Update access info
                entry["last_accessed"] = datetime.utcnow().isoformat()
                entry["access_count"] = entry.get("access_count", 0) + 1
                self._save_metadata()
                
                # Decompress and unpickle
                data = pickle.loads(gzip.decompress(compressed))
                return data
                
            except Exception as e:
                logger.error(f"Error reading from disk cache: {str(e)}")
                self.delete(key)
                return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Set item in disk cache"""
        with self.lock:
            try:
                # Serialize and compress
                serialized = pickle.dumps(data)
                compressed = gzip.compress(serialized, compresslevel=3)
                
                # Check size
                if len(compressed) > self.max_size_bytes:
                    logger.warning(f"Item too large for disk cache: {len(compressed)} bytes")
                    return False
                
                # Make space if needed
                self._make_space(len(compressed))
                
                # Write to disk
                file_path = self.cache_dir / f"{key}.cache"
                with open(file_path, 'wb') as f:
                    f.write(compressed)
                
                # Update metadata
                self.metadata[key] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "last_accessed": datetime.utcnow().isoformat(),
                    "size_bytes": len(compressed),
                    "access_count": 1,
                    "ttl": ttl or 86400,  # Default 24 hours
                }
                
                self._save_metadata()
                logger.debug(f"Disk cached: {key} ({len(compressed)} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"Error writing to disk cache: {str(e)}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete item from disk cache"""
        with self.lock:
            if key in self.metadata:
                file_path = self.cache_dir / f"{key}.cache"
                try:
                    if file_path.exists():
                        file_path.unlink()
                except:
                    pass
                
                del self.metadata[key]
                self._save_metadata()
                return True
            return False
    
    def cleanup(self) -> int:
        """Clean up expired entries and enforce size limits"""
        with self.lock:
            removed = 0
            
            # Remove expired
            for key in list(self.metadata.keys()):
                if self._is_expired(self.metadata[key]):
                    self.delete(key)
                    removed += 1
            
            # Remove oldest if over size limit
            self._enforce_size_limit()
            
            if removed > 0:
                logger.info(f"Disk cache cleanup removed {removed} entries")
            
            return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics"""
        with self.lock:
            total_size = sum(entry.get("size_bytes", 0) for entry in self.metadata.values())
            total_files = len(self.metadata)
            
            return {
                "entries": total_files,
                "size_bytes": total_size,
                "size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "cache_dir": str(self.cache_dir),
            }
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from disk"""
        try:
            if self.metadata_file.exists():
                import json
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            import json
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def _make_space(self, required_size: int):
        """Make space for new item"""
        current_size = sum(entry.get("size_bytes", 0) for entry in self.metadata.values())
        
        if current_size + required_size <= self.max_size_bytes:
            return
        
        # Sort by last accessed (oldest first)
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k].get("last_accessed", ""),
            reverse=False
        )
        
        # Remove oldest until we have enough space
        for key in sorted_keys:
            if current_size + required_size <= self.max_size_bytes:
                break
            
            entry_size = self.metadata[key].get("size_bytes", 0)
            self.delete(key)
            current_size -= entry_size
    
    def _enforce_size_limit(self):
        """Enforce size limit by removing oldest entries"""
        current_size = sum(entry.get("size_bytes", 0) for entry in self.metadata.values())
        
        if current_size <= self.max_size_bytes:
            return
        
        # Sort by last accessed (oldest first)
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k].get("last_accessed", ""),
            reverse=False
        )
        
        for key in sorted_keys:
            if current_size <= self.max_size_bytes:
                break
            
            entry_size = self.metadata[key].get("size_bytes", 0)
            self.delete(key)
            current_size -= entry_size
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        created_at = datetime.fromisoformat(entry.get("created_at", "1970-01-01"))
        ttl = entry.get("ttl", 86400)
        
        age = datetime.utcnow() - created_at
        return age.total_seconds() > ttl

class CacheManager:
    """Unified cache manager for both memory and disk caching"""
    
    def __init__(self):
        self.memory_cache = MemoryCache(max_size_mb=50, default_ttl=1800)  # 50MB, 30min
        self.disk_cache = DiskCache(
            cache_dir=settings.TEMP_PATH / "cache",
            max_size_mb=200  # 200MB
        )
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info("Cache manager initialized")
    
    def get(self, key: str, use_disk: bool = True) -> Optional[Any]:
        """Get item from cache (memory first, then disk)"""
        # Try memory cache first
        data = self.memory_cache.get(key)
        if data is not None:
            return data
        
        # Try disk cache if enabled
        if use_disk:
            data = self.disk_cache.get(key)
            if data is not None:
                # Store in memory cache for faster access
                self.memory_cache.set(key, data)
                return data
        
        return None
    
    def set(self, key: str, data: Any, use_disk: bool = True, ttl: Optional[int] = None):
        """Set item in cache"""
        # Always store in memory
        self.memory_cache.set(key, data, ttl)
        
        # Store in disk if enabled and data is large
        if use_disk:
            try:
                data_size = len(pickle.dumps(data))
                # Only store in disk if larger than 10KB
                if data_size > 10 * 1024:
                    self.disk_cache.set(key, data, ttl)
            except:
                pass
    
    def delete(self, key: str):
        """Delete item from all caches"""
        self.memory_cache.delete(key)
        self.disk_cache.delete(key)
    
    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        # Disk cache doesn't have clear, so we delete directory
        import shutil
        if self.disk_cache.cache_dir.exists():
            shutil.rmtree(self.disk_cache.cache_dir)
            self.disk_cache.cache_dir.mkdir(parents=True, exist_ok=True)
            self.disk_cache.metadata = {}
        
        logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        return {
            "memory": memory_stats,
            "disk": disk_stats,
            "total_entries": memory_stats["entries"] + disk_stats["entries"],
            "total_size_mb": memory_stats["size_mb"] + disk_stats["size_mb"],
        }
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            import time as time_module
            
            while True:
                try:
                    time_module.sleep(300)  # Cleanup every 5 minutes
                    
                    # Cleanup memory cache
                    memory_removed = self.memory_cache.cleanup()
                    
                    # Cleanup disk cache
                    disk_removed = self.disk_cache.cleanup()
                    
                    if memory_removed > 0 or disk_removed > 0:
                        logger.debug(f"Cache cleanup: memory={memory_removed}, disk={disk_removed}")
                        
                except Exception as e:
                    logger.error(f"Cache cleanup error: {str(e)}")
        
        thread = threading.Thread(target=cleanup_worker, daemon=True, name="CacheCleanup")
        thread.start()
        
        logger.info("Cache cleanup thread started")

# Global cache instance
cache_manager = CacheManager()