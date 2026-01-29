# app/jobs/manager.py
"""
Job queue manager for processing jobs in background
"""
import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

from ..core.config import settings
from ..utils.logger import logger
from .job_db import JobDatabase
from .models import JobType, JobStatus, JobPriority, JobResponse, JobCreate, JobUpdate, JobStats

class JobQueueManager:
    """Manages job queue and processing"""
    
    def __init__(self, max_workers: int = None):
        self.job_db = JobDatabase()
        self.max_workers = max_workers or settings.MAX_CONCURRENT_JOBS
        self.job_queue = Queue()
        self.workers = []
        self.is_running = False
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Job processors registry
        self.processors: Dict[JobType, Callable] = {}
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info(f"Job queue manager initialized (max workers: {self.max_workers})")
    
    def register_processor(self, job_type: JobType, processor_func: Callable):
        """Register a processor function for a specific job type"""
        self.processors[job_type] = processor_func
        logger.info(f"Registered processor for job type: {job_type.value}")
    
    def submit_job(self, job_type: JobType, parameters: Dict[str, Any], 
                   priority: JobPriority = JobPriority.NORMAL,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a new job to the queue"""
        job_id = f"{job_type.value}_{uuid.uuid4().hex[:12]}"
        
        # Create job in database
        job = self.job_db.create_job(
            job_id=job_id,
            job_type=job_type,
            parameters=parameters,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Add to queue
        self.job_queue.put((priority.value, job_id))
        
        logger.info(f"Job submitted: {job_id} (priority: {priority.value})")
        
        # Ensure workers are running
        self._ensure_workers_running()
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobResponse]:
        """Get the status of a job"""
        return self.job_db.get_job(job_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None, 
                  job_type: Optional[JobType] = None,
                  limit: int = 100, offset: int = 0) -> List[JobResponse]:
        """List jobs with optional filtering"""
        from .models import JobFilter
        
        filters = JobFilter(
            status=status,
            job_type=job_type,
            limit=limit,
            offset=offset
        )
        
        return self.job_db.list_jobs(filters)
    
    def get_job_stats(self) -> JobStats:
        """Get job statistics"""
        return self.job_db.get_job_stats()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        job = self.job_db.get_job(job_id)
        
        if not job:
            return False
        
        if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
            # Update job status
            update_data = {
                "status": JobStatus.CANCELLED,
                "completed_at": datetime.utcnow(),
                "error": "Job cancelled by user"
            }
            
            if job.status == JobStatus.PROCESSING:
                update_data["processing_time"] = time.time() - job.started_at.timestamp()
            
            self.job_db.update_job(job_id, update_data)
            logger.info(f"Job cancelled: {job_id}")
            return True
        
        return False
    
    def _ensure_workers_running(self):
        """Ensure worker threads are running"""
        with self.lock:
            if not self.is_running:
                self.start_workers()
    
    def start_workers(self):
        """Start worker threads"""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            
            # Start worker threads
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"JobWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
            
            logger.info(f"Started {self.max_workers} job worker threads")
    
    def stop_workers(self):
        """Stop worker threads"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # Clear queue to wake up workers
            while not self.job_queue.empty():
                try:
                    self.job_queue.get_nowait()
                except Empty:
                    break
            
            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=5.0)
            
            self.workers.clear()
            logger.info("Job workers stopped")
    
    def _worker_loop(self):
        """Worker thread main loop"""
        thread_name = threading.current_thread().name
        
        logger.debug(f"Worker thread started: {thread_name}")
        
        while self.is_running:
            try:
                # Get job from queue with timeout
                try:
                    _, job_id = self.job_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process the job
                self._process_job(job_id, thread_name)
                
                # Mark task as done
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker error in {thread_name}: {str(e)}")
                continue
    
    def _process_job(self, job_id: str, worker_name: str):
        """Process a single job"""
        job = self.job_db.get_job(job_id)
        
        if not job:
            logger.warning(f"Job not found: {job_id}")
            return
        
        # Check if job is already completed or cancelled
        if job.is_completed:
            logger.debug(f"Job already completed: {job_id}")
            return
        
        # Update job status to processing
        update_data = {
            "status": JobStatus.PROCESSING,
            "started_at": datetime.utcnow(),
            "progress": 0.0
        }
        
        self.job_db.update_job(job_id, update_data)
        
        logger.info(f"Worker {worker_name} processing job: {job_id} ({job.job_type.value})")
        
        try:
            # Get processor for this job type
            processor = self.processors.get(job.job_type)
            
            if not processor:
                raise ValueError(f"No processor registered for job type: {job.job_type.value}")
            
            # Process the job
            result = processor(job.parameters, self._create_progress_callback(job_id))
            
            # Update job as completed
            update_data = {
                "status": JobStatus.COMPLETED,
                "progress": 100.0,
                "completed_at": datetime.utcnow(),
                "result": result,
                "processing_time": time.time() - job.created_at.timestamp()
            }
            
            self.job_db.update_job(job_id, update_data)
            
            logger.info(f"Job completed successfully: {job_id} (worker: {worker_name})")
            
        except Exception as e:
            # Update job as failed
            update_data = {
                "status": JobStatus.FAILED,
                "progress": 0.0,
                "completed_at": datetime.utcnow(),
                "error": str(e),
                "processing_time": time.time() - job.created_at.timestamp()
            }
            
            self.job_db.update_job(job_id, update_data)
            
            logger.error(f"Job failed: {job_id} - {str(e)} (worker: {worker_name})")
    
    def _create_progress_callback(self, job_id: str) -> Callable[[float, Dict[str, Any]], None]:
        """Create a progress callback function for a job"""
        def progress_callback(progress: float, extra_info: Dict[str, Any] = None):
            update_data = {
                "progress": min(max(progress, 0.0), 100.0)
            }
            
            if extra_info:
                update_data["metadata"] = extra_info
            
            self.job_db.update_job(job_id, update_data)
            
            logger.debug(f"Job progress updated: {job_id} - {progress:.1f}%")
        
        return progress_callback
    
    def _start_cleanup_task(self):
        """Start periodic cleanup task"""
        def cleanup_task():
            import time as time_module
            
            while self.is_running:
                try:
                    # Wait for cleanup interval
                    time_module.sleep(settings.JOB_CLEANUP_INTERVAL)
                    
                    # Delete old jobs
                    self.job_db.delete_old_jobs(max_age_hours=24)
                    
                except Exception as e:
                    logger.error(f"Cleanup task error: {str(e)}")
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(
            target=cleanup_task,
            name="JobCleanup",
            daemon=True
        )
        cleanup_thread.start()
        
        logger.info("Job cleanup task started")
    
    def shutdown(self):
        """Shutdown the job manager"""
        self.stop_workers()
        self.thread_pool.shutdown(wait=True)
        logger.info("Job queue manager shutdown complete")