"""
In-memory job database for free tier compatibility
"""
import sqlite3
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from ..core.config import settings
from ..utils.logger import logger
from .models import JobType, JobStatus, JobPriority, JobResponse, JobFilter, JobStats

class JobDatabase:
    """In-memory SQLite database for job tracking"""
    
    def __init__(self):
        self.db_path = settings.TEMP_PATH / "jobs.db"
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress REAL,
                        priority INTEGER NOT NULL,
                        parameters TEXT NOT NULL,
                        result TEXT,
                        error TEXT,
                        metadata TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        processing_time REAL
                    )
                """)
                
                # Create indexes for faster queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_job_type ON jobs(job_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON jobs(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON jobs(priority)")
                
                conn.commit()
                
                logger.info(f"Job database initialized at: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize job database: {str(e)}")
            raise
    
    def create_job(self, job_id: str, job_type: JobType, parameters: Dict[str, Any], 
                   priority: JobPriority = JobPriority.NORMAL, 
                   metadata: Optional[Dict[str, Any]] = None) -> JobResponse:
        """Create a new job in the database"""
        try:
            created_at = datetime.utcnow()
            metadata = metadata or {}
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, job_type, status, progress, priority, 
                        parameters, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job_type.value,
                        JobStatus.PENDING.value,
                        0.0,
                        priority.value,
                        json.dumps(parameters),
                        json.dumps(metadata),
                        created_at.isoformat()
                    )
                )
                conn.commit()
            
            logger.info(f"Job created: {job_id} ({job_type.value})")
            
            return JobResponse(
                job_id=job_id,
                job_type=job_type,
                status=JobStatus.PENDING,
                progress=0.0,
                priority=priority,
                parameters=parameters,
                metadata=metadata,
                created_at=created_at
            )
            
        except Exception as e:
            logger.error(f"Failed to create job {job_id}: {str(e)}")
            raise
    
    def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Retrieve a job by ID"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (job_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_job_response(row)
                
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {str(e)}")
            return None
    
    def update_job(self, job_id: str, update_data: Dict[str, Any]) -> Optional[JobResponse]:
        """Update a job's properties"""
        try:
            # Build update query dynamically
            update_fields = []
            update_values = []
            
            for field, value in update_data.items():
                if field == "result" or field == "parameters" or field == "metadata":
                    # Serialize JSON fields
                    update_fields.append(f"{field} = ?")
                    update_values.append(json.dumps(value) if value else None)
                elif field == "status":
                    update_fields.append(f"{field} = ?")
                    update_values.append(value.value if hasattr(value, 'value') else value)
                elif field in ["progress", "processing_time"]:
                    update_fields.append(f"{field} = ?")
                    update_values.append(float(value) if value is not None else None)
                elif field in ["started_at", "completed_at"]:
                    update_fields.append(f"{field} = ?")
                    update_values.append(value.isoformat() if value else None)
                else:
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
            
            if not update_fields:
                return self.get_job(job_id)
            
            update_query = f"UPDATE jobs SET {', '.join(update_fields)} WHERE job_id = ?"
            update_values.append(job_id)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(update_query, update_values)
                conn.commit()
            
            logger.debug(f"Job updated: {job_id} - {list(update_data.keys())}")
            
            return self.get_job(job_id)
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {str(e)}")
            return None
    
    def list_jobs(self, filters: Optional[JobFilter] = None) -> List[JobResponse]:
        """List jobs with optional filtering"""
        try:
            query = "SELECT * FROM jobs WHERE 1=1"
            query_params = []
            
            if filters:
                if filters.status:
                    query += " AND status = ?"
                    query_params.append(filters.status.value)
                
                if filters.job_type:
                    query += " AND job_type = ?"
                    query_params.append(filters.job_type.value)
                
                if filters.min_priority is not None:
                    query += " AND priority >= ?"
                    query_params.append(filters.min_priority.value)
                
                if filters.created_after:
                    query += " AND created_at >= ?"
                    query_params.append(filters.created_after.isoformat())
                
                if filters.created_before:
                    query += " AND created_at <= ?"
                    query_params.append(filters.created_before.isoformat())
            
            # Order by priority (descending) and creation time (ascending)
            query += " ORDER BY priority DESC, created_at ASC"
            
            # Apply limit and offset
            query += " LIMIT ? OFFSET ?"
            query_params.extend([filters.limit if filters else 100, filters.offset if filters else 0])
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, query_params)
                rows = cursor.fetchall()
                
                return [self._row_to_job_response(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            return []
    
    def get_job_stats(self) -> JobStats:
        """Get job statistics"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                
                # Total counts
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                        SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled
                    FROM jobs
                """)
                counts = cursor.fetchone()
                
                # Average processing time for completed jobs
                cursor = conn.execute("""
                    SELECT AVG(processing_time) as avg_time 
                    FROM jobs 
                    WHERE status = 'completed' AND processing_time IS NOT NULL
                """)
                avg_time = cursor.fetchone()
                
                # Success rate
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_completed,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful
                    FROM jobs 
                    WHERE status IN ('completed', 'failed', 'cancelled')
                """)
                success_data = cursor.fetchone()
                
                total_completed = success_data['total_completed'] if success_data and success_data['total_completed'] else 0
                successful = success_data['successful'] if success_data and success_data['successful'] else 0
                
                success_rate = (successful / total_completed * 100) if total_completed > 0 else None
                
                return JobStats(
                    total_jobs=counts['total'] if counts else 0,
                    pending_jobs=counts['pending'] if counts else 0,
                    processing_jobs=counts['processing'] if counts else 0,
                    completed_jobs=counts['completed'] if counts else 0,
                    failed_jobs=counts['failed'] if counts else 0,
                    cancelled_jobs=counts['cancelled'] if counts else 0,
                    average_processing_time=avg_time['avg_time'] if avg_time and avg_time['avg_time'] else None,
                    success_rate=success_rate
                )
                
        except Exception as e:
            logger.error(f"Failed to get job stats: {str(e)}")
            return JobStats()
    
    def delete_old_jobs(self, max_age_hours: int = 24):
        """Delete jobs older than specified hours"""
        try:
            from datetime import datetime, timedelta
            
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # Count before deletion for logging
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM jobs WHERE created_at < ?",
                    (cutoff_time.isoformat(),)
                )
                count_before = cursor.fetchone()['count']
                
                # Delete old jobs
                conn.execute(
                    "DELETE FROM jobs WHERE created_at < ?",
                    (cutoff_time.isoformat(),)
                )
                conn.commit()
                
                # Count after deletion
                cursor = conn.execute("SELECT COUNT(*) as count FROM jobs")
                count_after = cursor.fetchone()['count']
                
                deleted_count = count_before - count_after
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old jobs (older than {max_age_hours} hours)")
                    
        except Exception as e:
            logger.error(f"Failed to delete old jobs: {str(e)}")
    
    def cleanup(self):
        """Clean up database connection and file"""
        try:
            # Delete the database file
            if self.db_path.exists():
                self.db_path.unlink()
                logger.info(f"Job database cleaned up: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup job database: {str(e)}")
    
    def _row_to_job_response(self, row) -> JobResponse:
        """Convert database row to JobResponse object"""
        try:
            # Parse JSON fields
            parameters = json.loads(row['parameters']) if row['parameters'] else {}
            result = json.loads(row['result']) if row['result'] else None
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            
            # Parse dates
            created_at = datetime.fromisoformat(row['created_at'])
            started_at = datetime.fromisoformat(row['started_at']) if row['started_at'] else None
            completed_at = datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None
            
            return JobResponse(
                job_id=row['job_id'],
                job_type=JobType(row['job_type']),
                status=JobStatus(row['status']),
                progress=row['progress'],
                priority=JobPriority(row['priority']),
                parameters=parameters,
                result=result,
                error=row['error'],
                metadata=metadata,
                created_at=created_at,
                started_at=started_at,
                completed_at=completed_at,
                processing_time=row['processing_time']
            )
            
        except Exception as e:
            logger.error(f"Failed to parse job row: {str(e)}")
            raise