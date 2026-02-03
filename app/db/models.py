# app/db/models.py
"""
Database models for storing processing results
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class ProcessedMedia(Base):
    __tablename__ = "processed_media"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    original_filename = Column(String)
    processed_filename = Column(String)
    processed_image_url = Column(String, nullable=True)  # The static/public URL
    django_media_id = Column(Integer, nullable=True)  # Link to Django's MediaUpload id
    django_user_id = Column(Integer, nullable=True)   # Link to Django's User id
    
    # Processing results
    processing_time = Column(Float, default=0.0)
    detection_count = Column(Integer, default=0)
    image_size = Column(String, nullable=True)
    models_used = Column(Text, nullable=True)  # JSON list of models used
    success = Column(Integer, default=1)  # 1 = success, 0 = failed
    
    # Raw data
    detections_json = Column(Text, nullable=True)  # Store detection results as JSON
    warnings = Column(Text, nullable=True)  # JSON list of warnings
    advanced_results = Column(Text, nullable=True)  # JSON object
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProcessedMedia(id={self.id}, job_id={self.job_id}, filename={self.original_filename})>"

# Initialize database - using SQLite for simplicity
# In production, you might want to use the same database as Django
engine = create_engine("sqlite:///./processed_media.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()