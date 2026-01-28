"""
Custom exceptions for the application
"""
from fastapi import HTTPException, status
from typing import Any, Dict, Optional

class ProcessingException(Exception):
    """Base exception for processing errors"""
    def __init__(self, message: str, code: str = "PROCESSING_ERROR", 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class FileValidationError(ProcessingException):
    """Exception for file validation errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "FILE_VALIDATION_ERROR", details)

class ModelLoadingError(ProcessingException):
    """Exception for model loading errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_LOADING_ERROR", details)

class JobQueueError(ProcessingException):
    """Exception for job queue errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "JOB_QUEUE_ERROR", details)

def create_http_exception(
    status_code: int,
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
) -> HTTPException:
    """Create a standardized HTTP exception"""
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "error_code": error_code,
            "details": details or {},
        }
    )

# Common HTTP exceptions
def unauthorized_error(message: str = "Unauthorized access") -> HTTPException:
    return create_http_exception(
        status_code=status.HTTP_401_UNAUTHORIZED,
        message=message,
        error_code="UNAUTHORIZED",
    )

def forbidden_error(message: str = "Access forbidden") -> HTTPException:
    return create_http_exception(
        status_code=status.HTTP_403_FORBIDDEN,
        message=message,
        error_code="FORBIDDEN",
    )

def not_found_error(message: str = "Resource not found") -> HTTPException:
    return create_http_exception(
        status_code=status.HTTP_404_NOT_FOUND,
        message=message,
        error_code="NOT_FOUND",
    )

def validation_error(message: str, details: Optional[Dict[str, Any]] = None) -> HTTPException:
    return create_http_exception(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message=message,
        error_code="VALIDATION_ERROR",
        details=details,
    )

def rate_limit_error() -> HTTPException:
    return create_http_exception(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        message="Rate limit exceeded",
        error_code="RATE_LIMIT_EXCEEDED",
        details={"retry_after": 900},  # 15 minutes
    )

def internal_server_error(message: str = "Internal server error") -> HTTPException:
    return create_http_exception(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message=message,
        error_code="INTERNAL_SERVER_ERROR",
    )