# Add to app/utils/compression.py
"""
Response compression utilities
"""
import gzip
import json
from typing import Any, Dict
from fastapi.responses import Response

def compress_response(data: Any, min_size: int = 1024) -> Response:
    """Compress response data if large enough"""
    # Convert to JSON if not already string/bytes
    if not isinstance(data, (str, bytes)):
        json_data = json.dumps(data)
    else:
        json_data = data if isinstance(data, str) else data.decode()
    
    # Check if worth compressing
    if len(json_data) < min_size:
        return Response(content=json_data, media_type="application/json")
    
    # Compress
    compressed = gzip.compress(json_data.encode(), compresslevel=6)
    
    return Response(
        content=compressed,
        media_type="application/json",
        headers={"Content-Encoding": "gzip"}
    )