from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import random

app = FastAPI(title="FastAPI Test Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API key check middleware
API_KEY = "a3f8e97b12c450d6f34a8921b567d0e9f12a34b5678c9d0e1f23a45b67c89d012"

@app.middleware("http")
async def check_api_key(request, call_next):
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"error": True, "message": "Invalid API key"}
        )
    return await call_next(request)

@app.post("/api/v1/process/image")
async def process_image(
    image: UploadFile = File(...),
    detection_types: str = "person,vehicle"
):
    """Process an image and return mock detections."""
    # Read the image file
    contents = await image.read()
    
    # Simulate processing time
    time.sleep(1)
    
    # Return mock detections
    return {
        "success": True,
        "processing_time": 1.2,
        "detections": [
            {
                "label": "person",
                "confidence": 0.92,
                "bbox": [120, 85, 310, 480],
                "class": "person",
                "type": "person"
            },
            {
                "label": "car",
                "confidence": 0.88,
                "bbox": [450, 200, 600, 350],
                "class": "car",
                "type": "vehicle"
            }
        ],
        "detection_count": 2,
        "image_size": "1920x1080",
        "detection_summary": {
            "person": 1,
            "vehicle": 1
        }
    }

@app.get("/health")
async def health():
    return {"healthy": True, "status": "online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)