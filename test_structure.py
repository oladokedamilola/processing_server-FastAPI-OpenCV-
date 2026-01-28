"""
Test the project structure
"""
import os
import sys

def check_structure():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print("Project Structure Check")
    print("=" * 60)
    
    # Required directories
    directories = [
        "app",
        "app/api",
        "app/api/v1", 
        "app/api/v1/endpoints",
        "app/core",
        "app/utils",
        "app/models",
    ]
    
    for dir_path in directories:
        full_path = os.path.join(project_root, dir_path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}/")
        
        # Create if doesn't exist
        if not exists:
            os.makedirs(full_path, exist_ok=True)
            print(f"   Created {dir_path}/")
    
    print("\nRequired Files Check")
    print("=" * 60)
    
    # Required files with content
    files_to_create = {
        "app/__init__.py": "# Empty - makes app a package\n",
        "app/api/__init__.py": "# Empty - makes api a package\n",
        "app/api/v1/__init__.py": """\"\"\"
API v1 endpoints
\"\"\"
from fastapi import APIRouter
from .endpoints import health

# Create main router
router = APIRouter(prefix="/api/v1")

# Include routers
router.include_router(health.router, tags=["health"])
""",
        "app/api/v1/endpoints/__init__.py": "# Empty - makes endpoints a package\n",
        "app/api/v1/endpoints/health.py": """\"\"\"
Health check endpoints
\"\"\"
from fastapi import APIRouter
import psutil
import platform
import time
from datetime import datetime

router = APIRouter(prefix="/health", tags=["health"])

# Track server start time
SERVER_START_TIME = time.time()

@router.get("/")
async def health_check():
    \"\"\"Basic health check endpoint\"\"\"
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FastAPI Processing Server",
        "version": "1.0.0",
        "environment": "development",
        "uptime": time.time() - SERVER_START_TIME,
    }

@router.get("/system")
async def system_info():
    \"\"\"System information endpoint\"\"\"
    return {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "active_jobs": 0,
        "total_processed": 0,
        "server_uptime": time.time() - SERVER_START_TIME,
    }
""",
        "app/core/__init__.py": "# Empty - makes core a package\n",
        "app/utils/__init__.py": "# Empty - makes utils a package\n",
        "app/models/__init__.py": "# Empty - makes models a package\n",
    }
    
    for file_path, content in files_to_create.items():
        full_path = os.path.join(project_root, file_path)
        
        # Check if file exists
        exists = os.path.exists(full_path)
        
        if exists:
            # Check if file has content
            with open(full_path, 'r') as f:
                file_content = f.read()
            has_content = len(file_content.strip()) > 0
            status = "✅" if has_content else "⚠️ (empty)"
            print(f"{status} {file_path}")
            
            # Update if empty
            if not has_content:
                with open(full_path, 'w') as f:
                    f.write(content)
                print(f"   Added content to {file_path}")
        else:
            print(f"❌ {file_path}")
            # Create file with content
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            print(f"   Created {file_path}")
    
    print("\n" + "=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    # Test imports
    sys.path.insert(0, project_root)
    
    tests = [
        ("app.core.config.settings", "from app.core.config import settings"),
        ("app.api.v1.router", "from app.api.v1 import router"),
        ("app.api.v1.endpoints.health", "from app.api.v1.endpoints import health"),
    ]
    
    for test_name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {test_name}")
        except Exception as e:
            print(f"❌ {test_name}: {e}")

if __name__ == "__main__":
    check_structure()