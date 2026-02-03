# Create a script to fix the directory structure
import os
import sys

# Get the base directory from the error path
base_dir = "C:\\Users\\PC\\Desktop\\processing_server (FastAPI + OpenCV)\\processing_server"

# Create necessary directories
directories = [
    "temp/videos/jobs",
    "temp/videos/uploads",
    "processed/videos",
    "logs",
    "models"
]

for directory in directories:
    full_path = os.path.join(base_dir, directory)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created directory: {full_path}")

print("\nDirectory structure created successfully!")