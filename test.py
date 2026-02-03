import os
from pathlib import Path

# The path where the file actually exists
actual_path = r"C:\Users\PC\Desktop\processing_server (FastAPI + OpenCV)\processing_server\processed\images\tmp8jsw3z2x_processed_20260201_210954_dnsdfk.jpg"

# What the FastAPI app thinks is the path
config_path = Path(__file__).parent.parent.parent / "processed" / "images" / "tmp8jsw3z2x_processed_20260201_210954_dnsdfk.jpg"

print(f"Actual file exists: {os.path.exists(actual_path)}")
print(f"Config path file exists: {os.path.exists(config_path)}")
print(f"Actual path: {actual_path}")
print(f"Config path: {config_path}")