# test_paths.py
import sys
from pathlib import Path

print("Testing path resolution...")
print(f"Current working directory: {Path.cwd()}")
print(f"__file__ (if run as script): {__file__ if '__file__' in globals() else 'N/A'}")

# Simulate what config.py does
config_file = Path(__file__).resolve() if '__file__' in globals() else Path.cwd() / "app" / "core" / "config.py"
print(f"Config file path: {config_file}")

# Go up 3 levels
base_path = config_file.parent.parent.parent
print(f"Base path (parent.parent.parent): {base_path}")

# Check what exists
expected_dirs = ["app", "processed", "uploads", "models"]
for dir_name in expected_dirs:
    test_path = base_path / dir_name
    exists = test_path.exists()
    print(f"  {dir_name}/ exists: {exists} ({test_path})")

# Alternative: Check current directory
print(f"\nChecking current directory: {Path.cwd()}")
for dir_name in expected_dirs:
    test_path = Path.cwd() / dir_name
    exists = test_path.exists()
    print(f"  {dir_name}/ exists: {exists} ({test_path})")