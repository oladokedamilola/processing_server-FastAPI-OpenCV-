import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Current memory usage: {memory_mb:.1f}MB")

# Check virtual memory
vm = psutil.virtual_memory()
print(f"Total system memory: {vm.total / 1024 / 1024:.1f}MB")
print(f"Available system memory: {vm.available / 1024 / 1024:.1f}MB")
print(f"System memory used: {vm.percent}%")