FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy application code and model
COPY app/ ./app/
COPY start_server.py .
COPY app.py .
COPY models/ ./models/ 

# Create necessary directories
RUN mkdir -p /app/uploads /app/processed /app/temp /app/logs

# Set environment variables
ENV PORT=7860
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models

# Expose the port HF Spaces expects
EXPOSE 7860

# Health check (optional but good practice)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health', timeout=2)"

# Start the server
CMD ["python", "start_server.py"]