FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y file ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For production, copy the worker code
# For development with volume mounts, this will be overridden
COPY *.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the worker
CMD ["python", "c3_render_worker.py"]
