FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY c3_render_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For production, copy the worker code
# For development with volume mounts, this will be overridden
COPY c3_render_worker/c3_render_worker.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the worker
CMD ["python", "c3_render_worker.py"] 