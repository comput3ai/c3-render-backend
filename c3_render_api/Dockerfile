FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY c3_render_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For production, copy the API code
# For development with volume mounts, this will be overridden
COPY c3_render_api/c3_render_api.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=c3_render_api.py
ENV FLASK_ENV=development

# Expose port
EXPOSE 5000

# Run the server
CMD ["python", "c3_render_api.py"] 