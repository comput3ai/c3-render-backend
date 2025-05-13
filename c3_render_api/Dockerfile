FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For production, copy the API code
# For development with volume mounts, this will be overridden
COPY c3_render_api.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=c3_render_api.py

# Expose port
EXPOSE 5000

# Run the server using Gunicorn
# Assumes the Flask app instance in c3_render_api.py is named 'app'
# Uses 4 workers as a default. Adjust as needed or use an env var.
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "c3_render_api:app"]