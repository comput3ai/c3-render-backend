#!/usr/bin/env python3
import os
import uuid
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from redis import Redis
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Redis configuration
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_client = Redis(host=redis_host, port=redis_port, decode_responses=True)

# Single job queue name
JOB_QUEUE = "queue:jobs"

def create_job(job_type: str, data: Dict[str, Any]) -> str:
    """Create a new job and add it to Redis queue"""
    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "data": json.dumps(data)  # Store data as JSON string
    }
    
    # Store job data in Redis
    redis_client.hset(f"job:{job_id}", mapping=job_data)
    
    # Add to the single processing queue
    redis_client.rpush(JOB_QUEUE, job_id)
    
    return job_id

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status from Redis"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return None
    return job_data

@app.route("/csm", methods=["POST"])
def text_to_speech():
    """Text-to-speech endpoint with voice cloning"""
    data = request.get_json()
    
    # Validate required fields
    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: text"}), 400
        
    # Validate audio_url and audio_text if provided
    if "audio_url" in data and "audio_text" not in data:
        return jsonify({"error": "audio_text is required when audio_url is provided"}), 400
    
    job_id = create_job("csm", data)
    return jsonify({"id": job_id})

@app.route("/whisper", methods=["POST"])
def speech_to_text():
    """Speech-to-text endpoint"""
    data = request.get_json()
    
    # Validate required fields
    if not data or "audio_url" not in data:
        return jsonify({"error": "Missing required field: audio_url"}), 400
    
    # Set default model if not provided
    if "model" not in data:
        data["model"] = "medium"
    
    job_id = create_job("whisper", data)
    return jsonify({"id": job_id})

@app.route("/portrait", methods=["POST"])
def portrait_video():
    """Portrait video generation endpoint"""
    data = request.get_json()
    
    # Validate required fields
    if not data or "image_url" not in data:
        return jsonify({"error": "Missing required field: image_url"}), 400
    
    job_id = create_job("portrait", data)
    return jsonify({"id": job_id})

@app.route("/analyze", methods=["POST"])
def image_analysis():
    """Image analysis endpoint"""
    data = request.get_json()
    
    # Validate required fields
    if not data or "image_url" not in data:
        return jsonify({"error": "Missing required field: image_url"}), 400
    
    job_id = create_job("analyze", data)
    return jsonify({"id": job_id})

@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    """Get job status endpoint"""
    job_data = get_job_status(job_id)
    if not job_data:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({"status": job_data["status"]})

@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id: str):
    """Get job result endpoint"""
    job_data = get_job_status(job_id)
    if not job_data:
        return jsonify({"error": "Job not found"}), 404
    
    if job_data["status"] != "success":
        return jsonify({"error": "Job not completed"}), 400
    
    # For text-based results (whisper, analyze)
    if job_data["type"] in ["whisper", "analyze"]:
        return jsonify({"text": job_data.get("result")})
    
    # For media-based results (csm, portrait)
    if job_data["type"] in ["csm", "portrait"]:
        return jsonify({"result_url": job_data.get("result_url")})
    
    return jsonify({"error": "Unknown job type"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000"))) 