#!/usr/bin/env python3
import os
import uuid
import json
import logging
import re
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from redis import Redis
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import time

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

# Get API key from environment variables
C3_API_KEY = os.getenv("C3_API_KEY")

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
    
    logger.info(f"Created {job_type} job {job_id}")
    
    return job_id

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status from Redis"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return None
    return job_data

def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
        
def validate_float(value: Any, min_val: float = None, max_val: float = None) -> bool:
    """Validate if a value is a valid float within the specified range"""
    try:
        float_val = float(value)
        if min_val is not None and float_val < min_val:
            return False
        if max_val is not None and float_val > max_val:
            return False
        return True
    except:
        return False
        
def validate_int(value: Any, min_val: int = None, max_val: int = None) -> bool:
    """Validate if a value is a valid integer within the specified range"""
    try:
        int_val = int(value)
        if min_val is not None and int_val < min_val:
            return False
        if max_val is not None and int_val > max_val:
            return False
        return True
    except:
        return False

def validate_csm_params(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate parameters for CSM text-to-speech"""
    # Check that at least one of text or monologue is provided
    if not data or ("text" not in data and "monologue" not in data):
        return False, "Missing required field: either text or monologue must be provided"
    
    # Validate text if provided
    if "text" in data:
        if not isinstance(data["text"], str):
            return False, "text must be a string"
        if not data["text"].strip():
            return False, "text cannot be empty"
            
    # Validate monologue if provided
    if "monologue" in data:
        if not isinstance(data["monologue"], list):
            return False, "monologue must be an array of strings"
        if not data["monologue"]:
            return False, "monologue array cannot be empty"
        for sentence in data["monologue"]:
            if not isinstance(sentence, str):
                return False, "all items in monologue array must be strings"
            if not sentence.strip():
                return False, "monologue array cannot contain empty strings"
    
    # If both text and monologue are provided, return an error
    if "text" in data and "monologue" in data:
        return False, "Cannot provide both text and monologue parameters. Use only one."
    
    # Validate voice option
    if "voice" in data:
        # For API consistency, map "random_voice" to "random" if provided
        if data["voice"] == "random_voice":
            data["voice"] = "random"
            
        valid_voices = ["random", "conversational_a", "conversational_b", "clone"]
        if data["voice"] not in valid_voices:
            return False, f"Invalid voice option. Must be one of: {', '.join(valid_voices)}"
        
        # For voice cloning, validate required parameters
        if data["voice"] == "clone":
            if "reference_audio_url" not in data:
                return False, "reference_audio_url is required for voice cloning"
            if not validate_url(data["reference_audio_url"]):
                return False, "reference_audio_url must be a valid URL"
                
            if "reference_text" not in data:
                return False, "reference_text is required for voice cloning"
            if not data["reference_text"].strip():
                return False, "reference_text cannot be empty"
    else:
        # Default to "random" if voice is not specified
        data["voice"] = "random"
    
    # Validate optional parameters if present
    if "temperature" in data and not validate_float(data["temperature"], 0, 2):
        return False, "temperature must be a float between 0 and 2"
        
    if "topk" in data and not validate_int(data["topk"], 1, 100):
        return False, "topk must be an integer between 1 and 100"
        
    if "max_audio_length" in data and not validate_int(data["max_audio_length"], 1000, 30000):
        return False, "max_audio_length must be an integer between 1000 and 30000"
        
    if "pause_duration" in data and not validate_int(data["pause_duration"], 0, 500):
        return False, "pause_duration must be an integer between 0 and 500"
    
    # Validate notify_url if present
    if "notify_url" in data and not validate_url(data["notify_url"]):
        return False, "notify_url must be a valid URL"
    
    return True, ""

def validate_whisper_params(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate parameters for Whisper speech-to-text"""
    # Check for required field
    if not data or "audio_url" not in data:
        return False, "Missing required field: audio_url"
    
    # Validate audio_url
    if not validate_url(data["audio_url"]):
        return False, "audio_url must be a valid URL"
    
    # Validate model if provided
    if "model" in data:
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if data["model"] not in valid_models:
            return False, f"Invalid model. Must be one of: {', '.join(valid_models)}"
    
    # Validate task if provided
    if "task" in data:
        valid_tasks = ["transcribe", "translate"]
        if data["task"] not in valid_tasks:
            return False, f"Invalid task. Must be one of: {', '.join(valid_tasks)}"
    
    # Language validation is minimal since we allow empty strings for auto-detection
    # and many different language codes
    if "language" in data and not isinstance(data["language"], str):
        return False, "language must be a string"
    
    # Validate notify_url if present
    if "notify_url" in data and not validate_url(data["notify_url"]):
        return False, "notify_url must be a valid URL"
    
    return True, ""

def validate_portrait_params(data: Dict[str, Any]) -> Dict[str, str]:
    """Validate parameters for portrait job"""
    required_fields = ["image_url", "audio_url"]
    errors = {}
    
    # Check for required fields
    for field in required_fields:
        if field not in data or not data[field]:
            errors[field] = f"Missing required field: {field}"
    
    # Validate URLs
    if "image_url" in data and data["image_url"]:
        if not validate_url(data["image_url"]):
            errors["image_url"] = "Invalid URL format for image_url"
    
    if "audio_url" in data and data["audio_url"]:
        if not validate_url(data["audio_url"]):
            errors["audio_url"] = "Invalid URL format for audio_url"
    
    return errors

def validate_analyze_params(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate parameters for image analysis"""
    # Check for required field
    if not data or "image_url" not in data:
        return False, "Missing required field: image_url"
    
    # Validate image_url
    if not validate_url(data["image_url"]):
        return False, "image_url must be a valid URL"
    
    # Validate notify_url if present
    if "notify_url" in data and not validate_url(data["notify_url"]):
        return False, "notify_url must be a valid URL"
    
    return True, ""

@app.route("/csm", methods=["POST"])
def text_to_speech():
    """Text-to-speech endpoint with voice cloning"""
    try:
        data = request.get_json()
        
        # Validate input parameters
        is_valid, error_message = validate_csm_params(data)
        if not is_valid:
            logger.error(f"CSM validation error: {error_message}")
            return jsonify({"error": error_message}), 400
        
        # Create job after validation
        job_id = create_job("csm", data)
        return jsonify({"id": job_id, "status": "queued"})
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.exception(f"Error processing CSM request: {str(e)}\n{error_details}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/whisper", methods=["POST"])
def speech_to_text():
    """Speech-to-text endpoint"""
    try:
        data = request.get_json()
        
        # Validate input parameters
        is_valid, error_message = validate_whisper_params(data)
        if not is_valid:
            logger.error(f"Whisper validation error: {error_message}")
            return jsonify({"error": error_message}), 400
        
        # Set default values if not provided
        if "model" not in data:
            data["model"] = "medium"
            
        if "task" not in data:
            data["task"] = "transcribe"
            
        if "language" not in data:
            data["language"] = ""  # Empty string for auto-detection
        
        # Create job after validation
        job_id = create_job("whisper", data)
        return jsonify({"id": job_id, "status": "queued"})
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.exception(f"Error processing Whisper request: {str(e)}\n{error_details}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/portrait", methods=["POST"])
def portrait_endpoint():
    """Text-to-video portrait endpoint"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            logger.error("Portrait request missing request body")
            return jsonify({"error": "Missing request body"}), 400
        
        # Validate parameters
        validation_errors = validate_portrait_params(data)
        if validation_errors:
            logger.error(f"Portrait validation errors: {json.dumps(validation_errors)}")
            return jsonify({"errors": validation_errors}), 400
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Store job in Redis
        job_data = {
            "id": job_id,
            "type": "portrait",
            "status": "queued",
            "data": json.dumps(data),
            "created_at": time.time()
        }
        
        redis_client.hset(f"job:{job_id}", mapping=job_data)
        redis_client.rpush(JOB_QUEUE, job_id)
        
        logger.info(f"Created portrait job {job_id}")
        
        # Return job ID
        return jsonify({"id": job_id, "status": "queued"}), 200
    except Exception as e:
        error_details = traceback.format_exc()
        logger.exception(f"Error processing Portrait request: {str(e)}\n{error_details}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/analyze", methods=["POST"])
def image_analysis():
    """Image analysis endpoint"""
    try:
        data = request.get_json()
        
        # Validate input parameters
        is_valid, error_message = validate_analyze_params(data)
        if not is_valid:
            logger.error(f"Analyze validation error: {error_message}")
            return jsonify({"error": error_message}), 400
        
        # Create job after validation
        job_id = create_job("analyze", data)
        return jsonify({"id": job_id, "status": "queued"})
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.exception(f"Error processing Analyze request: {str(e)}\n{error_details}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    """Get job status endpoint"""
    try:
        # Validate job ID format
        if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', job_id, re.IGNORECASE):
            logger.error(f"Invalid job ID format: {job_id}")
            return jsonify({"error": "Invalid job ID format"}), 400
            
        job_data = get_job_status(job_id)
        if not job_data:
            logger.warning(f"Job not found: {job_id}")
            return jsonify({"error": "Job not found"}), 404
        
        response = {"id": job_id, "status": job_data["status"]}
        
        # Include error message if job failed
        if job_data["status"] == "failed" and "error" in job_data:
            response["error"] = job_data["error"]
            logger.info(f"Job {job_id} status request - failed with error: {job_data['error']}")
        else:
            logger.info(f"Job {job_id} status request - current status: {job_data['status']}")
            
        return jsonify(response)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.exception(f"Error getting job status: {str(e)}\n{error_details}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id: str):
    """Get job result endpoint"""
    try:
        # Validate job ID format
        if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', job_id, re.IGNORECASE):
            logger.error(f"Invalid job ID format: {job_id}")
            return jsonify({"error": "Invalid job ID format"}), 400
            
        job_data = get_job_status(job_id)
        if not job_data:
            logger.warning(f"Job not found: {job_id}")
            return jsonify({"error": "Job not found"}), 404
        
        if job_data["status"] != "success" and job_data["status"] != "failed":
            logger.warning(f"Job {job_id} not completed yet, status: {job_data['status']}")
            error_msg = "Job not completed"
            if job_data["status"] == "failed" and "error" in job_data:
                error_msg = job_data["error"]
            return jsonify({"error": error_msg, "status": job_data["status"]}), 400
        
        result = {}
        
        # Include job ID and status in all responses
        result["id"] = job_id
        result["status"] = job_data["status"]
        
        # For text-based results (whisper, analyze)
        if job_data["type"] in ["whisper", "analyze"]:
            result["text"] = job_data.get("result")
        
        # For media-based results (csm, portrait)
        if job_data["type"] in ["csm", "portrait"]:
            result["result_url"] = job_data.get("result_url")
        
        # Include error message if it exists
        if "error" in job_data:
            result["error"] = job_data["error"]
            logger.info(f"Job {job_id} result request - includes error: {job_data['error']}")
        else:
            logger.info(f"Job {job_id} result request - success")
        
        return jsonify(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.exception(f"Error getting job result: {str(e)}\n{error_details}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def root():
    """Root endpoint showing API is running"""
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return jsonify({"status": "C3 Render API is running", "timestamp": current_time})

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error: {request.path}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    logger.warning(f"405 error: {request.method} {request.path}")
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_server_error(e):
    error_details = traceback.format_exc()
    logger.exception(f"500 error: {str(e)}\n{error_details}")
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000"))) 