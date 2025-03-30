#!/usr/bin/env python3
import os
import time
import json
import logging
import redis
import sys
import threading
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Redis configuration
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

# Single job queue name
JOB_QUEUE = "queue:jobs"

# Get API key from environment variables
api_key = os.getenv("C3_API_KEY")
if not api_key:
    logger.error("❌ Error: C3_API_KEY not found in environment")
    sys.exit(1)

# Base API endpoint for Comput3.ai
comput3_base_url = "https://api.comput3.ai/api/v0"

# Headers for Comput3.ai API
comput3_headers = {
    "X-C3-API-KEY": api_key,
    "Content-Type": "application/json",
    "Origin": "https://launch.comput3.ai"
}

# Initialize GPU state
gpu_instance = None
last_job_time = datetime.now()
gpu_monitor_thread = None
gpu_monitor_active = False

def get_running_workloads():
    """Get all currently running workloads"""
    url = f"{comput3_base_url}/workloads"
    data = {"running": True}

    try:
        response = requests.post(url, headers=comput3_headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"❌ Error getting workloads: {response.status_code}")
            logger.error(response.text)
            return []
    except Exception as e:
        logger.error(f"❌ Error connecting to Comput3.ai: {str(e)}")
        return []

def launch_workload(workload_type="media:fast"):
    """Launch a new workload of the specified type"""
    url = f"{comput3_base_url}/launch"

    # Always set expiration to current time + 3600 seconds (1 hour)
    current_time = int(time.time())
    expires = current_time + 3600

    # Create launch data
    data = {
        "type": workload_type,
        "expires": expires
    }

    try:
        response = requests.post(url, headers=comput3_headers, json=data)

        if response.status_code == 200:
            result = response.json()
            result["type"] = workload_type  # Add type to result for easier tracking
            result["expires"] = expires     # Add expiration for tracking
            return result
        else:
            logger.error(f"❌ Error launching workload: {response.status_code}")
            logger.error(response.text)
            return None
    except Exception as e:
        logger.error(f"❌ Error connecting to Comput3.ai: {str(e)}")
        return None

def stop_workload(workload_id):
    """Stop a running workload by its ID"""
    url = f"{comput3_base_url}/stop"

    data = {"workload": workload_id}

    try:
        response = requests.post(url, headers=comput3_headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"❌ Error stopping workload: {response.status_code}")
            logger.error(response.text)
            return None
    except Exception as e:
        logger.error(f"❌ Error connecting to Comput3.ai: {str(e)}")
        return None

def check_node_health(node):
    """Check if a node is responding"""
    node_url = f"https://{node}"
    headers = {"X-C3-API-KEY": api_key}

    try:
        response = requests.get(node_url, headers=headers, timeout=5)
        return response.status_code == 200
    except (requests.RequestException, Exception) as e:
        logger.warning(f"🔍 Health check failed for {node}: {str(e)}")
        return False

def start_gpu_monitor():
    """Start a thread to monitor GPU instance idle time"""
    global gpu_monitor_thread, gpu_monitor_active
    
    if gpu_monitor_thread is not None and gpu_monitor_thread.is_alive():
        return
    
    gpu_monitor_active = True
    gpu_monitor_thread = threading.Thread(target=monitor_gpu_idle, daemon=True)
    gpu_monitor_thread.start()
    logger.info("Started GPU idle monitor thread")

def monitor_gpu_idle():
    """Monitor GPU instance and shut it down if idle for 5 minutes"""
    global gpu_instance, last_job_time, gpu_monitor_active
    
    idle_threshold = timedelta(minutes=5)
    
    while gpu_monitor_active and gpu_instance is not None:
        # Check if the instance has been idle for too long
        idle_time = datetime.now() - last_job_time
        
        if idle_time > idle_threshold:
            logger.info(f"GPU instance {gpu_instance.get('node')} has been idle for {idle_time}, shutting down")
            shutdown_gpu_instance()
            break
        
        # Log idle status every minute
        if idle_time.seconds % 60 == 0:
            logger.info(f"GPU instance idle for {idle_time}")
        
        # Check every 10 seconds
        time.sleep(10)
    
    logger.info("GPU idle monitor stopped")

def get_gpu_instance():
    """Get or create a GPU instance for processing"""
    global gpu_instance, last_job_time
    
    # If we already have an instance, update the last job time and return it
    if gpu_instance is not None:
        # Check if the instance is healthy
        node_hostname = gpu_instance.get('node')
        if check_node_health(node_hostname):
            last_job_time = datetime.now()
            logger.info(f"Using existing GPU instance: {node_hostname}")
            return gpu_instance
        else:
            logger.warning(f"Existing GPU instance {node_hostname} is unhealthy, launching new one")
            shutdown_gpu_instance()
    
    # Launch a new GPU instance
    logger.info("Launching new GPU instance (media:fast)...")
    result = launch_workload(workload_type="media:fast")
    
    if result:
        gpu_instance = result
        last_job_time = datetime.now()
        node_hostname = result.get('node')
        logger.info(f"Successfully launched GPU instance: {node_hostname}")
        
        # Start the GPU monitor if not already running
        start_gpu_monitor()
        
        # Wait a bit for the instance to initialize
        logger.info("Waiting for GPU instance to initialize...")
        time.sleep(5)
        
        return gpu_instance
    else:
        logger.error("Failed to launch GPU instance")
        return None

def shutdown_gpu_instance():
    """Shut down the current GPU instance"""
    global gpu_instance, gpu_monitor_active
    
    if gpu_instance is None:
        return
    
    node_hostname = gpu_instance.get('node')
    node_id = gpu_instance.get('workload')
    
    logger.info(f"Shutting down GPU instance {node_hostname} (ID: {node_id})...")
    result = stop_workload(node_id)
    
    if result:
        stopped_time = datetime.fromtimestamp(result.get('stopped')).strftime('%Y-%m-%d %H:%M:%S')
        refund = result.get('refund_amount', 0)
        logger.info(f"Successfully stopped GPU instance {node_hostname} at {stopped_time} (Refund: {refund})")
    else:
        logger.error(f"Failed to stop GPU instance {node_hostname}")
    
    # Reset GPU instance state
    gpu_instance = None
    gpu_monitor_active = False

def send_webhook_notification(job_id, job_data, status, **kwargs):
    """Send webhook notification if URL is provided with retry logic"""
    try:
        # Parse job data to get the notification URL
        job_params = json.loads(job_data.get("data", "{}"))
        notify_url = job_params.get("notify_url")
        
        if not notify_url:
            return False
            
        logger.info(f"Sending webhook notification to {notify_url}")
        
        # Prepare webhook payload
        payload = {
            "id": job_id,
            "status": status
        }
        
        # Add any additional data to the payload
        for key, value in kwargs.items():
            payload[key] = value
            
        # Try to send the webhook up to 5 times
        max_attempts = 5
        delay_seconds = 5
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Webhook attempt {attempt}/{max_attempts}")
                response = requests.post(notify_url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"Webhook notification sent successfully on attempt {attempt}")
                    return True
                else:
                    logger.warning(f"Webhook attempt {attempt} failed with status code {response.status_code}")
                    
                    # If this isn't the last attempt, wait before retrying
                    if attempt < max_attempts:
                        logger.info(f"Retrying in {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
                    
            except requests.RequestException as e:
                logger.warning(f"Webhook attempt {attempt} failed with error: {str(e)}")
                
                # If this isn't the last attempt, wait before retrying
                if attempt < max_attempts:
                    logger.info(f"Retrying in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
        
        logger.error(f"Webhook notification failed after {max_attempts} attempts")
        return False
            
    except Exception as e:
        logger.exception(f"Error sending webhook notification: {str(e)}")
        return False

def process_csm_job(job_id, job_data):
    """Process a text-to-speech job"""
    logger.info(f"Processing CSM job: {job_id}")
    
    # Extract job parameters
    job_params = json.loads(job_data.get("data", "{}"))
    text = job_params.get("text", "")
    audio_url = job_params.get("audio_url")
    audio_text = job_params.get("audio_text")
    
    logger.info(f"Job parameters: text='{text[:50]}...', audio_url={audio_url is not None}")
    
    # Get or create a GPU instance
    instance = get_gpu_instance()
    if not instance:
        error_msg = "Failed to get GPU instance"
        update_job_status(job_id, "failed", error=error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        return
    
    # TODO: Connect to Gradio client for text-to-speech processing
    # TODO: If audio_url is present, use it for voice cloning
    # TODO: Store result in Minio/S3
    
    # For now, just simulate success
    result_url = "https://example.com/results/sample.mp3"
    update_job_status(job_id, "success", result_url=result_url)
    send_webhook_notification(job_id, job_data, "success", result_url=result_url)
    
def process_whisper_job(job_id, job_data):
    """Process a speech-to-text job"""
    logger.info(f"Processing Whisper job: {job_id}")
    
    # Extract job parameters
    job_params = json.loads(job_data.get("data", "{}"))
    audio_url = job_params.get("audio_url")
    model = job_params.get("model", "medium")
    
    logger.info(f"Job parameters: audio_url={audio_url}, model={model}")
    
    # Get or create a GPU instance
    instance = get_gpu_instance()
    if not instance:
        error_msg = "Failed to get GPU instance"
        update_job_status(job_id, "failed", error=error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        return
    
    # TODO: Connect to Gradio client for speech-to-text processing
    # TODO: Download audio from URL
    # TODO: Process with Whisper model
    
    # For now, just simulate success
    result_text = "This is a simulated transcription result"
    update_job_status(job_id, "success", result=result_text)
    send_webhook_notification(job_id, job_data, "success", text=result_text)

def process_portrait_job(job_id, job_data):
    """Process a portrait video job"""
    logger.info(f"Processing Portrait job: {job_id}")
    
    # Extract job parameters
    job_params = json.loads(job_data.get("data", "{}"))
    image_url = job_params.get("image_url")
    audio_url = job_params.get("audio_url")
    
    logger.info(f"Job parameters: image_url={image_url}, audio_url={audio_url}")
    
    # Get or create a GPU instance
    instance = get_gpu_instance()
    if not instance:
        error_msg = "Failed to get GPU instance"
        update_job_status(job_id, "failed", error=error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        return
    
    # TODO: Connect to ComfyUI for video generation
    # TODO: Download image and audio from URLs
    # TODO: Run portrait generation workflow
    # TODO: Store result in Minio/S3
    
    # For now, just simulate success
    result_url = "https://example.com/results/portrait.mp4"
    update_job_status(job_id, "success", result_url=result_url)
    send_webhook_notification(job_id, job_data, "success", result_url=result_url)

def process_analyze_job(job_id, job_data):
    """Process an image analysis job"""
    logger.info(f"Processing Analyze job: {job_id}")
    
    # Extract job parameters
    job_params = json.loads(job_data.get("data", "{}"))
    image_url = job_params.get("image_url")
    
    logger.info(f"Job parameters: image_url={image_url}")
    
    # Get or create a GPU instance
    instance = get_gpu_instance()
    if not instance:
        error_msg = "Failed to get GPU instance"
        update_job_status(job_id, "failed", error=error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        return
    
    # TODO: Connect to vision model API
    # TODO: Download image from URL
    # TODO: Process with vision model
    
    # For now, just simulate success
    result_text = "This is a simulated image analysis result"
    update_job_status(job_id, "success", result=result_text)
    send_webhook_notification(job_id, job_data, "success", text=result_text)

def update_job_status(job_id, status, **kwargs):
    """Update job status in Redis"""
    redis_client.hset(f"job:{job_id}", "status", status)
    
    # Update any additional fields
    for key, value in kwargs.items():
        redis_client.hset(f"job:{job_id}", key, value)
    
    logger.info(f"Updated job {job_id} status to {status}")

def process_job(job_id):
    """Process a job based on its type"""
    logger.info(f"Processing job {job_id}")
    
    # Get job data from Redis
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    if not job_data:
        logger.error(f"Job {job_id} not found in Redis")
        return
    
    # Update job status to running
    redis_client.hset(f"job:{job_id}", "status", "running")
    
    try:
        # Process based on job type
        job_type = job_data.get("type")
        
        if job_type == "csm":
            process_csm_job(job_id, job_data)
        elif job_type == "whisper":
            process_whisper_job(job_id, job_data)
        elif job_type == "portrait":
            process_portrait_job(job_id, job_data)
        elif job_type == "analyze":
            process_analyze_job(job_id, job_data)
        else:
            error_msg = f"Unknown job type: {job_type}"
            logger.error(error_msg)
            redis_client.hset(f"job:{job_id}", "status", "failed")
            redis_client.hset(f"job:{job_id}", "error", error_msg)
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)
    
    except Exception as e:
        error_msg = f"Error processing job: {str(e)}"
        logger.exception(error_msg)
        redis_client.hset(f"job:{job_id}", "status", "failed")
        redis_client.hset(f"job:{job_id}", "error", error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)

def main():
    """Main worker loop"""
    logger.info("Starting C3 Render Worker")
    
    try:
        while True:
            # Check the job queue for waiting jobs
            result = redis_client.blpop([JOB_QUEUE], timeout=1)
            
            if result:
                _, job_id = result
                process_job(job_id)
            
            # Small delay to avoid hammering Redis if no jobs
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down worker")
        if gpu_instance:
            shutdown_gpu_instance()

if __name__ == "__main__":
    main() 