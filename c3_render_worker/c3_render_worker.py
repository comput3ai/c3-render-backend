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

# Import directly from local files
import csm
import comfyui

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Output directory for temporary files
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Using output directory: {OUTPUT_DIR}")

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

# Current job tracking
current_job_id = None
current_job_data = None

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
    """Check if a node is responding with retries"""
    node_url = f"https://{node}"
    headers = {"X-C3-API-KEY": api_key}
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = requests.get(node_url, headers=headers, timeout=5)
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"🔍 Health check attempt {attempt+1}/{max_retries} failed for {node}: status code {response.status_code}")
        except (requests.RequestException, Exception) as e:
            logger.warning(f"🔍 Health check attempt {attempt+1}/{max_retries} failed for {node}: {str(e)}")
        
        # If not the last attempt, wait before retry
        if attempt < max_retries - 1:
            time.sleep(2)
    
    return False

def verify_node_in_workloads(node_id):
    """Verify if the node is still in the workloads API response"""
    workloads = get_running_workloads()
    workload_ids = [w.get('workload') for w in workloads]
    
    return node_id in workload_ids

def is_gpu_healthy():
    """Check if the current GPU instance is healthy"""
    if gpu_instance is None:
        return False
    
    node_hostname = gpu_instance.get('node')
    node_id = gpu_instance.get('workload')
    
    # Check 1: Verify the node is still in workloads
    if not verify_node_in_workloads(node_id):
        logger.warning(f"⚠️ GPU instance {node_hostname} is no longer in workloads API response")
        return False
    
    # Check 2: Verify the node is responding to health checks
    if not check_node_health(node_hostname):
        logger.warning(f"⚠️ GPU instance {node_hostname} failed all health check attempts")
        return False
    
    return True

def start_gpu_monitor():
    """Start a thread to monitor GPU instance health and idle time"""
    global gpu_monitor_thread, gpu_monitor_active
    
    if gpu_monitor_thread is not None and gpu_monitor_thread.is_alive():
        return
    
    gpu_monitor_active = True
    gpu_monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    gpu_monitor_thread.start()
    logger.info("Started GPU monitor thread")

def monitor_gpu():
    """Monitor GPU instance health and shut it down if idle for too long or unhealthy"""
    global gpu_instance, last_job_time, gpu_monitor_active, current_job_id, current_job_data
    
    idle_threshold = timedelta(minutes=5)
    
    while gpu_monitor_active and gpu_instance is not None:
        # Check if the instance is healthy
        if not is_gpu_healthy():
            logger.warning(f"⚠️ GPU instance {gpu_instance.get('node')} is unhealthy")
            
            # If there's an active job, mark it as failed
            if current_job_id is not None and current_job_data is not None:
                logger.error(f"⚠️ GPU instance failed during job execution for job {current_job_id}")
                
                error_msg = "GPU instance became unhealthy during job execution"
                update_job_status(current_job_id, "failed", error=error_msg)
                send_webhook_notification(current_job_id, current_job_data, "failed", error=error_msg)
            
            # Shut down the GPU instance
            shutdown_gpu_instance()
            break
        
        # If no active job, check idle time
        if current_job_id is None:
            idle_time = datetime.now() - last_job_time
            
            if idle_time > idle_threshold:
                logger.info(f"GPU instance {gpu_instance.get('node')} has been idle for {idle_time}, shutting down")
                shutdown_gpu_instance()
                break
            
            # Log idle status every minute
            if idle_time.seconds % 60 == 0 and idle_time.seconds > 0:
                logger.info(f"GPU instance idle for {idle_time}")
        
        # Check every 15 seconds for health, slightly faster checking during jobs
        # and slightly slower during idle to balance responsiveness and efficiency
        time.sleep(15)
    
    logger.info("GPU monitor stopped")

def get_gpu_instance():
    """Get or create a GPU instance for processing"""
    global gpu_instance, last_job_time
    
    # If we already have an instance, update the last job time and return it
    if gpu_instance is not None:
        # Check if the instance is healthy
        if is_gpu_healthy():
            last_job_time = datetime.now()
            logger.info(f"Using existing GPU instance: {gpu_instance.get('node')}")
            return gpu_instance
        else:
            logger.warning(f"Existing GPU instance {gpu_instance.get('node')} is unhealthy, launching new one")
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
        
        # Wait for the instance to initialize
        logger.info("Waiting for GPU instance to initialize...")
        time.sleep(5)
        
        # Check if the instance is healthy after initialization
        if is_gpu_healthy():
            logger.info(f"GPU instance {node_hostname} is healthy and ready")
            return gpu_instance
        else:
            logger.error(f"GPU instance {node_hostname} failed health check after initialization")
            shutdown_gpu_instance()
            return None
    else:
        logger.error("Failed to launch GPU instance")
        return None

def shutdown_gpu_instance():
    """Shut down the current GPU instance"""
    global gpu_instance, gpu_monitor_active, current_job_id, current_job_data
    
    if gpu_instance is None:
        return
    
    node_hostname = gpu_instance.get('node')
    node_id = gpu_instance.get('workload')
    
    # Reset job tracking
    current_job_id = None
    current_job_data = None
    
    # Stop GPU monitoring
    gpu_monitor_active = False
    
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

def upload_to_storage(file_path):
    """Save file to storage and return the URL"""
    logger.info(f"Saving file to output directory: {file_path}")
    
    try:
        # Just return the local file path for now
        # In a real environment, this would be replaced with actual storage upload
        # and returning a publicly accessible URL
        
        # For development, construct a placeholder URL
        filename = os.path.basename(file_path)
        result_url = f"file://{file_path}"
        logger.info(f"File saved. URL: {result_url}")
        
        return result_url
        
    except Exception as e:
        logger.exception(f"Unexpected error during file handling: {str(e)}")
        # Return a fallback URL for now
        return f"file://{file_path}"

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
    global current_job_id, current_job_data
    
    logger.info(f"Processing CSM job: {job_id}")
    
    try:
        # Get or create a GPU instance
        instance = get_gpu_instance()
        if not instance:
            error_msg = "Failed to get GPU instance"
            update_job_status(job_id, "failed", error=error_msg)
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)
            return
        
        # Set current job for monitoring
        current_job_id = job_id
        current_job_data = job_data
        
        # Generate speech from text using CSM
        audio_file = csm.text_to_speech_with_csm(job_data.get("data", "{}"), job_id, instance, api_key, OUTPUT_DIR)
        
        # Save the output path (we're not uploading to Minio yet)
        result_url = upload_to_storage(audio_file)
        
        # Update job status and send notification
        update_job_status(job_id, "success", result_url=result_url)
        send_webhook_notification(job_id, job_data, "success", result_url=result_url, local_path=audio_file)
        
        # Note: We're not cleaning up the file since we want to keep it in the output directory
        
    except Exception as e:
        error_msg = f"Error processing CSM job: {str(e)}"
        logger.exception(error_msg)
        update_job_status(job_id, "failed", error=error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        # Shut down GPU instance on job failure
        shutdown_gpu_instance()
    finally:
        # Clear job tracking
        current_job_id = None
        current_job_data = None

def process_whisper_job(job_id, job_data):
    """Process a speech-to-text job"""
    global current_job_id, current_job_data
    
    logger.info(f"Processing Whisper job: {job_id}")
    
    try:
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
        
        # Set current job for monitoring
        current_job_id = job_id
        current_job_data = job_data
        
        # TODO: Connect to Gradio client for speech-to-text processing
        # TODO: Download audio from URL to output directory with job ID filename
        # output_path = os.path.join(OUTPUT_DIR, f"{job_id}_input.mp3")
        # TODO: Process with Whisper model
        # TODO: Save results to output directory with job ID filename
        # text_output_path = os.path.join(OUTPUT_DIR, f"{job_id}.txt")
        
        # For now, just simulate success
        result_text = "This is a simulated transcription result"
        update_job_status(job_id, "success", result=result_text)
        send_webhook_notification(job_id, job_data, "success", text=result_text)
        
    except Exception as e:
        error_msg = f"Error processing Whisper job: {str(e)}"
        logger.exception(error_msg)
        update_job_status(job_id, "failed", error=error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        # Shut down GPU instance on job failure
        shutdown_gpu_instance()
    finally:
        # Clear job tracking
        current_job_id = None
        current_job_data = None

def process_portrait_job(job_id, job_data):
    """Process a portrait video job using ComfyUI"""
    global current_job_id, current_job_data
    
    logger.info(f"Processing Portrait job: {job_id}")
    
    try:
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
        
        # Set current job for monitoring
        current_job_id = job_id
        current_job_data = job_data
        
        # Run portrait generation using ComfyUI
        video_output_path = comfyui.generate_portrait_video(image_url, audio_url, job_id, instance, api_key, OUTPUT_DIR)
        
        if video_output_path:
            # Save the result URL
            result_url = upload_to_storage(video_output_path)
            
            # Update job status and send notification
            update_job_status(job_id, "success", result_url=result_url)
            send_webhook_notification(job_id, job_data, "success", result_url=result_url)
        else:
            error_msg = "Failed to generate portrait video"
            update_job_status(job_id, "failed", error=error_msg)
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        
    except Exception as e:
        error_msg = f"Error processing Portrait job: {str(e)}"
        logger.exception(error_msg)
        update_job_status(job_id, "failed", error=error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        # Shut down GPU instance on job failure
        shutdown_gpu_instance()
    finally:
        # Clear job tracking
        current_job_id = None
        current_job_data = None

def process_analyze_job(job_id, job_data):
    """Process an image analysis job"""
    global current_job_id, current_job_data
    
    logger.info(f"Processing Analyze job: {job_id}")
    
    try:
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
        
        # Set current job for monitoring
        current_job_id = job_id
        current_job_data = job_data
        
        # TODO: Connect to vision model API
        # TODO: Download image from URL to output directory with job ID filename
        # image_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_input.jpg")
        # TODO: Process with vision model
        # TODO: Save results to output directory with job ID filename
        # text_output_path = os.path.join(OUTPUT_DIR, f"{job_id}.txt")
        
        # For now, just simulate success
        result_text = "This is a simulated image analysis result"
        update_job_status(job_id, "success", result=result_text)
        send_webhook_notification(job_id, job_data, "success", text=result_text)
        
    except Exception as e:
        error_msg = f"Error processing Analyze job: {str(e)}"
        logger.exception(error_msg)
        update_job_status(job_id, "failed", error=error_msg)
        send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        # Shut down GPU instance on job failure
        shutdown_gpu_instance()
    finally:
        # Clear job tracking
        current_job_id = None
        current_job_data = None

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
        # Shut down GPU instance on job failure
        shutdown_gpu_instance()

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