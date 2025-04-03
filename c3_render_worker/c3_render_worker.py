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
from gradio_client import Client
import re
from urllib.parse import urlparse

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

def split_text_into_sentences(text):
    """
    Split text into individual sentences for better CSM processing.
    
    This improves the quality of speech generation by allowing
    the model to handle each sentence as a separate unit.
    """
    # Basic sentence splitting using regex
    # This handles periods, question marks, and exclamation marks followed by a space or end of text
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    logger.info(f"Split text into {len(sentences)} sentences")
    return sentences

def text_to_speech_with_csm(text, job_id):
    """Generate speech from text using CSM with configurable voice options"""
    logger.info(f"Generating speech with CSM: {text[:50]}...")
    
    # Extract job parameters
    job_data = redis_client.hgetall(f"job:{job_id}")
    job_params = json.loads(job_data.get("data", "{}"))
    
    # Get CSM-specific parameters with defaults
    voice = job_params.get("voice", "random")  # Options: random, conversational_a, conversational_b, or "clone" for voice cloning
    
    # Map API voice parameter to CSM internal parameter
    csm_voice_map = {
        "random": "random_voice",
        "conversational_a": "conversational_a",
        "conversational_b": "conversational_b",
        "clone": "custom_voice"
    }
    
    # Convert API voice name to CSM internal name
    csm_voice = csm_voice_map.get(voice, "random_voice")
    
    temperature = float(job_params.get("temperature", 0.9))
    topk = int(job_params.get("topk", 50))
    max_audio_length = int(job_params.get("max_audio_length", 10000))
    pause_duration = int(job_params.get("pause_duration", 150))
    
    # Voice cloning parameters
    reference_audio_url = job_params.get("reference_audio_url")
    reference_text = job_params.get("reference_text")
    
    # Get node hostname from GPU instance
    node_hostname = gpu_instance.get('node')
    
    # Construct CSM endpoint URL using the node hostname
    csm_url = f"https://{node_hostname}/csm/"
    logger.info(f"Using CSM endpoint: {csm_url}")
    
    # Define retry parameters
    max_retries = 5
    retry_delay = 5  # seconds
    
    # Create Gradio client with retries
    client = None
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries} to connect to CSM service...")
            
            # Create Gradio client with API key header
            client = Client(
                csm_url,
                headers={"X-C3-API-KEY": api_key}
            )
            
            # If we get here, the client was created successfully
            logger.info(f"Successfully connected to CSM service on attempt {attempt}")
            break
            
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to connect to CSM service on attempt {attempt}: {str(e)}")
            
            # If this isn't the last attempt, wait before retrying
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
    
    # If we couldn't create the client after all retries, raise the last error
    if client is None:
        logger.error(f"Failed to connect to CSM service after {max_retries} attempts")
        raise last_error
    
    try:
        # Split text into sentences for better speech quality
        sentences = split_text_into_sentences(text)
        
        # Prepare the monologue text as a JSON array of sentences
        monologue_json = json.dumps(sentences)
        
        # Handle voice cloning if reference audio is provided
        reference_audio_file = None
        if voice == "clone" and reference_audio_url and reference_text:
            # Download reference audio for voice cloning
            logger.info(f"Downloading reference audio for voice cloning: {reference_audio_url}")
            
            try:
                # Create a temporary file for the reference audio
                reference_audio_file = os.path.join(OUTPUT_DIR, f"{job_id}_reference.mp3")
                
                # Download the reference audio
                response = requests.get(reference_audio_url, timeout=30)
                if response.status_code == 200:
                    with open(reference_audio_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded reference audio to {reference_audio_file}")
                else:
                    logger.error(f"Failed to download reference audio: {response.status_code}")
                    raise Exception(f"Failed to download reference audio: {response.status_code}")
            except Exception as e:
                logger.exception(f"Error downloading reference audio: {str(e)}")
                raise
        
        # Log the CSM parameters
        logger.info(f"CSM parameters: voice={voice}, temperature={temperature}, topk={topk}, " +
                   f"max_audio_length={max_audio_length}, pause_duration={pause_duration}")
        if voice == "clone":
            logger.info(f"Voice cloning parameters: reference_audio_url={reference_audio_url}, " +
                       f"reference_text={reference_text}")
        
        # Generate speech using the monologue mode with specified parameters
        logger.info(f"Calling CSM with {voice} and {len(sentences)} sentences...")
        
        # Parameters for prediction
        predict_params = {
            "monologue_json": monologue_json,
            "temperature": temperature,
            "topk": topk,
            "max_audio_length": max_audio_length,
            "pause_duration": pause_duration,
            "api_name": "/generate_monologue_audio"
        }
        
        # Configure voice parameters based on type
        if voice == "clone" and reference_audio_file:
            # Use custom voice with reference audio and text
            predict_params["speaker_voice"] = "custom_voice"
            predict_params["speaker_text"] = reference_text
            predict_params["speaker_audio"] = reference_audio_file
        else:
            # Use built-in voice options (random or conversational)
            predict_params["speaker_voice"] = csm_voice
            predict_params["speaker_text"] = ""
            predict_params["speaker_audio"] = None
        
        # Call the CSM service to generate audio
        result = client.predict(**predict_params)
        
        # Result is a file path to the generated audio
        logger.info(f"CSM successfully generated audio: {result}")
        
        # Generate a filename using the job ID
        filename = f"{job_id}.mp3"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Copy the file to our output directory
        with open(result, 'rb') as src_file, open(output_path, 'wb') as dest_file:
            dest_file.write(src_file.read())
        
        logger.info(f"Saved audio to {output_path}")
        
        # Clean up reference audio file if it exists
        if reference_audio_file and os.path.exists(reference_audio_file):
            os.unlink(reference_audio_file)
            logger.info(f"Cleaned up reference audio file {reference_audio_file}")
        
        return output_path
        
    except Exception as e:
        logger.exception(f"Error generating speech with CSM: {str(e)}")
        
        # Clean up reference audio file if it exists
        if 'reference_audio_file' in locals() and reference_audio_file and os.path.exists(reference_audio_file):
            os.unlink(reference_audio_file)
            logger.info(f"Cleaned up reference audio file {reference_audio_file}")
            
        raise

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
        # Extract job parameters
        job_params = json.loads(job_data.get("data", "{}"))
        
        # For backward compatibility, support both 'text' and 'monologue'
        monologue_text = job_params.get("monologue", job_params.get("text", ""))
        
        logger.info(f"Job parameters: monologue='{monologue_text[:50]}...'")
        
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
        audio_file = text_to_speech_with_csm(monologue_text, job_id)
        
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

def download_file(url, output_path=None, preserve_extension=True):
    """Download a file from URL to the specified path
    
    Args:
        url: URL to download
        output_path: Path to save the file (if preserve_extension is True, the extension may be changed)
        preserve_extension: Whether to preserve the file extension from the URL
        
    Returns:
        The path to the downloaded file or False if download failed
    """
    logger.info(f"Downloading file from {url}")
    
    try:
        if preserve_extension and output_path:
            # Extract the extension from the URL
            url_path = urlparse(url).path
            _, ext = os.path.splitext(url_path)
            
            if ext:
                # Replace the extension in the output path
                output_base, _ = os.path.splitext(output_path)
                output_path = output_base + ext
                logger.info(f"Using extension from URL: {ext}")
        
        logger.info(f"Saving to {output_path}")
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded file to {output_path}")
            return output_path
        else:
            logger.error(f"Failed to download file: {response.status_code}")
            return False
    except Exception as e:
        logger.exception(f"Error downloading file: {str(e)}")
        return False

def check_comfyui_readiness(node_hostname, max_retries=12, retry_delay=10):
    """Check if ComfyUI is ready and wait until it's available
    
    Args:
        node_hostname: Hostname of the node running ComfyUI
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        bool: True if ComfyUI is ready, False otherwise
    """
    comfyui_healthcheck_url = f"https://{node_hostname}/system_stats"
    logger.info(f"Checking ComfyUI readiness at {comfyui_healthcheck_url}")
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"ComfyUI readiness check attempt {attempt}/{max_retries}")
            
            response = requests.get(
                comfyui_healthcheck_url,
                headers={"X-C3-API-KEY": api_key},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"ComfyUI is ready on attempt {attempt}")
                return True
            else:
                logger.warning(f"ComfyUI not ready yet (status code: {response.status_code})")
        except Exception as e:
            logger.warning(f"ComfyUI readiness check failed on attempt {attempt}: {str(e)}")
        
        # If this isn't the last attempt, wait before retrying
        if attempt < max_retries:
            logger.info(f"Waiting {retry_delay} seconds before retrying ComfyUI readiness check...")
            time.sleep(retry_delay)
    
    logger.error(f"ComfyUI not ready after {max_retries} attempts")
    return False

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
        
        # Download files from URLs with the correct naming convention
        image_output_base = os.path.join(OUTPUT_DIR, f"{job_id}_portrait")
        audio_output_base = os.path.join(OUTPUT_DIR, f"{job_id}_audio")
        
        # Download files with extension preservation
        downloaded_image = download_file(image_url, image_output_base)
        if not downloaded_image:
            error_msg = "Failed to download image file"
            update_job_status(job_id, "failed", error=error_msg)
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)
            return
        
        downloaded_audio = download_file(audio_url, audio_output_base)
        if not downloaded_audio:
            error_msg = "Failed to download audio file"
            update_job_status(job_id, "failed", error=error_msg)
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)
            # Clean up downloaded image
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            return
        
        # Run portrait generation using ComfyUI
        video_output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        
        # Get ComfyUI endpoint URLs from the GPU instance - direct to the root without /comfyui prefix
        node_hostname = gpu_instance.get('node')
        comfyui_base_url = f"https://{node_hostname}"
        comfyui_prompt_url = f"{comfyui_base_url}/prompt"
        comfyui_upload_url = f"{comfyui_base_url}/upload/image"
        comfyui_upload_audio_url = f"{comfyui_base_url}/upload/audio"
        comfyui_history_url = f"{comfyui_base_url}/history"
        
        # Wait for ComfyUI to be ready
        logger.info("Waiting for ComfyUI container to be ready...")
        if not check_comfyui_readiness(node_hostname):
            error_msg = "ComfyUI container failed to become ready"
            update_job_status(job_id, "failed", error=error_msg)
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)
            # Clean up downloaded files
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            if os.path.exists(downloaded_audio):
                os.unlink(downloaded_audio)
            return
        
        # File names to use on ComfyUI server - preserve the extensions
        server_image_name = os.path.basename(downloaded_image)
        server_audio_name = os.path.basename(downloaded_audio)
        
        # Upload files to the ComfyUI server
        logger.info(f"Uploading image to ComfyUI at {comfyui_upload_url}")
        
        try:
            # Upload image file - determine MIME type based on extension
            image_ext = os.path.splitext(downloaded_image)[1].lower()
            image_mime = "image/jpeg"  # Default
            if image_ext == ".png":
                image_mime = "image/png"
            elif image_ext == ".jpg" or image_ext == ".jpeg":
                image_mime = "image/jpeg"
            elif image_ext == ".webp":
                image_mime = "image/webp"
            
            with open(downloaded_image, 'rb') as f:
                files = {
                    'image': (server_image_name, f, image_mime)
                }
                headers = {"X-C3-API-KEY": api_key}
                
                response = requests.post(
                    comfyui_upload_url,
                    files=files,
                    headers=headers,
                    timeout=60
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded image to ComfyUI")
                else:
                    error_msg = f"Failed to upload image to ComfyUI: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            
            # Upload audio file - determine MIME type based on extension
            audio_ext = os.path.splitext(downloaded_audio)[1].lower()
            audio_mime = "audio/mpeg"  # Default
            if audio_ext == ".mp3":
                audio_mime = "audio/mpeg"
            elif audio_ext == ".wav":
                audio_mime = "audio/wav"
            elif audio_ext == ".ogg":
                audio_mime = "audio/ogg"
            
            with open(downloaded_audio, 'rb') as f:
                files = {
                    'audio': (server_audio_name, f, audio_mime)
                }
                headers = {"X-C3-API-KEY": api_key}
                
                response = requests.post(
                    comfyui_upload_audio_url,
                    files=files,
                    headers=headers,
                    timeout=60
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded audio to ComfyUI")
                else:
                    error_msg = f"Failed to upload audio to ComfyUI: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            
            # Prepare ComfyUI workflow
            comfyui_workflow = {
                "13": {
                    "inputs": {
                        "frame_rate": [
                            "31",
                            1
                        ],
                        "loop_count": 0,
                        "filename_prefix": f"Video_{job_id}",
                        "format": "video/h264-mp4",
                        "pix_fmt": "yuv420p",
                        "crf": 19,
                        "save_metadata": True,
                        "trim_to_audio": True,
                        "pingpong": False,
                        "save_output": True,
                        "images": [
                            "31",
                            0
                        ],
                        "audio": [
                            "26",
                            0
                        ]
                    },
                    "class_type": "VHS_VideoCombine",
                    "_meta": {
                        "title": "Video Combine 🎥🅥🅗🅢"
                    }
                },
                "18": {
                    "inputs": {
                        "image": server_image_name
                    },
                    "class_type": "LoadImage",
                    "_meta": {
                        "title": "Load a portrait Image (Face Closeup)"
                    }
                },
                "21": {
                    "inputs": {
                        "images": [
                            "46",
                            0
                        ]
                    },
                    "class_type": "PreviewImage",
                    "_meta": {
                        "title": "Preview Image after Resize"
                    }
                },
                "26": {
                    "inputs": {
                        "audio": server_audio_name
                    },
                    "class_type": "LoadAudio",
                    "_meta": {
                        "title": "LoadAudio"
                    }
                },
                "31": {
                    "inputs": {
                        "seed": 2054408119,
                        "inference_steps": 25,
                        "dynamic_scale": 1,
                        "fps": 25,
                        "model": [
                            "34",
                            0
                        ],
                        "data_dict": [
                            "35",
                            0
                        ]
                    },
                    "class_type": "SONICSampler",
                    "_meta": {
                        "title": "SONICSampler"
                    }
                },
                "32": {
                    "inputs": {
                        "ckpt_name": "svd_xt.safetensors"
                    },
                    "class_type": "ImageOnlyCheckpointLoader",
                    "_meta": {
                        "title": "Image Only Checkpoint Loader (img2vid model)"
                    }
                },
                "33": {
                    "inputs": {
                        "min_resolution": 576,
                        "duration": 4,
                        "expand_ratio": 1,
                        "clip_vision": [
                            "32",
                            1
                        ],
                        "vae": [
                            "32",
                            2
                        ],
                        "audio": [
                            "26",
                            0
                        ],
                        "image": [
                            "46",
                            0
                        ],
                        "weight_dtype": [
                            "34",
                            1
                        ]
                    },
                    "class_type": "SONIC_PreData",
                    "_meta": {
                        "title": "SONIC_PreData"
                    }
                },
                "34": {
                    "inputs": {
                        "sonic_unet": "unet.pth",
                        "ip_audio_scale": 1,
                        "use_interframe": True,
                        "dtype": "fp16",
                        "model": [
                            "32",
                            0
                        ]
                    },
                    "class_type": "SONICTLoader",
                    "_meta": {
                        "title": "SONICTLoader"
                    }
                },
                "35": {
                    "inputs": {
                        "anything": [
                            "33",
                            0
                        ]
                    },
                    "class_type": "easy cleanGpuUsed",
                    "_meta": {
                        "title": "Clean VRAM Used"
                    }
                },
                "46": {
                    "inputs": {
                        "mode": "resize",
                        "supersample": "true",
                        "resampling": "lanczos",
                        "rescale_factor": 1,
                        "resize_width": 500,
                        "resize_height": 500,
                        "image": [
                            "18",
                            0
                        ]
                    },
                    "class_type": "Image Resize",
                    "_meta": {
                        "title": "Image Resize"
                    }
                }
            }
            
            # Queue the prompt in ComfyUI
            logger.info(f"Sending workflow to ComfyUI at {comfyui_prompt_url}")
            
            # Prepare the prompt data
            prompt_data = {
                "prompt": comfyui_workflow
            }
            
            # Send the request to ComfyUI
            response = requests.post(
                comfyui_prompt_url,
                json=prompt_data,
                headers={"X-C3-API-KEY": api_key},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                logger.info(f"ComfyUI workflow queued with prompt ID: {prompt_id}")
                
                # Poll for workflow completion
                max_wait_time = 300  # 5 minutes
                poll_interval = 5     # 5 seconds
                timeout = time.time() + max_wait_time
                
                completed = False
                output_filename = None
                
                while time.time() < timeout and not completed:
                    # Check workflow history
                    logger.info(f"Checking workflow status for prompt ID: {prompt_id}")
                    
                    history_response = requests.get(
                        comfyui_history_url,
                        headers={"X-C3-API-KEY": api_key},
                        timeout=30
                    )
                    
                    if history_response.status_code == 200:
                        history = history_response.json()
                        
                        # Check if our prompt is in history and completed
                        if prompt_id in history:
                            prompt_info = history[prompt_id]
                            
                            # Check if execution is complete
                            if "outputs" in prompt_info and prompt_info.get("status", {}).get("completed", False):
                                logger.info(f"Workflow completed for prompt ID: {prompt_id}")
                                completed = True
                                
                                # Get the output filename
                                # The output node is node 13 (VideoCombine) which produces a video file
                                if "13" in prompt_info.get("outputs", {}):
                                    video_output = prompt_info["outputs"]["13"]
                                    if video_output and isinstance(video_output, list) and len(video_output) > 0:
                                        output_filename = video_output[0].get("filename")
                                        logger.info(f"Found output filename: {output_filename}")
                            
                            elif prompt_info.get("status", {}).get("error", False):
                                error_msg = prompt_info.get("status", {}).get("error_message", "Unknown error in ComfyUI workflow")
                                logger.error(f"Workflow failed: {error_msg}")
                                raise Exception(f"ComfyUI workflow failed: {error_msg}")
                    
                    if not completed:
                        logger.info(f"Workflow still in progress, checking again in {poll_interval} seconds...")
                        time.sleep(poll_interval)
                
                if not completed:
                    raise Exception(f"Timed out waiting for ComfyUI workflow to complete after {max_wait_time} seconds")
                
                if not output_filename:
                    raise Exception("ComfyUI workflow completed but no output filename was found")
                
                # Download the output file
                output_url = f"{comfyui_base_url}/output/{output_filename}"
                logger.info(f"Downloading output video from {output_url}")
                
                response = requests.get(
                    output_url,
                    headers={"X-C3-API-KEY": api_key},
                    timeout=120
                )
                
                if response.status_code == 200:
                    with open(video_output_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Successfully downloaded output video to {video_output_path}")
                    
                    # Save the result URL
                    result_url = upload_to_storage(video_output_path)
                    
                    # Update job status and send notification
                    update_job_status(job_id, "success", result_url=result_url)
                    send_webhook_notification(job_id, job_data, "success", result_url=result_url)
                else:
                    error_msg = f"Failed to download output video: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            else:
                error_msg = f"Failed to queue ComfyUI workflow: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"Error running ComfyUI workflow: {str(e)}"
            logger.exception(error_msg)
            update_job_status(job_id, "failed", error=error_msg)
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        
        # Clean up input files
        if os.path.exists(downloaded_image):
            os.unlink(downloaded_image)
            logger.info(f"Cleaned up input image file: {downloaded_image}")
        
        if os.path.exists(downloaded_audio):
            os.unlink(downloaded_audio)
            logger.info(f"Cleaned up input audio file: {downloaded_audio}")
        
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