#!/usr/bin/env python3
import os
import time
import json
import logging
import redis
import sys
import threading
import requests
import random
import socket
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
import io
import magic  # For MIME type detection
from urllib.parse import urlparse, urlunparse

# Import from constants file
from constants import (
    GPU_IDLE_TIMEOUT,
    RENDER_POLLING_INTERVAL,
    GPU_WORKER_DELAY,
    NO_GPU_WORKER_DELAY,
    LOCK_RETRY_INTERVAL,
    QUEUE_CHECK_INTERVAL,
    DEFAULT_GPU_LAUNCH,
    DEFAULT_GPU_REPLACE
)

# Generate a unique worker ID (hostname is sufficient with lock expiration)
WORKER_ID = socket.gethostname()

# Import directly from local files
import csm
import comfyui
import whisper

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

# Log the loaded constants
logger.info(f"Using GPU_IDLE_TIMEOUT: {GPU_IDLE_TIMEOUT}s, RENDER_POLLING_INTERVAL: {RENDER_POLLING_INTERVAL}s")

# Redis configuration
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

# MinIO configuration
# These variables are typically injected via the environment section
# in docker-compose.production.yaml, which often substitutes values
# from a .env file loaded by docker compose.
# Common production values might look like:
# MINIO_ENDPOINT=your_public_minio.com
# MINIO_ACCESS_KEY=your_access_key
# MINIO_SECRET_KEY=your_secret_key
# MINIO_BUCKET=your_bucket_name
# MINIO_SECURE=true
# MINIO_PUBLIC_URL=https://your_public_minio.com/
#
# NOTE for same-host Docker Compose setups (where worker and minio run
# on the same docker network but possibly different compose files):
# Use the service name and default port: MINIO_ENDPOINT=minio:9000
# Use HTTP: MINIO_SECURE=false
minio_endpoint = os.getenv("MINIO_ENDPOINT")
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
minio_bucket = os.getenv("MINIO_BUCKET", "c3-render-output")
minio_public_url = os.getenv("MINIO_PUBLIC_URL") # Base public URL for constructing final URLs

# Initialize MinIO clients
internal_minio_client = None
public_minio_client = None

# 1. Initialize Internal Client (for uploads)
if all([minio_endpoint, minio_access_key, minio_secret_key, minio_bucket]):
    try:
        logger.info(f"Attempting to initialize INTERNAL MinIO client for endpoint: {minio_endpoint} (secure={minio_secure})")
        internal_minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure
        )
        # Check bucket existence using the internal client
        found = internal_minio_client.bucket_exists(minio_bucket)
        if not found:
            logger.error(f"❌ Configured MinIO bucket '{minio_bucket}' does not exist! Disabling MinIO uploads.")
            internal_minio_client = None # Disable internal client if bucket missing
        else:
            logger.info(f"Internal MinIO client initialized successfully. Bucket '{minio_bucket}' found.")

    except S3Error as conn_err:
        logger.error(f"❌ Error connecting INTERNAL MinIO client ({minio_endpoint}) or checking bucket: {conn_err}")
        internal_minio_client = None
    except Exception as e:
        logger.error(f"❌ Unexpected error initializing INTERNAL MinIO client for {minio_endpoint}: {e}")
        internal_minio_client = None
else:
     logger.warning("⚠️ MinIO configuration incomplete for internal client (endpoint, keys, or bucket missing). File uploads via MinIO will be disabled.")

# 2. Initialize Public Client (for presigned URL generation) - only if internal client is ready and public URL is set
if internal_minio_client and minio_public_url:
    try:
        parsed_public_endpoint = urlparse(minio_public_url)
        public_endpoint_netloc = parsed_public_endpoint.netloc or parsed_public_endpoint.path
        public_secure = parsed_public_endpoint.scheme == 'https'

        logger.info(f"Attempting to initialize PUBLIC MinIO client for endpoint: {public_endpoint_netloc} (secure={public_secure})")
        public_minio_client = Minio(
            public_endpoint_netloc,
            access_key=minio_access_key, # Use same credentials
            secret_key=minio_secret_key,
            secure=public_secure
        )
        # Optional: Could add a light check here like list_buckets(limit=1) if needed, but maybe not necessary
        # as it's only used for URL generation. Let's assume credentials are valid if internal client worked.
        logger.info("Public MinIO client initialized successfully for presigned URL generation.")

    except Exception as e:
        logger.error(f"❌ Failed to initialize PUBLIC MinIO client for {minio_public_url}: {e}. Presigned URLs may use internal hostname.")
        public_minio_client = None # Ensure it's None if init fails

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
gpu_launch_time = None  # Track when the GPU was launched

# Monitor thread
monitor_thread = None
monitor_active = False

# Current job tracking
current_job_id = None
current_job_data = None
job_start_time = None
job_thread = None
job_cancel_flag = False  # Flag to signal job cancellation

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
    """Launch a new workload of the specified type with retries"""
    url = f"{comput3_base_url}/launch"
    retry_delays = [5, 15, 30, 45, 60]  # Delays in seconds

    for attempt, delay in enumerate(retry_delays, 1):
        logger.info(f"Attempt {attempt}/{len(retry_delays)} to launch workload type: {workload_type}")

        # Always set expiration to current time + DEFAULT_GPU_LAUNCH seconds
        current_time = int(time.time())
        expires = current_time + DEFAULT_GPU_LAUNCH

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
                logger.info(f"Successfully launched workload on attempt {attempt}")
                return result
            else:
                # Check if response is HTML (likely CloudFlare error page)
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type or response.text.strip().startswith('<!DOCTYPE html>'):
                    logger.error(f"❌ Attempt {attempt} failed - Error launching workload: {response.status_code} (CloudFlare error page)")
                else:
                    logger.error(f"❌ Attempt {attempt} failed - Error launching workload: {response.status_code}")
                    # Only log the first 100 characters of the response to avoid log spam
                    if response.text:
                        truncated_response = response.text[:100] + ('...' if len(response.text) > 100 else '')
                        logger.error(f"Response: {truncated_response}")

        except Exception as e:
            logger.error(f"❌ Attempt {attempt} failed - Error connecting to Comput3.ai: {str(e)}")

        # If not the last attempt, wait before retrying
        if attempt < len(retry_delays):
            logger.info(f"Waiting {delay} seconds before next launch attempt...")
            time.sleep(delay)

    logger.error(f"❌ Failed to launch workload after {len(retry_delays)} attempts.")
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

def start_monitor():
    """Start a unified monitor thread for GPU health and job timeouts"""
    global monitor_thread, monitor_active

    if monitor_thread is not None and monitor_thread.is_alive():
        return

    monitor_active = True
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    logger.info("Started unified monitor thread")

def monitor():
    """Monitor GPU health, job timeouts, and idle time"""
    global gpu_instance, last_job_time, monitor_active, current_job_id, current_job_data, job_start_time, job_cancel_flag

    idle_threshold = timedelta(seconds=GPU_IDLE_TIMEOUT)
    max_gpu_runtime = timedelta(seconds=DEFAULT_GPU_LAUNCH)  # Maximum GPU runtime

    while monitor_active and gpu_instance is not None:
        try:
            # 1. Check GPU health
            if not is_gpu_healthy():
                logger.warning(f"GPU instance {gpu_instance.get('node')} is unhealthy")

                # Handle unhealthy GPU for active job
                if current_job_id is not None and current_job_data is not None:
                    error_msg = "GPU instance became unhealthy during job execution"

                    # Check if the job already has a result_url from partial completion
                    result_url = redis_client.hget(f"job:{current_job_id}", "result_url")

                    # Mark job as cancelling - this will prevent the job thread from sending a webhook
                    redis_client.hset(f"job:{current_job_id}", "cancelling", "true")

                    # Update job status to failed
                    update_job_status(current_job_id, "failed", error=error_msg)

                    # Prepare webhook data
                    webhook_data = {"error": error_msg}
                    if result_url:
                        logger.info(f"Including partial result_url in failure webhook for job {current_job_id}")
                        webhook_data["result_url"] = result_url

                    # Send webhook
                    send_webhook_notification(current_job_id, current_job_data, "failed", **webhook_data)

                    # Signal job to terminate
                    job_cancel_flag = True

                shutdown_gpu_instance()
                break

            # 2. Check if GPU has been running too long
            if gpu_launch_time:
                gpu_runtime = datetime.now() - gpu_launch_time
                gpu_time_left = max_gpu_runtime - gpu_runtime

                if gpu_runtime > max_gpu_runtime:
                    logger.info(f"GPU instance {gpu_instance.get('node')} has been running for over {DEFAULT_GPU_LAUNCH} seconds.")

                    if current_job_id is None:
                        logger.info("Shutting down to refresh before approaching expiration limit.")
                        shutdown_gpu_instance()
                        break
                    else:
                        logger.info("GPU is approaching expiration but has an active job. Will shut down after job completion.")
                else:
                    # Log GPU time remaining every minute
                    if gpu_runtime.seconds % 60 == 0 and gpu_runtime.seconds > 0:
                        logger.info(f"GPU instance running for {gpu_runtime.seconds}s, {gpu_time_left.seconds}s remaining until refresh")

            # 3. Check active job status
            if current_job_id is not None and current_job_data is not None and job_start_time is not None:
                # Get current time
                now = time.time()
                now_readable = datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')

                # Calculate elapsed time
                elapsed_time = datetime.now() - job_start_time

                # Calculate time remaining for both constraints
                max_time = int(current_job_data.get("max_time", 1200))  # Default 20 minutes if not provided
                max_time_remaining = max_time - elapsed_time.total_seconds()

                complete_by = int(current_job_data.get("complete_by", 0))
                complete_by_remaining = complete_by - now if complete_by > 0 else float('inf')

                # Define complete_by_readable here before it's used
                complete_by_readable = datetime.fromtimestamp(complete_by).strftime('%Y-%m-%d %H:%M:%S') if complete_by > 0 else "None"

                # Log status on every monitor thread iteration (already runs every 5 seconds)
                logger.info(f"🏃 JOB RUNNING - Job {current_job_id}: elapsed={elapsed_time.total_seconds():.1f}s, " +
                           f"max_time_remaining={max_time_remaining:.1f}s, " +
                           f"complete_by_remaining={complete_by_remaining:.1f}s " +
                           f"(now={now:.0f}/{now_readable}, complete_by={complete_by}/{complete_by_readable})")

                # 3a. Check for max_time exceeded
                if elapsed_time.total_seconds() > max_time:
                    error_msg = f"Job exceeded maximum runtime of {max_time} seconds"
                    logger.error(f"Job {current_job_id}: {error_msg}")

                    # Check if the job already has a result_url from partial completion
                    result_url = redis_client.hget(f"job:{current_job_id}", "result_url")

                    # Mark job as cancelling - this will prevent the job thread from sending a webhook
                    redis_client.hset(f"job:{current_job_id}", "cancelling", "true")

                    # Update job status to failed
                    update_job_status(current_job_id, "failed", error=error_msg)

                    # Prepare webhook data
                    webhook_data = {"error": error_msg}
                    if result_url:
                        logger.info(f"Including partial result_url in failure webhook for job {current_job_id}")
                        webhook_data["result_url"] = result_url

                    # Send webhook
                    send_webhook_notification(current_job_id, current_job_data, "failed", **webhook_data)

                    # Signal job to terminate
                    job_cancel_flag = True

                    # Reset job tracking
                    current_job_id = None
                    current_job_data = None
                    job_start_time = None

                # 3b. Check for complete_by deadline, but first make sure job hasn't been reset
                elif complete_by > 0 and current_job_id is not None:
                    now_readable = datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
                    complete_by_readable = datetime.fromtimestamp(complete_by).strftime('%Y-%m-%d %H:%M:%S')

                    # Get created_at timestamp if available
                    created_at = int(current_job_data.get("created_at", 0))
                    created_at_readable = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S') if created_at > 0 else "unknown"

                    logger.debug(f"Complete_by check: now={now} ({now_readable}) vs complete_by={complete_by} ({complete_by_readable}), " +
                                f"now > complete_by = {now > complete_by}, created_at={created_at} ({created_at_readable})")

                    if now > complete_by:
                        error_msg = f"Job exceeded completion deadline (created_at: {created_at_readable}, complete_by: {complete_by_readable})"
                        logger.error(f"Job {current_job_id}: {error_msg}")

                        # Check if the job already has a result_url from partial completion
                        result_url = redis_client.hget(f"job:{current_job_id}", "result_url")

                        # Mark job as cancelling - this will prevent the job thread from sending a webhook
                        redis_client.hset(f"job:{current_job_id}", "cancelling", "true")

                        # Update job status to failed
                        update_job_status(current_job_id, "failed", error=error_msg)

                        # Prepare webhook data
                        webhook_data = {"error": error_msg}
                        if result_url:
                            logger.info(f"Including partial result_url in failure webhook for job {current_job_id}")
                            webhook_data["result_url"] = result_url

                        # Send webhook
                        send_webhook_notification(current_job_id, current_job_data, "failed", **webhook_data)

                        # Signal job to terminate
                        job_cancel_flag = True

                        # Reset job tracking
                        current_job_id = None
                        current_job_data = None
                        job_start_time = None

            # 4. Check idle time if no active job
            elif current_job_id is None:
                idle_time = datetime.now() - last_job_time

                if idle_time > idle_threshold:
                    logger.info(f"GPU instance {gpu_instance.get('node')} has been idle for {idle_time}, shutting down")
                    shutdown_gpu_instance()
                    break

                # Log idle status every minute
                if idle_time.seconds % 60 == 0 and idle_time.seconds > 0:
                    logger.info(f"GPU instance idle for {idle_time}")

            # Check every 5 seconds for better responsiveness to timeouts
            time.sleep(5)
        except Exception as e:
            logger.exception(f"Error in monitor thread: {e}")
            # Continue the loop to maintain monitoring despite errors

    logger.info("Monitor stopped")

def get_gpu_instance():
    """Get or create a GPU instance for processing"""
    global gpu_instance, last_job_time, gpu_launch_time

    # If we already have an instance, update the last job time and return it
    if gpu_instance is not None:
        # Check if the instance is healthy
        if is_gpu_healthy():
            # Check if GPU has been running for more than DEFAULT_GPU_LAUNCH
            if gpu_launch_time and (datetime.now() - gpu_launch_time).total_seconds() > DEFAULT_GPU_LAUNCH:
                logger.info(f"GPU instance {gpu_instance.get('node')} has been running for over {DEFAULT_GPU_LAUNCH} seconds. Shutting down for refresh.")
                shutdown_gpu_instance()
                # Continue to launch a new instance
            else:
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
        gpu_launch_time = datetime.now()  # Set the launch time
        node_hostname = result.get('node')
        logger.info(f"Successfully launched GPU instance: {node_hostname} at {gpu_launch_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Start the monitor if not already running
        start_monitor()

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
    global gpu_instance, monitor_active, current_job_id, current_job_data, gpu_launch_time

    if gpu_instance is None:
        return

    node_hostname = gpu_instance.get('node')
    node_id = gpu_instance.get('workload')

    # Reset job tracking
    current_job_id = None
    current_job_data = None

    # Stop GPU monitoring
    monitor_active = False

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
    gpu_launch_time = None  # Reset launch time

def upload_to_storage(file_path):
    """Upload file to MinIO storage using internal client and return public presigned URL"""
    # Check if the internal client (needed for upload) is available
    if not internal_minio_client:
        logger.error("Internal MinIO client not available. Cannot upload file.")
        return f"file://{file_path}"

    if not os.path.exists(file_path):
        logger.error(f"❌ File not found for upload: {file_path}")
        return None # Indicate upload failure clearly

    filename = os.path.basename(file_path)
    # Use a simple object name structure: output/<original_filename>
    object_name = f"output/{filename}"

    logger.info(f"Uploading {file_path} to MinIO bucket '{minio_bucket}' as '{object_name}' using INTERNAL client.")

    try:
        # Detect MIME type using python-magic
        mime_type = magic.from_file(file_path, mime=True)
        logger.info(f"Detected MIME type: {mime_type}")

        # Get file size
        file_size = os.path.getsize(file_path)

        # *** Use internal client for the actual upload ***
        internal_minio_client.fput_object(
            minio_bucket,
            object_name,
            file_path,
            content_type=mime_type,
        )
        logger.info(f"Successfully uploaded {filename} ({file_size} bytes) using internal client.")

        # *** Generate presigned URL using the PUBLIC client if available ***
        url_client = public_minio_client or internal_minio_client # Fallback to internal if public failed/not configured

        if public_minio_client:
            logger.info("Generating presigned URL using PUBLIC client.")
        else:
            logger.warning("Public MinIO client not available. Generating presigned URL using INTERNAL client (URL might not be publicly accessible).")

        try:
            presigned_url = url_client.presigned_get_object(
                minio_bucket,
                object_name,
                expires=timedelta(days=7)
            )
            logger.info(f"Generated presigned URL (valid for 7 days): {presigned_url}")
            return presigned_url
        except Exception as url_err:
            logger.error(f"❌ Failed to generate presigned URL for {object_name} using {'public' if public_minio_client else 'internal'} client: {url_err}")
            # Fallback to minio path if URL generation fails completely
            return f"minio://{minio_bucket}/{object_name}"

    except S3Error as e:
        logger.error(f"❌ MinIO S3 Error during upload using internal client: {e}")
        return f"file://{file_path}" # Fallback to local path on S3 error
    except FileNotFoundError:
        # This check is technically redundant due to the check at the start,
        # but kept for robustness in case of race conditions.
        logger.error(f"❌ File disappeared before upload: {file_path}")
        return None # Indicate upload failure clearly
    except Exception as e:
        logger.exception(f"❌ Unexpected error during MinIO upload/URL generation: {str(e)}")
        return f"file://{file_path}" # Fallback on unexpected error

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

def process_csm_job(job_id, job_data, gpu_instance, cancel_checker=None):
    """Process a text-to-speech job"""
    logger.info(f"Processing CSM job: {job_id}")

    try:
        # Generate speech from text using CSM
        audio_file = csm.text_to_speech_with_csm(job_data.get("data", "{}"), job_id, gpu_instance, api_key, OUTPUT_DIR, cancel_checker)

        # Save the output path (we're not uploading to Minio yet)
        result_url = upload_to_storage(audio_file)

        # Update job status in Redis (no webhook)
        update_job_status(job_id, "success", result_url=result_url)

        # Note: We're not cleaning up the file since we want to keep it in the output directory
        return True

    except Exception as e:
        error_msg = f"Error processing CSM job: {str(e)}"
        logger.exception(error_msg)

        # Only update the job status in Redis (no webhook)
        update_job_status(job_id, "failed", error=error_msg)
        return False

def process_whisper_job(job_id, job_data, gpu_instance, cancel_checker=None):
    """Process a speech-to-text job"""
    logger.info(f"Processing Whisper job: {job_id}")

    try:
        # Extract job parameters
        job_params = json.loads(job_data.get("data", "{}"))
        audio_url = job_params.get("audio_url")
        model = job_params.get("model", "medium")

        logger.info(f"Job parameters: audio_url={audio_url}, model={model}")

        # Process with Whisper
        transcription = whisper.speech_to_text_with_whisper(job_data.get("data", "{}"), job_id, gpu_instance, api_key, OUTPUT_DIR, cancel_checker)

        # Update job status (no webhook)
        update_job_status(job_id, "success", result=transcription)
        return True

    except Exception as e:
        error_msg = f"Error processing Whisper job: {str(e)}"
        logger.exception(error_msg)

        # Only update job status in Redis (no webhook)
        update_job_status(job_id, "failed", error=error_msg)
        return False

def process_portrait_job(job_id, job_data, gpu_instance, cancel_checker=None):
    """Process a portrait video job using ComfyUI"""
    logger.info(f"Processing Portrait job: {job_id}")

    try:
        # Extract job parameters
        job_params = json.loads(job_data.get("data", "{}"))
        image_url = job_params.get("image_url")
        audio_url = job_params.get("audio_url")

        logger.info(f"Job parameters: image_url={image_url}, audio_url={audio_url}")

        # Run portrait generation using ComfyUI
        result = comfyui.generate_portrait_video(image_url, audio_url, job_id, gpu_instance, api_key, OUTPUT_DIR, cancel_checker)

        # Handle result based on return type
        if isinstance(result, tuple) and len(result) == 2 and result[0] is False:
            # This is an error with detailed message
            _, detailed_error = result

            # Check if the detailed error is a dictionary with error and result_url
            result_url = None
            if isinstance(detailed_error, dict) and 'error' in detailed_error:
                error_msg = f"Failed to generate portrait video: {detailed_error['error']}"
                if 'result_url' in detailed_error:
                    result_url = detailed_error['result_url']
                    logger.info(f"Got partial result URL from ComfyUI cancellation: {result_url}")
                    # Save the result URL for future reference
                    redis_client.hset(f"job:{job_id}", "result_url", result_url)
            else:
                error_msg = f"Failed to generate portrait video: {detailed_error}"

            logger.error(error_msg)

            # Update job status in Redis (no webhook)
            update_job_status(job_id, "failed", error=error_msg)
            return False

        elif result:
            # This is a successful result with video path
            video_output_path = result
            # Save the result URL
            result_url = upload_to_storage(video_output_path)

            # Update job status in Redis (no webhook)
            update_job_status(job_id, "success", result_url=result_url)
            return True
        else:
            # Backward compatibility for old return format (just False)
            error_msg = "Failed to generate portrait video"

            # Update job status in Redis (no webhook)
            update_job_status(job_id, "failed", error=error_msg)
            return False

    except Exception as e:
        error_msg = f"Error processing Portrait job: {str(e)}"
        logger.exception(error_msg)

        # Update job status in Redis (no webhook)
        update_job_status(job_id, "failed", error=error_msg)
        return False

def process_analyze_job(job_id, job_data, gpu_instance, cancel_checker=None):
    """Process an image analysis job"""
    logger.info(f"Processing Analyze job: {job_id}")

    try:
        # Extract job parameters
        job_params = json.loads(job_data.get("data", "{}"))
        image_url = job_params.get("image_url")

        logger.info(f"Job parameters: image_url={image_url}")

        # TODO: Connect to vision model API
        # TODO: Download image from URL to output directory with job ID filename
        # image_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_input.jpg")
        # TODO: Process with vision model
        # TODO: Save results to output directory with job ID filename
        # text_output_path = os.path.join(OUTPUT_DIR, f"{job_id}.txt")

        # For now, just simulate success
        result_text = "This is a simulated image analysis result"
        update_job_status(job_id, "success", result=result_text)
        return True

    except Exception as e:
        error_msg = f"Error processing Analyze job: {str(e)}"
        logger.exception(error_msg)
        update_job_status(job_id, "failed", error=error_msg)
        return False

def update_job_status(job_id, status, **kwargs):
    """Update job status in Redis"""
    redis_client.hset(f"job:{job_id}", "status", status)

    # Update any additional fields
    for key, value in kwargs.items():
        redis_client.hset(f"job:{job_id}", key, value)

    if status == "running":
        logger.info(f"🏃 JOB RUNNING - Updated job {job_id} status to {status}")
    else:
        logger.info(f"Updated job {job_id} status to {status}")

def safely_peek_job():
    """Check if there's a job in the queue without removing it"""
    try:
        # Use lrange to peek at the first job in the queue
        jobs = redis_client.lrange(JOB_QUEUE, 0, 0)
        if jobs:
            return jobs[0]  # Return the job ID without removing it
        return None
    except Exception as e:
        logger.error(f"Error peeking at job queue: {str(e)}")
        return None

def safely_claim_job(job_id):
    """Atomically claim a job from the queue if it matches the expected ID"""
    try:
        # Use a Lua script to atomically check and remove the job
        script = """
        local job_id = ARGV[1]
        local queue_key = KEYS[1]

        -- Check if the first job in the queue matches our expected job_id
        local first_job = redis.call('lindex', queue_key, 0)
        if first_job == job_id then
            -- Remove and return the job
            return redis.call('lpop', queue_key)
        else
            -- Job is no longer first in queue or was taken by another worker
            return nil
        end
        """

        result = redis_client.eval(script, 1, JOB_QUEUE, job_id)
        if result == job_id:
            logger.info(f"Successfully claimed job {job_id}")
            return True
        else:
            logger.info(f"Failed to claim job {job_id} - it was likely taken by another worker")
            return False

    except Exception as e:
        logger.error(f"Error claiming job: {str(e)}")
        return False

def process_job(job_id, gpu_instance):
    """Set up job for processing in a separate thread"""
    global current_job_id, current_job_data, job_start_time, job_thread, job_cancel_flag

    # Get job data from Redis
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        logger.error(f"Job {job_id} not found in Redis")
        return True

    # Reset cancel flag
    job_cancel_flag = False

    # Set tracking variables
    current_job_id = job_id
    current_job_data = job_data
    job_start_time = datetime.now()

    # Update job status to running
    update_job_status(job_id, "running")

    # Create thread for job processing
    job_thread = threading.Thread(
        target=_job_runner_thread,
        args=(job_id, job_data, gpu_instance)
    )
    job_thread.daemon = True
    job_thread.start()

    logger.info(f"🏃 JOB RUNNING - Started processing thread for job {job_id}")
    return True

def _job_runner_thread(job_id, job_data, gpu_instance):
    """Thread that runs the actual job processing"""
    global job_cancel_flag, current_job_id, current_job_data, job_start_time

    try:
        # Get job type
        job_type = job_data.get("type")
        logger.info(f"🏃 JOB RUNNING - Job thread started for {job_type} job {job_id}")

        # Create a cancel check function to pass to render engines
        def check_cancellation():
            return job_cancel_flag

        # Process based on job type
        success = False
        if job_type == "csm":
            success = process_csm_job(job_id, job_data, gpu_instance, check_cancellation)
        elif job_type == "whisper":
            success = process_whisper_job(job_id, job_data, gpu_instance, check_cancellation)
        elif job_type == "portrait":
            success = process_portrait_job(job_id, job_data, gpu_instance, check_cancellation)
        elif job_type == "analyze":
            success = process_analyze_job(job_id, job_data, gpu_instance, check_cancellation)
        else:
            error_msg = f"Unknown job type: {job_type}"
            logger.error(error_msg)
            update_job_status(job_id, "failed", error=error_msg)

        # Retrieve the current status and info from Redis
        job_info = redis_client.hgetall(f"job:{job_id}")
        job_status = job_info.get("status", "unknown")

        # Check if job is already being cancelled by the monitor thread
        is_cancelling = job_info.get("cancelling") == "true"

        # Only send webhook if the monitor thread hasn't already sent one
        if not is_cancelling:
            # Get any result URL or error message
            result_url = job_info.get("result_url")
            error_msg = job_info.get("error")
            result_text = job_info.get("result")

            # Send webhook with the appropriate data
            if job_status == "success":
                logger.info(f"Sending success webhook for job {job_id}")
                webhook_data = {}

                # Include the appropriate result data based on job type
                if result_url:
                    webhook_data["result_url"] = result_url
                if result_text:
                    webhook_data["text"] = result_text

                send_webhook_notification(job_id, job_data, "success", **webhook_data)
            elif job_status == "failed":
                logger.info(f"Sending failure webhook for job {job_id}")
                webhook_data = {}

                if error_msg:
                    webhook_data["error"] = error_msg
                if result_url:
                    logger.info(f"Including partial result_url in failure webhook for job {job_id}")
                    webhook_data["result_url"] = result_url

                send_webhook_notification(job_id, job_data, "failed", **webhook_data)
        else:
            logger.info(f"Job {job_id} is marked as cancelling, monitor thread is handling the webhook")

        # Check if GPU has been running longer than DEFAULT_GPU_REPLACE after job completion
        if gpu_launch_time and (datetime.now() - gpu_launch_time).total_seconds() > DEFAULT_GPU_REPLACE:
            logger.info(f"GPU instance {gpu_instance.get('node')} has been running for more than {DEFAULT_GPU_REPLACE} seconds after job completion. Shutting down for refresh.")
            shutdown_gpu_instance()

        logger.info(f"Job thread completed for {job_id} with success={success}")
        
        # Reset the job tracking variables
        current_job_id = None
        current_job_data = None
        job_start_time = None
        
    except Exception as e:
        error_msg = f"Error in job thread: {str(e)}"
        logger.exception(error_msg)

        # Update job status
        update_job_status(job_id, "failed", error=error_msg)

        # Check if job is being cancelled by the monitor thread
        is_cancelling = redis_client.hget(f"job:{job_id}", "cancelling") == "true"

        # Only send webhook if not already being cancelled
        if not is_cancelling:
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)
        
        # Reset the job tracking variables after an error too
        current_job_id = None
        current_job_data = None
        job_start_time = None

def lock_job(job_id, worker_id, lock_timeout=300):
    """
    Try to lock a job for a worker

    Args:
        job_id: The job ID to lock
        worker_id: Unique ID of this worker instance
        lock_timeout: Seconds until lock expires (prevents stuck jobs if worker crashes)

    Returns:
        True if lock acquired, False otherwise
    """
    # Use Lua script for atomic operations
    lock_script = """
    local job_key = KEYS[1]
    local worker_id = ARGV[1]
    local lock_timeout = ARGV[2]

    -- Check if job exists
    if redis.call('EXISTS', job_key) == 0 then
        return 0
    end

    -- Check if already locked
    local locked_by = redis.call('HGET', job_key, 'locked_by')
    if locked_by and locked_by ~= '' then
        return 0
    end

    -- Set lock
    redis.call('HSET', job_key, 'locked_by', worker_id)
    redis.call('HSET', job_key, 'locked_at', redis.call('TIME')[1])

    -- Set lock expiration (separate key with same timeout)
    local lock_key = job_key .. ':lock'
    redis.call('SET', lock_key, worker_id, 'EX', lock_timeout)

    return 1
    """

    try:
        result = redis_client.eval(
            lock_script,
            1,  # 1 key
            f"job:{job_id}",  # KEYS[1]
            worker_id,  # ARGV[1]
            lock_timeout  # ARGV[2]
        )
        return result == 1
    except Exception as e:
        logger.error(f"Error locking job {job_id}: {e}")
        return False

def release_job_lock(job_id, worker_id):
    """Release a job lock if it's held by this worker"""
    release_script = """
    local job_key = KEYS[1]
    local lock_key = job_key .. ':lock'
    local worker_id = ARGV[1]

    -- Check if we own the lock
    local locked_by = redis.call('HGET', job_key, 'locked_by')
    if locked_by == worker_id then
        -- Release the lock
        redis.call('HDEL', job_key, 'locked_by')
        redis.call('HDEL', job_key, 'locked_at')
        redis.call('DEL', lock_key)
        return 1
    end

    return 0
    """

    try:
        result = redis_client.eval(
            release_script,
            1,  # 1 key
            f"job:{job_id}",  # KEYS[1]
            worker_id  # ARGV[1]
        )
        return result == 1
    except Exception as e:
        logger.error(f"Error releasing job lock {job_id}: {e}")
        return False

def dequeue_job(job_id, worker_id):
    """
    Permanently remove job from queue if we have the lock
    Returns True if dequeued, False otherwise
    """
    dequeue_script = """
    local job_id = ARGV[1]
    local queue_key = KEYS[1]
    local job_key = KEYS[2]
    local worker_id = ARGV[2]

    -- Check if we own the lock
    local locked_by = redis.call('HGET', job_key, 'locked_by')
    if locked_by == worker_id then
        -- Remove from queue
        redis.call('LREM', queue_key, 1, job_id)
        return 1
    end

    return 0
    """

    try:
        result = redis_client.eval(
            dequeue_script,
            2,  # 2 keys
            JOB_QUEUE,  # KEYS[1]
            f"job:{job_id}",  # KEYS[2]
            job_id,  # ARGV[1]
            worker_id  # ARGV[2]
        )
        return result == 1
    except Exception as e:
        logger.error(f"Error dequeuing job {job_id}: {e}")
        return False

def check_job_expired(job_id):
    """Check if a job has expired based on complete_by timestamp"""
    try:
        job_data = redis_client.hgetall(f"job:{job_id}")
        if not job_data:
            return False  # Job not found, can't be expired

        complete_by = int(job_data.get("complete_by", 0))
        if complete_by > 0 and time.time() > complete_by:
            logger.info(f"Job {job_id} has expired (complete_by: {complete_by})")
            return True

        return False
    except Exception as e:
        logger.error(f"Error checking job expiration: {e}")
        return False  # Assume not expired on error

def handle_expired_job(job_id):
    """Mark a job as failed due to expiration and remove from queue"""
    try:
        # Get job data
        job_data = redis_client.hgetall(f"job:{job_id}")
        if not job_data:
            return False

        # Mark as failed
        error_msg = "Job expired before processing could begin"
        update_job_status(job_id, "failed", error=error_msg)

        # Only send webhook if not already being handled
        is_cancelling = job_data.get("cancelling") == "true"
        if not is_cancelling:
            # Send webhook notification
            send_webhook_notification(job_id, job_data, "failed", error=error_msg)

        # Remove from queue
        redis_client.lrem(JOB_QUEUE, 1, job_id)

        logger.info(f"Removed expired job {job_id} from queue")
        return True
    except Exception as e:
        logger.error(f"Error handling expired job: {e}")
        return False

def main():
    """Main worker loop"""
    global gpu_instance, last_job_time

    logger.info(f"Starting C3 Render Worker with ID: {WORKER_ID}")
    logger.info(f"Using polling interval: Queue check={QUEUE_CHECK_INTERVAL}s")

    try:
        while True:
            # Skip claiming new jobs if we're already processing one
            if current_job_id is not None:
                logger.debug(f"Currently processing job {current_job_id}, waiting before checking queue again")
                time.sleep(QUEUE_CHECK_INTERVAL)
                continue

            # Get all jobs from the queue (up to 10 to avoid excessive load)
            jobs = redis_client.lrange(JOB_QUEUE, 0, 9)

            if jobs:
                # Jobs exist in queue, try to process one
                processed_job = False

                # Apply standard delay based on GPU availability
                # This ensures workers with GPUs get priority by responding faster
                delay = GPU_WORKER_DELAY if (gpu_instance and is_gpu_healthy()) else NO_GPU_WORKER_DELAY
                logger.debug(f"Applying {delay}s delay for job processing (has_gpu={gpu_instance is not None})")
                time.sleep(delay)

                # Try each job in the queue until we successfully process one
                for job_id in jobs:
                    # Check if expired first
                    if check_job_expired(job_id):
                        handle_expired_job(job_id)
                        continue

                    # Make sure job is still in the queue
                    if job_id not in redis_client.lrange(JOB_QUEUE, 0, 9):
                        logger.debug(f"Job {job_id} was claimed by another worker")
                        continue

                    logger.debug(f"Attempting to lock job {job_id}")

                    # Try to lock the job
                    if lock_job(job_id, WORKER_ID):
                        logger.info(f"Successfully locked job {job_id}")

                        try:
                            # Now we have the lock, try to get a GPU if needed
                            if gpu_instance is None or not is_gpu_healthy():
                                logger.info("Launching GPU for locked job")
                                gpu_instance = get_gpu_instance()

                            if gpu_instance:
                                # Successfully got GPU, dequeue the job
                                if dequeue_job(job_id, WORKER_ID):
                                    logger.info(f"Successfully dequeued job {job_id}")

                                    # Initialize/start the monitor if not running
                                    start_monitor()

                                    # Process the job
                                    process_job(job_id, gpu_instance)
                                    last_job_time = datetime.now()
                                    processed_job = True
                                    break  # Exit the loop after processing a job
                                else:
                                    logger.warning(f"Failed to dequeue job {job_id} despite having lock")
                                    release_job_lock(job_id, WORKER_ID)
                            else:
                                logger.error("Failed to get GPU for locked job")
                                release_job_lock(job_id, WORKER_ID)
                        except Exception as e:
                            logger.exception(f"Error processing locked job: {e}")
                            release_job_lock(job_id, WORKER_ID)
                    else:
                        logger.debug(f"Job {job_id} is locked by another worker, trying next job")
                        # No sleep here, continue to next job immediately

                # If no job could be processed after trying all in the queue, add a short sleep
                if not processed_job:
                    logger.debug("Could not process any job in the queue, all locked or expired")
                    time.sleep(QUEUE_CHECK_INTERVAL)  # Use consistent QUEUE_CHECK_INTERVAL instead of LOCK_RETRY_INTERVAL
            else:
                # No jobs in queue, small sleep to avoid hammering Redis
                time.sleep(QUEUE_CHECK_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Shutting down worker")
        if gpu_instance:
            shutdown_gpu_instance()

if __name__ == "__main__":
    main()