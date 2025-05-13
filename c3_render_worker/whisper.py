# Module for Whisper speech-to-text transcription
import os
import time
import logging
import json
import requests
from gradio_client import Client, handle_file

# Import from constants file
from constants import RENDER_POLLING_INTERVAL

logger = logging.getLogger(__name__)

def download_audio(audio_url, output_path):
    """Download audio file from URL to the specified path"""
    logger.info(f"Downloading audio from URL: {audio_url}")

    try:
        response = requests.get(audio_url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded audio to {output_path}")
            return True
        else:
            logger.error(f"Failed to download audio: {response.status_code}")
            return False
    except Exception as e:
        logger.exception(f"Error downloading audio: {str(e)}")
        return False

def speech_to_text_with_whisper(text, job_id, gpu_instance, api_key, output_dir, cancel_callback=None):
    """Transcribe speech to text using Whisper with configurable model options"""
    logger.info(f"Starting Whisper transcription for job: {job_id}")

    # Extract job parameters
    job_data = json.loads(text)
    job_params = job_data if isinstance(job_data, dict) else {}

    # Get Whisper-specific parameters with defaults
    audio_url = job_params.get("audio_url")
    model = job_params.get("model", "medium")
    task = job_params.get("task", "transcribe")
    language = job_params.get("language", "")  # Empty string for auto-detection

    if not audio_url:
        logger.error("No audio_url provided in job parameters")
        raise ValueError("audio_url is required")

    # Get node hostname from GPU instance
    node_hostname = gpu_instance.get('node')

    # Construct Whisper endpoint URL using the node hostname
    whisper_url = f"https://{node_hostname}/whisper/"
    logger.info(f"Using Whisper endpoint: {whisper_url}")

    # Create Gradio client with retries
    client = None
    last_error = None

    for attempt in range(1, 6):
        try:
            logger.info(f"Attempt {attempt}/5 to connect to Whisper service...")

            # Create Gradio client with API key header
            client = Client(
                whisper_url,
                headers={"X-C3-API-KEY": api_key}
            )

            # If we get here, the client was created successfully
            logger.info(f"Successfully connected to Whisper service on attempt {attempt}")
            break

        except Exception as e:
            last_error = e
            logger.warning(f"Failed to connect to Whisper service on attempt {attempt}: {str(e)}")

            # If this isn't the last attempt, wait before retrying
            if attempt < 5:
                logger.info(f"Waiting 5 seconds before retrying...")
                time.sleep(5)

    # If we couldn't create the client after all retries, raise the last error
    if client is None:
        logger.error(f"Failed to connect to Whisper service after 5 attempts")
        raise last_error

    try:
        # Download the audio file
        audio_input_path = os.path.join(output_dir, f"{job_id}_input.mp3")
        if not download_audio(audio_url, audio_input_path):
            raise Exception(f"Failed to download audio from {audio_url}")

        # Log the Whisper parameters
        logger.info(f"Whisper parameters: model={model}, task={task}, language={language}")

        # Call Whisper API with the downloaded audio file
        logger.info(f"Calling Whisper API with model {model}...")

        # Parameters for prediction
        predict_params = {
            "audio_file": handle_file(audio_input_path),  # Format file for Gradio
            "model": model,
            "task": task,
            "language": language,
            "device": "cuda",
            "compute_type": "float16",
            "batch_size": 16,
            "vad_method": "pyannote",
            "word_timestamps": True,
            "segment_resolution": "sentence",
            "api_name": "/transcribe_audio"
        }

        # Set up timing tracking for logging purposes only
        start_time = time.time()

        # Use the client with a custom progress tracking function
        logger.info("Starting Whisper transcription")

        result = None

        # Submit the prediction request and start tracking
        job = client.submit(**predict_params)

        # Poll for completion
        while True:
            # Check for cancellation request
            if cancel_callback and cancel_callback():
                logger.warning("Job cancellation detected - terminating Whisper transcription")
                # Try to cancel the job if possible
                try:
                    job.cancel()
                    logger.info("Sent cancellation request to Whisper")
                except Exception as e:
                    logger.warning(f"Error cancelling Whisper job: {e}")

                # Get the specific error message from Redis if available
                try:
                    from redis import Redis
                    redis_host = os.getenv("REDIS_HOST", "localhost")
                    redis_port = int(os.getenv("REDIS_PORT", "6379"))
                    redis_client = Redis(host=redis_host, port=redis_port, decode_responses=True)

                    job_data = redis_client.hgetall(f"job:{job_id}")
                    if job_data and "error" in job_data:
                        error_msg = job_data["error"]
                        logger.info(f"Using specific error reason from Redis: {error_msg}")
                    else:
                        error_msg = "Job was cancelled by system"
                except Exception as e:
                    logger.warning(f"Error retrieving specific error message: {e}")
                    error_msg = "Job was cancelled by system"

                # Clean up resources
                if os.path.exists(audio_input_path):
                    os.unlink(audio_input_path)

                raise Exception(error_msg)

            elapsed_time = time.time() - start_time

            # Check if job is done
            if job.done():
                logger.info("Whisper transcription completed")
                result = job.result()
                break

            # Log progress and wait before checking again
            logger.info(f"Whisper transcription in progress (elapsed: {elapsed_time:.1f}s)... Checking again in {RENDER_POLLING_INTERVAL}s")
            time.sleep(RENDER_POLLING_INTERVAL)

        # Unpack the result (transcript text and file paths)
        transcript, download_files = result

        logger.info(f"Transcription completed successfully. Result length: {len(transcript)}")

        # Save the transcript to a text file
        output_path = os.path.join(output_dir, f"{job_id}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)

        logger.info(f"Saved transcript to {output_path}")

        # Clean up the input audio file
        if os.path.exists(audio_input_path):
            os.unlink(audio_input_path)
            logger.info(f"Cleaned up input audio file {audio_input_path}")

        # Return the transcript text
        return transcript

    except Exception as e:
        logger.exception(f"Error during Whisper transcription: {str(e)}")

        # Clean up the input audio file
        if 'audio_input_path' in locals() and os.path.exists(audio_input_path):
            os.unlink(audio_input_path)
            logger.info(f"Cleaned up input audio file {audio_input_path}")

        raise