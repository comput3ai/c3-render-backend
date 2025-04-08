# Module for Whisper speech-to-text transcription
import os
import time
import logging
import json
import requests
from gradio_client import Client, handle_file

# Import from constants file
from constants import MAX_RENDER_TIME, RENDER_POLLING_INTERVAL

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

def speech_to_text_with_whisper(text, job_id, gpu_instance, api_key, output_dir):
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

    # Define retry parameters
    max_retries = 5
    retry_delay = 5  # seconds

    # Create Gradio client with retries
    client = None
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries} to connect to Whisper service...")

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
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)

    # If we couldn't create the client after all retries, raise the last error
    if client is None:
        logger.error(f"Failed to connect to Whisper service after {max_retries} attempts")
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

        # Set up timeout tracking
        start_time = time.time()
        max_wait_time = MAX_RENDER_TIME

        # Use the client with a custom progress tracking function to handle timeouts
        logger.info(f"Starting Whisper transcription with {max_wait_time}s timeout")

        result = None

        # Submit the prediction request and start tracking
        job = client.submit(**predict_params)

        # Poll for completion with timeout
        while True:
            elapsed_time = time.time() - start_time

            if elapsed_time > max_wait_time:
                logger.error(f"Whisper transcription timed out after {max_wait_time} seconds")
                # Try to cancel the job if possible
                try:
                    job.cancel()
                except:
                    pass
                raise Exception(f"Transcription timed out after {max_wait_time} seconds")

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