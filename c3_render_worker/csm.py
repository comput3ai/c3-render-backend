# Module for CSM text-to-speech generation
import os
import time
import logging
import json
import requests
import re
from gradio_client import Client

logger = logging.getLogger(__name__)

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

def text_to_speech_with_csm(text, job_id, gpu_instance, api_key, output_dir):
    """Generate speech from text using CSM with configurable voice options"""
    logger.info(f"Generating speech with CSM: {text[:50]}...")
    
    # Extract job parameters
    job_data = json.loads(text)
    job_params = job_data if isinstance(job_data, dict) else {}
    
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
    monologue_text = job_params.get("monologue", job_params.get("text", ""))
    
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
        sentences = split_text_into_sentences(monologue_text)
        
        # Prepare the monologue text as a JSON array of sentences
        monologue_json = json.dumps(sentences)
        
        # Handle voice cloning if reference audio is provided
        reference_audio_file = None
        if voice == "clone" and reference_audio_url and reference_text:
            # Download reference audio for voice cloning
            logger.info(f"Downloading reference audio for voice cloning: {reference_audio_url}")
            
            try:
                # Create a temporary file for the reference audio
                reference_audio_file = os.path.join(output_dir, f"{job_id}_reference.mp3")
                
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
        output_path = os.path.join(output_dir, filename)
        
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