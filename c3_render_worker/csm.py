# Module for CSM text-to-speech generation
import os
import time
import logging
import json
import requests
import re
import subprocess
from gradio_client import Client, handle_file

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

def convert_to_mono(audio_path):
    """
    Convert audio file to mono format if it's in stereo.
    Returns path to a mono version of the audio file.
    """
    try:
        # Create new filename for mono version
        filename, ext = os.path.splitext(audio_path)
        mono_path = f"{filename}_mono{ext}"
        
        # Use ffmpeg to check if audio is already mono
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0', 
            '-show_entries', 'stream=channels', '-of', 'json', audio_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                channels = json.loads(result.stdout)['streams'][0]['channels']
                if channels == 1:
                    logger.info(f"Audio file is already mono: {audio_path}")
                    return audio_path
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning(f"Error parsing ffprobe output: {str(e)}. Will convert to be safe.")
        
        # Use ffmpeg directly to convert to mono
        convert_cmd = ['ffmpeg', '-i', audio_path, '-ac', '1', '-y', mono_path]
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Saved mono audio to: {mono_path}")
            return mono_path
        else:
            logger.warning(f"Error converting to mono with ffmpeg: {result.stderr}")
            return audio_path
    except Exception as e:
        logger.warning(f"Error converting audio to mono: {str(e)}. Using original audio.")
        return audio_path

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
        "clone": "upload_voice"
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
        # Prepare the monologue text as a JSON array with single item
        # This matches exactly how the working script formats text
        monologue_json = json.dumps([monologue_text])
        logger.info(f"Prepared monologue text as single-item JSON array")
        
        # Handle voice cloning if reference audio is provided
        reference_audio_file = None
        mono_audio_file = None
        audio_file = None  # For handle_file result
        
        if voice == "clone" and reference_audio_url and reference_text:
            logger.info("=== Using upload_voice workflow ===")
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
                    
                    # Convert to mono if in stereo format (important for voice cloning to work)
                    mono_audio_file = convert_to_mono(reference_audio_file)
                    logger.info(f"Using mono audio file for voice cloning: {mono_audio_file}")
                    
                    # Format reference_audio properly - exactly as in working script
                    logger.info(f"Preparing reference audio with handle_file")
                    audio_file = handle_file(mono_audio_file)
                    
                    logger.info(f"Reference text: '{reference_text}'")
                else:
                    logger.error(f"Failed to download reference audio: {response.status_code}")
                    raise Exception(f"Failed to download reference audio: {response.status_code}")
            except Exception as e:
                logger.exception(f"Error downloading reference audio: {str(e)}")
                raise
        
        # Log the CSM parameters
        logger.info(f"CSM parameters: voice={voice}, temperature={temperature}, topk={topk}, " +
                   f"max_audio_length={max_audio_length}, pause_duration={pause_duration}")
        
        # Generate audio based on voice type - exact match to working script structure
        if csm_voice == "upload_voice" and audio_file and reference_text:
            # Voice cloning call path - exactly matching working script
            logger.info("Generating audio with cloned voice...")
            
            result = client.predict(
                speaker_voice="upload_voice",
                speaker_text=reference_text,
                speaker_audio=audio_file,
                monologue_json=monologue_json,
                temperature=temperature,
                topk=topk,
                max_audio_length=max_audio_length,
                pause_duration=pause_duration,
                api_name="/generate_monologue_audio"
            )
        else:
            # Standard voice call path - exactly matching working script
            logger.info(f"=== Using {csm_voice} workflow ===")
            
            result = client.predict(
                speaker_voice=csm_voice,
                speaker_text="",  # No reference text needed for built-in voices
                speaker_audio=None,  # No reference audio needed for built-in voices
                monologue_json=monologue_json,
                temperature=temperature,
                topk=topk,
                max_audio_length=max_audio_length,
                pause_duration=pause_duration,
                api_name="/generate_monologue_audio"
            )
        
        # Result is a file path to the generated audio
        logger.info(f"CSM successfully generated audio: {result}")
        
        # Generate a filename using the job ID
        filename = f"{job_id}.mp3"
        output_path = os.path.join(output_dir, filename)
        
        # Copy the file to our output directory
        with open(result, 'rb') as src_file, open(output_path, 'wb') as dest_file:
            dest_file.write(src_file.read())
        
        logger.info(f"Saved audio to {output_path}")
        
        # Clean up reference audio files if they exist
        if reference_audio_file and os.path.exists(reference_audio_file):
            os.unlink(reference_audio_file)
            logger.info(f"Cleaned up reference audio file {reference_audio_file}")
        
        if mono_audio_file and mono_audio_file != reference_audio_file and os.path.exists(mono_audio_file):
            os.unlink(mono_audio_file)
            logger.info(f"Cleaned up mono audio file {mono_audio_file}")
        
        return output_path
        
    except Exception as e:
        logger.exception(f"Error generating speech with CSM: {str(e)}")
        
        # Clean up reference audio files if they exist
        if 'reference_audio_file' in locals() and reference_audio_file and os.path.exists(reference_audio_file):
            os.unlink(reference_audio_file)
            logger.info(f"Cleaned up reference audio file {reference_audio_file}")
            
        if 'mono_audio_file' in locals() and mono_audio_file and mono_audio_file != reference_audio_file and os.path.exists(mono_audio_file):
            os.unlink(mono_audio_file)
            logger.info(f"Cleaned up mono audio file {mono_audio_file}")
            
        raise