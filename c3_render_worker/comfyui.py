# Module for ComfyUI portrait video generation
import os
import time
import logging
import json
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def download_file(url, output_path=None, preserve_extension=True):
    """Download a file from URL to the specified path
    
    Args:
        url: URL to download
        output_path: Path to save the file (if preserve_extension is True, the extension may be changed)
        preserve_extension: Whether to preserve the file extension from the URL
        
    Returns:
        The path to the downloaded file or False if download failed
    """
    logger.info(f"Downloading file from URL (truncated): {url[:100]}...")
    
    try:
        # Create a temporary path if none was provided
        temp_path = output_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_download")
        
        parsed_url = urlparse(url)
        original_filename = None
        ext = None
        
        # Try to extract the filename from query parameters (works with signed URLs)
        query_params = parsed_url.query.split('&')
        for param in query_params:
            if '=' in param:
                key, value = param.split('=', 1)
                # Look for common filename parameters
                if key.lower() in ['filename', 'file', 'name']:
                    original_filename = value
                    _, ext = os.path.splitext(original_filename)
                    logger.info(f"Found filename in query param: {original_filename}")
                    break
        
        # If no filename in query params, try the URL path
        if not original_filename:
            path = parsed_url.path
            potential_filename = os.path.basename(path)
            if '.' in potential_filename and len(potential_filename.split('.')[-1]) <= 5:
                original_filename = potential_filename
                _, ext = os.path.splitext(original_filename)
                logger.info(f"Extracted filename from URL path: {original_filename}")
        
        # If we have a valid extension from the URL and want to preserve it
        if preserve_extension and ext and len(ext) < 10 and ext != '.php':
            output_base, _ = os.path.splitext(output_path)
            output_path = output_base + ext
            logger.info(f"Using extension from URL: {ext}")
        
        # Start the download
        logger.info(f"Downloading to path: {output_path}")
        
        # Download the file
        response = requests.get(url, stream=True, timeout=60)
        
        if response.status_code == 200:
            # Save to original path first
            temp_output = output_path
            with open(temp_output, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # If no extension was found or we need to verify it
            if preserve_extension and (not ext or ext == '.php' or len(ext) >= 10):
                # Try content-type header
                content_type = response.headers.get('Content-Type', '')
                logger.info(f"Content-Type from response: {content_type}")
                
                # Map content types to file extensions
                content_type_map = {
                    'image/jpeg': '.jpg',
                    'image/jpg': '.jpg',
                    'image/png': '.png',
                    'image/webp': '.webp',
                    'audio/mpeg': '.mp3',
                    'audio/mp3': '.mp3', 
                    'audio/wav': '.wav',
                    'audio/x-wav': '.wav',
                    'audio/ogg': '.ogg',
                    'video/mp4': '.mp4'
                }
                
                new_ext = None
                if content_type in content_type_map:
                    new_ext = content_type_map[content_type]
                    
                # As a last resort, try to detect the file type using python-magic if available
                if not new_ext:
                    try:
                        import magic
                        mime = magic.Magic(mime=True)
                        detected_type = mime.from_file(temp_output)
                        logger.info(f"Detected MIME type: {detected_type}")
                        
                        if detected_type in content_type_map:
                            new_ext = content_type_map[detected_type]
                    except ImportError:
                        # python-magic not available, try using file command
                        try:
                            import subprocess
                            result = subprocess.run(['file', '--mime-type', temp_output], 
                                                  capture_output=True, text=True, check=True)
                            file_output = result.stdout.strip()
                            detected_type = file_output.split(': ')[-1].strip()
                            logger.info(f"Detected MIME type using file command: {detected_type}")
                            
                            if detected_type in content_type_map:
                                new_ext = content_type_map[detected_type]
                        except Exception as e:
                            logger.warning(f"Could not detect file type using file command: {e}")
                
                # If we found a better extension, update the path
                if new_ext:
                    output_base, _ = os.path.splitext(output_path)
                    new_path = output_base + new_ext
                    
                    # Only rename if paths differ
                    if new_path != temp_output:
                        logger.info(f"Renaming from {temp_output} to {new_path} based on detected type")
                        os.rename(temp_output, new_path)
                        output_path = new_path
            
            logger.info(f"Successfully downloaded file to {output_path}")
            return output_path
        else:
            logger.error(f"Failed to download file: {response.status_code}")
            return False
    except Exception as e:
        logger.exception(f"Error downloading file: {str(e)}")
        return False

def generate_portrait_video(image_url, audio_url, job_id, gpu_instance, api_key, output_dir):
    """Process a portrait video job using ComfyUI
    
    Args:
        image_url: URL of the portrait image
        audio_url: URL of the audio file
        job_id: Job ID
        gpu_instance: GPU instance information
        api_key: API key for authentication
        output_dir: Directory to save output files
        
    Returns:
        Path to the output video file or False if failed
    """
    logger.info(f"Generating portrait video for job {job_id}")
    
    try:
        # Download files from URLs with the correct naming convention
        image_output_base = os.path.join(output_dir, f"{job_id}_portrait")
        audio_output_base = os.path.join(output_dir, f"{job_id}_audio")
        
        # Download files with extension preservation
        downloaded_image = download_file(image_url, image_output_base)
        if not downloaded_image:
            error_msg = "Failed to download image file"
            logger.error(error_msg)
            return False
        
        downloaded_audio = download_file(audio_url, audio_output_base)
        if not downloaded_audio:
            error_msg = "Failed to download audio file"
            logger.error(error_msg)
            # Clean up downloaded image
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            return False
        
        # Get ComfyUI endpoint URLs from the GPU instance
        node_hostname = gpu_instance.get('node')
        # ComfyUI is hosted at ui-{node_hostname}
        comfyui_hostname = f"ui-{node_hostname}"
        comfyui_url = f"https://{comfyui_hostname}"
        
        # Ensure ComfyUI is ready by checking system_stats endpoint
        logger.info(f"Checking if ComfyUI is ready at {comfyui_url}")
        max_retries = 12
        retry_delay = 10
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"ComfyUI readiness check attempt {attempt}/{max_retries}")
                response = requests.get(
                    f"{comfyui_url}/system_stats",
                    headers={"X-C3-API-KEY": api_key},
                    timeout=30
                )
                
                if response.status_code == 200:
                    logger.info(f"ComfyUI is ready on attempt {attempt}")
                    break
                else:
                    logger.warning(f"ComfyUI not ready yet (status code: {response.status_code})")
            except Exception as e:
                logger.warning(f"ComfyUI readiness check failed on attempt {attempt}: {str(e)}")
            
            # If this is the last attempt, fail
            if attempt == max_retries:
                error_msg = "ComfyUI didn't become ready in time"
                logger.error(error_msg)
                
                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)
                
                return False
            
            # Wait before retrying
            logger.info(f"Waiting {retry_delay} seconds before retrying...")
            time.sleep(retry_delay)
        
        # Upload files to ComfyUI
        logger.info("Uploading files to ComfyUI")
        
        # Upload image - simply get the basename of the downloaded file
        image_filename = os.path.basename(downloaded_image)
        
        with open(downloaded_image, 'rb') as f:
            files = {'image': (image_filename, f)}
            response = requests.post(
                f"{comfyui_url}/upload/image",
                files=files,
                headers={"X-C3-API-KEY": api_key},
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to upload image: {response.status_code} - {response.text}"
                logger.error(error_msg)
                
                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)
                
                return False
            
            uploaded_image_name = response.json().get('name', image_filename)
            logger.info(f"Image uploaded successfully as {uploaded_image_name}")
        
        # Upload audio - also use the image upload endpoint
        audio_filename = os.path.basename(downloaded_audio)
        
        with open(downloaded_audio, 'rb') as f:
            files = {'image': (audio_filename, f)}
            response = requests.post(
                f"{comfyui_url}/upload/image",
                files=files,
                headers={"X-C3-API-KEY": api_key},
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to upload audio: {response.status_code} - {response.text}"
                logger.error(error_msg)
                
                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)
                
                return False
            
            uploaded_audio_name = response.json().get('name', audio_filename)
            logger.info(f"Audio uploaded successfully as {uploaded_audio_name}")
        
        # Configure workflow
        # First, determine audio duration to set in the SONIC_PreData node
        from mutagen import File as MutagenFile
        try:
            audio = MutagenFile(downloaded_audio)
            if audio:
                # Get the raw duration in seconds
                raw_duration = audio.info.length
                # Round up to the next second (math.ceil equivalent)
                rounded_duration = int(raw_duration) + (1 if raw_duration % 1 > 0 else 0)
                # Add 1 second safety margin
                audio_duration = rounded_duration + 1
                logger.info(f"Audio duration: {raw_duration:.2f}s → {audio_duration}s (rounded up + safety margin)")
            else:
                audio_duration = 5  # Default fallback
                logger.warning(f"Could not determine audio duration, using default: {audio_duration}s")
        except Exception as e:
            audio_duration = 5  # Default fallback on error
            logger.warning(f"Error getting audio duration: {e}, using default: {audio_duration}s")
        
        # Prepare the workflow
        workflow = {
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
                    "image": uploaded_image_name
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
                    "audio": uploaded_audio_name
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
                    "duration": audio_duration,
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
        
        # Queue the workflow
        logger.info("Queueing ComfyUI workflow")
        
        # Generate a client ID (uuid)
        import uuid
        client_id = str(uuid.uuid4())
        
        payload = {
            "prompt": workflow,
            "client_id": client_id
        }
        
        max_queue_retries = 3
        for attempt in range(1, max_queue_retries + 1):
            try:
                response = requests.post(
                    f"{comfyui_url}/prompt",
                    json=payload,
                    headers={"X-C3-API-KEY": api_key},
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prompt_id = result.get("prompt_id")
                    logger.info(f"Workflow queued with ID: {prompt_id}")
                    break
                else:
                    logger.warning(f"Failed to queue workflow: {response.status_code} - {response.text}")
                    
                    # If this is the last attempt, fail
                    if attempt == max_queue_retries:
                        error_msg = f"Failed to queue workflow after {max_queue_retries} attempts"
                        logger.error(error_msg)
                        
                        # Clean up downloaded files
                        if os.path.exists(downloaded_image):
                            os.unlink(downloaded_image)
                        if os.path.exists(downloaded_audio):
                            os.unlink(downloaded_audio)
                        
                        return False
                    
                    # Wait before retrying
                    time.sleep(10)
            except Exception as e:
                logger.error(f"Error queueing workflow: {str(e)}")
                
                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)
                
                return False
        
        # Wait for workflow completion
        logger.info(f"Waiting for workflow completion (prompt ID: {prompt_id})")
        
        max_wait_time = 600  # 10 minutes
        poll_interval = 15    # 15 seconds
        start_time = time.time()
        
        # Initially wait a bit for the workflow to start
        time.sleep(5)
        
        completed = False
        output_filename = None
        
        while time.time() - start_time < max_wait_time:
            elapsed_time = time.time() - start_time
            logger.info(f"Checking workflow status (elapsed: {elapsed_time:.1f}s)...")
            
            # Check workflow history
            try:
                response = requests.get(
                    f"{comfyui_url}/history/{prompt_id}",
                    headers={"X-C3-API-KEY": api_key},
                    timeout=30
                )
                
                if response.status_code == 200:
                    history = response.json()
                    
                    # Handle different response formats (direct or nested)
                    if prompt_id in history:
                        history_data = history[prompt_id]
                    else:
                        history_data = history
                    
                    # Check for errors
                    if "status" in history_data:
                        status = history_data["status"]
                        if status.get("status_str") == "error":
                            error_message = "Workflow execution failed"
                            
                            # Try to get detailed error information
                            for msg in status.get("messages", []):
                                if msg[0] == "execution_error" and len(msg) > 1:
                                    error_details = msg[1]
                                    node_id = error_details.get("node_id", "unknown")
                                    node_type = error_details.get("node_type", "unknown")
                                    exception = error_details.get("exception_message", "Unknown error")
                                    error_message = f"Error in node {node_id} ({node_type}): {exception}"
                            
                            logger.error(f"Workflow failed with error: {error_message}")
                            
                            # Clean up downloaded files
                            if os.path.exists(downloaded_image):
                                os.unlink(downloaded_image)
                            if os.path.exists(downloaded_audio):
                                os.unlink(downloaded_audio)
                            
                            return False
                    
                    # Check if execution is complete
                    if "outputs" in history_data:
                        # Check for VHS_VideoCombine output in node 13 
                        node_13_output = history_data["outputs"].get("13", {})
                        
                        # Check for videos in 'gifs' field (primary location for VHS_VideoCombine)
                        if "gifs" in node_13_output and node_13_output["gifs"]:
                            completed = True
                            output_filename = node_13_output["gifs"][0].get("filename")
                            logger.info(f"Found output video in 'gifs': {output_filename}")
                            break
                        
                        # Check for videos in 'videos' field (alternative location)
                        if "videos" in node_13_output and node_13_output["videos"]:
                            completed = True
                            output_filename = node_13_output["videos"][0].get("filename")
                            logger.info(f"Found output video in 'videos': {output_filename}")
                            break
                
                # Check if we're still in the queue
                try:
                    queue_response = requests.get(
                        f"{comfyui_url}/queue",
                        headers={"X-C3-API-KEY": api_key},
                        timeout=30
                    )
                    
                    if queue_response.status_code == 200:
                        queue_data = queue_response.json()
                        
                        # Debug log to understand the structure 
                        logger.debug(f"Queue data structure: {str(queue_data)[:200]}...")
                        
                        # Just look at the execution state directly without checking queue
                        if "executing" in queue_data and prompt_id in str(queue_data["executing"]):
                            logger.info("Workflow is currently executing...")
                        else:
                            logger.info("Workflow is not currently executing...")
                            
                            # Check if it's waiting in a queue
                            queued = False
                            if "queue_pending" in queue_data:
                                for item in queue_data["queue_pending"]:
                                    if isinstance(item, dict) and item.get("prompt_id") == prompt_id:
                                        queued = True
                                        break
                                    elif item == prompt_id:
                                        queued = True
                                        break
                            
                            if queued:
                                logger.info("Workflow is queued, waiting to run...")
                            else:
                                logger.info("Workflow is not in queue...")
                                
                                # If we're not in any queue and don't have output files, just keep waiting
                                # The completion will be determined by output files appearing
                except Exception as e:
                    logger.warning(f"Error checking queue status (will continue waiting): {str(e)}")
            
            except Exception as e:
                logger.warning(f"Error checking workflow status: {str(e)}")
            
            # If we've already completed, break the loop
            if completed:
                break
            
            # Wait before checking again
            logger.info(f"Still in progress... Checking again in {poll_interval} seconds")
            time.sleep(poll_interval)
        
        # If we timed out without completing
        if not completed:
            error_msg = f"Workflow processing timed out after {max_wait_time} seconds"
            logger.error(error_msg)
            
            # Clean up downloaded files
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            if os.path.exists(downloaded_audio):
                os.unlink(downloaded_audio)
            
            return False
        
        # If we completed but didn't get an output filename
        if not output_filename:
            error_msg = "Workflow completed but no output file was found"
            logger.error(error_msg)
            
            # Clean up downloaded files
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            if os.path.exists(downloaded_audio):
                os.unlink(downloaded_audio)
            
            return False
        
        # Download the output file
        logger.info(f"Downloading output video: {output_filename}")
        
        # Construct URL parameters
        import urllib.parse
        params = {"filename": output_filename}
        download_url = f"{comfyui_url}/view?{urllib.parse.urlencode(params)}"
        
        output_video_path = os.path.join(output_dir, f"{job_id}.mp4")
        
        try:
            response = requests.get(
                download_url,
                headers={"X-C3-API-KEY": api_key},
                stream=True,
                timeout=120
            )
            
            if response.status_code == 200:
                with open(output_video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Video downloaded successfully to {output_video_path}")
                
                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                    logger.info(f"Cleaned up input image: {downloaded_image}")
                
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)
                    logger.info(f"Cleaned up input audio: {downloaded_audio}")
                
                return output_video_path
            else:
                error_msg = f"Failed to download output video: {response.status_code}"
                logger.error(error_msg)
                
                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)
                
                return False
        
        except Exception as e:
            error_msg = f"Error downloading output video: {str(e)}"
            logger.error(error_msg)
            
            # Clean up downloaded files
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            if os.path.exists(downloaded_audio):
                os.unlink(downloaded_audio)
            
            return False
    
    except Exception as e:
        error_msg = f"Error in portrait video generation: {str(e)}"
        logger.exception(error_msg)
        
        # Clean up any downloaded files
        if 'downloaded_image' in locals() and downloaded_image and os.path.exists(downloaded_image):
            os.unlink(downloaded_image)
        
        if 'downloaded_audio' in locals() and downloaded_audio and os.path.exists(downloaded_audio):
            os.unlink(downloaded_audio)
        
        return False