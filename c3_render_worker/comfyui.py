# Module for ComfyUI portrait video generation
import os
import time
import logging
import json
import requests
import urllib.parse
from urllib.parse import urlparse
from PIL import Image  # Import Pillow for image processing
import subprocess

# Import from constants file
from constants import RENDER_POLLING_INTERVAL

# Import job control from worker
# from c3_render_worker import job_cancel_flag

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

def extract_audio_from_video(video_path, output_dir):
    """Extract audio from video file using FFmpeg and save as MP3

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the extracted audio

    Returns:
        Path to the extracted audio file or None if extraction failed
    """
    logger.info(f"Extracting audio from video file: {video_path}")

    try:
        # Generate output filename
        filename = os.path.basename(video_path)
        name, _ = os.path.splitext(filename)
        audio_path = os.path.join(output_dir, f"{name}_audio.mp3")

        # Use FFmpeg to extract audio - force MP3 output
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',              # Disable video
            '-acodec', 'libmp3lame',  # Force MP3 codec
            '-q:a', '2',        # High quality (0-highest, 9-lowest)
            '-ac', '2',         # Stereo output
            '-ar', '44100',     # 44.1kHz sample rate
            '-y',               # Overwrite output file
            audio_path
        ]

        # Run FFmpeg command
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully extracted audio to: {audio_path}")
            return audio_path
        else:
            logger.error(f"Failed to extract audio: {result.stderr}")
            return None

    except Exception as e:
        logger.exception(f"Error extracting audio from video: {str(e)}")
        return None

def generate_portrait_video(image_url, audio_url, job_id, gpu_instance, api_key, output_dir, cancel_callback=None):
    """Process a portrait video job using ComfyUI

    Args:
        image_url: URL of the portrait image
        audio_url: URL of the audio file
        job_id: Job ID
        gpu_instance: GPU instance information
        api_key: API key for authentication
        output_dir: Directory to save output files
        cancel_callback: Optional function that returns True if job should be cancelled

    Returns:
        Path to the output video file or tuple (False, error_message) if failed
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
            return False, error_msg

        # Get image dimensions and calculate resize values for ComfyUI workflow
        processed_image = downloaded_image  # By default, use the original image
        try:
            with Image.open(downloaded_image) as img:
                orig_width, orig_height = img.size
                logger.info(f"Original image dimensions: {orig_width}x{orig_height}")

                # Check if image exceeds ComfyUI's dimension limits (2048 px)
                if orig_width > 2048 or orig_height > 2048:
                    # Calculate scale to fit within 2048x2048
                    scale = min(2048 / orig_width, 2048 / orig_height)
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)

                    logger.info(f"Image exceeds ComfyUI limits. Preprocessing to {new_width}x{new_height}")

                    # Create preprocessed image filename
                    img_basename = os.path.basename(downloaded_image)
                    name, ext = os.path.splitext(img_basename)
                    processed_image = os.path.join(output_dir, f"{name}_processed{ext}")

                    # Resize and save
                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    resized_img.save(processed_image)
                    logger.info(f"Saved preprocessed image to {processed_image}")

                    # Update dimensions for future reference
                    orig_width, orig_height = new_width, new_height

                # Calculate scaling factors for workflow node resizing
                width_scale = 1280 / orig_width
                height_scale = 720 / orig_height

                # Use the smaller scaling factor to fit within constraints
                scale = min(width_scale, height_scale)

                # Calculate new dimensions for workflow node
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)

                logger.info(f"ComfyUI workflow will resize to {new_width}x{new_height} (maintaining aspect ratio)")
        except Exception as e:
            logger.warning(f"Error processing image dimensions: {e}. Using default resize values.")
            processed_image = downloaded_image
            new_width = 500
            new_height = 500

        downloaded_audio = download_file(audio_url, audio_output_base)
        if not downloaded_audio:
            error_msg = "Failed to download audio file"
            logger.error(error_msg)
            # Clean up downloaded and processed images
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            if processed_image != downloaded_image and os.path.exists(processed_image):
                os.unlink(processed_image)
            return False, error_msg

        # Check if the downloaded file is a video and extract audio if needed
        try:
            import magic
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(downloaded_audio)

            # Check for any video mime type
            is_video = file_type.startswith('video/')

            # Also check for problematic audio containers that might need conversion
            problematic_audio = file_type in [
                'audio/x-m4a',
                'audio/aac',
                'audio/webm',
                'audio/ogg',
                'application/ogg'
            ]

            if is_video or problematic_audio:
                if is_video:
                    logger.info(f"Detected video file: {file_type}. Extracting audio to MP3...")
                else:
                    logger.info(f"Detected audio format that needs conversion: {file_type}. Converting to MP3...")

                extracted_audio = extract_audio_from_video(downloaded_audio, output_dir)

                if extracted_audio:
                    # Keep a reference to the original file for cleanup
                    original_audio = downloaded_audio
                    # Use the extracted audio file instead
                    downloaded_audio = extracted_audio
                    logger.info(f"Using converted audio: {downloaded_audio}")
                else:
                    error_msg = "Failed to extract/convert audio file"
                    logger.error(error_msg)

                    # Clean up downloaded files
                    if os.path.exists(downloaded_image):
                        os.unlink(downloaded_image)
                    if processed_image != downloaded_image and os.path.exists(processed_image):
                        os.unlink(processed_image)
                    if os.path.exists(downloaded_audio):
                        os.unlink(downloaded_audio)

                    return False, error_msg
            else:
                logger.info(f"Downloaded file is usable audio format: {file_type}")
                original_audio = None
        except ImportError:
            # Fall back to checking file extension if python-magic is not available
            _, ext = os.path.splitext(downloaded_audio.lower())
            video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp']
            audio_extensions_to_convert = ['.m4a', '.aac', '.ogg', '.opus', '.wma', '.wav']

            if ext in video_extensions:
                logger.info(f"File has video extension: {ext}. Extracting audio to MP3...")
                extracted_audio = extract_audio_from_video(downloaded_audio, output_dir)
                need_conversion = True
            elif ext in audio_extensions_to_convert:
                logger.info(f"File has audio extension that needs conversion: {ext}. Converting to MP3...")
                extracted_audio = extract_audio_from_video(downloaded_audio, output_dir)
                need_conversion = True
            else:
                logger.info(f"File has supported audio extension: {ext}")
                need_conversion = False
                original_audio = None

            if need_conversion:
                if extracted_audio:
                    # Keep a reference to the original file for cleanup
                    original_audio = downloaded_audio
                    # Use the extracted audio file instead
                    downloaded_audio = extracted_audio
                    logger.info(f"Using converted audio: {downloaded_audio}")
                else:
                    error_msg = "Failed to extract/convert audio file"
                    logger.error(error_msg)

                    # Clean up downloaded files
                    if os.path.exists(downloaded_image):
                        os.unlink(downloaded_image)
                    if processed_image != downloaded_image and os.path.exists(processed_image):
                        os.unlink(processed_image)
                    if os.path.exists(downloaded_audio):
                        os.unlink(downloaded_audio)

                    return False, error_msg
        except Exception as e:
            logger.warning(f"Error checking file type, proceeding with file as-is: {str(e)}")
            original_audio = None

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
                if processed_image != downloaded_image and os.path.exists(processed_image):
                    os.unlink(processed_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)

                return False, error_msg

            # Wait before retrying
            logger.info(f"Waiting {retry_delay} seconds before retrying...")
            time.sleep(retry_delay)

        # Upload files to ComfyUI
        logger.info("Uploading files to ComfyUI")

        # Upload image - simply get the basename of the processed file
        image_filename = os.path.basename(processed_image)

        with open(processed_image, 'rb') as f:
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
                if processed_image != downloaded_image and os.path.exists(processed_image):
                    os.unlink(processed_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)

                return False, error_msg

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
                if processed_image != downloaded_image and os.path.exists(processed_image):
                    os.unlink(processed_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)

                return False, error_msg

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
                # Add 3 seconds safety margin to prevent audio cutting
                audio_duration = rounded_duration + 3
                logger.info(f"Audio duration: {raw_duration:.2f}s → {audio_duration}s (rounded up + 3s safety margin)")
            else:
                audio_duration = 8  # Default fallback (increased from 5)
                logger.warning(f"Could not determine audio duration, using default: {audio_duration}s")
        except Exception as e:
            audio_duration = 8  # Default fallback on error (increased from 5)
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
                        "58",
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
                    "audio": uploaded_audio_name,
                    "audioUI": ""
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
                    "fps": 24,
                    "model": [
                        "34",
                        0
                    ],
                    "data_dict": [
                        "33",
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
                        "58",
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
                    "dtype": "bf16",
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
            "58": {
                "inputs": {
                    "width": new_width,
                    "height": new_height,
                    "interpolation": "nearest",
                    "method": "stretch",
                    "condition": "always",
                    "multiple_of": 0,
                    "image": [
                        "18",
                        0
                    ]
                },
                "class_type": "ImageResize+",
                "_meta": {
                    "title": "🔧 Image Resize"
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
                        error_msg = f"Failed to queue workflow after {max_queue_retries} attempts: {response.status_code} - {response.text}"
                        logger.error(error_msg)

                        # Clean up downloaded files
                        if os.path.exists(downloaded_image):
                            os.unlink(downloaded_image)
                        if processed_image != downloaded_image and os.path.exists(processed_image):
                            os.unlink(processed_image)
                        if os.path.exists(downloaded_audio):
                            os.unlink(downloaded_audio)

                        return False, error_msg

                    # Wait before retrying
                    time.sleep(10)
            except Exception as e:
                error_msg = f"Error queueing workflow: {str(e)}"
                logger.error(error_msg)

                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                if processed_image != downloaded_image and os.path.exists(processed_image):
                    os.unlink(processed_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)

                return False, error_msg

        # Wait for workflow completion
        logger.info(f"Waiting for workflow completion (prompt ID: {prompt_id})")

        poll_interval = RENDER_POLLING_INTERVAL  # Use environment variable for polling interval
        start_time = time.time()

        # Initially wait a bit for the workflow to start
        time.sleep(5)

        completed = False
        output_filename = None

        while True:
            # Check if job has been cancelled by the monitor thread
            if cancel_callback and cancel_callback():
                logger.warning("Job cancellation detected - terminating workflow")
                # Try to cancel the job if possible
                try:
                    cancel_url = f"{comfyui_url}/cancel"
                    cancel_data = {"prompt_id": prompt_id}
                    requests.post(cancel_url, headers={"X-C3-API-KEY": api_key}, json=cancel_data, timeout=10)
                    logger.info(f"Sent cancellation request for prompt ID: {prompt_id}")
                except Exception as e:
                    logger.warning(f"Error sending cancellation request: {str(e)}")

                # Get the specific error message from Redis if available
                try:
                    from redis import Redis
                    redis_host = os.getenv("REDIS_HOST", "localhost")
                    redis_port = int(os.getenv("REDIS_PORT", "6379"))
                    redis_client = Redis(host=redis_host, port=redis_port, decode_responses=True)

                    # Extract job_id from the job_id param or from output path
                    job_id_match = job_id
                    job_data = redis_client.hgetall(f"job:{job_id_match}")

                    # Check for existing result_url
                    result_url = job_data.get("result_url")

                    if job_data and "error" in job_data:
                        error_msg = job_data["error"]
                        logger.info(f"Using specific error reason from Redis: {error_msg}")
                    else:
                        error_msg = "Job was cancelled by system"
                except Exception as e:
                    logger.warning(f"Error retrieving specific error message: {e}")
                    error_msg = "Job was cancelled by system"
                    result_url = None

                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                if processed_image != downloaded_image and os.path.exists(processed_image):
                    os.unlink(processed_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)

                # Let the caller know if we have a partial result
                if 'result_url' in locals() and result_url:
                    logger.info(f"Including partial result_url in cancel response: {result_url}")
                    # Store the tuple with the error message and the result URL
                    error_data = {
                        "error": error_msg,
                        "result_url": result_url
                    }
                    return False, error_data
                else:
                    return False, error_msg

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
                            error_message = "ComfyUI workflow execution failed"

                            # Try to get detailed error information
                            for msg in status.get("messages", []):
                                if msg[0] == "execution_error" and len(msg) > 1:
                                    error_details = msg[1]
                                    node_id = error_details.get("node_id", "unknown")
                                    node_type = error_details.get("node_type", "unknown")
                                    exception = error_details.get("exception_message", "Unknown error")
                                    error_message = f"ComfyUI error in node {node_id} ({node_type}): {exception}"

                            logger.error(f"Workflow failed with error: {error_message}")

                            # Clean up downloaded files
                            if os.path.exists(downloaded_image):
                                os.unlink(downloaded_image)
                            if processed_image != downloaded_image and os.path.exists(processed_image):
                                os.unlink(processed_image)
                            if os.path.exists(downloaded_audio):
                                os.unlink(downloaded_audio)

                            return False, error_message

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

        # If we completed but didn't get an output filename
        if not output_filename:
            error_msg = "ComfyUI workflow completed but no output file was found"
            logger.error(error_msg)

            # Clean up downloaded files
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            if processed_image != downloaded_image and os.path.exists(processed_image):
                os.unlink(processed_image)
            if os.path.exists(downloaded_audio):
                os.unlink(downloaded_audio)

            return False, error_msg

        # Download the output file
        logger.info(f"Downloading output video: {output_filename}")

        # Construct URL parameters
        params = {"filename": output_filename}
        from urllib.parse import urlencode
        download_url = f"{comfyui_url}/view?{urlencode(params)}"

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

                if processed_image != downloaded_image and os.path.exists(processed_image):
                    os.unlink(processed_image)
                    logger.info(f"Cleaned up processed image: {processed_image}")

                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)
                    logger.info(f"Cleaned up input audio: {downloaded_audio}")

                # Clean up original video file if we extracted audio
                if 'original_audio' in locals() and original_audio and os.path.exists(original_audio):
                    os.unlink(original_audio)
                    logger.info(f"Cleaned up original video file: {original_audio}")

                return output_video_path
            else:
                error_msg = f"Failed to download output video: {response.status_code} - {response.text}"
                logger.error(error_msg)

                # Clean up downloaded files
                if os.path.exists(downloaded_image):
                    os.unlink(downloaded_image)
                if processed_image != downloaded_image and os.path.exists(processed_image):
                    os.unlink(processed_image)
                if os.path.exists(downloaded_audio):
                    os.unlink(downloaded_audio)
                # Clean up original video file if we extracted audio
                if 'original_audio' in locals() and original_audio and os.path.exists(original_audio):
                    os.unlink(original_audio)

                return False, error_msg

        except Exception as e:
            error_msg = f"Error downloading output video: {str(e)}"
            logger.error(error_msg)

            # Clean up downloaded files
            if os.path.exists(downloaded_image):
                os.unlink(downloaded_image)
            if processed_image != downloaded_image and os.path.exists(processed_image):
                os.unlink(processed_image)
            if os.path.exists(downloaded_audio):
                os.unlink(downloaded_audio)
            # Clean up original video file if we extracted audio
            if 'original_audio' in locals() and original_audio and os.path.exists(original_audio):
                os.unlink(original_audio)

            return False, error_msg

    except Exception as e:
        error_msg = f"Error in portrait video generation: {str(e)}"
        logger.exception(error_msg)

        # Clean up any downloaded files
        if 'downloaded_image' in locals() and downloaded_image and os.path.exists(downloaded_image):
            os.unlink(downloaded_image)

        if 'processed_image' in locals() and processed_image != downloaded_image and processed_image and os.path.exists(processed_image):
            os.unlink(processed_image)

        if 'downloaded_audio' in locals() and downloaded_audio and os.path.exists(downloaded_audio):
            os.unlink(downloaded_audio)

        return False, error_msg