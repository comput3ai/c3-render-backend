# 🚀 C3 Render API & Worker

A simple distributed system for managing AI rendering tasks including text-to-speech, speech-to-text, portrait video generation, and image analysis. The system consists of a Flask API server for job submission and a worker component that processes jobs using GPU instances from Comput3.AI.

## ✨ Components

1. **C3 Render API**: Flask application that handles job submission, status tracking, and result retrieval
2. **C3 Render Worker**: Python script that processes jobs, manages GPU instances, and sends webhook notifications

## 🔧 Setup

### Local Development

1. Set up environment variables:
   - For the API: Copy `c3_render_api/.env.sample` to `c3_render_api/.env`. (Note: Authentication is typically bypassed for local development)
   - For the Worker: Copy `c3_render_worker/.env.sample` to `c3_render_worker/.env` and add your Comput3.AI API key and Minio configuration

**Note on MinIO for Local Development (Same-Host Docker):**
If you are running the MinIO server in a separate Docker container (or compose setup) on the same host machine as the worker, and they share a Docker network, you typically need to configure the worker's `.env` file as follows:
- `MINIO_ENDPOINT=minio:9000` (using the service name and default MinIO port)
- `MINIO_SECURE=false` (as the connection is internal over HTTP)

```bash
# For API
cp c3_render_api/.env.sample c3_render_api/.env

# For Worker
cp c3_render_worker/.env.sample c3_render_worker/.env
# Edit c3_render_worker/.env to set your C3_API_KEY and Minio configuration
```

2. Start the services using Docker Compose:
```bash
docker-compose up -d
```

This will start:
- API service on port 5000 (accessible via `http://localhost:5000`)
- Redis on port 6379
- Worker service (not exposed to host)

3. Check logs to ensure everything is running:
```bash
# For all services
docker-compose logs -f

# For specific service
docker-compose logs -f api
docker-compose logs -f worker
```

4. Access the services:
- API: `http://localhost:5000` (No authentication usually required for local setup)

### Manual Setup

(Follow similar steps as local Docker setup, running services directly)

1. Install dependencies for API and Worker.
2. Set up environment variables (API `.env` and Worker `.env`).
3. Start Redis (e.g., `docker-compose up -d redis`).
4. Run the API server (`cd c3_render_api && python c3_render_api.py`).
5. Run the worker (`cd c3_render_worker && python c3_render_worker.py`).

## 🌐 Production Deployment

- **URL:** The API is expected to be served over **HTTPS** (e.g., `https://your-render-api.example.com`).
- **Authentication:** Requests to the production API **must** include an API key via one of the following headers:
  - `X-C3-RENDER-KEY: your_api_key_here`
  - `Authorization: Bearer your_api_key_here`
  (The key value should be configured on the API server, typically via the `C3_RENDER_API_KEY` environment variable in its `.env` file).
- **WSGI Server:** Use a production-grade WSGI server like `gunicorn` (as configured in the API Dockerfile).

## 🏗️ Architecture

- **Smart job queue processing**: Workers prioritize reusing existing GPU instances to minimize costs and startup times
- **Randomized startup delays**: Workers use random delays (15-30 seconds by default) before launching GPUs to prevent simultaneous launches
- **GPU Instance Management**: Workers maintain GPU instances for configurable idle time (default 5 minutes) before shutting down
- **Robust GPU Monitoring**: A dedicated monitoring thread performs health checks, with multiple retry attempts and verification against the Comput3.AI API
- **Webhook Notifications**: Workers send webhook notifications with 5 retry attempts at 5-second intervals
- **Job Status Tracking**: All job status information is stored in Redis
- **Storage**: Uses Minio S3-compatible storage for storing job results (configured via environment variables). Access to results is provided via **presigned URLs** valid for 7 days.
- **Modular Design**: Worker functionality is separated into task-specific modules (csm.py, comfyui.py)
- **Automatic Image Processing**: Workers automatically process images to maintain aspect ratios and handle oversized images

## ⏱️ Job Timing and Execution Constraints

All jobs support the following timing parameters to control execution constraints:

- **max_time**: Maximum allowed runtime for a job in seconds (60-7200, default: 1200/20 minutes)
  - If a job exceeds this runtime, it will be automatically terminated and marked as failed
  - This measures actual processing time from when the worker starts the job

- **complete_in**: Number of seconds from job creation when the job must complete (60-86400, default: 3600/1 hour)
  - This is converted to an absolute `complete_by` timestamp in the worker
  - If current time exceeds this timestamp, the job will be terminated and marked as failed
  - Jobs already in the queue that have exceeded their `complete_by` time will be skipped and marked as expired

These parameters provide flexible options for controlling job execution:
- Use `max_time` to limit resource usage for individual jobs
- Use `complete_in` to set hard deadlines for time-sensitive operations

Example usage:
```json
{
  "text": "This is an example with timing constraints.",
  "max_time": 300,        // Must complete within 5 minutes of processing time
  "complete_in": 1800     // Must complete within 30 minutes of being submitted
}
```

## 🔩 Worker Configuration

The worker component can be configured with the following environment variables:

- **C3_API_KEY**: Your Comput3.AI API key (required)
- **REDIS_HOST**: Redis server hostname (default: localhost)
- **REDIS_PORT**: Redis server port (default: 6379)
- **OUTPUT_DIR**: Directory for temporary output files (default: /app/output)
- **GPU_IDLE_TIMEOUT**: Time in seconds to keep a GPU alive after its last job (default: 300)
- **PRE_LAUNCH_TIMEOUT**: Minimum wait time in seconds before launching a new GPU (default: 15)
- **PRE_LAUNCH_TIMEOUT_MAX**: Maximum wait time in seconds before launching a new GPU (default: 30)
- **MAX_RENDER_TIME**: Maximum time in seconds allowed for a render job before timing out (default: 1800)
- **RENDER_POLLING_INTERVAL**: Interval in seconds to check job status during rendering (default: 5)

## 🔌 API Endpoints

**Note:** All examples below assume a **production deployment**. For **local development**, use `http://localhost:5000` and omit the `X-C3-RENDER-KEY` header.

### POST /csm
Generate speech using the CSM (Collaborative Speech Model) text-to-speech system with configurable voice options, including voice cloning.

```bash
# Basic text-to-speech request with text parameter (Production)
curl -X POST https://your-render-api.example.com/csm \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "text": "Hello, this is a test of the C3 Render API text to speech system using CSM.",
    "notify_url": "https://your-webhook-endpoint.com/callback",
    "max_time": 600,
    "complete_in": 1800
  }'

# Text-to-speech with monologue parameter as array of sentences (Production)
curl -X POST https://your-render-api.example.com/csm \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "monologue": ["This is the first sentence.", "This is the second sentence."],
    "voice": "conversational_a",
    "temperature": 0.7,
    "topk": 40,
    "max_audio_length": 8000,
    "pause_duration": 200,
    "notify_url": "https://your-webhook-endpoint.com/callback",
    "max_time": 900,
    "complete_in": 3600
  }'
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **text** | string | - | The text to convert to speech (use either text OR monologue) |
| **monologue** | array of strings | - | Array of sentences to convert to speech (use either text OR monologue) |
| **notify_url** | string | Optional | Webhook URL to receive job status updates |
| **voice** | string | "random" | Voice type to use. Options: "random", "conversational_a", "conversational_b", or "clone" for voice cloning |
| **reference_audio_url** | string | None | URL to reference audio file for voice cloning (required when voice="clone") |
| **reference_text** | string | None | Text content of the reference audio file for voice cloning (required when voice="clone") |
| **temperature** | float | 0.9 | Controls randomness of output (0.0-2.0). Higher values produce more random outputs |
| **topk** | integer | 50 | Number of highest probability tokens to consider at each generation step |
| **max_audio_length** | integer | 10000 | Maximum length of generated audio in milliseconds |
| **pause_duration** | integer | 150 | Duration of pauses between sentences in milliseconds |
| **max_time** | integer | 1200 | Maximum allowed processing time in seconds (60-7200) |
| **complete_in** | integer | 3600 | Job must complete within this many seconds of creation (60-86400) |

> Note: You must provide EITHER the `text` parameter (as a string) OR the `monologue` parameter (as an array of strings), but not both. The `monologue` format is preferred for better sentence pauses and phrasing.

#### Example with Voice Cloning

```bash
curl -X POST https://your-render-api.example.com/csm \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "monologue": ["This is a voice cloning test.", "The system will try to mimic the voice in the reference audio."],
    "voice": "clone",
    "reference_audio_url": "https://example.com/reference-voice.mp3",
    "reference_text": "This is a sample of my voice for cloning purposes.",
    "notify_url": "https://your-webhook-endpoint.com/callback",
    "max_time": 1800,
    "complete_in": 3600
  }'
```

#### Example Response

```json
{
  "id": "c1e8f9a0-1b2c-4d3e-9f4g-5h6i7j8k9l0m",
  "status": "queued"
}
```

The worker will process the job and generate speech using the CSM service on Comput3 GPU instances. Upon completion, a webhook will be sent to the `notify_url` with the result (a presigned URL valid for 7 days).

Webhook payload on success:
```json
{
  "id": "c1e8f9a0-1b2c-4d3e-9f4g-5h6i7j8k9l0m",
  "status": "success",
  "result_url": "https://minio-endpoint.com/results/audio.mp3?X-Amz-Algorithm=...&X-Amz-Expires=604800&..." // Example Presigned URL
}
```

Webhook payload on failure:
```json
{
  "id": "c1e8f9a0-1b2c-4d3e-9f4g-5h6i7j8k9l0m",
  "status": "failed",
  "error": "Error message describing the failure"
}
```

### POST /whisper

Transcribe audio to text.

```
POST /whisper
```

#### Request Body
```json
{
  "audio_url": "https://storage.example.com/recording.mp3",
  "model": "medium",  // Optional, defaults to "medium"
  "task": "transcribe",  // Optional, defaults to "transcribe", can also be "translate"
  "language": "",  // Optional, defaults to auto-detection
  "notify_url": "https://myapp.example.com/webhooks/job-complete",  // Optional
  "max_time": 600,  // Optional, default is 1200 seconds (20 minutes)
  "complete_in": 1800  // Optional, default is 3600 seconds (1 hour)
}
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **audio_url** | string | Required | URL to the audio file to transcribe |
| **model** | string | "medium" | Whisper model to use. Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3" |
| **task** | string | "transcribe" | Task type, can be "transcribe" or "translate" |
| **language** | string | "" | Language code (empty for auto-detection) |
| **notify_url** | string | Optional | Webhook URL to receive job status updates |
| **max_time** | integer | 1200 | Maximum allowed processing time in seconds (60-7200) |
| **complete_in** | integer | 3600 | Job must complete within this many seconds of creation (60-86400) |

#### cURL Example
```bash
# Basic whisper request (Production)
curl -X POST https://your-render-api.example.com/whisper \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "audio_url": "https://example.com/path/to/audio.mp3",
    "notify_url": "https://your-webhook-endpoint.com/callback",
    "max_time": 600,
    "complete_in": 1800
  }'

# Whisper request with model specification (Production)
curl -X POST https://your-render-api.example.com/whisper \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "audio_url": "https://example.com/path/to/audio.mp3",
    "model": "large-v3",
    "task": "translate",
    "notify_url": "https://your-webhook-endpoint.com/callback",
    "max_time": 1800,
    "complete_in": 3600
  }'
```

#### Example Response
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "status": "queued"
}
```

Upon completion, a webhook will be sent to the `notify_url` with the transcription result:

Webhook payload on success:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "status": "success",
  "text": "This is the transcribed text from the audio file."
}
```

Webhook payload on failure:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "status": "failed",
  "error": "Error message describing the failure"
}
```

### Portrait Video Generation

Create a speaking portrait video from an image and audio.

```
POST /portrait
```

#### Request Body
```json
{
  "image_url": "https://storage.example.com/portrait.jpg",
  "audio_url": "https://storage.example.com/speech.mp3",
  "notify_url": "https://myapp.example.com/webhooks/job-complete",  // Optional
  "max_time": 1800,  // Optional, default is 1200 seconds (20 minutes)
  "complete_in": 3600  // Optional, default is 3600 seconds (1 hour)
}
```

#### Example Curl Command
```bash
curl -X POST https://your-render-api.example.com/portrait \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "image_url": "https://example.com/path/to/portrait.jpg",
    "audio_url": "https://example.com/path/to/audio.mp3",
    "notify_url": "https://your-webhook-endpoint.com/callback",
    "max_time": 1800,
    "complete_in": 3600
  }'
```

#### Example Response
```json
{
  "id": "c1e8f9a0-1b2c-4d3e-9f4g-5h6i7j8k9l0m",
  "status": "queued"
}
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **image_url** | string | Required | URL to the portrait image (face) to animate |
| **audio_url** | string | Required | URL to the audio file that will be used to animate the portrait |
| **notify_url** | string | Optional | Webhook URL to receive job status updates |
| **max_time** | integer | 1200 | Maximum allowed processing time in seconds (60-7200) |
| **complete_in** | integer | 3600 | Job must complete within this many seconds of creation (60-86400) |

#### cURL Example (Production)
```bash
curl -X POST https://your-render-api.example.com/portrait \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "image_url": "https://storage.example.com/portrait.jpg",
    "audio_url": "https://storage.example.com/speech.mp3",
    "notify_url": "https://myapp.example.com/webhooks/job-complete",
    "max_time": 1800,
    "complete_in": 7200
  }'
```

Response:
```json
{
  "id": "c1e8f9a0-1b2c-4d3e-9f4g-5h6i7j8k9l0m",
  "status": "queued"
}
```

#### Implementation Details

Portrait video generation uses ComfyUI with the SONIC model to create animated talking face videos. The process:

1. The worker downloads the provided image and audio files to the output directory
2. ComfyUI's SONIC model processes the files to generate a talking face video that matches the audio
3. The resulting MP4 video is saved to the output directory
4. The video file is uploaded to MinIO storage.
5. A webhook notification is sent upon completion with a **presigned URL** (valid for 7 days) to the generated video.

The implementation uses a specialized workflow with the following components:
- ImageOnlyCheckpointLoader: Loads the SVD_XT model for image animation
- SONIC_PreData: Prepares the input data for the SONIC model
- SONICSampler: Generates the animated frames based on audio input
- VHS_VideoCombine: Combines the frames with the audio to create the final video

### Image Analysis

Analyze an image using a vision model.

```
POST /analyze
```

#### Request Body
```json
{
  "image_url": "https://storage.example.com/image.jpg",
  "notify_url": "https://myapp.example.com/webhooks/job-complete",  // Optional
  "max_time": 300,  // Optional, default is 1200 seconds (20 minutes)
  "complete_in": 1800  // Optional, default is 3600 seconds (1 hour)
}
```

#### cURL Example (Production)
```bash
curl -X POST https://your-render-api.example.com/analyze \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "image_url": "https://example.com/path/to/image.jpg",
    "notify_url": "https://your-webhook-endpoint.com/callback",
    "max_time": 300,
    "complete_in": 1800
  }'
```

#### Example Response
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "status": "queued"
}
```

### Job Status

Check the status of a submitted job.

```
GET /status/{id}
```

#### cURL Example (Production)
```bash
# Replace JOB_ID with the actual job ID returned when submitting a job
curl -X GET https://your-render-api.example.com/status/JOB_ID \
  -H "X-C3-RENDER-KEY: your_api_key_here"

# Example with an actual UUID
curl -X GET https://your-render-api.example.com/status/550e8400-e29b-41d4-a716-446655440000 \
  -H "X-C3-RENDER-KEY: your_api_key_here"
```

#### Example Response
```json
{
  "status": "queued"  // Possible values: "queued", "running", "success", "failed"
}
```

### Update Job Status

Manually update the status of a job, particularly for marking jobs as timed out or cancelled by the frontend.

```
POST /job/{id}
```

#### Request Body
```json
{
  "status": "timed_out",  // Required: "timed_out", "cancelled", or "failed"
  "error": "Job timed out after 30 minutes",  // Optional: custom error message
  "remove_data": false  // Optional: whether to remove the job data from Redis
}
```

#### cURL Example (Production)
```bash
# Replace JOB_ID with the actual job ID to update
curl -X POST https://your-render-api.example.com/job/JOB_ID \
  -H "Content-Type: application/json" \
  -H "X-C3-RENDER-KEY: your_api_key_here" \
  -d '{
    "status": "timed_out",
    "error": "Job timed out after waiting too long"
  }'
```

#### Example Response
```json
{
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 marked as timed_out"
}
```

### Job Result

Retrieve the result of a completed job.

```
GET /result/{id}
```

#### cURL Example (Production)
```bash
# Replace JOB_ID with the actual job ID returned when submitting a job
curl -X GET https://your-render-api.example.com/result/JOB_ID \
  -H "X-C3-RENDER-KEY: your_api_key_here"

# Example with an actual UUID
curl -X GET https://your-render-api.example.com/result/550e8400-e29b-41d4-a716-446655440000 \
  -H "X-C3-RENDER-KEY: your_api_key_here"
```

#### Example Response for Media Jobs (CSM, Portrait)
```json
{
  "result_url": "https://minio-endpoint.com/results/JOB_ID.mp4?X-Amz-Algorithm=...&X-Amz-Expires=604800&..." // Example Presigned URL
}
```

#### Example Response for Text Jobs (Whisper, Analyze)
```json
{
  "text": "The transcribed text or analysis result"
}
```

## 📫 Webhook Notifications

When a job completes and a `notify_url` was provided, the worker will send a POST request to the specified URL with the following payload:

For successful jobs:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "result_url": "https://minio-endpoint.com/results/output.mp4?X-Amz-Algorithm=...&X-Amz-Expires=604800&..."  // for media jobs (csm, portrait), PRESIGNED URL
}
```

or

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "text": "Result text for text-based outputs"  // for text jobs (whisper, analyze)
}
```

For failed jobs:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error": "Error message describing what went wrong"
}
```

The webhook will only include fields that are relevant to the specific job type and status. Text-based jobs (whisper, analyze) will include a `text` field with the result, while media-based jobs (csm, portrait) will include a `result_url` field containing the **presigned URL** to the generated media file (valid for 7 days).

### Webhook Reliability

The system implements a robust notification mechanism:
- 5 retry attempts for failed webhook deliveries
- 5-second intervals between retry attempts
- Detailed logging of webhook delivery attempts
- Graceful handling of webhook failures

## 🖥️ GPU Instance Management

The worker component implements intelligent GPU instance management:

### Instance Initialization
- When a new GPU instance is launched, the system waits 5 seconds for the instance to initialize
- After initialization, the worker performs an initial health check to verify the instance is ready

### Health Monitoring
- A dedicated background thread monitors GPU instance health
- During job execution, health checks are performed every 15 seconds
- Health checks include:
  - Multiple retry attempts (up to 3) for failed health checks
  - Verification that the instance is still active in the Comput3.AI workloads API
  - HTTP connectivity tests to ensure the instance is responding
- If an instance fails health checks, it is automatically shut down
- When an instance fails during job execution, the current job is marked as failed with appropriate error messages

### Idle Management
- Instances are kept alive for 5 minutes after the last job
- The monitoring thread tracks idle time and automatically shuts down unused instances
- This approach balances cost efficiency with responsiveness

### Error Handling
- Failed instances are gracefully shut down and replaced
- All instance state changes are logged for troubleshooting
- Jobs are automatically marked as failed when GPU instances become unhealthy during processing
- When jobs fail during execution, GPU instances are shut down to conserve resources and costs

## 🧰 Development

The system is designed to be simple and maintainable:
- Single job queue in Redis for all job types
- GPU instances are reused between jobs (with 5-minute idle timeout)
- Workers handle webhook notifications with retry logic
- Clear separation between API (job submission) and worker (job processing)
- Modular approach with separate files for CSM and ComfyUI integrations

## 📂 Project Structure

The application is split into two main components:

1. **C3 Render API** (`c3_render_api/`): A Flask-based REST API that handles client requests and schedules jobs for processing.
2. **C3 Render Worker** (`c3_render_worker/`): A background worker that processes the jobs and interacts with Comput3.AI GPU services.

## 🐳 Docker Setup

The application is containerized using Docker, with services defined in `docker-compose.yml` (for development) and potentially `docker-compose.production.yaml` (for production).

### Key Services:

- **API**: The Flask API endpoint for client requests
- **Worker**: Background worker that processes jobs
- **Redis**: For job queueing and data storage
- **MinIO** (Optional, may run separately): S3-compatible storage service.

### Output Directory

The project includes a shared volume mount for output files:

```
./output:/app/output
```

This directory is used by the worker to store *temporary* files during processing (e.g., downloaded inputs, intermediate files) and potentially the final output *before* uploading to MinIO. Files saved in this directory may persist depending on the Docker setup, making it useful for debugging and local development.

#### File Naming Convention (Local Output / Temporary)

Files temporarily stored in the output directory might follow this convention:

- Text-to-speech (CSM) output: `{job_id}.mp3`
- Speech-to-text (Whisper) input: `{job_id}_input.mp3`
- Speech-to-text (Whisper) output: `{job_id}.txt`
- Portrait video generation image input: `{job_id}_portrait.jpg` (or original extension)
- Portrait video generation audio input: `{job_id}_audio.mp3` (or original extension)
- Portrait video generation output: `{job_id}.mp4`
- Image analysis input: `{job_id}_input.jpg` (or original extension)
- Image analysis output: `{job_id}.txt`

Note: The final persistent storage is MinIO, accessed via presigned URLs.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.