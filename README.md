# C3 Render API & Worker

A simple distributed system for managing AI rendering tasks including text-to-speech, speech-to-text, portrait video generation, and image analysis. The system consists of a Flask API server for job submission and a worker component that processes jobs using GPU instances from Comput3.ai.

## Components

1. **C3 Render API**: Flask application that handles job submission, status tracking, and result retrieval
2. **C3 Render Worker**: Python script that processes jobs, manages GPU instances, and sends webhook notifications

## Setup

### Local Development

1. Set up environment variables:
   - For the API: Copy `c3_render_api/.env.sample` to `c3_render_api/.env`
   - For the Worker: Copy `c3_render_worker/.env.sample` to `c3_render_worker/.env` and add your Comput3.ai API key and Minio configuration

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
- API service on port 5000
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
- API: http://localhost:5000

### Manual Setup

1. Install dependencies:
```bash
# For API
cd c3_render_api
pip install -r requirements.txt

# For Worker
cd c3_render_worker
pip install -r requirements.txt
```

2. Set up environment variables by copying the sample files:
```bash
# For API
cp c3_render_api/.env.sample c3_render_api/.env

# For Worker
cp c3_render_worker/.env.sample c3_render_worker/.env
# Edit .env files as needed for your environment, including C3_API_KEY and Minio configuration
```

3. Start Redis service (using Docker Compose):
```bash
docker-compose up -d redis
```

4. Run the API server:
```bash
cd c3_render_api
python c3_render_api.py
```

5. Run the worker (in a separate terminal):
```bash
cd c3_render_worker
python c3_render_worker.py
```

## Architecture

- **Single job queue**: All jobs go into a single Redis queue (`queue:jobs`) regardless of type
- **GPU Instance Management**: Workers maintain GPU instances for 5 minutes of idle time before shutting down
- **Robust GPU Monitoring**: A dedicated monitoring thread performs health checks every 10 seconds, with multiple retry attempts for failed checks and verification against the Comput3.ai API
- **Webhook Notifications**: Workers send webhook notifications with 5 retry attempts at 5-second intervals
- **Job Status Tracking**: All job status information is stored in Redis
- **Storage**: Uses Minio S3-compatible storage for storing job results (configured via environment variables)
- **Modular Design**: Worker functionality is separated into task-specific modules (csm.py, comfyui.py)

## API Endpoints

### POST /csm
Generate speech using the CSM (Collaborative Speech Model) text-to-speech system with configurable voice options, including voice cloning.

```bash
# Basic text-to-speech request
curl -X POST http://localhost:5000/csm \
  -H "Content-Type: application/json" \
  -d '{
    "monologue": "Hello, this is a test of the C3 Render API text to speech system using CSM.",
    "notify_url": "https://your-webhook-endpoint.com/callback"
  }'

# Text-to-speech with voice customization
curl -X POST http://localhost:5000/csm \
  -H "Content-Type: application/json" \
  -d '{
    "monologue": "This example uses the conversational voice type A with custom temperature and other parameters.",
    "voice": "conversational_a",
    "temperature": 0.7,
    "topk": 40,
    "max_audio_length": 8000,
    "pause_duration": 200,
    "notify_url": "https://your-webhook-endpoint.com/callback"
  }'
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **monologue** | string | Required | The text to convert to speech |
| **notify_url** | string | Optional | Webhook URL to receive job status updates |
| **voice** | string | "random" | Voice type to use. Options: "random", "conversational_a", "conversational_b", or "clone" for voice cloning |
| **reference_audio_url** | string | None | URL to reference audio file for voice cloning (required when voice="clone") |
| **reference_text** | string | None | Text content of the reference audio file for voice cloning (required when voice="clone") |
| **temperature** | float | 0.9 | Controls randomness of output (0.0-2.0). Higher values produce more random outputs |
| **topk** | integer | 50 | Number of highest probability tokens to consider at each generation step |
| **max_audio_length** | integer | 10000 | Maximum length of generated audio in milliseconds |
| **pause_duration** | integer | 150 | Duration of pauses between sentences in milliseconds |

> Note: For backward compatibility, the `text` field is still supported and will be treated the same as `monologue`.

#### Example with Voice Cloning

```bash
curl -X POST http://localhost:5000/csm \
  -H "Content-Type: application/json" \
  -d '{
    "monologue": "This is a voice cloning test. The system will try to mimic the voice in the reference audio.",
    "voice": "clone",
    "reference_audio_url": "https://example.com/reference-voice.mp3",
    "reference_text": "This is a sample of my voice for cloning purposes.",
    "notify_url": "https://your-webhook-endpoint.com/callback"
  }'
```

#### Example Response

```json
{
  "id": "c1e8f9a0-1b2c-4d3e-9f4g-5h6i7j8k9l0m",
  "status": "queued"
}
```

The worker will process the job and generate speech using the CSM service on Comput3 GPU instances. Upon completion, a webhook will be sent to the `notify_url` with the result URL.

Webhook payload on success:
```json
{
  "id": "c1e8f9a0-1b2c-4d3e-9f4g-5h6i7j8k9l0m",
  "status": "success",
  "result_url": "https://storage-endpoint.com/results/audio.mp3"
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

### POST /api/v1/jobs/whisper

Transcribe audio to text.

```
POST /api/v1/jobs/whisper
```

#### Request Body
```json
{
  "audio_url": "https://storage.example.com/recording.mp3",
  "model": "medium",  // Optional, defaults to "medium"
  "notify_url": "https://myapp.example.com/webhooks/job-complete"  // Optional
}
```

#### cURL Example
```bash
# Basic whisper request
curl -X POST http://localhost:5000/api/v1/jobs/whisper \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/path/to/audio.mp3",
    "notify_url": "https://your-webhook-endpoint.com/callback"
  }'

# Whisper request with model specification
curl -X POST http://localhost:5000/api/v1/jobs/whisper \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/path/to/audio.mp3",
    "model": "large",
    "notify_url": "https://your-webhook-endpoint.com/callback"
  }'
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
  "notify_url": "https://myapp.example.com/webhooks/job-complete"  // Optional
}
```

#### Example Curl Command
```bash
curl -X POST http://localhost:5000/portrait \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/path/to/portrait.jpg",
    "audio_url": "https://example.com/path/to/audio.mp3",
    "notify_url": "https://your-webhook-endpoint.com/callback"
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

#### cURL Example
```bash
curl -X POST http://localhost:5000/portrait \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://storage.example.com/portrait.jpg",
    "audio_url": "https://storage.example.com/speech.mp3",
    "notify_url": "https://myapp.example.com/webhooks/job-complete"
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
4. A webhook notification is sent upon completion with the URL to the generated video

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
  "notify_url": "https://myapp.example.com/webhooks/job-complete"  // Optional
}
```

#### cURL Example
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/path/to/image.jpg",
    "notify_url": "https://your-webhook-endpoint.com/callback"
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

#### cURL Example
```bash
# Replace JOB_ID with the actual job ID returned when submitting a job
curl -X GET http://localhost:5000/status/JOB_ID

# Example with an actual UUID
curl -X GET http://localhost:5000/status/550e8400-e29b-41d4-a716-446655440000
```

#### Example Response
```json
{
  "status": "queued"  // Possible values: "queued", "running", "success", "failed"
}
```

### Job Result

Retrieve the result of a completed job.

```
GET /result/{id}
```

#### cURL Example
```bash
# Replace JOB_ID with the actual job ID returned when submitting a job
curl -X GET http://localhost:5000/result/JOB_ID

# Example with an actual UUID
curl -X GET http://localhost:5000/result/550e8400-e29b-41d4-a716-446655440000
```

#### Example Response for Media Jobs (CSM, Portrait)
```json
{
  "result_url": "https://storage.example.com/results/JOB_ID.mp4"
}
```

#### Example Response for Text Jobs (Whisper, Analyze)
```json
{
  "text": "The transcribed text or analysis result"
}
```

## Webhook Notifications

When a job completes and a `notify_url` was provided, the worker will send a POST request to the specified URL with the following payload:

For successful jobs:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "result_url": "https://example.com/results/output.mp4"  // for media jobs (csm, portrait)
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

The webhook will only include fields that are relevant to the specific job type and status. Text-based jobs (whisper, analyze) will include a `text` field with the result, while media-based jobs (csm, portrait) will include a `result_url` field pointing to the generated media file.

### Webhook Reliability

The system implements a robust notification mechanism:
- 5 retry attempts for failed webhook deliveries
- 5-second intervals between retry attempts
- Detailed logging of webhook delivery attempts
- Graceful handling of webhook failures

## GPU Instance Management

The worker component implements intelligent GPU instance management:

### Instance Initialization
- When a new GPU instance is launched, the system waits 5 seconds for the instance to initialize
- After initialization, the worker performs an initial health check to verify the instance is ready

### Health Monitoring
- A dedicated background thread monitors GPU instance health every 10 seconds when idle
- During job execution, health checks are performed every 15 seconds
- Health checks include:
  - Multiple retry attempts (up to 3) for failed health checks
  - Verification that the instance is still active in the Comput3.ai workloads API
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

## Development

The system is designed to be simple and maintainable:
- Single job queue in Redis for all job types
- GPU instances are reused between jobs (with 5-minute idle timeout)
- Workers handle webhook notifications with retry logic
- Clear separation between API (job submission) and worker (job processing)
- Modular approach with separate files for CSM and ComfyUI integrations

## Project Structure

The application is split into two main components:

1. **C3 Render API** (`c3_render_api/`): A Flask-based REST API that handles client requests and schedules jobs for processing.
2. **C3 Render Worker** (`c3_render_worker/`): A background worker that processes the jobs and interacts with Comput3.ai GPU services.

## Docker Setup

The application is containerized using Docker, with services defined in `docker-compose.yml`.

### Key Services:

- **API**: The Flask API endpoint for client requests
- **Worker**: Background worker that processes jobs
- **Redis**: For job queueing and data storage

### Output Directory

The project includes a shared volume mount for output files:

```
./output:/app/output
```

This directory is used to store temporary files generated by the worker, such as audio files from the CSM text-to-speech service. Files saved in this directory persist even after container restarts, making it useful for debugging and local development.

#### File Naming Convention

Files in the output directory are named using the job ID as follows:

- Text-to-speech (CSM) output: `{job_id}.mp3`
- Speech-to-text (Whisper) input: `{job_id}_input.mp3`
- Speech-to-text (Whisper) output: `{job_id}.txt`
- Portrait video generation image input: `{job_id}_portrait.jpg`
- Portrait video generation audio input: `{job_id}_audio.mp3`
- Portrait video generation output: `{job_id}.mp4`
- Image analysis input: `{job_id}_input.jpg`
- Image analysis output: `{job_id}.txt`

This naming convention ensures that files are easily traceable to their originating jobs and prevents file naming conflicts.

## License

MIT