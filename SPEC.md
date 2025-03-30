# GPU AI Rendering Queue API Documentation

This API provides batch processing capabilities for various AI rendering tasks including text-to-speech, speech-to-text, portrait video generation, and image analysis.

### Implementation Notes

- System intentionally avoids complex class hierarchies or abstractions
- Jobs tracked by UUIDs in Redis
- API designed for simplicity and ease of integration
- Comput3.ai API functions integrated directly into worker
- Single unified job queue for all job types
- Robust webhook notification system with 5 retry attempts

## Base URL

```
https://api.example.com/api/v0
```

## Implementation Details

### Technology Stack

- **API Layer**: Flask (simple single-file implementation)
- **Queue Management**: Redis
- **Storage**: Minio client for S3-compatible object storage
- **GPU Processing**: Gradio clients for AI model interaction, ComfyUI for video workflows
- **GPU Resources**: Comput3.ai for on-demand GPU instances ("media:fast" workload type)
- **Containerization**: Simple Docker Compose setup

### GPU Instance Management

The worker component has direct integration with the Comput3.ai API for GPU instance management, implementing functions for launching, monitoring, and stopping workloads.

#### Integration Approach

1. **Integrated Comput3.ai API functions**:
   - `launch_workload()` - For starting "media:fast" GPU instances
   - `check_node_health()` - For verifying instance availability 
   - `stop_workload()` - For releasing instances after job completion

2. **Worker implementation pattern**:
   - Worker pulls job from a single Redis queue
   - Uses integrated functions to provision a GPU instance (reuses existing instance if available)
   - Processes the job on the instance (via Gradio or ComfyUI)
   - Updates job status in Redis
   - Sends webhook notifications with retry logic
   - Maintains GPU instances for 5 minutes of idle time before shutdown

3. **Environment setup**:
   - Uses C3_API_KEY environment variable for Comput3.ai API authentication

4. **Error handling**:
   - Implements robust health check mechanisms
   - Handles instance failures gracefully and updates job status accordingly
   - Includes 5 retry attempts for webhook notifications with 5-second intervals

### Implementation Approach

The implementation is deliberately kept minimal and straightforward:

#### API Server
- Single Flask application file that defines all endpoints
- Simple JSON validation for requests
- Redis for job queuing and status tracking
- No direct use of Minio or GPU resources

#### Worker Process
- Basic script that pulls jobs from a single Redis queue
- Has integrated Comput3.ai API functions for GPU instance management
- Implements intelligent GPU instance management with idle timeout
- Includes robust webhook notification system
- Handles multiple job types from a single queue

#### File Storage
- Uses client-provided URLs directly to download resources for processing
- Results to be stored in Minio/S3 with shareable URLs
- Workers will handle all storage operations

#### GPU Processing
- Workers use Gradio clients to interact with AI models for text and audio tasks
- Video tasks (portrait) use ComfyUI workflow JSONs with the requests library
- Models are accessed through interfaces running on Comput3.ai instances
- Instances are kept alive between jobs to maximize efficiency

### Kubernetes Deployment

The system can be deployed to Kubernetes with the following components:

#### Kubernetes Resources
1. **Flask API Deployment/Service**: Handles all API endpoints, generates job IDs, and adds jobs to Redis
2. **Redis StatefulSet/Service**: Manages job queue and stores job status information
3. **Worker Deployment**: Processes jobs from Redis, launches GPU instances, and updates job status

These components interact within the Kubernetes cluster:
- The API service receives requests and places them in Redis
- Worker pods pick up jobs from Redis and process them
- Results are stored in Minio and URLs are updated in Redis

Kubernetes provides several advantages for this architecture:
- Horizontal scaling of worker pods based on queue depth
- Health checks and automatic restarts of failed components
- Resource allocation and limits for stable operation
- Rolling updates for zero-downtime deployments

## Endpoints

### Text-to-Speech (CSM Model)

Generate speech from text, with optional voice cloning.

```
POST /csm
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | Yes | The text to convert to speech |
| audio_url | string | No | URL to audio file for voice cloning |
| audio_text | string | No* | Text content of the audio file (*required if audio_url is provided) |
| notify_url | string | No | Webhook URL for job completion notification |

#### Example Request

```json
{
  "text": "Hello world, this is a test of the text to speech system.",
  "audio_url": "https://storage.example.com/sample-voice.mp3",
  "audio_text": "Hello, my name is John and this is my voice sample.",
  "notify_url": "https://myapp.example.com/webhooks/job-complete"
}
```

#### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Speech-to-Text (Whisper)

Transcribe audio to text.

```
POST /whisper
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_url | string | Yes | URL to the audio file for transcription |
| model | string | No | Model to use for transcription (defaults to "medium") |
| notify_url | string | No | Webhook URL for job completion notification |

#### Example Request

```json
{
  "audio_url": "https://storage.example.com/recording.mp3",
  "model": "large",
  "notify_url": "https://myapp.example.com/webhooks/job-complete"
}
```

#### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001"
}
```

### Portrait Video Generation

Create a speaking portrait video from an image and audio.

```
POST /portrait
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image_url | string | Yes | URL to the image to animate |
| audio_url | string | No | URL to the audio for the portrait to speak (generates random voice if not provided) |
| notify_url | string | No | Webhook URL for job completion notification |

#### Example Request

```json
{
  "image_url": "https://storage.example.com/portrait.jpg",
  "audio_url": "https://storage.example.com/speech.mp3",
  "notify_url": "https://myapp.example.com/webhooks/job-complete"
}
```

#### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440002"
}
```

### Image Analysis

Analyze an image using a vision model.

```
POST /analyze
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image_url | string | Yes | URL to the image to be analyzed |
| notify_url | string | No | Webhook URL for job completion notification |

#### Example Request

```json
{
  "image_url": "https://storage.example.com/image.jpg",
  "notify_url": "https://myapp.example.com/webhooks/job-complete"
}
```

#### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003"
}
```

### Job Status

Check the status of a submitted job.

```
GET /status/{id}
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| id | string | **Required.** The UUID of the job |

#### Response

```json
{
  "status": "queued" // Possible values: "queued", "running", "failed", "success"
}
```

### Job Result

Retrieve the result of a completed job.

```
GET /result/{id}
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| id | string | **Required.** The UUID of the job |

#### Response

For text-based results (like /whisper or /analyze):
```json
{
  "text": "The transcription or analysis result text"
}
```

For media-based results (like /csm or /portrait):
```json
{
  "result_url": "https://storage.example.com/results/550e8400-e29b-41d4-a716-446655440003.mp4"
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
