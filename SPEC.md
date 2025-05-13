# C3 Render Service Specification

This API provides batch processing capabilities for various AI rendering tasks including text-to-speech, speech-to-text, portrait video generation, and image analysis.

## System Architecture

The C3 Render system consists of three main components:

1. **API Server**: A Flask application that handles client requests
2. **Worker**: A Python service that processes jobs using Comput3.ai GPU instances
3. **Redis**: A message queue and job data store

### Workflow

1. Client submits a job via the API
2. API validates the request, generates a job ID, and adds the job to the queue
3. Worker pulls jobs from the queue and processes them
4. Worker uses Comput3.ai GPU instances to execute the job
5. Worker updates job status in Redis and sends webhook notifications
6. Client can check job status or retrieve results via the API

## Implementation Details

### Technology Stack

- **API Layer**: Flask (simple single-file implementation)
- **Queue Management**: Redis
- **Storage**: Minio client for S3-compatible object storage
- **GPU Processing**: Gradio clients for AI model interaction, ComfyUI for video workflows
- **GPU Resources**: Comput3.ai for on-demand GPU instances ("media:fast" workload type)
- **Containerization**: Simple Docker Compose setup

### Worker Architecture

The worker component has been refactored into a modular design:

1. **c3_render_worker.py**: Main worker file that handles job queue and dispatching
2. **csm.py**: Module for text-to-speech generation using CSM
3. **comfyui.py**: Module for portrait video generation using ComfyUI

This modular approach improves code organization, maintainability, and makes it easier to add new job types in the future.

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
   - Processes the job on the instance (via Gradio for CSM or ComfyUI for portraits)
   - Updates job status in Redis
   - Sends webhook notifications with retry logic
   - Maintains GPU instances for 5 minutes of idle time before shutdown

3. **Environment setup**:
   - Uses C3_API_KEY environment variable for Comput3.ai API authentication

4. **Error handling**:
   - Implements robust health check mechanisms
   - Handles instance failures gracefully and updates job status accordingly
   - Includes 5 retry attempts for webhook notifications with 5-second intervals

## API Endpoints

### Text-to-Speech (CSM)

```
POST /csm
```

Generate speech from text using CSM (Collaborative Speech Model).

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| monologue | string | Yes | The text to convert to speech |
| voice | string | No | Voice type (random, conversational_a, conversational_b, clone) |
| reference_audio_url | string | For voice=clone | URL to reference audio file for voice cloning |
| reference_text | string | For voice=clone | Text content of the reference audio |
| temperature | float | No | Controls randomness of output (0.0-2.0) |
| topk | integer | No | Tokens to consider at each generation step |
| max_audio_length | integer | No | Maximum audio length in milliseconds |
| pause_duration | integer | No | Duration of pauses between sentences |
| notify_url | string | No | Webhook URL for job completion notification |

### Portrait Video Generation

```
POST /portrait
```

Create a talking head video from a portrait image and audio.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_url | string | Yes | URL to the portrait image |
| audio_url | string | Yes | URL to the audio file |
| notify_url | string | No | Webhook URL for job completion notification |

### Speech-to-Text (Whisper)

```
POST /api/v1/jobs/whisper
```

Transcribe audio to text using Whisper.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| audio_url | string | Yes | URL to the audio file |
| model | string | No | Whisper model size (default: medium) |
| notify_url | string | No | Webhook URL for job completion notification |

### Image Analysis

```
POST /analyze
```

Analyze an image using vision models.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_url | string | Yes | URL to the image |
| notify_url | string | No | Webhook URL for job completion notification |

### Job Status

```
GET /status/{job_id}
```

Get the current status of a job.

### Job Result

```
GET /result/{job_id}
```

Retrieve the result of a completed job.

## Job Types and Processing

### CSM (Text-to-Speech)

The CSM module (csm.py) handles text-to-speech generation:

1. Downloads reference audio if voice cloning is requested
2. Splits input text into sentences for better processing
3. Connects to CSM service on Comput3.ai GPU instance
4. Configures voice parameters based on request
5. Generates speech audio file
6. Returns path to generated audio file

### ComfyUI (Portrait Video Generation)

The ComfyUI module (comfyui.py) handles portrait video generation:

1. Downloads portrait image and audio files
2. Uploads files to ComfyUI running on Comput3.ai GPU instance
3. Configures and runs the SONIC animation workflow
4. Monitors workflow execution
5. Downloads the generated video
6. Returns path to generated video file

## Webhook Notifications

When a job completes and a `notify_url` is provided, the system sends a webhook with the following fields:

### Success (Media Jobs)
```json
{
  "id": "job_id",
  "status": "success",
  "result_url": "https://example.com/results/output.mp4"
}
```

### Success (Text Jobs)
```json
{
  "id": "job_id",
  "status": "success",
  "text": "Transcription or analysis result text"
}
```

### Failure
```json
{
  "id": "job_id",
  "status": "failed",
  "error": "Error message"
}
```

### Webhook Reliability

The system implements a robust notification mechanism:
- 5 retry attempts for failed webhook deliveries
- 5-second intervals between retry attempts
- Detailed logging of webhook delivery attempts
- Graceful handling of webhook failures

## File Storage

All output files are stored in the configured output directory, with the following naming convention:

- Text-to-speech: `{job_id}.mp3`
- Portrait video: `{job_id}.mp4`
- Speech-to-text input: `{job_id}_input.mp3`
- Speech-to-text output: `{job_id}.txt`
- Image analysis input: `{job_id}_input.jpg`
- Image analysis output: `{job_id}.txt`

## Implementation Notes

- System intentionally avoids complex class hierarchies or abstractions
- Jobs tracked by UUIDs in Redis
- API designed for simplicity and ease of integration
- Comput3.ai API functions integrated directly into worker
- Single unified job queue for all job types
- Robust webhook notification system with 5 retry attempts
- Modular codebase with separate files for CSM and ComfyUI integrations