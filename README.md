# C3 Render API & Worker

A simple distributed system for managing AI rendering tasks including text-to-speech, speech-to-text, portrait video generation, and image analysis. The system consists of a Flask API server for job submission and a worker component that processes jobs using GPU instances from Comput3.ai.

## Components

1. **C3 Render API**: Flask application that handles job submission, status tracking, and result retrieval
2. **C3 Render Worker**: Python script that processes jobs, manages GPU instances, and sends webhook notifications

## Setup

### Local Development

1. Set up environment variables:
   - For the API: Copy `c3_render_api/.env.sample` to `c3_render_api/.env`
   - For the Worker: Copy `c3_render_worker/.env.sample` to `c3_render_worker/.env` and add your Comput3.ai API key

```bash
# For API
cp c3_render_api/.env.sample c3_render_api/.env

# For Worker
cp c3_render_worker/.env.sample c3_render_worker/.env
# Edit c3_render_worker/.env to set your C3_API_KEY
```

2. Start the services using Docker Compose:
```bash
docker-compose up -d
```

This will start:
- API service on port 5000
- Redis on port 6379
- Minio (object storage) on ports 9000 (API) and 9001 (web console)
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
- Minio Console: http://localhost:9001 (login with minioadmin/minioadmin)

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
# Edit .env files as needed for your environment
```

3. Start Redis and Minio services (using Docker Compose):
```bash
docker-compose up -d redis minio
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
- **Webhook Notifications**: Workers send webhook notifications with 5 retry attempts at 5-second intervals
- **Job Status Tracking**: All job status information is stored in Redis

## API Endpoints

### Text-to-Speech (CSM Model)

Generate speech from text, with optional voice cloning.

```
POST /csm
```

#### Request Body
```json
{
  "text": "Hello world, this is a test of the text to speech system.",
  "audio_url": "https://storage.example.com/sample-voice.mp3",  // Optional
  "audio_text": "Hello, my name is John and this is my voice sample.",  // Required if audio_url is provided
  "notify_url": "https://myapp.example.com/webhooks/job-complete"  // Optional
}
```

### Speech-to-Text (Whisper)

Transcribe audio to text.

```
POST /whisper
```

#### Request Body
```json
{
  "audio_url": "https://storage.example.com/recording.mp3",
  "model": "medium",  // Optional, defaults to "medium"
  "notify_url": "https://myapp.example.com/webhooks/job-complete"  // Optional
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
  "audio_url": "https://storage.example.com/speech.mp3",  // Optional
  "notify_url": "https://myapp.example.com/webhooks/job-complete"  // Optional
}
```

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

### Job Status

Check the status of a submitted job.

```
GET /status/{id}
```

### Job Result

Retrieve the result of a completed job.

```
GET /result/{id}
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

## Development

The system is designed to be simple and maintainable:
- Single job queue in Redis for all job types
- GPU instances are reused between jobs (with 5-minute idle timeout)
- Workers handle webhook notifications with retry logic
- Clear separation between API (job submission) and worker (job processing)

## TODOs

The following features are planned for implementation:

### Worker Implementation
- [ ] Connect to Gradio clients for text-to-speech processing (CSM)
- [ ] Implement voice cloning with audio_url and audio_text
- [ ] Connect to Gradio clients for speech-to-text processing (Whisper)
- [ ] Connect to ComfyUI for portrait video generation
- [ ] Implement image analysis with vision models
- [ ] Add Minio S3 storage for results

### Infrastructure
- [ ] Kubernetes deployment configuration
- [ ] Worker horizontal scaling based on queue depth
- [ ] Monitoring and alerting for job failures
- [ ] Define resource requests and limits

### Security
- [ ] Add API authentication
- [ ] HTTPS encryption for API endpoints
- [ ] Secure storage of credentials
- [ ] Implement rate limiting

## License

MIT 