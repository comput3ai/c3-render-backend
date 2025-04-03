# C3 Render Worker

A worker service that processes rendering jobs queued in Redis. The worker manages GPU instances from Comput3.ai and processes various job types including text-to-speech, portrait video generation, speech-to-text, and image analysis.

## Features

- Processes jobs from Redis queue
- Manages Comput3.ai GPU instances
- Handles multiple job types:
  - CSM text-to-speech with voice options
  - ComfyUI portrait video generation
  - Whisper speech-to-text (coming soon)
  - Image analysis (coming soon)
- Sends webhook notifications on job completion
- Monitors GPU instance health
- Auto-scales GPU instances based on demand

## Components

- **c3_render_worker.py**: Main worker file that handles the job queue and dispatches jobs
- **csm.py**: Module for text-to-speech generation using CSM
- **comfyui.py**: Module for portrait video generation using ComfyUI
- **Dockerfile**: Container definition for the worker
- **env.sample**: Example environment variables configuration
- **requirements.txt**: Python dependencies

## Usage

### Environment Setup

Copy the sample environment file and configure your settings:

```bash
cp env.sample .env
```

Required environment variables:
- `C3_API_KEY`: Your Comput3.ai API key
- `REDIS_HOST`: Redis server hostname (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `OUTPUT_DIR`: Directory for temporary output files (default: /app/output)

### Running the Worker

```bash
python c3_render_worker.py
```

The worker will connect to Redis and start processing jobs from the queue.

### Docker

The worker can be run as a Docker container:

```bash
docker build -t c3-render-worker .
docker run -d --name c3-render-worker --env-file .env c3-render-worker
```

## Job Processing Flow

1. Worker pulls job from Redis queue
2. Worker checks job type (csm, portrait, whisper, analyze)
3. Worker obtains a GPU instance from Comput3.ai (or reuses existing instance)
4. Job is processed on the GPU instance
5. Results are saved to the output directory
6. Webhook notification is sent if a callback URL was provided
7. Job status is updated in Redis

## GPU Instance Management

- GPU instances are kept alive for 5 minutes after the last job
- A monitoring thread checks instance health regularly
- Unhealthy instances are shut down and replaced
- During job execution, the worker checks instance health more frequently