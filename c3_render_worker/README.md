# C3 Render Worker

A worker service that processes rendering jobs queued in Redis. The worker manages GPU instances from Comput3.ai and processes various job types including text-to-speech, portrait video generation, speech-to-text, and image analysis.

## Features

- Processes jobs from Redis queue
- Manages Comput3.ai GPU instances
- Handles multiple job types:
  - CSM text-to-speech with voice options
  - ComfyUI portrait video generation
  - Whisper speech-to-text
  - Image analysis (coming soon)
- Sends webhook notifications on job completion
- Monitors GPU instance health
- Auto-scales GPU instances based on demand
- Smart GPU allocation to minimize costs
- Automatic image preprocessing for optimal results

## Components

- **c3_render_worker.py**: Main worker file that handles the job queue and dispatches jobs
- **csm.py**: Module for text-to-speech generation using CSM
- **comfyui.py**: Module for portrait video generation using ComfyUI
- **whisper.py**: Module for speech-to-text transcription using Whisper
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

Optional environment variables:
- `GPU_IDLE_TIMEOUT`: Time in seconds to keep a GPU alive after its last job (default: 300)
- `PRE_LAUNCH_TIMEOUT`: Minimum wait time in seconds before launching a new GPU (default: 15)
- `PRE_LAUNCH_TIMEOUT_MAX`: Maximum wait time in seconds before launching a new GPU (default: 30)
- `MAX_RENDER_TIME`: Maximum time in seconds allowed for a render job before timing out (default: 1800)
- `RENDER_POLLING_INTERVAL`: Interval in seconds to check job status during rendering (default: 5)

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

1. Worker peeks at the job queue without removing jobs
2. If the worker has an active GPU, it claims the job immediately
3. If no GPU is available, worker waits for a random time between `PRE_LAUNCH_TIMEOUT` and `PRE_LAUNCH_TIMEOUT_MAX` seconds before launching one
4. After waiting, if the job is still available, the worker launches a GPU and processes the job
5. Results are saved to the output directory
6. Webhook notification is sent if a callback URL was provided
7. Job status is updated in Redis

## GPU Instance Management

- GPU instances are kept alive for `GPU_IDLE_TIMEOUT` seconds after the last job (default: 5 minutes)
- Workers with existing GPUs claim jobs immediately, prioritizing warm GPU instances
- New GPU launches use random delays to avoid all workers launching GPUs simultaneously when there aren't enough running
- A monitoring thread checks instance health regularly
- Unhealthy instances are shut down and replaced

## Image Processing

- Portrait images are automatically resized while maintaining aspect ratio
- Maximum dimensions are constrained to 1280px width and 720px height
- Large images (>2048px in either dimension) are preprocessed to prevent ComfyUI failures
- All image processing is handled with PIL for optimum performance