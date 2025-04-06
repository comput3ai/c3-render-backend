# C3 Render API

A Flask-based API server for submitting jobs to the C3 Render service. This API provides endpoints for various rendering tasks including text-to-speech, portrait video generation, speech-to-text, and image analysis.

## Features

- RESTful API for job submission
- Job status tracking
- Result retrieval
- Multiple job types support:
  - CSM text-to-speech with voice options
  - Portrait video generation
  - Whisper speech-to-text
  - Image analysis

## Components

- **Flask application**: Handles HTTP requests and responses
- **Redis integration**: Queues jobs for processing by worker nodes
- **Job tracking**: Maintains job status and results

## API Endpoints

### Text-to-Speech (CSM)

```
POST /csm
```

Generate speech from text using the CSM service.

### Portrait Video Generation

```
POST /portrait
```

Create a talking head video from a portrait image and audio file.

### Speech-to-Text (Whisper)

```
POST /whisper
```

Transcribe speech from an audio file to text.

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **audio_url** | string | Required | URL to the audio file to transcribe |
| **model** | string | "medium" | Whisper model to use. Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3" |
| **task** | string | "transcribe" | Task type, can be "transcribe" or "translate" |
| **language** | string | "" | Language code (empty for auto-detection) |
| **notify_url** | string | Optional | Webhook URL to receive job status updates |

### Image Analysis

```
POST /analyze
```

Analyze an image using vision models.

### Job Status

```
GET /status/{job_id}
```

Get the current status of a submitted job.

### Job Result

```
GET /result/{job_id}
```

Retrieve the result of a completed job.

## Usage

### Environment Setup

Copy the sample environment file and configure your settings:

```bash
cp env.sample .env
```

Required environment variables:
- `REDIS_HOST`: Redis server hostname (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `FLASK_SECRET_KEY`: Secret key for Flask application
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 5000)

### Running the API Server

```bash
python c3_render_api.py
```

The API server will start and listen for connections on the configured host and port.

### Docker

The API can be run as a Docker container:

```bash
docker build -t c3-render-api .
docker run -d --name c3-render-api -p 5000:5000 --env-file .env c3-render-api
```

## Job Submission Flow

1. Client submits a job via the API
2. API validates the request parameters
3. API generates a unique job ID
4. API stores job information in Redis
5. API adds the job to the processing queue
6. API returns the job ID and initial status to the client
7. Worker nodes process the job asynchronously
8. Clients can check job status or retrieve results using the job ID