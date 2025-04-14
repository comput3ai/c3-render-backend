#!/usr/bin/env python3
"""
Shared constants for the C3 render system.
"""
import os

# Timeout configurations
GPU_IDLE_TIMEOUT = int(os.getenv("GPU_IDLE_TIMEOUT", "300"))  # Default: 5 minutes (300 seconds)
RENDER_POLLING_INTERVAL = int(os.getenv("RENDER_POLLING_INTERVAL", "5"))  # Default: 5 seconds

# Worker delay constants
GPU_WORKER_DELAY = int(os.getenv("GPU_WORKER_DELAY", "1"))     # 1 second for workers with GPU already running
NO_GPU_WORKER_DELAY = int(os.getenv("NO_GPU_WORKER_DELAY", "5"))  # 5 seconds for workers without GPU

# Queue polling intervals
LOCK_RETRY_INTERVAL = float(os.getenv("LOCK_RETRY_INTERVAL", "0.5"))  # Default: 0.5 seconds to retry when job is locked
QUEUE_CHECK_INTERVAL = float(os.getenv("QUEUE_CHECK_INTERVAL", "0.5"))  # Default: 0.5 seconds between queue checks when idle