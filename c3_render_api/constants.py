#!/usr/bin/env python3
"""
Shared constants for the C3 render system.
"""
import os

# Timeout configurations
GPU_IDLE_TIMEOUT = int(os.getenv("GPU_IDLE_TIMEOUT", "300"))  # Default: 5 minutes (300 seconds)
PRE_LAUNCH_TIMEOUT_MIN = int(os.getenv("PRE_LAUNCH_TIMEOUT", "15"))  # Default minimum: 15 seconds
PRE_LAUNCH_TIMEOUT_MAX = int(os.getenv("PRE_LAUNCH_TIMEOUT_MAX", "30"))  # Default maximum: 30 seconds
MAX_RENDER_TIME = int(os.getenv("MAX_RENDER_TIME", "1800"))  # Default: 30 minutes (1800 seconds) 
RENDER_POLLING_INTERVAL = int(os.getenv("RENDER_POLLING_INTERVAL", "5"))  # Default: 5 seconds 