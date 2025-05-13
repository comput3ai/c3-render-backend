#!/usr/bin/env python3
"""
Shared constants for the C3 render system.
"""
import os

# Timeout configurations
GPU_IDLE_TIMEOUT = int(os.getenv("GPU_IDLE_TIMEOUT", "300"))  # Default: 5 minutes (300 seconds)
RENDER_POLLING_INTERVAL = int(os.getenv("RENDER_POLLING_INTERVAL", "5"))  # Default: 5 seconds

# Default job timing constraints
DEFAULT_COMPLETE_BY = int(os.getenv("DEFAULT_COMPLETE_BY", "3600"))  # 1 hour from now
DEFAULT_MAX_TIME = int(os.getenv("DEFAULT_MAX_TIME", "1200"))  # 20 minutes max runtime