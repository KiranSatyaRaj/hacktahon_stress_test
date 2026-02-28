#!/usr/bin/env python3
"""
Convenience launcher for the Sustained Performance Stress Challenge.

Usage:
    python run_stress.py [--duration 3600] [--port 8765] [--cpu-only] [--gpu-only]
"""
from stress_challenge.main import main

if __name__ == "__main__":
    main()
