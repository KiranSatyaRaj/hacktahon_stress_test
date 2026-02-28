"""
Configuration constants for the Sustained Performance Stress Challenge.
"""
import os
import multiprocessing

# ── Timing ───────────────────────────────────────────────────────────
DEFAULT_DURATION_SECONDS = 3600        # 60 minutes
DEFAULT_SAMPLE_INTERVAL = 2           # seconds between metric snapshots
LIVE_BROADCAST_INTERVAL = 2           # seconds between WebSocket pushes

# ── Workload tuning ──────────────────────────────────────────────────
CPU_MATRIX_SIZE = 1024                # NxN matrix for CPU stress
CPU_WORKER_COUNT = multiprocessing.cpu_count()  # one per logical core
GPU_MATRIX_SIZE = 2048                # NxN matrix for GPU stress (numpy fallback)

# ── Thermal thresholds ───────────────────────────────────────────────
CPU_THROTTLE_TEMP_C = 95
GPU_THROTTLE_TEMP_C = 85

# ── Output ───────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
CSV_FILENAME = "metrics.csv"
REPORT_FILENAME = "analysis_report.txt"

# ── Dashboard ────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8765
