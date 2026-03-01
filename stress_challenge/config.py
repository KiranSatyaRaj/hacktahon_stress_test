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

# ── Core affinity (combined CPU+GPU mode) ────────────────────────────
# Reserve 2 P-cores for the GPU subprocess so CUDA kernel submission
# doesn't get starved when all cores are under CPU load.
GPU_RESERVED_CORES = [0, 1]           # logical cores pinned to GPU subprocess
CPU_COMBINED_WORKER_COUNT = max(1, multiprocessing.cpu_count() - len(GPU_RESERVED_CORES))
ALL_CORES = list(range(multiprocessing.cpu_count()))
CPU_ALLOWED_CORES = [c for c in ALL_CORES if c not in GPU_RESERVED_CORES]

# ── Thermal thresholds ───────────────────────────────────────────────
CPU_THROTTLE_TEMP_C = 95
GPU_THROTTLE_TEMP_C = 85

# ── Adaptive Controller ─────────────────────────────────────────────
# Thermal zones (°C) — the controller takes action based on these
CPU_SAFE_TEMP = 75                    # below = no action needed
CPU_WARNING_TEMP = 80                 # above = reduce workload gently
CPU_CRITICAL_TEMP = 85                # above = aggressive reduction
GPU_SAFE_TEMP = 72
GPU_WARNING_TEMP = 78
# Controller timing
CONTROLLER_TICK_INTERVAL = 5          # seconds between controller evaluations
RECOVERY_COOLDOWN_SECS = 30           # wait this long in SAFE before recovering
# Risk thresholds
RISK_WARNING = 0.50
RISK_CRITICAL = 0.65

# ── Output ───────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
CSV_FILENAME = "metrics.csv"
REPORT_FILENAME = "analysis_report.txt"

# ── Dashboard ────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8765
