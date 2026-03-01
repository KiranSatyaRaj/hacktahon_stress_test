# UnderStress — Sustained Performance Stress Challenge

A hardware stress testing tool with an adaptive feedback controller that maintains consistent CPU and GPU performance under sustained thermal load.

## Problem

Under sustained full load, CPUs reach thermal limits (95C+) and the firmware forces clock reductions (throttling). Performance becomes unpredictable and uncontrollable.

## Solution

UnderStress runs controlled CPU and GPU workloads while an adaptive controller monitors temperatures and injects micro-sleeps to keep hardware below throttling thresholds. The result is lower but **consistent** throughput instead of chaotic thermal oscillation.

## Quick Start

```bash
pip install numpy torch psutil pynvml fastapi uvicorn
```

**Baseline run (no controller):**
```bash
python run_stress.py --duration 1200 --log-interval 5
```

**Controlled run:**
```bash
python run_stress.py --duration 1200 --log-interval 5 --controller
```

**Compare results:** Open `http://localhost:8765/compare` after both runs.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--duration` | 3600 | Test duration in seconds |
| `--log-interval` | 5 | Console print interval |
| `--controller` | off | Enable adaptive controller |
| `--cpu-only` | - | CPU stress only |
| `--gpu-only` | - | GPU stress only |
| `--port` | 8765 | Dashboard port |

## How It Works

- **CPU Stress:** 18 worker processes running continuous matrix multiplication (NumPy BLAS), core-pinned for isolation.
- **GPU Stress:** 4 concurrent CUDA streams — FP16 Tensor Core GEMM, batched matmul, and transcendental ops — auto-sized to 85% VRAM.
- **Controller:** Evaluates a risk score every 5 seconds based on CPU temperature (60%), GPU power (20%), GPU thermal throttle (10%), and throughput degradation (10%). Applies graduated sleep injection to both CPU and GPU workloads. Worker reduction used only as a last resort.
- **Dashboard:** Real-time metrics via WebSocket, comparison page with overlaid charts.

## Output

```
output/baseline/       — metrics.csv, analysis_report.txt, stress_test.log
output/controlled/     — metrics.csv, analysis_report.txt, stress_test.log, controller_decisions.csv
```

## Project Structure

```
stress_challenge/
  main.py          — FastAPI server, CLI entry, orchestration
  config.py        — Thresholds and constants
  workloads.py     — CPU and GPU stress engines
  controller.py    — Adaptive feedback controller
  metrics.py       — Metrics collection and logging
  analyzer.py      — Post-test statistical analysis
  dashboard/       — Live monitoring and comparison UI
```

## Tech Stack

Python, NumPy, PyTorch CUDA, psutil, pynvml (NVML), FastAPI, Chart.js

## License

MIT
