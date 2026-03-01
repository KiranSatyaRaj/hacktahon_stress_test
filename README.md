# 🔥 GigaHeal — Adaptive Compute Stabilizer

A real-time thermal management system that maintains **consistent hardware performance** under sustained stress by dynamically controlling CPU and GPU workloads through an intelligent feedback loop.

## The Problem

When hardware runs under sustained full load:
- CPU reaches **95°C** → firmware forces clock reduction (thermal throttling)
- Performance becomes **unpredictable** — GFLOPS oscillates wildly
- The application has **no control** over when or how much throttling occurs

```
Without GigaHeal:  GFLOPS: 75 → 62 → 55 → 68 → 50 → 63  (chaotic)
With GigaHeal:     GFLOPS: 75 → 72 → 65 → 65 → 65 → 65  (stable)
```

## The Solution

GigaHeal implements a **3-level adaptive controller** that monitors thermal metrics in real-time and applies graduated interventions:

| Level | Trigger | CPU Action | GPU Action |
|-------|---------|------------|------------|
| **SAFE** | Risk < 0.50 | No action | No action |
| **WARNING** | Risk ≥ 0.50 | +5ms sleep/tick (cap 50ms) | +2.5ms sleep/tick |
| **CRITICAL** | Risk ≥ 0.65 | +10ms sleep/tick (cap 80ms), then -1 worker | +5ms sleep/tick |
| **RECOVERY** | 30s stable in SAFE | -5ms sleep/tick, then +1 worker | Proportional reduction |

### Risk Score Formula
```
risk = 0.60 × (cpu_temp / 85°C) + 0.20 × (gpu_power / power_limit) +
       0.10 × thermal_throttle + 0.10 × throughput_degradation
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Main Thread: FastAPI + WebSocket (uvicorn)                    │
├──────────────────────────────────────────────────────────────┤
│ Thread 1: MetricsCollector (2s interval)                     │
│ Thread 2: ConsoleLogger (5s interval)                        │
│ Thread 3: EventLogger (threshold alarms)                     │
│ Thread 4: AdaptiveController (5s tick)                       │
│ Thread 5: Auto-stop timer                                    │
├──────────────────────────────────────────────────────────────┤
│ Process 1-18: CPU Workers (NumPy BLAS matmul, core-pinned)   │
│ Subprocess:   GPU Stress (PyTorch FP16 Tensor Core GEMM)     │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
```bash
pip install numpy torch psutil pynvml fastapi uvicorn
```

### Run Baseline (no controller)
```bash
python run_stress.py --duration 1200 --log-interval 5
```

### Run with Controller
```bash
python run_stress.py --duration 1200 --log-interval 5 --controller
```

### Compare Results
Open `http://localhost:8765/compare` after both runs complete.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--duration` | 3600 | Test duration in seconds |
| `--log-interval` | 5 | Console print interval |
| `--controller` | off | Enable adaptive controller |
| `--cpu-only` | — | CPU stress only |
| `--gpu-only` | — | GPU stress only |
| `--port` | 8765 | Dashboard port |

## Key Features

### CPU Stress Engine
- **18 worker processes** running continuous `1024×1024` matrix multiplication (NumPy BLAS)
- Core-pinned via `os.sched_setaffinity()` for isolation
- Shared iteration counter for real-time GFLOPS calculation
- Controller injects micro-sleeps via `multiprocessing.Value`

### GPU Stress Engine
- **4 concurrent CUDA streams** for maximum utilization:
  - Stream 0-1: FP16 8192×8192 GEMM (Tensor Core stress)
  - Stream 2: 128-batch matmul (memory bandwidth)
  - Stream 3: sin/cos transcendentals (SFU + CUDA core stress)
- Auto-sizes matrices to 85% of VRAM
- Controller injects sleep via a polled control file

### Real-Time Dashboard
- Live metrics via WebSocket at `http://localhost:8765`
- Comparison page at `/compare` with overlaid charts
- Verdict banner showing CoV improvement

### Metrics Collected
- CPU: per-core utilization, frequency, package temperature, per-core temps
- GPU: utilization, temperature, power draw, clock speeds, throttle reasons, VRAM usage
- Performance: CPU GFLOPS, GPU TFLOPS, iteration rates
- Controller: risk score, action level, active workers, sleep values

## Output Files

```
output/
├── baseline/
│   ├── metrics.csv              # All snapshots
│   ├── analysis_report.txt      # Statistical analysis
│   └── stress_test.log          # Threshold events
└── controlled/
    ├── metrics.csv
    ├── analysis_report.txt
    ├── stress_test.log
    └── controller_decisions.csv  # Every controller action
```

## Project Structure

```
stress_challenge/
├── main.py          # FastAPI server, CLI entry, orchestration
├── config.py        # All thresholds and constants
├── workloads.py     # CPU + GPU stress engines
├── controller.py    # Adaptive feedback controller
├── metrics.py       # MetricsCollector, ConsoleLogger, EventLogger
├── analyzer.py      # Post-test statistical analysis
└── dashboard/
    ├── index.html   # Live monitoring dashboard
    ├── compare.html # Baseline vs Controlled comparison
    ├── app.js       # Dashboard logic
    └── style.css    # Dashboard styling
```

## Tech Stack

- **Python 3.10+** with `multiprocessing` for CPU stress
- **NumPy** (BLAS) for matrix multiplication
- **PyTorch CUDA** for GPU stress (FP16 Tensor Core GEMM)
- **psutil** + **pynvml (NVML)** for system metrics
- **FastAPI** + **uvicorn** for dashboard and API
- **Chart.js** for real-time visualization

## License

MIT
