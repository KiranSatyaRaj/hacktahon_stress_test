# Hackathon Coverage — Sustained Performance Stress Challenge

> **Last updated:** 2026-02-27

## Problem Statement (restated)
> Design and build a compute-intensive workload that runs continuously for **60 minutes** while maintaining consistent performance. The solution must **collect and log system metrics** (CPU usage, GPU utilization, thermal behavior) and **analyze** how the system behaves under sustained load.
>
> *Example: A long-running AI training or simulation workload that records performance and temperature data throughout execution and uses this data to identify bottlenecks, thermal limits, and optimization strategies.*

---

## Coverage Scorecard

| Requirement | Status | Implementation |
|---|---|---|
| **Compute-intensive CPU workload** | ✅ Done | [CPUWorkload](file:///c:/Users/pramo/Downloads/stress_test-main/stress_test-main/stress_challenge/workloads.py#54-84): 1 process/core, NumPy 1024×1024 `matmul` + `sin`/`cos` loop |
| **Compute-intensive GPU workload** | ✅ Done | [GPUWorkload](file:///c:/Users/pramo/Downloads/stress_test-main/stress_test-main/stress_challenge/workloads.py#249-347) → PyTorch CUDA 4096×4096 `torch.mm` loop (primary) |
| **Runs for 60 minutes** | ✅ Done | `--duration 3600` default, [_auto_stop_timer](file:///c:/Users/pramo/Downloads/stress_test-main/stress_test-main/stress_challenge/main.py#289-298) handles auto-shutdown |
| **Consistent sustained performance** | ⚠️ Partial | Workloads run continuously but no adaptive throttling guard |
| **Collect CPU usage metrics** | ✅ Done | `psutil` per-core %, avg %, frequency every 2 s |
| **Collect GPU utilization metrics** | ✅ Done | `pynvml`: util%, temp, power, VRAM, SM clock, mem clock |
| **Collect thermal behavior** | ✅ Done (GPU) | GPU temp via NVML. CPU temp requires `lm-sensors` (Linux only) |
| **Log metrics to console** | ✅ Done | [ConsoleLogger](file:///c:/Users/pramo/Downloads/stress_test-main/stress_test-main/stress_challenge/metrics.py#251-349): bordered table every 5 s |
| **Log metrics to file** | ✅ Done | `MetricsCollector.save_csv()` → `output/metrics.csv` |
| **Structured event logs** | ✅ Done | [EventLogger](file:///c:/Users/pramo/Downloads/stress_test-main/stress_test-main/stress_challenge/metrics.py#351-547): threshold alarms → `output/stress_test.log` |
| **Analyze system behavior** | ✅ Done | `PerformanceAnalyzer` → `output/analysis_report.txt` post-run |
| **Identify bottlenecks** | ⚠️ Partial | Analyzer runs post-hoc. No real-time bottleneck insight |
| **Identify thermal limits** | ✅ Done | [EventLogger](file:///c:/Users/pramo/Downloads/stress_test-main/stress_test-main/stress_challenge/metrics.py#351-547) flags WARN at 75 °C, CRITICAL at 85 °C |
| **Optimization strategies** | ❌ Missing | Analyzer doesn't suggest concrete optimizations yet |
| **Live monitoring dashboard** | ✅ Bonus | FastAPI + WebSocket → browser dashboard with live charts |

---

## GPU Workload Method — Detailed Explanation

The project tries **4 strategies in priority order**, using the first that succeeds:

```
Strategy 1 (PRIMARY on Windows+CUDA): PyTorch CUDA
  → spawns subprocess running:
      size = 4096
      a = torch.randn(size, size, device='cuda:0', dtype=float32)
      b = torch.randn(size, size, device='cuda:0', dtype=float32)
      while True:
          c = torch.mm(a, b)          # 4096×4096 matmul on GPU
          a = torch.sin(c) + 0.001
          b = torch.cos(c) + 0.001
          torch.cuda.synchronize()    # blocks until GPU finishes

Strategy 2: Vulkan (vkcube, 4 instances, PRIME offload) — Linux
Strategy 3: OpenGL (glxgears, 8 fullscreen instances) — Linux
Strategy 4: nvidia-smi clock boost + CPU matmul — last resort
```

> **You can check which method is active** via `GET /api/status` → `gpu_method` field,
> or the banner printed after auto-start.

---

## What's Still Missing (Gap Analysis)

### 🔴 High Priority (affects hackathon score)

1. **Optimization recommendations in the report**
   - [analyzer.py](file:///c:/Users/pramo/Downloads/stress_test-main/stress_test-main/stress_challenge/analyzer.py) calculates stats but doesn't suggest *what to do about them*
   - Add rules like: "GPU was thermally throttled for 12% of the run → reduce batch size or add cooling"

2. **CPU thermal on Windows**
   - Currently uses Linux `sensors` command → returns nothing on Windows
   - Add `wmi` or `OpenHardwareMonitor` fallback for Windows CPU temps

3. **Real-time bottleneck annotation in dashboard**
   - Dashboard graphs don't show threshold markers (e.g. red line at 85 °C)
   - EventLogger fires events but they're not shown in the live UI

### 🟡 Medium Priority (nice to have)

4. **Workload throughput metric**
   - CPU/GPU usage % tells you load but not *how much work was done*
   - Log iteration count from [CPUWorkload](file:///c:/Users/pramo/Downloads/stress_test-main/stress_test-main/stress_challenge/workloads.py#54-84) workers → compute ops/sec

5. **Sustained consistency score**
   - Report coefficient of variation (stddev/mean) of CPU/GPU util over time
   - A low CoV = consistent performance; a high one = thermal throttling causing drops

6. **Thermal throttle correlation**
   - Correlate GPU temp spikes with GPU util drops (throttle signature)
   - This is the core insight the problem statement asks for

### 🟢 Low Priority

7. GPU memory fragmentation tracking (`torch.cuda.memory_reserved()` vs `allocated()`)
8. NVLink / PCIe bandwidth (with DCGM on Linux)
9. Auto-save intermediate CSVs every 5 min to avoid data loss on crash

---

## Summary Score: ~65% of the problem statement covered

The **core requirements** (workload, data collection, logging, basic analysis) are solid.
The **differentiating factors** for a hackathon (optimization insights, thermal correlation, consistency scoring) are the gaps to close.
