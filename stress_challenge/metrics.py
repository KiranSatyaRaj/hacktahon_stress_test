"""
MetricsCollector — Continuous system metrics sampling.

Collects CPU, GPU, and thermal data in a background thread and
exposes the latest snapshot for the WebSocket dashboard.
"""
import time
import csv
import os
import threading
import subprocess
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import psutil

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

from . import config


@dataclass
class MetricSnapshot:
    """Single point-in-time measurement of all tracked metrics."""
    timestamp: float = 0.0
    elapsed_seconds: float = 0.0

    # CPU
    cpu_avg_percent: float = 0.0
    cpu_per_core: list = field(default_factory=list)
    cpu_freq_current_mhz: float = 0.0
    cpu_freq_max_mhz: float = 0.0

    # Memory
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0

    # CPU Thermals
    cpu_temp_package: float = 0.0
    cpu_temp_cores: list = field(default_factory=list)

    # Fans (all sensors: CPU + GPU cooler, etc.)
    fan_readings: list = field(default_factory=list)   # [{label, rpm}]

    # GPU
    gpu_available: bool = False
    gpu_util_percent: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_power_w: float = 0.0
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_clock_sm_mhz: float = 0.0
    gpu_clock_mem_mhz: float = 0.0
    gpu_perf_state: int = -1                # P-state: 0=P0 (max), 8=P8 (idle)
    gpu_throttle_reasons: str = ""          # decoded throttle reason bitmask
    gpu_power_limit_w: float = 0.0          # max allowed power draw

    # Workload performance
    cpu_iter_sec: float = 0.0               # aggregate CPU matmul iterations/sec
    cpu_gflops: float = 0.0                 # derived CPU GFLOPS
    gpu_iter_sec: float = 0.0               # GPU stress iterations/sec
    gpu_tflops: float = 0.0                 # derived GPU FP16 TFLOPS

    # CUDA memory fragmentation
    cuda_mem_allocated_mb: float = 0.0      # actively allocated VRAM
    cuda_mem_reserved_mb: float = 0.0       # reserved by caching allocator
    cuda_mem_frag_mb: float = 0.0           # reserved - allocated = wasted

    # CPU time breakdown
    cpu_user_pct: float = 0.0               # % of CPU time in user mode
    cpu_kernel_pct: float = 0.0             # % of CPU time in kernel mode

    def to_dict(self):
        d = asdict(self)
        # Flatten lists for CSV / JSON
        for i, v in enumerate(d.pop("cpu_per_core", [])):
            d[f"cpu_core_{i}_percent"] = v
        for i, v in enumerate(d.pop("cpu_temp_cores", [])):
            d[f"cpu_temp_core_{i}"] = v
        for reading in d.pop("fan_readings", []):
            label = reading.get("label", "fan").replace(" ", "_").lower()
            d[f"fan_{label}_rpm"] = reading.get("rpm", 0)
        return d


class MetricsCollector:
    """
    Runs in a background thread, sampling system metrics at a fixed interval.
    """

    def __init__(self, interval: float = config.DEFAULT_SAMPLE_INTERVAL,
                 output_dir: str = config.DEFAULT_OUTPUT_DIR):
        self.interval = interval
        self.output_dir = output_dir
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshots: list[MetricSnapshot] = []
        self._lock = threading.Lock()
        self._start_time: float = 0.0

        # Optional references to workload objects for throughput polling
        self._cpu_workload = None
        self._gpu_workload = None

        # NVML init
        self._nvml_handle = None
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None

    def set_workloads(self, cpu_workload=None, gpu_workload=None):
        """Give the collector references to workload objects for throughput polling."""
        self._cpu_workload = cpu_workload
        self._gpu_workload = gpu_workload

    # ── Public API ───────────────────────────────────────────────────

    def start(self):
        """Begin background metric collection."""
        self._start_time = time.time()
        # Prime psutil so the very first cpu_percent(interval=None) call
        # returns valid data (not 0.0 from a cold baseline).
        psutil.cpu_percent(interval=None, percpu=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the collector to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def get_latest(self) -> Optional[MetricSnapshot]:
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def get_all_snapshots(self) -> list[MetricSnapshot]:
        with self._lock:
            return list(self._snapshots)

    def save_csv(self) -> str:
        """Persist all snapshots to CSV. Returns the file path."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, config.CSV_FILENAME)

        snapshots = self.get_all_snapshots()
        if not snapshots:
            return path

        rows = [s.to_dict() for s in snapshots]
        fieldnames = list(rows[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    # ── Internal ─────────────────────────────────────────────────────

    def _run_loop(self):
        while not self._stop_event.is_set():
            snap = self._collect()
            with self._lock:
                self._snapshots.append(snap)
            self._stop_event.wait(self.interval)

    def _collect(self) -> MetricSnapshot:
        now = time.time()
        snap = MetricSnapshot(
            timestamp=now,
            elapsed_seconds=round(now - self._start_time, 2),
        )

        # CPU utilization
        # interval=None: non-blocking, measures since last call.
        # The primer call in start() ensures the very first reading is valid.
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        snap.cpu_per_core = per_cpu
        snap.cpu_avg_percent = round(sum(per_cpu) / len(per_cpu), 1) if per_cpu else 0.0

        # CPU frequency
        freq = psutil.cpu_freq()
        if freq:
            snap.cpu_freq_current_mhz = round(freq.current, 0)
            snap.cpu_freq_max_mhz = round(freq.max, 0)

        # RAM
        mem = psutil.virtual_memory()
        snap.ram_percent = mem.percent
        snap.ram_used_gb = round(mem.used / (1024 ** 3), 2)
        snap.ram_total_gb = round(mem.total / (1024 ** 3), 2)

        # CPU thermals (via lm_sensors)
        self._read_cpu_temps(snap)

        # Fan RPM (CPU + GPU cooler via hwmon/psutil)
        self._read_fan_metrics(snap)

        # GPU metrics via NVML
        self._read_gpu_metrics(snap)

        # Workload throughput + FLOPS
        self._read_workload_throughput(snap)

        # CPU user vs kernel time breakdown
        self._read_cpu_times(snap)

        return snap

    _temp_cache: tuple = (0.0, [])   # (package_temp, core_temps)
    _temp_cache_ts: float = 0.0       # timestamp of last successful read
    _TEMP_CACHE_TTL: float = 2.0      # seconds between actual queries

    _fan_cache: list = []             # [{label, rpm}]
    _fan_cache_ts: float = 0.0
    _FAN_CACHE_TTL: float = 2.0       # fan RPM changes slowly — 2 s is fine

    def _read_cpu_temps(self, snap: MetricSnapshot):
        """
        Read CPU temperatures using the best available method:
          1. psutil.sensors_temperatures() — Linux / macOS
          2. PowerShell WMI MSAcpi_ThermalZoneTemperature — Windows
        Results are cached for _TEMP_CACHE_TTL seconds to avoid overhead.
        """
        import sys

        now = time.time()
        if now - self._temp_cache_ts < self._TEMP_CACHE_TTL:
            # Return cached values to avoid calling PowerShell every tick
            snap.cpu_temp_package, snap.cpu_temp_cores = self._temp_cache
            return

        package_temp = 0.0
        core_temps: list = []

        # ── Method 1: psutil (Linux / macOS) ──────────────────────
        try:
            all_temps = psutil.sensors_temperatures()
            if all_temps:
                for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
                    if key in all_temps:
                        readings = all_temps[key]
                        for r in readings:
                            if "Package" in (r.label or ""):
                                package_temp = r.current
                            elif "Core" in (r.label or "") or not r.label:
                                core_temps.append(r.current)
                        if not package_temp and core_temps:
                            package_temp = max(core_temps)
                        break
        except AttributeError:
            pass  # Windows — psutil.sensors_temperatures() not available

        # ── Method 2: PowerShell WMI (Windows) ────────────────────
        # Uses Win32_PerfFormattedData_Counters_ThermalZoneInformation.
        # Temperature property is in whole Kelvin: value - 273.15 = °C
        if package_temp == 0.0 and sys.platform == "win32":
            try:
                cmd = [
                    "powershell", "-NonInteractive", "-Command",
                    "Get-WmiObject Win32_PerfFormattedData_Counters_ThermalZoneInformation"
                    " | Select-Object -ExpandProperty Temperature"
                ]
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=4
                )
                raw_values = [
                    int(v.strip())
                    for v in r.stdout.strip().splitlines()
                    if v.strip().lstrip("-").isdigit()
                ]
                if raw_values:
                    # Kelvin → Celsius  (NOT decikelvin — this class gives whole K)
                    celsius_values = [v - 273.15 for v in raw_values]
                    celsius_values = [round(t, 1) for t in celsius_values if 0 < t < 120]
                    if celsius_values:
                        package_temp = max(celsius_values)
                        core_temps = celsius_values
            except Exception:
                pass

        # ── Cache and store ────────────────────────────────────────
        self._temp_cache = (package_temp, core_temps)
        self._temp_cache_ts = now
        snap.cpu_temp_package = package_temp
        snap.cpu_temp_cores = core_temps

    def _read_fan_metrics(self, snap: MetricSnapshot):
        """
        Read fan RPM for all fans exposed by the kernel (CPU cooler, GPU
        cooler, chassis fans). Uses psutil.sensors_fans() which reads the
        Linux hwmon subsystem — same source as `lm-sensors`.
        Results are cached for _FAN_CACHE_TTL seconds.
        """
        now = time.time()
        if now - self._fan_cache_ts < self._FAN_CACHE_TTL:
            snap.fan_readings = list(self._fan_cache)
            return

        readings: list = []
        try:
            all_fans = psutil.sensors_fans()
            if all_fans:
                for sensor_name, entries in all_fans.items():
                    for entry in entries:
                        label = (
                            f"{sensor_name}_{entry.label}"
                            if entry.label
                            else sensor_name
                        )
                        readings.append({"label": label, "rpm": int(entry.current)})
        except AttributeError:
            pass  # Windows — psutil.sensors_fans() not available
        except Exception:
            pass

        self._fan_cache = readings
        self._fan_cache_ts = now
        snap.fan_readings = list(readings)

    def _read_gpu_metrics(self, snap: MetricSnapshot):
        """Read GPU metrics via pynvml."""
        if not self._nvml_handle:
            snap.gpu_available = False
            return

        try:
            snap.gpu_available = True

            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            snap.gpu_util_percent = util.gpu

            temp = pynvml.nvmlDeviceGetTemperature(
                self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            snap.gpu_temp_c = temp

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                snap.gpu_power_w = round(power / 1000.0, 2)
            except Exception:
                snap.gpu_power_w = 0.0

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            snap.gpu_mem_used_mb = round(mem_info.used / (1024 ** 2), 1)
            snap.gpu_mem_total_mb = round(mem_info.total / (1024 ** 2), 1)

            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(
                    self._nvml_handle, pynvml.NVML_CLOCK_SM
                )
                snap.gpu_clock_sm_mhz = sm_clock
            except Exception:
                snap.gpu_clock_sm_mhz = 0.0

            try:
                mem_clock = pynvml.nvmlDeviceGetClockInfo(
                    self._nvml_handle, pynvml.NVML_CLOCK_MEM
                )
                snap.gpu_clock_mem_mhz = mem_clock
            except Exception:
                snap.gpu_clock_mem_mhz = 0.0

            # GPU performance state (P0=max, P8=idle)
            try:
                snap.gpu_perf_state = pynvml.nvmlDeviceGetPerformanceState(self._nvml_handle)
            except Exception:
                snap.gpu_perf_state = -1

            # GPU throttle reasons (bitmask → human-readable)
            try:
                mask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._nvml_handle)
                snap.gpu_throttle_reasons = self._decode_throttle_reasons(mask)
            except Exception:
                snap.gpu_throttle_reasons = ""

            # GPU power limit (max allowed power draw)
            try:
                limit = pynvml.nvmlDeviceGetPowerManagementLimit(self._nvml_handle)
                snap.gpu_power_limit_w = round(limit / 1000.0, 2)
            except Exception:
                snap.gpu_power_limit_w = 0.0

        except Exception:
            snap.gpu_available = False

    @staticmethod
    def _decode_throttle_reasons(mask: int) -> str:
        """
        Decode NVML throttle reason bitmask into human-readable causes.
        Returns comma-separated string of active reasons, or 'none'.
        """
        reasons = []
        # Bitmask values from nvml.h
        if mask & 0x0000000000000001:
            reasons.append("GpuIdle")
        if mask & 0x0000000000000002:
            reasons.append("AppClocksSetting")
        if mask & 0x0000000000000004:
            reasons.append("SwPowerCap")
        if mask & 0x0000000000000008:
            reasons.append("HwSlowdown")
        if mask & 0x0000000000000010:
            reasons.append("SyncBoost")
        if mask & 0x0000000000000020:
            reasons.append("SwThermalSlowdown")
        if mask & 0x0000000000000040:
            reasons.append("HwThermalSlowdown")
        if mask & 0x0000000000000080:
            reasons.append("HwPowerBrakeSlowdown")
        if mask & 0x0000000000000100:
            reasons.append("DisplayClockSetting")
        return ",".join(reasons) if reasons else "none"

    def _read_workload_throughput(self, snap: MetricSnapshot):
        """
        Read throughput from workload objects (if set) and derive FLOPS.
        CPU: GFLOPS = iter/s × 2 × N³ / 1e9  (for NxN float64 matmul)
        GPU: read directly from GPUWorkload.get_tflops()
        """
        # CPU throughput + GFLOPS
        if self._cpu_workload is not None:
            try:
                rate = self._cpu_workload.get_throughput()
                snap.cpu_iter_sec = round(rate, 2)
                N = config.CPU_MATRIX_SIZE
                snap.cpu_gflops = round(rate * 2 * (N ** 3) / 1e9, 3)
            except Exception:
                pass

        # GPU throughput + TFLOPS
        if self._gpu_workload is not None:
            try:
                snap.gpu_iter_sec = round(self._gpu_workload.get_throughput(), 2)
                snap.gpu_tflops = round(self._gpu_workload.get_tflops(), 3)
                # CUDA memory fragmentation
                snap.cuda_mem_allocated_mb = round(self._gpu_workload.cuda_mem_allocated_mb, 1)
                snap.cuda_mem_reserved_mb = round(self._gpu_workload.cuda_mem_reserved_mb, 1)
                snap.cuda_mem_frag_mb = round(self._gpu_workload.cuda_mem_frag_mb, 1)
            except Exception:
                pass

    def _read_cpu_times(self, snap: MetricSnapshot):
        """
        Read CPU user vs kernel time percentages.
        Uses psutil.cpu_times_percent() to get the breakdown since last call.
        """
        try:
            ct = psutil.cpu_times_percent(interval=None)
            snap.cpu_user_pct = round(ct.user + getattr(ct, 'nice', 0.0), 1)
            snap.cpu_kernel_pct = round(ct.system + getattr(ct, 'iowait', 0.0), 1)
        except Exception:
            pass

    def shutdown_nvml(self):
        if HAS_NVML and self._nvml_handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class ConsoleLogger:
    """
    Periodically prints a formatted snapshot of all key metrics to stdout.

    Usage::

        logger = ConsoleLogger(collector, interval=5)
        logger.start()
        ...
        logger.stop()
    """

    _DIVIDER = "─" * 62

    def __init__(self, collector: "MetricsCollector", interval: float = 5.0):
        self._collector = collector
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Public ────────────────────────────────────────────────────

    def start(self):
        """Start the background logging thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ConsoleLogger")
        self._thread.start()

    def stop(self):
        """Stop logging and wait for the thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    # ── Internal ─────────────────────────────────────────────────

    def _run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self._interval)
            if self._stop_event.is_set():
                break
            snap = self._collector.get_latest()
            if snap is not None:
                self._print(snap)

    def _print(self, s: "MetricSnapshot"):
        lines = [
            "",
            f"  ┌{'─' * 60}┐",
            f"  │{'  📊  METRICS SNAPSHOT':^60}│",
            f"  │  Elapsed: {s.elapsed_seconds:>7.1f}s{'':<41}│",
            f"  ├{'─' * 60}┤",
            f"  │  {'CPU':}{'':40}│",
            f"  │    Avg Usage  : {s.cpu_avg_percent:>5.1f} %{'':<33}│",
            f"  │    Frequency  : {s.cpu_freq_current_mhz:>7.0f} / {s.cpu_freq_max_mhz:.0f} MHz{'':<20}│",
        ]

        # Per-core percentages (up to 16, in groups of 4)
        cores = s.cpu_per_core
        for row_start in range(0, min(len(cores), 16), 4):
            chunk = cores[row_start: row_start + 4]
            cells = "  ".join(f"C{row_start + i:02d}:{v:>5.1f}%" for i, v in enumerate(chunk))
            lines.append(f"  │    {cells:<56}│")

        # CPU thermals
        if s.cpu_temp_package:
            lines.append(f"  │    Pkg Temp    : {s.cpu_temp_package:>5.1f} °C{'':<34}│")
        if s.cpu_temp_cores:
            core_str = "  ".join(f"{t:.0f}°" for t in s.cpu_temp_cores[:8])
            lines.append(f"  │    Core Temps  : {core_str:<42}│")

        lines += [
            f"  ├{'─' * 60}┤",
            f"  │  {'RAM':}{'':40}│",
            f"  │    Used       : {s.ram_used_gb:>5.2f} / {s.ram_total_gb:.2f} GB  ({s.ram_percent:.1f} %){'':<14}│",
        ]

        # GPU block
        lines.append(f"  ├{'─' * 60}┤")
        if s.gpu_available:
            lines += [
                f"  │  {'GPU (NVML)':}{'':40}│",
                f"  │    Utilisation: {s.gpu_util_percent:>5.1f} %{'':<33}│",
                f"  │    Temperature: {s.gpu_temp_c:>5.1f} °C{'':<33}│",
                f"  │    Power      : {s.gpu_power_w:>6.1f} / {s.gpu_power_limit_w:.0f} W{'':<27}│",
                f"  │    VRAM       : {s.gpu_mem_used_mb:>7.1f} / {s.gpu_mem_total_mb:.1f} MB{'':<20}│",
                f"  │    SM Clock   : {s.gpu_clock_sm_mhz:>6.0f} MHz{'':<32}│",
                f"  │    Mem Clock  : {s.gpu_clock_mem_mhz:>6.0f} MHz{'':<32}│",
            ]
            # CUDA memory fragmentation (only if GPU workload is active)
            if s.cuda_mem_allocated_mb > 0:
                lines.append(
                    f"  │    CUDA Mem   : {s.cuda_mem_allocated_mb:>6.0f} alloc"
                    f" / {s.cuda_mem_reserved_mb:.0f} rsv"
                    f"  frag={s.cuda_mem_frag_mb:.0f} MB{'':<6}│"
                )
            # Perf state + throttle reason on one line
            pstate = f"P{s.gpu_perf_state}" if s.gpu_perf_state >= 0 else "N/A"
            throttle = s.gpu_throttle_reasons or "none"
            if len(throttle) > 30:
                throttle = throttle[:28] + ".."
            lines.append(f"  │    Perf State : {pstate:<5} Throttle: {throttle:<25}│")
        else:
            lines.append(f"  │  GPU          : not available / NVML unavailable{'':<11}│")

        # Fans block
        if s.fan_readings:
            lines.append(f"  ├{'─' * 60}┤")
            lines.append(f"  │  {'FANS':}{'':40}│")
            for fan in s.fan_readings:
                lbl = fan.get("label", "fan")[:22]
                rpm = fan.get("rpm", 0)
                lines.append(f"  │    {lbl:<22}: {rpm:>5} RPM{'':<24}│")

        # Performance block (throughput + FLOPS + CPU time)
        has_perf = s.cpu_iter_sec > 0 or s.gpu_iter_sec > 0
        if has_perf:
            lines.append(f"  ├{'─' * 60}┤")
            lines.append(f"  │  {'PERFORMANCE':}{'':40}│")
            if s.cpu_iter_sec > 0:
                lines.append(
                    f"  │    CPU : {s.cpu_iter_sec:>7.1f} iter/s"
                    f"  │  {s.cpu_gflops:>7.2f} GFLOPS{'':<14}│"
                )
            if s.gpu_iter_sec > 0:
                lines.append(
                    f"  │    GPU : {s.gpu_iter_sec:>7.1f} iter/s"
                    f"  │  {s.gpu_tflops:>7.3f} TFLOPS{'':<14}│"
                )
            # CPU user vs kernel time
            if s.cpu_user_pct > 0:
                lines.append(
                    f"  │    CPU Time: user {s.cpu_user_pct:>5.1f}%"
                    f"  kernel {s.cpu_kernel_pct:>5.1f}%{'':<19}│"
                )

        lines += [
            f"  └{'─' * 60}┘",
            "",
        ]

        print("\n".join(lines), flush=True)


class EventLogger:
    """
    Monitors metric thresholds and emits structured log events.

    Output:
      - Console  : one-liner via print() (same as ConsoleLogger, no interference)
      - Log file : output/stress_test.log  (plain text, one line per event)

    Fires immediately on threshold crossings (not every interval), plus
    a periodic INFO summary line every `interval` seconds.

    Thresholds:
        CPU avg      > 80 % → WARN   | > 90 % → CRITICAL
        GPU util     > 80 % → WARN   | > 90 % → CRITICAL
        GPU temp     > 75 °C → WARN  | > config.GPU_THROTTLE_TEMP_C → CRITICAL
        RAM          > 70 % → WARN   | > 85 % → CRITICAL
        GPU power    > 200 W → WARN  (spike alert)
    """

    def __init__(
        self,
        collector: "MetricsCollector",
        output_dir: str = config.DEFAULT_OUTPUT_DIR,
        interval: float = 5.0,
    ):
        self._collector = collector
        self._output_dir = output_dir
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._states: dict = {}          # attr → "ok" | "warn" | "critical"
        self._start_time: float = 0.0
        self._log_file = None            # open file handle

    # ── Public ────────────────────────────────────────────────────

    def start(self):
        """Open log file and begin the monitoring thread."""
        self._start_time = time.time()
        os.makedirs(self._output_dir, exist_ok=True)
        log_path = os.path.join(self._output_dir, "stress_test.log")
        self._log_file = open(log_path, "a", encoding="utf-8", buffering=1)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="EventLogger"
        )
        self._thread.start()
        self._emit("INFO", f"EventLogger started — log_file={log_path}")

    def stop(self):
        """Stop monitoring and flush/close the log file."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._log_file:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    # ── Internal ─────────────────────────────────────────────────

    def _elapsed(self) -> str:
        secs = int(time.time() - self._start_time)
        return f"+{secs // 60}m{secs % 60:02d}s"

    def _emit(self, level: str, msg: str, snap: "Optional[MetricSnapshot]" = None):
        """
        Write a structured log line to the terminal and log file.
        Uses print() directly — no Python logging module involved.
        """
        import datetime
        t_elapsed = self._elapsed()
        t_wall = datetime.datetime.now().strftime("%H:%M:%S")

        if snap:
            metrics_str = (
                f"CPU={snap.cpu_avg_percent:.1f}% "
                f"RAM={snap.ram_percent:.1f}% "
                f"GPU={snap.gpu_util_percent:.0f}% "
                f"TEMP={snap.gpu_temp_c:.0f}°C "
                f"PWR={snap.gpu_power_w:.0f}W"
            )
        else:
            metrics_str = ""

        # Build colour prefix for terminal readability
        if level == "CRITICAL":
            prefix = "🔴 [CRITICAL]"
        elif level == "WARN":
            prefix = "🟡 [WARN    ]"
        else:
            prefix = "🔵 [INFO    ]"

        console_line = (
            f"  {prefix} t={t_elapsed}"
            + (f" | {metrics_str}" if metrics_str else "")
            + f" | {msg}"
        )
        file_line = (
            f"{t_wall} [{level:<8}] t={t_elapsed}"
            + (f" | {metrics_str}" if metrics_str else "")
            + f" | {msg}\n"
        )

        print(console_line, flush=True)
        if self._log_file:
            self._log_file.write(file_line)

    def _run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self._interval)
            if self._stop_event.is_set():
                break

            snap = self._collector.get_latest()
            if snap is None:
                continue

            # Periodic summary
            self._emit("INFO", "periodic snapshot", snap)
            # Threshold checks
            self._check_thresholds(snap)

    def _check_thresholds(self, snap: "MetricSnapshot"):
        """Emit events when metrics cross or recover from thresholds."""
        checks = [
            ("cpu_avg_percent",  80.0, 90.0,  "%",  "CPU"),
            ("gpu_util_percent", 80.0, 90.0,  "%",  "GPU_UTIL"),
            ("gpu_temp_c",       75.0, float(config.GPU_THROTTLE_TEMP_C), "°C", "GPU_TEMP"),
            ("ram_percent",      70.0, 85.0,  "%",  "RAM"),
        ]

        for attr, warn_val, crit_val, unit, label in checks:
            val = getattr(snap, attr, 0.0)
            prev = self._states.get(attr, "ok")

            if val >= crit_val:
                new_state = "critical"
                if prev != "critical":
                    self._emit(
                        "CRITICAL",
                        f"{label} critical threshold crossed "
                        f"({val:.1f}{unit} ≥ {crit_val}{unit})",
                        snap,
                    )
            elif val >= warn_val:
                new_state = "warn"
                if prev == "ok":
                    self._emit(
                        "WARN",
                        f"{label} warning threshold crossed "
                        f"({val:.1f}{unit} ≥ {warn_val}{unit})",
                        snap,
                    )
            else:
                new_state = "ok"
                if prev in ("warn", "critical"):
                    self._emit(
                        "INFO",
                        f"{label} recovered below threshold "
                        f"({val:.1f}{unit} < {warn_val}{unit})",
                        snap,
                    )
            self._states[attr] = new_state

        # GPU power spike
        if snap.gpu_available and snap.gpu_mem_total_mb > 0:
            if snap.gpu_power_w > 200 and self._states.get("gpu_power") != "spike":
                self._states["gpu_power"] = "spike"
                self._emit("WARN", f"GPU power spike: {snap.gpu_power_w:.0f}W", snap)
            elif snap.gpu_power_w <= 200:
                self._states["gpu_power"] = "ok"
