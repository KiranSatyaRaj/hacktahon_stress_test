"""
Adaptive Feedback Controller — closed-loop workload stabiliser.

Monitors thermal/power metrics and dynamically adjusts workload intensity
to maintain consistent performance under sustained load.

Risk Score = weighted blend of:
  - CPU temperature vs critical threshold  (40%)
  - GPU power draw vs power limit          (30%)
  - GPU throttle active flag               (20%)
  - Throughput degradation percentage       (10%)

Three levels:
  SAFE     (risk < 0.65) → hold or recover workload
  WARNING  (0.65–0.80)   → gentle reduction (micro-sleep)
  CRITICAL (> 0.80)      → aggressive reduction
"""
import csv
import os
import time
import threading
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .metrics import MetricsCollector, MetricSnapshot
    from .workloads import CPUWorkload, GPUWorkload


class AdaptiveController:
    """Real-time feedback controller that adjusts workload to maintain steady performance."""

    LEVELS = ("SAFE", "WARNING", "CRITICAL")

    def __init__(
        self,
        collector: "MetricsCollector",
        cpu_workload: "CPUWorkload | None" = None,
        gpu_workload: "GPUWorkload | None" = None,
        output_dir: str = config.DEFAULT_OUTPUT_DIR,
    ):
        self._collector = collector
        self._cpu = cpu_workload
        self._gpu = gpu_workload
        self._output_dir = output_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # State
        self._level = "SAFE"
        self._risk = 0.0
        self._last_action = ""
        self._safe_since: float = 0.0       # timestamp when risk dropped to SAFE
        self._peak_gflops: float = 0.0       # baseline for degradation calc
        self._decisions: list[dict] = []

        # Cumulative actions applied
        self._cpu_sleep_ms: float = 0.0

        # Decision log CSV
        os.makedirs(output_dir, exist_ok=True)
        self._csv_path = os.path.join(output_dir, "controller_decisions.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "elapsed_s", "risk", "level",
            "cpu_temp", "gpu_power_w", "gpu_throttle", "cpu_gflops",
            "action", "cpu_sleep_ms",
        ])

    # ── Public API ────────────────────────────────────────────────

    @property
    def level(self) -> str:
        return self._level

    @property
    def risk(self) -> float:
        return self._risk

    @property
    def last_action(self) -> str:
        return self._last_action

    def start(self):
        self._stop.clear()
        self._safe_since = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self._csv_file.close()
        except Exception:
            pass

    # ── Core Loop ─────────────────────────────────────────────────

    def _run_loop(self):
        while not self._stop.is_set():
            snap = self._collector.get_latest()
            if snap:
                self._evaluate(snap)
            self._stop.wait(config.CONTROLLER_TICK_INTERVAL)

    def _evaluate(self, snap: "MetricSnapshot"):
        """Compute risk, determine level, take action."""
        risk = self._compute_risk(snap)
        self._risk = round(risk, 3)

        # Determine level
        if risk >= config.RISK_CRITICAL:
            new_level = "CRITICAL"
        elif risk >= config.RISK_WARNING:
            new_level = "WARNING"
        else:
            new_level = "SAFE"

        # Track peak GFLOPS for degradation calc
        if snap.cpu_gflops > self._peak_gflops:
            self._peak_gflops = snap.cpu_gflops

        # Take action based on level
        action = ""
        if new_level == "CRITICAL":
            action = self._action_critical(snap)
            self._safe_since = 0.0
        elif new_level == "WARNING":
            action = self._action_warning(snap)
            self._safe_since = 0.0
        else:  # SAFE
            if self._safe_since == 0.0:
                self._safe_since = time.time()
            # Try recovery if we've been safe long enough
            elapsed_safe = time.time() - self._safe_since
            if elapsed_safe >= config.RECOVERY_COOLDOWN_SECS and self._cpu_sleep_ms > 0:
                action = self._action_recover(snap)

        self._level = new_level
        self._last_action = action

        # Log the decision
        self._log_decision(snap, action)

    # ── Risk Computation ──────────────────────────────────────────

    def _compute_risk(self, snap: "MetricSnapshot") -> float:
        """Weighted risk score between 0.0 and ~1.0."""
        # Component 1: CPU temperature (40%)
        cpu_temp = snap.cpu_temp_package
        temp_risk = min(1.0, cpu_temp / config.CPU_CRITICAL_TEMP) if cpu_temp > 0 else 0

        # Component 2: GPU power vs limit (30%)
        if snap.gpu_power_limit_w > 0 and snap.gpu_power_w > 0:
            power_risk = min(1.0, snap.gpu_power_w / snap.gpu_power_limit_w)
        else:
            power_risk = 0

        # Component 3: GPU throttle active (20%) — binary
        throttle = snap.gpu_throttle_reasons or ""
        throttle_risk = 1.0 if (throttle and throttle != "none") else 0.0

        # Component 4: Throughput degradation (10%)
        if self._peak_gflops > 0 and snap.cpu_gflops > 0:
            drop = max(0.0, (self._peak_gflops - snap.cpu_gflops) / self._peak_gflops)
            degrad_risk = min(1.0, drop)
        else:
            degrad_risk = 0

        return (
            0.4 * temp_risk +
            0.3 * power_risk +
            0.2 * throttle_risk +
            0.1 * degrad_risk
        )

    # ── Healing Actions ───────────────────────────────────────────

    def _action_warning(self, snap: "MetricSnapshot") -> str:
        """Level 1: Gentle reduction — add micro-sleep to CPU workers."""
        if self._cpu is None:
            return ""

        # Add 3ms of sleep (cumulative)
        new_sleep = min(self._cpu_sleep_ms + 3.0, 20.0)  # cap at 20ms
        if new_sleep != self._cpu_sleep_ms:
            self._cpu_sleep_ms = new_sleep
            self._cpu.set_sleep_ms(new_sleep)
            msg = f"⚠ WARNING — risk {self._risk:.2f} | CPU {snap.cpu_temp_package:.0f}°C → CPU sleep {new_sleep:.0f}ms"
            print(f"    🔧 CONTROLLER: {msg}", flush=True)
            return f"cpu_sleep→{new_sleep:.0f}ms"
        return ""

    def _action_critical(self, snap: "MetricSnapshot") -> str:
        """Level 2: Aggressive — larger sleep increase."""
        if self._cpu is None:
            return ""

        # Jump to higher sleep
        new_sleep = min(self._cpu_sleep_ms + 8.0, 30.0)  # cap at 30ms
        if new_sleep != self._cpu_sleep_ms:
            self._cpu_sleep_ms = new_sleep
            self._cpu.set_sleep_ms(new_sleep)
            msg = f"🚨 CRITICAL — risk {self._risk:.2f} | CPU {snap.cpu_temp_package:.0f}°C → CPU sleep {new_sleep:.0f}ms"
            print(f"    🔧 CONTROLLER: {msg}", flush=True)
            return f"cpu_sleep→{new_sleep:.0f}ms"
        return ""

    def _action_recover(self, snap: "MetricSnapshot") -> str:
        """Level 0: Gradual recovery — reduce sleep when safe."""
        if self._cpu is None:
            return ""

        new_sleep = max(0.0, self._cpu_sleep_ms - 2.0)
        if new_sleep != self._cpu_sleep_ms:
            self._cpu_sleep_ms = new_sleep
            self._cpu.set_sleep_ms(new_sleep)
            msg = f"✅ RECOVERY — risk {self._risk:.2f} | CPU {snap.cpu_temp_package:.0f}°C → CPU sleep {new_sleep:.0f}ms"
            print(f"    🔧 CONTROLLER: {msg}", flush=True)
            self._safe_since = time.time()  # reset cooldown timer
            return f"recover→sleep={new_sleep:.0f}ms"
        return ""

    # ── Logging ───────────────────────────────────────────────────

    def _log_decision(self, snap: "MetricSnapshot", action: str):
        """Write decision to CSV."""
        try:
            self._csv_writer.writerow([
                f"{snap.timestamp:.2f}",
                f"{snap.elapsed_seconds}",
                f"{self._risk:.3f}",
                self._level,
                f"{snap.cpu_temp_package:.1f}",
                f"{snap.gpu_power_w:.1f}",
                snap.gpu_throttle_reasons or "none",
                f"{snap.cpu_gflops:.2f}",
                action or "hold",
                f"{self._cpu_sleep_ms:.0f}",
            ])
            self._csv_file.flush()
        except Exception:
            pass
