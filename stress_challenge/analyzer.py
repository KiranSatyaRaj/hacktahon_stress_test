"""
Post-run analysis engine.

Analyzes collected metric snapshots and produces a structured report
identifying thermal behavior, performance trends, throttling events,
and optimization recommendations.
"""
import os
import statistics
from typing import Optional

from .metrics import MetricSnapshot
from . import config


class PerformanceAnalyzer:
    """Analyzes metric snapshots to identify bottlenecks and trends."""

    def __init__(self, snapshots: list[MetricSnapshot],
                 output_dir: str = config.DEFAULT_OUTPUT_DIR):
        self.snapshots = snapshots
        self.output_dir = output_dir
        self.report_lines: list[str] = []

    def analyze(self) -> str:
        """Run full analysis and return the report text."""
        self.report_lines = []
        self._header()
        self._cpu_analysis()
        self._thermal_analysis()
        self._gpu_analysis()
        self._memory_analysis()
        self._performance_trends()
        self._bottleneck_summary()
        self._recommendations()

        report = "\n".join(self.report_lines)
        self._save_report(report)
        return report

    # ── Section Builders ─────────────────────────────────────────────

    def _header(self):
        if not self.snapshots:
            self._section("ERROR", "No metric data to analyze.")
            return
        duration = self.snapshots[-1].elapsed_seconds
        mins = int(duration // 60)
        secs = int(duration % 60)
        self._section("STRESS TEST ANALYSIS REPORT",
                      f"Duration: {mins}m {secs}s  |  Samples: {len(self.snapshots)}  |  "
                      f"Interval: ~{self.snapshots[1].elapsed_seconds - self.snapshots[0].elapsed_seconds:.1f}s"
                      if len(self.snapshots) > 1 else f"Duration: {mins}m {secs}s  |  Samples: {len(self.snapshots)}")

    def _cpu_analysis(self):
        vals = [s.cpu_avg_percent for s in self.snapshots]
        freqs = [s.cpu_freq_current_mhz for s in self.snapshots if s.cpu_freq_current_mhz > 0]
        self._section("CPU UTILIZATION")
        self._stats("Average CPU %", vals)
        if freqs:
            self._stats("CPU Frequency (MHz)", freqs)
            # Detect frequency drops > 10% from max
            max_freq = max(freqs)
            drops = [f for f in freqs if f < max_freq * 0.9]
            if drops:
                pct = len(drops) / len(freqs) * 100
                self._line(f"  ⚠  Frequency throttling detected: {pct:.1f}% of samples below 90% of peak ({max_freq:.0f} MHz)")

    def _thermal_analysis(self):
        pkg_temps = [s.cpu_temp_package for s in self.snapshots if s.cpu_temp_package > 0]
        if not pkg_temps:
            self._section("THERMAL (CPU)", "  No CPU thermal data available.")
            return

        self._section("THERMAL (CPU)")
        self._stats("Package Temp (°C)", pkg_temps)

        # Time to steady state (within 2°C of final avg for 30s)
        steady_state_idx = self._find_steady_state(pkg_temps)
        if steady_state_idx is not None and len(self.snapshots) > steady_state_idx:
            ss_time = self.snapshots[steady_state_idx].elapsed_seconds
            self._line(f"  Time to thermal steady-state: ~{ss_time:.0f}s")

        # Throttle events (temp >= threshold)
        throttle_events = sum(1 for t in pkg_temps if t >= config.CPU_THROTTLE_TEMP_C)
        if throttle_events:
            self._line(f"  🔴 THERMAL THROTTLE EVENTS: {throttle_events} samples at/above {config.CPU_THROTTLE_TEMP_C}°C")
        else:
            self._line(f"  ✅ No CPU thermal throttling detected (threshold: {config.CPU_THROTTLE_TEMP_C}°C)")

    def _gpu_analysis(self):
        gpu_snaps = [s for s in self.snapshots if s.gpu_available]
        if not gpu_snaps:
            self._section("GPU", "  GPU metrics not available.")
            return

        self._section("GPU UTILIZATION")
        self._stats("GPU Util %", [s.gpu_util_percent for s in gpu_snaps])
        self._stats("GPU Temp (°C)", [s.gpu_temp_c for s in gpu_snaps])
        self._stats("GPU Power (W)", [s.gpu_power_w for s in gpu_snaps])
        self._stats("GPU SM Clock (MHz)", [s.gpu_clock_sm_mhz for s in gpu_snaps if s.gpu_clock_sm_mhz > 0])
        self._stats("GPU Mem Clock (MHz)", [s.gpu_clock_mem_mhz for s in gpu_snaps if s.gpu_clock_mem_mhz > 0])

        # GPU throttle detection
        gpu_temps = [s.gpu_temp_c for s in gpu_snaps]
        throttle_events = sum(1 for t in gpu_temps if t >= config.GPU_THROTTLE_TEMP_C)
        if throttle_events:
            self._line(f"  🔴 GPU THERMAL THROTTLE EVENTS: {throttle_events} samples at/above {config.GPU_THROTTLE_TEMP_C}°C")
        else:
            self._line(f"  ✅ No GPU thermal throttling detected (threshold: {config.GPU_THROTTLE_TEMP_C}°C)")

    def _memory_analysis(self):
        ram_vals = [s.ram_percent for s in self.snapshots]
        self._section("MEMORY")
        self._stats("RAM Usage %", ram_vals)

        gpu_snaps = [s for s in self.snapshots if s.gpu_available]
        if gpu_snaps:
            vram = [s.gpu_mem_used_mb for s in gpu_snaps]
            total = gpu_snaps[0].gpu_mem_total_mb
            self._stats(f"VRAM Usage (MB / {total:.0f} MB total)", vram)

    def _performance_trends(self):
        """Detect performance degradation over time via simple linear regression."""
        self._section("PERFORMANCE TRENDS")

        if len(self.snapshots) < 10:
            self._line("  Insufficient data for trend analysis (need 10+ samples).")
            return

        # CPU frequency trend
        freqs = [s.cpu_freq_current_mhz for s in self.snapshots if s.cpu_freq_current_mhz > 0]
        if len(freqs) >= 10:
            slope = self._linear_slope(freqs)
            direction = "declining" if slope < -1 else "stable" if abs(slope) <= 1 else "increasing"
            self._line(f"  CPU Frequency trend: {direction} ({slope:+.2f} MHz/sample)")

        # GPU util trend
        gpu_utils = [s.gpu_util_percent for s in self.snapshots if s.gpu_available]
        if len(gpu_utils) >= 10:
            slope = self._linear_slope(gpu_utils)
            direction = "declining" if slope < -0.5 else "stable" if abs(slope) <= 0.5 else "increasing"
            self._line(f"  GPU Utilization trend: {direction} ({slope:+.2f} %/sample)")

        # CPU temp trend
        temps = [s.cpu_temp_package for s in self.snapshots if s.cpu_temp_package > 0]
        if len(temps) >= 10:
            slope = self._linear_slope(temps)
            direction = "rising" if slope > 0.05 else "stable" if abs(slope) <= 0.05 else "cooling"
            self._line(f"  CPU Temperature trend: {direction} ({slope:+.3f} °C/sample)")

    def _bottleneck_summary(self):
        self._section("BOTTLENECK IDENTIFICATION")
        bottlenecks = []

        # CPU thermal bottleneck
        pkg_temps = [s.cpu_temp_package for s in self.snapshots if s.cpu_temp_package > 0]
        if pkg_temps and max(pkg_temps) >= config.CPU_THROTTLE_TEMP_C:
            bottlenecks.append("CPU THERMAL: Package temperature reached thermal limit, causing frequency throttling.")

        # GPU thermal bottleneck
        gpu_temps = [s.gpu_temp_c for s in self.snapshots if s.gpu_available]
        if gpu_temps and max(gpu_temps) >= config.GPU_THROTTLE_TEMP_C:
            bottlenecks.append("GPU THERMAL: GPU temperature reached thermal limit, reducing boost clocks.")

        # Power limit
        gpu_powers = [s.gpu_power_w for s in self.snapshots if s.gpu_available and s.gpu_power_w > 0]
        if gpu_powers:
            max_power = max(gpu_powers)
            high_power_count = sum(1 for p in gpu_powers if p > max_power * 0.95)
            if high_power_count > len(gpu_powers) * 0.5:
                bottlenecks.append(f"GPU POWER LIMIT: GPU spent >50% of time near its power ceiling ({max_power:.0f}W).")

        # Memory pressure
        ram_vals = [s.ram_percent for s in self.snapshots]
        if ram_vals and max(ram_vals) > 90:
            bottlenecks.append("RAM PRESSURE: System RAM usage exceeded 90%, potential swap/OOM risk.")

        if bottlenecks:
            for b in bottlenecks:
                self._line(f"  🔴 {b}")
        else:
            self._line("  ✅ No significant bottlenecks detected during this test run.")

    def _recommendations(self):
        self._section("OPTIMIZATION RECOMMENDATIONS")
        recs = []

        pkg_temps = [s.cpu_temp_package for s in self.snapshots if s.cpu_temp_package > 0]
        if pkg_temps:
            max_temp = max(pkg_temps)
            if max_temp >= 95:
                recs.append("🧊 CPU thermal paste may need replacement — package temps hit critical levels.")
                recs.append("🌀 Ensure laptop cooling vents are unobstructed; consider a cooling pad.")
            elif max_temp >= 85:
                recs.append("🌡️  CPU runs warm under full load — consider undervolting for better thermals.")

        gpu_temps = [s.gpu_temp_c for s in self.snapshots if s.gpu_available]
        if gpu_temps:
            max_gt = max(gpu_temps)
            if max_gt >= 85:
                recs.append("🎮 GPU thermal limit reached — consider undervolting GPU via nvidia-smi.")
            elif max_gt >= 75:
                recs.append("🎮 GPU thermals acceptable but elevated — a cooling pad may help sustained workloads.")

        freqs = [s.cpu_freq_current_mhz for s in self.snapshots if s.cpu_freq_current_mhz > 0]
        if len(freqs) >= 10:
            slope = self._linear_slope(freqs)
            if slope < -2:
                recs.append("📉 CPU frequency declining over time — system may be power/thermal limited long-term.")

        if not recs:
            recs.append("✅ System appears well-optimized for sustained workloads — no critical issues found.")

        for r in recs:
            self._line(f"  {r}")

    # ── Helpers ──────────────────────────────────────────────────────

    def _section(self, title: str, subtitle: str = ""):
        self.report_lines.append("")
        self.report_lines.append("=" * 72)
        self.report_lines.append(f"  {title}")
        self.report_lines.append("=" * 72)
        if subtitle:
            self.report_lines.append(subtitle)

    def _line(self, text: str):
        self.report_lines.append(text)

    def _stats(self, label: str, values: list[float]):
        if not values:
            self._line(f"  {label}: No data")
            return
        self._line(f"  {label}:")
        self._line(f"    Mean: {statistics.mean(values):.1f}  |  "
                   f"Std: {statistics.stdev(values) if len(values) > 1 else 0:.1f}  |  "
                   f"Min: {min(values):.1f}  |  Max: {max(values):.1f}")
        if len(values) >= 20:
            sorted_v = sorted(values)
            p95 = sorted_v[int(len(sorted_v) * 0.95)]
            p99 = sorted_v[int(len(sorted_v) * 0.99)]
            self._line(f"    P95: {p95:.1f}  |  P99: {p99:.1f}")

    def _find_steady_state(self, values: list[float], window: int = 15) -> Optional[int]:
        """Find index where values stabilize (std-dev < 2°C over window)."""
        if len(values) < window:
            return None
        for i in range(window, len(values)):
            window_vals = values[i - window:i]
            if statistics.stdev(window_vals) < 2.0:
                return i - window
        return None

    @staticmethod
    def _linear_slope(values: list[float]) -> float:
        """Simple least-squares slope (value per sample)."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator else 0.0

    def _save_report(self, report: str):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, config.REPORT_FILENAME)
        with open(path, "w") as f:
            f.write(report)
