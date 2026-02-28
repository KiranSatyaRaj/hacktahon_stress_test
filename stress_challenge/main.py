"""
Main orchestrator — FastAPI server with WebSocket for live dashboard.

Coordinates workloads, metrics collection, and serves the dashboard UI.
"""
import asyncio
import json
import os
import signal
import sys
import threading
import time
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from . import config
from .metrics import MetricsCollector, ConsoleLogger, EventLogger
from .workloads import CPUWorkload, GPUWorkload
from .analyzer import PerformanceAnalyzer

# ── Global State ─────────────────────────────────────────────────────
collector: MetricsCollector | None = None
console_logger: ConsoleLogger | None = None
event_logger: EventLogger | None = None
cpu_workload: CPUWorkload | None = None
gpu_workload: GPUWorkload | None = None
test_config = {
    "duration": config.DEFAULT_DURATION_SECONDS,
    "interval": config.DEFAULT_SAMPLE_INTERVAL,
    "log_interval": 5.0,          # seconds between console metric prints
    "output_dir": config.DEFAULT_OUTPUT_DIR,
    "cpu_enabled": True,
    "gpu_enabled": True,
    "start_time": 0.0,
    "running": False,
    "completed": False,
}
connected_clients: set[WebSocket] = set()
_shutdown_event = threading.Event()


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background broadcast task."""
    task = asyncio.create_task(broadcast_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

# Serve the dashboard directory
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = os.path.join(DASHBOARD_DIR, "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(DASHBOARD_DIR, "style.css"), media_type="text/css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(os.path.join(DASHBOARD_DIR, "app.js"), media_type="application/javascript")


@app.get("/api/status")
async def get_status():
    return {
        "running": test_config["running"],
        "completed": test_config["completed"],
        "duration": test_config["duration"],
        "elapsed": time.time() - test_config["start_time"] if test_config["running"] else 0,
        "cpu_enabled": test_config["cpu_enabled"],
        "gpu_enabled": test_config["gpu_enabled"],
        "gpu_method": gpu_workload.method if gpu_workload else "none",
    }


@app.post("/api/start")
async def start_test():
    """Begin the stress test."""
    global collector, console_logger, event_logger, cpu_workload, gpu_workload

    if test_config["running"]:
        return {"error": "Test already running"}

    os.makedirs(test_config["output_dir"], exist_ok=True)

    # Initialize components
    collector = MetricsCollector(
        interval=test_config["interval"],
        output_dir=test_config["output_dir"]
    )

    test_config["start_time"] = time.time()
    test_config["running"] = True
    test_config["completed"] = False

    # Start metrics collection
    collector.start()

    # Start periodic console logger
    console_logger = ConsoleLogger(collector, interval=test_config["log_interval"])
    console_logger.start()

    # Start event logger (threshold alarms → stdout + log file)
    event_logger = EventLogger(
        collector,
        output_dir=test_config["output_dir"],
        interval=test_config["log_interval"],
    )
    event_logger.start()

    # Start workloads — in combined mode, pin cores for isolation
    combined = test_config["cpu_enabled"] and test_config["gpu_enabled"]

    if test_config["cpu_enabled"]:
        cpu_workload = CPUWorkload(
            allowed_cores=config.CPU_ALLOWED_CORES if combined else None,
        )
        cpu_workload.start()

    if test_config["gpu_enabled"]:
        gpu_workload = GPUWorkload(combined_mode=combined)
        gpu_workload.start()

    # Give the collector references to poll throughput/FLOPS
    collector.set_workloads(
        cpu_workload=cpu_workload if test_config["cpu_enabled"] else None,
        gpu_workload=gpu_workload if test_config["gpu_enabled"] else None,
    )

    # Schedule auto-stop
    threading.Thread(target=_auto_stop_timer, daemon=True).start()

    return {"status": "started", "duration": test_config["duration"]}


@app.post("/api/stop")
async def stop_test():
    """Stop the stress test early."""
    _stop_all()
    return {"status": "stopped"}


@app.get("/api/report")
async def get_report():
    """Get the analysis report after test completion."""
    if not test_config["completed"]:
        return {"error": "Test not completed yet"}

    report_path = os.path.join(test_config["output_dir"], config.REPORT_FILENAME)
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return {"report": f.read()}
    return {"error": "Report not found"}


@app.get("/api/history")
async def get_history():
    """Return all collected metric snapshots for chart rendering."""
    if collector is None:
        return {"snapshots": []}
    snapshots = collector.get_all_snapshots()
    return {"snapshots": [s.to_dict() for s in snapshots]}


# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await ws.receive_text()
            # Client can send commands via WS if needed
    except WebSocketDisconnect:
        connected_clients.discard(ws)
    except Exception:
        connected_clients.discard(ws)


async def broadcast_loop():
    """Periodically sends latest metrics to all connected WebSocket clients."""
    while True:
        await asyncio.sleep(config.LIVE_BROADCAST_INTERVAL)
        if not connected_clients or collector is None:
            continue

        snap = collector.get_latest()
        if snap is None:
            continue

        data = snap.to_dict()
        data["test_running"] = test_config["running"]
        data["test_completed"] = test_config["completed"]
        data["test_duration"] = test_config["duration"]
        data["gpu_method"] = gpu_workload.method if gpu_workload else "none"

        message = json.dumps(data)
        dead = set()
        for ws in connected_clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        connected_clients.difference_update(dead)


# ── Internal ─────────────────────────────────────────────────────────

def _cli_auto_start():
    """
    Called in a daemon thread when --auto-start is used.
    Waits 1 s for uvicorn to bind, then starts the stress test and
    immediately prints the first console snapshot (t=0 baseline).
    """
    global collector, console_logger, event_logger, cpu_workload, gpu_workload

    time.sleep(1.0)  # let uvicorn finish binding

    if test_config["running"]:
        return  # already started via browser

    os.makedirs(test_config["output_dir"], exist_ok=True)

    collector = MetricsCollector(
        interval=test_config["interval"],
        output_dir=test_config["output_dir"],
    )

    test_config["start_time"] = time.time()
    test_config["running"] = True
    test_config["completed"] = False

    collector.start()

    # Give the collector one sample interval to gather the first reading
    time.sleep(max(test_config["interval"], 2.0))

    # Print immediate t=0 snapshot so the user sees output right away
    snap = collector.get_latest()
    if snap:
        from .metrics import ConsoleLogger as _CL
        _CL(collector, interval=test_config["log_interval"])._print(snap)

    # Now start the periodic logger
    console_logger = ConsoleLogger(collector, interval=test_config["log_interval"])
    console_logger.start()

    # Start event logger
    event_logger = EventLogger(
        collector,
        output_dir=test_config["output_dir"],
        interval=test_config["log_interval"],
    )
    event_logger.start()

    # Start workloads — in combined mode, pin cores for isolation
    combined = test_config["cpu_enabled"] and test_config["gpu_enabled"]

    if test_config["cpu_enabled"]:
        cpu_workload = CPUWorkload(
            allowed_cores=config.CPU_ALLOWED_CORES if combined else None,
        )
        cpu_workload.start()

    if test_config["gpu_enabled"]:
        gpu_workload = GPUWorkload(combined_mode=combined)
        gpu_workload.start()

    # Give the collector references to poll throughput/FLOPS
    collector.set_workloads(
        cpu_workload=cpu_workload if test_config["cpu_enabled"] else None,
        gpu_workload=gpu_workload if test_config["gpu_enabled"] else None,
    )

    # Schedule auto-stop
    threading.Thread(target=_auto_stop_timer, daemon=True).start()

    print(f"    ✅ Stress test running — console updates every "
          f"{test_config['log_interval']:.0f}s | dashboard: "
          f"http://localhost:{test_config.get('port', config.DASHBOARD_PORT)}\n",
          flush=True)


def _auto_stop_timer():
    """Wait for duration then stop everything."""
    start = test_config["start_time"]
    duration = test_config["duration"]
    while time.time() - start < duration:
        if _shutdown_event.is_set() or not test_config["running"]:
            return
        time.sleep(1)
    _stop_all()


def _stop_all():
    """Gracefully stop workloads, collect final metrics, run analysis."""
    global collector, console_logger, event_logger, cpu_workload, gpu_workload

    if not test_config["running"]:
        return

    test_config["running"] = False

    # Stop workloads
    if cpu_workload:
        cpu_workload.stop()
    if gpu_workload:
        gpu_workload.stop()

    # Stop console logger
    if console_logger:
        console_logger.stop()

    # Stop event logger and flush log file
    if event_logger:
        event_logger.stop()

    # Stop metrics collection
    if collector:
        collector.stop()
        csv_path = collector.save_csv()
        print(f"\n    📁 Metrics CSV saved → {csv_path}")
        log_path = os.path.join(test_config["output_dir"], "stress_test.log")
        print(f"    📋 Event log saved   → {log_path}")

        # Run analysis
        snapshots = collector.get_all_snapshots()
        if snapshots:
            analyzer = PerformanceAnalyzer(
                snapshots=snapshots,
                output_dir=test_config["output_dir"]
            )
            analyzer.analyze()

        collector.shutdown_nvml()

    test_config["completed"] = True


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🔥 Sustained Performance Stress Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # 60-min full stress test
  %(prog)s --duration 120           # 2-minute quick test
  %(prog)s --cpu-only               # CPU stress only
  %(prog)s --gpu-only               # GPU stress only
  %(prog)s --port 9000              # Custom dashboard port
        """
    )
    parser.add_argument("--duration", type=int, default=config.DEFAULT_DURATION_SECONDS,
                        help="Test duration in seconds (default: 3600)")
    parser.add_argument("--interval", type=float, default=config.DEFAULT_SAMPLE_INTERVAL,
                        help="Metric sampling interval in seconds (default: 2)")
    parser.add_argument("--log-interval", type=float, default=5.0,
                        help="Console metrics print interval in seconds (default: 5)")
    parser.add_argument("--output", type=str, default=config.DEFAULT_OUTPUT_DIR,
                        help="Output directory for logs and reports")
    parser.add_argument("--cpu-only", action="store_true", help="Only run CPU workload")
    parser.add_argument("--gpu-only", action="store_true", help="Only run GPU workload")
    parser.add_argument("--host", type=str, default=config.DASHBOARD_HOST,
                        help="Dashboard host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=config.DASHBOARD_PORT,
                        help="Dashboard port (default: 8765)")
    parser.add_argument("--auto-start", action="store_true", default=True,
                        help="Auto-start stress test immediately on launch (default: True)")
    parser.add_argument("--no-auto-start", dest="auto_start", action="store_false",
                        help="Wait for browser dashboard to start the test")

    args = parser.parse_args()

    # Apply config
    test_config["duration"] = args.duration
    test_config["interval"] = args.interval
    test_config["log_interval"] = args.log_interval
    test_config["output_dir"] = args.output
    test_config["port"] = args.port

    if args.cpu_only:
        test_config["gpu_enabled"] = False
    if args.gpu_only:
        test_config["cpu_enabled"] = False

    _auto_start_flag = args.auto_start

    mins = args.duration // 60
    secs = args.duration % 60

    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║        🔥  SUSTAINED PERFORMANCE STRESS CHALLENGE  🔥       ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Duration  : {:>4}m {:>2}s                                    ║
    ║  CPU       : {}                                          ║
    ║  GPU       : {}                                          ║
    ║  Dashboard : http://{}:{:<5}                       ║
    ║  Output    : {:<46} ║
    ╚══════════════════════════════════════════════════════════════╝
    """.format(
        mins, secs,
        "Enabled " if test_config["cpu_enabled"] else "Disabled",
        "Enabled " if test_config["gpu_enabled"] else "Disabled",
        "localhost", args.port,
        args.output[:46],
    ))

    if _auto_start_flag:
        print(f"    🚀 Auto-start enabled — test begins immediately")
        print(f"    📊 Metrics logged to console every {args.log_interval:.0f}s")
    print(f"    🌐 Open http://localhost:{args.port} in your browser for the live dashboard")
    print(f"    Press Ctrl+C to stop the test early\n")

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n    ⏹  Stopping stress test...")
        _shutdown_event.set()
        _stop_all()
        print("    ✅ Test stopped. Check output directory for results.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Auto-start: kick off the test 1 s after uvicorn is ready
    if _auto_start_flag:
        threading.Thread(target=_cli_auto_start, daemon=True).start()

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
