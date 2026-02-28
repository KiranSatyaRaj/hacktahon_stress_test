"""
Compute-intensive workloads for CPU and GPU stress testing.

CPU: Multi-process NumPy matrix multiplications (GIL-free).
GPU: Multiple strategies attempted in order:
     1. PyTorch CUDA tensor ops (if torch+CUDA available)
     2. Vulkan compute via vkcube (if available)
     3. OpenGL stress via glxgears with PRIME offload
     4. nvidia-smi triggered GPU clocks
"""
import multiprocessing
import threading
import subprocess
import time
import os
import signal
import sys
import shutil

import numpy as np

from . import config


# ════════════════════════════════════════════════════════════════════════
#  CPU WORKLOAD
# ════════════════════════════════════════════════════════════════════════

def _cpu_worker(stop_event: multiprocessing.Event, worker_id: int):
    """
    Single process that performs continuous matrix multiplication.
    Each process pins to its own NumPy computation (no GIL contention).
    """
    size = config.CPU_MATRIX_SIZE
    rng = np.random.default_rng(seed=worker_id)

    # Pre-allocate two random matrices
    a = rng.random((size, size), dtype=np.float64)
    b = rng.random((size, size), dtype=np.float64)

    iteration = 0
    while not stop_event.is_set():
        # Continuous heavy compute: matmul + element-wise ops
        c = np.dot(a, b)
        a = np.sin(c[:size, :size]) + 0.001
        b = np.cos(c[:size, :size]) + 0.001
        iteration += 1

        if iteration % 5 == 0:
            if stop_event.is_set():
                break


class CPUWorkload:
    """Manages a pool of CPU stress worker processes."""

    def __init__(self, num_workers: int = config.CPU_WORKER_COUNT):
        self.num_workers = num_workers
        self._stop_event = multiprocessing.Event()
        self._processes: list[multiprocessing.Process] = []

    def start(self):
        self._stop_event.clear()
        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=_cpu_worker,
                args=(self._stop_event, i),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

    def stop(self):
        self._stop_event.set()
        for p in self._processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._processes.clear()

    @property
    def is_running(self) -> bool:
        return any(p.is_alive() for p in self._processes)


# ════════════════════════════════════════════════════════════════════════
#  GPU WORKLOAD
# ════════════════════════════════════════════════════════════════════════

# Inline Python script for CUDA stress via torch (spawned as subprocess
# to isolate CUDA runtime from the main process & use system python if needed)
TORCH_GPU_STRESS_SCRIPT = '''
import sys, os, time, signal
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

keep_running = True
def handler(sig, frame):
    global keep_running
    keep_running = False
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

try:
    import torch
    if not torch.cuda.is_available():
        print("TORCH_CUDA_UNAVAILABLE", flush=True)
        sys.exit(1)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    vram_bytes = props.total_memory
    print(f"TORCH_CUDA_OK {props.name}", flush=True)

    # ── Determine sizing based on available VRAM ──────────────────────
    # Use up to 85% of VRAM split across multiple work buffers
    budget_bytes = int(vram_bytes * 0.85)

    # Tensor Core FP16 GEMM: choose largest N such that 6 * N^2 * 2 bytes <= budget
    # (a, b, c for 2 streams = 6 matrices of float16)
    import math
    N = min(8192, int(math.isqrt(budget_bytes // (6 * 2))))
    N = (N // 64) * 64  # round to multiple of 64 for Tensor Core alignment
    N = max(N, 2048)

    print(
        f"GPU_BUDGET  vram={vram_bytes} budget={budget_bytes}"
        f" ({budget_bytes/1024**3:.2f} GB)  matrix_N={N}"
        f" ({N*N*2/1024**2:.0f} MB per FP16 matrix)",
        flush=True,
    )

    # Batch matmul sizing: B batches of MxK @ KxN, float16
    B, M, K = 64, 1024, 1024

    # ── Allocate persistent buffers (avoid per-iteration alloc overhead) ─
    # Stream 0 & 1: large 2D FP16 GEMM → hammers Tensor Cores
    a0 = torch.randn(N, N, device=device, dtype=torch.float16)
    b0 = torch.randn(N, N, device=device, dtype=torch.float16)
    c0 = torch.empty(N, N, device=device, dtype=torch.float16)

    a1 = torch.randn(N, N, device=device, dtype=torch.float16)
    b1 = torch.randn(N, N, device=device, dtype=torch.float16)
    c1 = torch.empty(N, N, device=device, dtype=torch.float16)

    # Stream 2: batched matmul → keeps memory bandwidth saturated
    ba = torch.randn(B, M, K, device=device, dtype=torch.float16)
    bb = torch.randn(B, K, M, device=device, dtype=torch.float16)

    # Stream 3: large element-wise ops → stresses L2 cache & bandwidth
    big = torch.randn(N * 2, device=device, dtype=torch.float32)

    # ── CUDA streams for concurrent kernel execution ──────────────────
    streams = [torch.cuda.Stream(device=device) for _ in range(4)]

    torch.cuda.synchronize()

    while keep_running:
        # Stream 0: FP16 GEMM A → hammers Tensor Cores
        with torch.cuda.stream(streams[0]):
            torch.mm(a0, b0, out=c0)
            torch.add(c0, 0.0001, out=a0)   # perturb to stay alive

        # Stream 1: FP16 GEMM B (independent → fills remaining SMs)
        with torch.cuda.stream(streams[1]):
            torch.mm(a1, b1, out=c1)
            torch.add(c1, 0.0001, out=b0)

        # Stream 2: batched matmul → VRAM bandwidth pressure
        with torch.cuda.stream(streams[2]):
            torch.bmm(ba, bb, out=torch.empty(B, M, M, device=device, dtype=torch.float16))

        # Stream 3: element-wise transcendentals → saturate CUDA cores
        with torch.cuda.stream(streams[3]):
            torch.sin_(big)
            torch.cos_(big)

        # Sync once per iteration to keep all streams coordinated
        torch.cuda.synchronize()

except Exception as e:
    print(f"TORCH_ERROR {e}", flush=True)
    sys.exit(1)
'''

# Inline Vulkan compute shader stress script
VULKAN_STRESS_SCRIPT = '''
import subprocess, signal, sys, time
# Run multiple vkcube instances with PRIME offload to stress the GPU
procs = []
env = dict(__NV_PRIME_RENDER_OFFLOAD="1", __VK_LAYER_NV_optimus="NVIDIA_only",
           VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json")
import os
full_env = {**os.environ, **env}

keep_running = True
def handler(sig, frame):
    global keep_running
    keep_running = False
    for p in procs:
        try: p.terminate()
        except: pass
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

print("VULKAN_STARTING", flush=True)
for i in range(4):
    try:
        p = subprocess.Popen(
            ["vkcube", "--c", "99999999"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=full_env
        )
        procs.append(p)
    except: pass

if not procs:
    print("VULKAN_FAILED", flush=True)
    sys.exit(1)
print(f"VULKAN_OK {len(procs)} instances", flush=True)
while keep_running:
    time.sleep(1)
    if all(p.poll() is not None for p in procs):
        break
for p in procs:
    try: p.terminate()
    except: pass
'''

# Inline OpenGL stress script using glxgears with PRIME offload
OPENGL_STRESS_SCRIPT = '''
import subprocess, signal, sys, time, os
procs = []
env = {**os.environ, "__NV_PRIME_RENDER_OFFLOAD": "1", "__GLX_VENDOR_LIBRARY_NAME": "nvidia"}

keep_running = True
def handler(sig, frame):
    global keep_running
    keep_running = False
    for p in procs:
        try: p.terminate()
        except: pass
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

# Spawn multiple glxgears with high-res for more GPU load
print("OPENGL_STARTING", flush=True)
for i in range(8):
    try:
        p = subprocess.Popen(
            ["glxgears", "-fullscreen"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env
        )
        procs.append(p)
    except: pass

if not procs:
    print("OPENGL_FAILED", flush=True)
    sys.exit(1)
print(f"OPENGL_OK {len(procs)} instances", flush=True)
while keep_running:
    time.sleep(1)
    if all(p.poll() is not None for p in procs):
        break
for p in procs:
    try: p.terminate()
    except: pass
'''

# Pure nvidia-smi power draw stress (last resort)
NVIDIA_COMPUTE_STRESS_SCRIPT = '''
import subprocess, signal, sys, time, os
keep_running = True
def handler(sig, frame):
    global keep_running
    keep_running = False
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

# Set max GPU clocks to increase power draw and utilization
try:
    subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True)
    result = subprocess.run(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
                          capture_output=True, text=True)
    # Try to set max clocks
    subprocess.run(["nvidia-smi", "-lgc", "0,9999"], capture_output=True)
    subprocess.run(["nvidia-smi", "-lmc", "0,9999"], capture_output=True)
except: pass

print("NVIDIA_COMPUTE_STARTING", flush=True)

# Create GPU load by running nvidia-smi repeatedly + memory allocation via CUDA
# This is a lightweight stress - it won't max the GPU but will show activity
import numpy as np
size = 8192
while keep_running:
    # Large matrix operations create SOME GPU activity through driver overhead
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    c = np.dot(a, b)
    # Also query GPU to keep it from sleeping
    subprocess.run(["nvidia-smi", "-q", "-d", "UTILIZATION"],
                  capture_output=True, timeout=5)
    if not keep_running:
        break

print("NVIDIA_COMPUTE_STOPPED", flush=True)
'''


class GPUWorkload:
    """
    GPU stress workload with fallback strategy chain.

    Tries methods in order until one works:
    1. PyTorch CUDA (real GPU compute)
    2. Vulkan compute (vkcube with PRIME offload)
    3. OpenGL (glxgears with PRIME offload)
    4. nvidia-smi clock boosting (minimal stress)
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._subprocess: subprocess.Popen | None = None
        self._method = "none"

    @property
    def method(self) -> str:
        return self._method

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_stress_chain, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._subprocess and self._subprocess.poll() is None:
            self._subprocess.send_signal(signal.SIGTERM)
            try:
                self._subprocess.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._subprocess.kill()
        if self._thread:
            self._thread.join(timeout=15)
        # Reset GPU clocks
        try:
            subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, timeout=5)
            subprocess.run(["nvidia-smi", "-rmc"], capture_output=True, timeout=5)
        except Exception:
            pass

    @staticmethod
    def _find_torch_python() -> str:
        """
        Return path to a Python interpreter that has torch + CUDA.
        Searches common venv locations relative to this file AND known
        system paths, falling back to sys.executable if nothing found.
        """
        import glob

        probe = (
            "import torch, sys; assert torch.cuda.is_available();"
            " print(sys.executable)"
        )
        # 1. The project's own venv
        candidates = [sys.executable]
        # 2. Sibling / parent venvs (common hackathon layout)
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidates += glob.glob(os.path.join(base, "*", ".venv", "bin", "python3"))
        candidates += glob.glob(os.path.join(base, "*", ".venv", "bin", "python"))
        # 3. Common user-level venvs
        home = os.path.expanduser("~")
        candidates += glob.glob(os.path.join(home, "**", ".venv", "bin", "python3"), recursive=True)
        candidates += glob.glob(os.path.join(home, ".local", "bin", "python3"))
        # 4. System pythons
        for name in ("python3.12", "python3.11", "python3.10", "python3"):
            found = shutil.which(name)
            if found:
                candidates.append(found)

        seen: set[str] = set()
        for py in candidates:
            if not py or py in seen:
                continue
            seen.add(py)
            try:
                result = subprocess.run(
                    [py, "-c", probe],
                    capture_output=True, text=True, timeout=8
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                pass
        return sys.executable  # last resort

    def _run_stress_chain(self):
        """Try each GPU stress method in order."""
        strategies = [
            ("torch_cuda", TORCH_GPU_STRESS_SCRIPT, "TORCH_CUDA_OK"),
            ("vulkan", VULKAN_STRESS_SCRIPT, "VULKAN_OK"),
            ("opengl", OPENGL_STRESS_SCRIPT, "OPENGL_OK"),
            ("nvidia_compute", NVIDIA_COMPUTE_STRESS_SCRIPT, "NVIDIA_COMPUTE_STARTING"),
        ]

        # Use the best Python that has torch+CUDA; fall back gracefully
        torch_python = self._find_torch_python()
        python_path = sys.executable  # default for non-torch strategies

        for name, script, success_marker in strategies:
            if self._stop_event.is_set():
                return

            # Use torch-capable interpreter only for the torch strategy
            interpreter = torch_python if name == "torch_cuda" else python_path

            try:
                self._subprocess = subprocess.Popen(
                    [interpreter, "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Wait for the first line of output to determine if it succeeded
                first_line = ""
                try:
                    import select
                    ready, _, _ = select.select([self._subprocess.stdout], [], [], 10)
                    if ready:
                        first_line = self._subprocess.stdout.readline().strip()
                except Exception:
                    pass

                if success_marker in first_line:
                    self._method = name
                    # Wait until stop event or process exits
                    while not self._stop_event.is_set():
                        if self._subprocess.poll() is not None:
                            break
                        self._stop_event.wait(1.0)
                    return
                else:
                    # This method failed, kill it and try next
                    self._subprocess.terminate()
                    try:
                        self._subprocess.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._subprocess.kill()

            except Exception:
                pass

        # All methods failed
        self._method = "none_available"
