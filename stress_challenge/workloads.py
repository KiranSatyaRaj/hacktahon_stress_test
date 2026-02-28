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

def _cpu_worker(stop_event: multiprocessing.Event, worker_id: int,
                iteration_counter: multiprocessing.Value,
                allowed_cores: list[int] | None = None):
    """
    Single process that performs continuous matrix multiplication.
    Each process pins to its own NumPy computation (no GIL contention).
    Increments a shared counter every iteration so the main process can
    compute throughput (iter/s) and GFLOPS.
    """
    # Pin this worker to specific core(s) if provided
    if allowed_cores:
        try:
            # Round-robin: each worker gets one core from the allowed set
            core = allowed_cores[worker_id % len(allowed_cores)]
            os.sched_setaffinity(0, {core})
        except (OSError, AttributeError):
            pass  # Windows or permission error — fall back to OS scheduling

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

        # Atomically bump the shared counter (lock-free on most platforms)
        with iteration_counter.get_lock():
            iteration_counter.value += 1

        if iteration % 5 == 0:
            if stop_event.is_set():
                break


class CPUWorkload:
    """Manages a pool of CPU stress worker processes."""

    def __init__(self, num_workers: int | None = None,
                 allowed_cores: list[int] | None = None):
        """
        Args:
            num_workers: Number of worker processes. Defaults to cpu_count()
                         or len(allowed_cores) if provided.
            allowed_cores: List of logical core IDs to pin workers to.
                          None = no pinning (use all cores).
        """
        self._allowed_cores = allowed_cores
        if num_workers is not None:
            self.num_workers = num_workers
        elif allowed_cores is not None:
            self.num_workers = len(allowed_cores)
        else:
            self.num_workers = config.CPU_WORKER_COUNT
        self._stop_event = multiprocessing.Event()
        self._processes: list[multiprocessing.Process] = []
        # Shared iteration counter: total iterations across ALL workers
        self._iteration_counter = multiprocessing.Value('L', 0)  # unsigned long
        self._last_count: int = 0
        self._last_ts: float = 0.0

    @property
    def iteration_count(self) -> int:
        return self._iteration_counter.value

    def get_throughput(self) -> float:
        """Return current iterations/sec (aggregate across all workers)."""
        now = time.time()
        current = self._iteration_counter.value
        if self._last_ts == 0.0:
            self._last_ts = now
            self._last_count = current
            return 0.0
        dt = now - self._last_ts
        if dt < 0.5:
            return 0.0  # too early, avoid division noise
        rate = (current - self._last_count) / dt
        self._last_count = current
        self._last_ts = now
        return rate

    def start(self):
        self._stop_event.clear()
        self._iteration_counter.value = 0
        self._last_count = 0
        self._last_ts = 0.0
        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=_cpu_worker,
                args=(self._stop_event, i, self._iteration_counter,
                      self._allowed_cores),
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

# ── Core affinity + priority (set by parent in combined mode) ─────
pin_cores = os.environ.get("GPU_PIN_CORES", "")
if pin_cores:
    try:
        cores = [int(c) for c in pin_cores.split(",") if c.strip()]
        os.sched_setaffinity(0, set(cores))
    except (OSError, AttributeError, ValueError):
        pass
try:
    os.nice(-5)  # elevate priority for CUDA kernel submission
except (OSError, PermissionError):
    pass

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

    # Batch matmul sizing: B batches of MxK @ KxN, float16
    B, M, K = 64, 1024, 1024

    print(
        f"GPU_BUDGET  vram={vram_bytes} budget={budget_bytes}"
        f" ({budget_bytes/1024**3:.2f} GB)  matrix_N={N}"
        f" ({N*N*2/1024**2:.0f} MB per FP16 matrix)"
        f" B={B} M={M} K={K}",
        flush=True,
    )

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

    # Iteration tracking — print "GPU_ITER <count> <elapsed>" every 2s
    # so the parent process can compute throughput and TFLOPS
    iters = 0
    t_start = time.time()
    t_last_report = t_start

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
        iters += 1

        # Report throughput + CUDA memory every ~2 seconds
        now = time.time()
        if now - t_last_report >= 2.0:
            alloc = torch.cuda.memory_allocated(0)
            resrv = torch.cuda.memory_reserved(0)
            frag = resrv - alloc
            print(f"GPU_ITER {iters} {now - t_start:.2f}", flush=True)
            print(f"GPU_MEM alloc={alloc} reserved={resrv} frag={frag}", flush=True)
            t_last_report = now

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

    def __init__(self, combined_mode: bool = False):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._subprocess: subprocess.Popen | None = None
        self._method = "none"
        self._combined_mode = combined_mode
        # Throughput tracking (parsed from subprocess stdout)
        self._gpu_iters: int = 0
        self._gpu_elapsed: float = 0.0
        self._gpu_iter_rate: float = 0.0       # current iter/sec
        self._last_iters: int = 0
        self._last_iter_ts: float = 0.0
        # Matrix sizes from GPU_BUDGET for TFLOPS derivation
        self._matrix_n: int = 0
        self._batch_b: int = 0
        self._batch_m: int = 0
        self._batch_k: int = 0
        self._stdout_reader: threading.Thread | None = None
        # CUDA memory fragmentation (from GPU_MEM stdout)
        self._cuda_alloc: int = 0       # bytes actively allocated
        self._cuda_reserved: int = 0    # bytes reserved by caching allocator
        self._cuda_frag: int = 0        # reserved - allocated = waste

    @property
    def method(self) -> str:
        return self._method

    @property
    def iteration_count(self) -> int:
        return self._gpu_iters

    def get_throughput(self) -> float:
        """Return current GPU iterations/sec."""
        return self._gpu_iter_rate

    def get_tflops(self) -> float:
        """Derive FP16 TFLOPS from current throughput and matrix sizes."""
        rate = self._gpu_iter_rate
        if rate <= 0 or self._matrix_n == 0:
            return 0.0
        N = self._matrix_n
        B, M, K = self._batch_b, self._batch_m, self._batch_k
        # Per iteration: 2× NxN GEMM + 1× batched BxMxK @ BxKxM
        ops_per_iter = 2 * (2 * N * N * N) + B * (2 * M * K * M)
        return rate * ops_per_iter / 1e12

    @property
    def cuda_mem_frag_mb(self) -> float:
        """CUDA memory fragmentation in MB (reserved - allocated)."""
        return self._cuda_frag / (1024 ** 2)

    @property
    def cuda_mem_allocated_mb(self) -> float:
        return self._cuda_alloc / (1024 ** 2)

    @property
    def cuda_mem_reserved_mb(self) -> float:
        return self._cuda_reserved / (1024 ** 2)

    def start(self):
        self._stop_event.clear()
        self._gpu_iters = 0
        self._gpu_elapsed = 0.0
        self._gpu_iter_rate = 0.0
        self._last_iters = 0
        self._last_iter_ts = 0.0
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
        if self._stdout_reader:
            self._stdout_reader.join(timeout=5)
        if self._thread:
            self._thread.join(timeout=15)
        # Reset GPU clocks
        try:
            subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, timeout=5)
            subprocess.run(["nvidia-smi", "-rmc"], capture_output=True, timeout=5)
        except Exception:
            pass

    def _read_subprocess_stdout(self):
        """
        Background thread that reads the subprocess stdout line by line.
        Parses GPU_ITER and GPU_BUDGET lines to update throughput tracking.
        """
        try:
            for line in self._subprocess.stdout:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("GPU_ITER "):
                    # Format: "GPU_ITER <total_iters> <elapsed_secs>"
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            total_iters = int(parts[1])
                            elapsed = float(parts[2])
                            # Compute instantaneous rate
                            now = time.time()
                            if self._last_iter_ts > 0:
                                dt = now - self._last_iter_ts
                                if dt > 0.1:
                                    self._gpu_iter_rate = (total_iters - self._last_iters) / dt
                            self._last_iters = total_iters
                            self._last_iter_ts = now
                            self._gpu_iters = total_iters
                            self._gpu_elapsed = elapsed
                        except (ValueError, IndexError):
                            pass

                elif line.startswith("GPU_BUDGET"):
                    # Parse matrix sizes: "... matrix_N=8192 ... B=32 M=1024 K=1024"
                    import re
                    m = re.search(r'matrix_N=(\d+)', line)
                    if m:
                        self._matrix_n = int(m.group(1))
                    m = re.search(r'B=(\d+)', line)
                    if m:
                        self._batch_b = int(m.group(1))
                    m = re.search(r'M=(\d+)', line)
                    if m:
                        self._batch_m = int(m.group(1))
                    m = re.search(r'K=(\d+)', line)
                    if m:
                        self._batch_k = int(m.group(1))

                elif line.startswith("GPU_MEM "):
                    # Format: "GPU_MEM alloc=X reserved=Y frag=Z"
                    import re
                    m = re.search(r'alloc=(\d+)', line)
                    if m:
                        self._cuda_alloc = int(m.group(1))
                    m = re.search(r'reserved=(\d+)', line)
                    if m:
                        self._cuda_reserved = int(m.group(1))
                    m = re.search(r'frag=(\d+)', line)
                    if m:
                        self._cuda_frag = int(m.group(1))

                if self._stop_event.is_set():
                    break
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
                # Build env with optional core pinning
                env = os.environ.copy()
                if self._combined_mode:
                    env["GPU_PIN_CORES"] = ",".join(
                        str(c) for c in config.GPU_RESERVED_CORES
                    )

                self._subprocess = subprocess.Popen(
                    [interpreter, "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
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

                    # Start background stdout reader to capture GPU_ITER + GPU_BUDGET
                    self._stdout_reader = threading.Thread(
                        target=self._read_subprocess_stdout, daemon=True
                    )
                    self._stdout_reader.start()

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

