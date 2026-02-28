/* ═══════════════════════════════════════════════════════════════
   STRESS CHALLENGE — LIVE DASHBOARD JavaScript
   Real-time WebSocket metrics, Chart.js charts, animated gauges
   ═══════════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────────────
let ws = null;
let charts = {};
let testRunning = false;
let testCompleted = false;
let testDuration = 3600;
let startTime = 0;
let timerInterval = null;

// Metric history (for charts)
const MAX_POINTS = 500;
const history = {
    labels: [],
    cpuAvg: [],
    cpuTemp: [],
    gpuUtil: [],
    gpuTemp: [],
    gpuPower: [],
    cpuFreq: [],
    gpuClock: [],
    ramPercent: [],
    cpuGflops: [],
    gpuTflops: [],
};

// Running stats
const stats = {
    peakCpuTemp: 0,
    peakGpuTemp: 0,
    cpuSum: 0,
    cpuCount: 0,
    maxGpuPower: 0,
    minCpuFreq: Infinity,
    maxCpuFreq: 0,
    throttleEvents: 0,
};

// ── Init ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initParticles();
    connectWebSocket();
    checkStatus();
});

// ── Background Particles ──────────────────────────────────────
function initParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.width = particle.style.height = (2 + Math.random() * 4) + 'px';
        particle.style.animationDuration = (15 + Math.random() * 25) + 's';
        particle.style.animationDelay = (Math.random() * 20) + 's';
        particle.style.background = `hsla(${230 + Math.random() * 60}, 70%, 60%, ${0.15 + Math.random() * 0.2})`;
        container.appendChild(particle);
    }
}

// ── Chart.js Setup ────────────────────────────────────────────
function initCharts() {
    // Chart.js global defaults
    Chart.defaults.color = '#8888aa';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.elements.point.radius = 0;
    Chart.defaults.elements.point.hoverRadius = 4;
    Chart.defaults.elements.line.tension = 0.35;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.animation.duration = 800;

    const commonScales = {
        x: {
            grid: { display: false },
            ticks: { maxTicksLimit: 8, font: { family: "'JetBrains Mono', monospace", size: 10 } },
        },
        y: {
            grid: { color: 'rgba(255,255,255,0.03)' },
            ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } },
        }
    };

    // CPU Usage Chart
    charts.cpu = new Chart(document.getElementById('cpuChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'CPU Average',
                data: [],
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                fill: true,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                ...commonScales,
                y: { ...commonScales.y, min: 0, max: 100, title: { display: true, text: '%', color: '#555577' } }
            },
            plugins: {
                legend: { display: true, position: 'top' },
            }
        }
    });

    // Temperature Chart
    charts.temp = new Chart(document.getElementById('tempChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CPU Package',
                    data: [],
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.08)',
                    fill: true,
                    borderWidth: 2,
                },
                {
                    label: 'GPU',
                    data: [],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.08)',
                    fill: true,
                    borderWidth: 2,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                ...commonScales,
                y: {
                    ...commonScales.y,
                    title: { display: true, text: '°C', color: '#555577' },
                    suggestedMin: 30,
                }
            },
            plugins: {
                legend: { display: true, position: 'top' },
                annotation: {
                    annotations: {
                        cpuThrottle: {
                            type: 'line',
                            yMin: 95,
                            yMax: 95,
                            borderColor: 'rgba(239, 68, 68, 0.4)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: { content: 'CPU Throttle', display: true, position: 'start', font: { size: 9 }, color: '#ef4444', backgroundColor: 'transparent' }
                        }
                    }
                }
            }
        }
    });

    // GPU Metrics Chart
    charts.gpu = new Chart(document.getElementById('gpuChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'GPU Util %',
                    data: [],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.08)',
                    fill: true,
                    borderWidth: 2,
                    yAxisID: 'y',
                },
                {
                    label: 'Power (W)',
                    data: [],
                    borderColor: '#ec4899',
                    borderWidth: 2,
                    borderDash: [4, 4],
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: commonScales.x,
                y: {
                    ...commonScales.y,
                    position: 'left',
                    min: 0,
                    max: 100,
                    title: { display: true, text: '%', color: '#555577' }
                },
                y1: {
                    ...commonScales.y,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'W', color: '#555577' },
                    suggestedMin: 0,
                }
            },
            plugins: { legend: { display: true, position: 'top' } }
        }
    });

    // Frequency & Power Chart
    charts.freq = new Chart(document.getElementById('freqChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CPU Freq (MHz)',
                    data: [],
                    borderColor: '#06b6d4',
                    borderWidth: 2,
                    yAxisID: 'y',
                },
                {
                    label: 'GPU SM Clock (MHz)',
                    data: [],
                    borderColor: '#a855f7',
                    borderWidth: 2,
                    borderDash: [4, 4],
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: commonScales.x,
                y: {
                    ...commonScales.y,
                    position: 'left',
                    title: { display: true, text: 'MHz (CPU)', color: '#555577' },
                    suggestedMin: 800,
                },
                y1: {
                    ...commonScales.y,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'MHz (GPU)', color: '#555577' },
                    suggestedMin: 0,
                }
            },
            plugins: { legend: { display: true, position: 'top' } }
        }
    });

    // Performance Throughput Chart
    charts.perf = new Chart(document.getElementById('perfChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CPU GFLOPS',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.08)',
                    fill: true,
                    borderWidth: 2,
                    yAxisID: 'y',
                },
                {
                    label: 'GPU TFLOPS',
                    data: [],
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.08)',
                    fill: true,
                    borderWidth: 2,
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: commonScales.x,
                y: {
                    ...commonScales.y,
                    position: 'left',
                    title: { display: true, text: 'GFLOPS', color: '#555577' },
                    suggestedMin: 0,
                },
                y1: {
                    ...commonScales.y,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'TFLOPS', color: '#555577' },
                    suggestedMin: 0,
                }
            },
            plugins: { legend: { display: true, position: 'top' } }
        }
    });
}

// ── WebSocket ─────────────────────────────────────────────────
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => console.log('🔌 WebSocket connected');
    ws.onclose = () => {
        console.log('🔌 WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 2000);
    };
    ws.onerror = (e) => console.error('WebSocket error:', e);

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMetricUpdate(data);
    };
}

// ── Metric Handler ────────────────────────────────────────────
function handleMetricUpdate(data) {
    // Update running state
    testRunning = data.test_running;
    testCompleted = data.test_completed;
    testDuration = data.test_duration || 3600;

    updateStatusBadge();
    updateButtons();

    if (!testRunning && !testCompleted) return;

    const elapsed = data.elapsed_seconds || 0;
    const label = formatTime(elapsed);

    // Push to history
    pushHistory('labels', label);
    pushHistory('cpuAvg', data.cpu_avg_percent || 0);
    pushHistory('cpuTemp', data.cpu_temp_package || 0);
    pushHistory('gpuUtil', data.gpu_util_percent || 0);
    pushHistory('gpuTemp', data.gpu_temp_c || 0);
    pushHistory('gpuPower', data.gpu_power_w || 0);
    pushHistory('cpuFreq', data.cpu_freq_current_mhz || 0);
    pushHistory('gpuClock', data.gpu_clock_sm_mhz || 0);
    pushHistory('ramPercent', data.ram_percent || 0);
    pushHistory('cpuGflops', data.cpu_gflops || 0);
    pushHistory('gpuTflops', data.gpu_tflops || 0);

    // Update gauges
    updateGauge('cpuGaugeFill', 'cpuGaugeValue', data.cpu_avg_percent || 0, 100);
    updateGauge('gpuGaugeFill', 'gpuGaugeValue', data.gpu_util_percent || 0, 100);
    updateGauge('ramGaugeFill', 'ramGaugeValue', data.ram_percent || 0, 100);
    updateTempGauge(data.cpu_temp_package || 0);

    // Gauge details
    document.getElementById('cpuGaugeDetail').textContent = `${data.cpu_freq_current_mhz || 0} MHz`;
    document.getElementById('gpuGaugeDetail').textContent = `${data.gpu_temp_c || 0}°C | ${data.gpu_power_w || 0}W`;
    document.getElementById('ramGaugeDetail').textContent = `${data.ram_used_gb || 0} / ${data.ram_total_gb || 0} GB`;
    document.getElementById('cpuTempDetail').textContent = `Package ${data.cpu_temp_package || 0}°C`;

    // Update charts
    updateChart(charts.cpu, history.labels, [history.cpuAvg]);
    updateChart(charts.temp, history.labels, [history.cpuTemp, history.gpuTemp]);
    updateChart(charts.gpu, history.labels, [history.gpuUtil, history.gpuPower]);
    updateChart(charts.freq, history.labels, [history.cpuFreq, history.gpuClock]);
    updateChart(charts.perf, history.labels, [history.cpuGflops, history.gpuTflops]);

    // Update progress
    updateProgress(elapsed, testDuration);

    // Update stats
    updateStats(data);
}

// ── Gauge Helpers ─────────────────────────────────────────────
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52; // ~326.73

function updateGauge(fillId, valueId, value, max) {
    const fill = document.getElementById(fillId);
    const valueEl = document.getElementById(valueId);
    const fraction = Math.min(value / max, 1);
    const offset = GAUGE_CIRCUMFERENCE * (1 - fraction);
    fill.style.strokeDashoffset = offset;
    valueEl.innerHTML = `${Math.round(value)}<span class="gauge-unit">${valueId.includes('Temp') ? '°C' : '%'}</span>`;
}

function updateTempGauge(temp) {
    const fill = document.getElementById('cpuTempGaugeFill');
    const valueEl = document.getElementById('cpuTempGaugeValue');
    const fraction = Math.min(temp / 105, 1);
    const offset = GAUGE_CIRCUMFERENCE * (1 - fraction);
    fill.style.strokeDashoffset = offset;
    valueEl.innerHTML = `${Math.round(temp)}<span class="gauge-unit">°C</span>`;

    // Color based on temperature
    fill.classList.remove('hot', 'warm');
    if (temp >= 90) {
        fill.classList.add('hot');
    } else if (temp >= 75) {
        fill.classList.add('warm');
    }
}

// ── Chart Update ──────────────────────────────────────────────
function updateChart(chart, labels, datasets) {
    chart.data.labels = labels;
    datasets.forEach((d, i) => {
        if (chart.data.datasets[i]) {
            chart.data.datasets[i].data = d;
        }
    });
    chart.update('none'); // no animation for smooth updates
}

function pushHistory(key, value) {
    history[key].push(value);
    if (history[key].length > MAX_POINTS) {
        history[key].shift();
    }
}

// ── Progress Bar ──────────────────────────────────────────────
function updateProgress(elapsed, duration) {
    const container = document.getElementById('progressContainer');
    const fill = document.getElementById('progressFill');
    const glow = document.getElementById('progressGlow');
    const percentEl = document.getElementById('progressPercent');
    const etaEl = document.getElementById('progressEta');

    if (testRunning) {
        container.classList.add('visible');
    }

    const pct = Math.min((elapsed / duration) * 100, 100);
    fill.style.width = pct + '%';
    glow.style.width = pct + '%';
    percentEl.textContent = pct.toFixed(1) + '%';

    const remaining = Math.max(duration - elapsed, 0);
    etaEl.textContent = 'ETA: ' + formatTime(remaining);
}

// ── Stats ─────────────────────────────────────────────────────
function updateStats(data) {
    // Peak CPU temp
    if (data.cpu_temp_package > stats.peakCpuTemp) {
        stats.peakCpuTemp = data.cpu_temp_package;
        document.getElementById('statPeakCpuTemp').textContent = stats.peakCpuTemp.toFixed(0) + '°C';
    }

    // Peak GPU temp
    if (data.gpu_temp_c > stats.peakGpuTemp) {
        stats.peakGpuTemp = data.gpu_temp_c;
        document.getElementById('statPeakGpuTemp').textContent = stats.peakGpuTemp.toFixed(0) + '°C';
    }

    // Average CPU
    stats.cpuSum += data.cpu_avg_percent || 0;
    stats.cpuCount++;
    const avgCpu = stats.cpuSum / stats.cpuCount;
    // This is the running mean across the whole test, not the current reading.
    // Current reading is shown in the CPU gauge above.
    document.getElementById('statAvgCpu').textContent = avgCpu.toFixed(1) + '% avg';

    // Max GPU power
    if (data.gpu_power_w > stats.maxGpuPower) {
        stats.maxGpuPower = data.gpu_power_w;
        document.getElementById('statMaxGpuPower').textContent = stats.maxGpuPower.toFixed(1) + 'W';
    }

    // CPU freq range
    const freq = data.cpu_freq_current_mhz || 0;
    if (freq > 0) {
        if (freq < stats.minCpuFreq) stats.minCpuFreq = freq;
        if (freq > stats.maxCpuFreq) stats.maxCpuFreq = freq;
        if (stats.minCpuFreq < Infinity) {
            document.getElementById('statCpuFreqRange').textContent =
                `${(stats.minCpuFreq / 1000).toFixed(1)}-${(stats.maxCpuFreq / 1000).toFixed(1)}G`;
        }
    }

    // Throttle events (CPU temp >= 95 or freq drop > 20%)
    if (data.cpu_temp_package >= 95) {
        stats.throttleEvents++;
    }

    // GPU perf state
    const gpuPs = data.gpu_perf_state;
    if (gpuPs !== undefined && gpuPs >= 0) {
        const el = document.getElementById('statGpuPerfState');
        el.textContent = 'P' + gpuPs;
        el.style.color = gpuPs === 0 ? '#22c55e' : gpuPs <= 2 ? '#eab308' : '#ef4444';
    }

    // GPU throttle reason
    const throttle = data.gpu_throttle_reasons || '';
    if (throttle) {
        const el = document.getElementById('statGpuThrottle');
        el.textContent = throttle === 'none' ? '✓ None' : throttle;
        el.style.color = throttle === 'none' ? '#22c55e' : '#ef4444';
    }

    // CPU GFLOPS (current)
    if (data.cpu_gflops > 0) {
        document.getElementById('statCpuGflops').textContent = data.cpu_gflops.toFixed(1);
    }

    // GPU TFLOPS (current)
    if (data.gpu_tflops > 0) {
        document.getElementById('statGpuTflops').textContent = data.gpu_tflops.toFixed(2);
    }

    // CUDA memory fragmentation
    if (data.cuda_mem_frag_mb !== undefined && data.cuda_mem_frag_mb > 0) {
        document.getElementById('statCudaFrag').textContent = data.cuda_mem_frag_mb.toFixed(0) + ' MB';
    }

    // CPU user/kernel time
    if (data.cpu_user_pct > 0) {
        document.getElementById('statCpuTime').textContent =
            data.cpu_user_pct.toFixed(0) + '% / ' + (data.cpu_kernel_pct || 0).toFixed(0) + '%';
    }

    // Controller Status
    const ctrlLevel = data.controller_level || '';
    if (ctrlLevel) {
        const el = document.getElementById('statController');
        const risk = (data.controller_risk || 0).toFixed(2);
        el.textContent = `${ctrlLevel} (Risk: ${risk})`;
        el.style.color = ctrlLevel === 'SAFE' ? '#22c55e' : ctrlLevel === 'WARNING' ? '#eab308' : '#ef4444';
    }
}

// ── UI State ──────────────────────────────────────────────────
function updateStatusBadge() {
    const badge = document.getElementById('statusBadge');
    const text = badge.querySelector('.status-text');

    badge.classList.remove('running', 'completed');

    if (testRunning) {
        badge.classList.add('running');
        text.textContent = 'RUNNING';
    } else if (testCompleted) {
        badge.classList.add('completed');
        text.textContent = 'COMPLETED';
    } else {
        text.textContent = 'IDLE';
    }
}

function updateButtons() {
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    const btnReport = document.getElementById('btnReport');

    btnStart.disabled = testRunning;
    btnStop.disabled = !testRunning;
    btnReport.disabled = !testCompleted;
}

// ── API Calls ─────────────────────────────────────────────────
async function startTest() {
    try {
        const res = await fetch('/api/start', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'started') {
            testRunning = true;
            startTime = Date.now();
            startTimer();
            updateStatusBadge();
            updateButtons();
            document.getElementById('progressContainer').classList.add('visible');

            // Reset stats
            Object.keys(stats).forEach(k => {
                if (k === 'minCpuFreq') stats[k] = Infinity;
                else stats[k] = 0;
            });

            // Reset history
            Object.keys(history).forEach(k => history[k] = []);
        }
    } catch (e) {
        console.error('Failed to start test:', e);
    }
}

async function stopTest() {
    try {
        const res = await fetch('/api/stop', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'stopped') {
            testRunning = false;
            testCompleted = true;
            stopTimer();
            updateStatusBadge();
            updateButtons();
        }
    } catch (e) {
        console.error('Failed to stop test:', e);
    }
}

async function viewReport() {
    try {
        const res = await fetch('/api/report');
        const data = await res.json();
        if (data.report) {
            document.getElementById('reportText').textContent = data.report;
            document.getElementById('reportModal').classList.add('visible');
        } else {
            alert(data.error || 'Report not available');
        }
    } catch (e) {
        console.error('Failed to fetch report:', e);
    }
}

function closeReport() {
    document.getElementById('reportModal').classList.remove('visible');
}

async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        testRunning = data.running;
        testCompleted = data.completed;
        testDuration = data.duration || 3600;
        updateStatusBadge();
        updateButtons();

        if (testRunning) {
            startTime = Date.now() - (data.elapsed * 1000);
            startTimer();
            document.getElementById('progressContainer').classList.add('visible');

            // Fetch existing history
            const histRes = await fetch('/api/history');
            const histData = await histRes.json();
            if (histData.snapshots && histData.snapshots.length > 0) {
                histData.snapshots.forEach(s => {
                    pushHistory('labels', formatTime(s.elapsed_seconds));
                    pushHistory('cpuAvg', s.cpu_avg_percent || 0);
                    pushHistory('cpuTemp', s.cpu_temp_package || 0);
                    pushHistory('gpuUtil', s.gpu_util_percent || 0);
                    pushHistory('gpuTemp', s.gpu_temp_c || 0);
                    pushHistory('gpuPower', s.gpu_power_w || 0);
                    pushHistory('cpuFreq', s.cpu_freq_current_mhz || 0);
                    pushHistory('gpuClock', s.gpu_clock_sm_mhz || 0);
                    pushHistory('ramPercent', s.ram_percent || 0);
                    pushHistory('cpuGflops', s.cpu_gflops || 0);
                    pushHistory('gpuTflops', s.gpu_tflops || 0);
                    updateStats(s);
                });

                updateChart(charts.cpu, history.labels, [history.cpuAvg]);
                updateChart(charts.temp, history.labels, [history.cpuTemp, history.gpuTemp]);
                updateChart(charts.gpu, history.labels, [history.gpuUtil, history.gpuPower]);
                updateChart(charts.freq, history.labels, [history.cpuFreq, history.gpuClock]);
                updateChart(charts.perf, history.labels, [history.cpuGflops, history.gpuTflops]);
            }
        }
    } catch (e) {
        console.error('Failed to check status:', e);
    }
}

// ── Timer ─────────────────────────────────────────────────────
function startTimer() {
    stopTimer();
    timerInterval = setInterval(() => {
        if (!testRunning) { stopTimer(); return; }
        const elapsed = (Date.now() - startTime) / 1000;
        document.getElementById('timer').textContent = formatTime(elapsed);
    }, 250);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// ── Helpers ───────────────────────────────────────────────────
function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

function pad(n) { return n.toString().padStart(2, '0'); }

// ── Keyboard shortcuts ────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeReport();
});
