# metrics_sampler.py
import time, threading, math, os
from collections import deque
import psutil

try:
    import pynvml as nvml   # provided by nvidia-ml-py
    nvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

class MetricsSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self._t = None
        self._stop = threading.Event()
        self.samples = []
        self.proc = psutil.Process(os.getpid())
        self.disk0 = psutil.disk_io_counters()
        self.net0  = psutil.net_io_counters()
        self.start_time = None
        self.end_time   = None

    def _gpu_snapshot(self):
        if not NVML_OK:
            return dict(gpu_power_w=None, gpu_util=None, gpu_mem_mb=None, gpu_temp=None, gpu_name=None)
        h = nvml.nvmlDeviceGetHandleByIndex(0)
        try:
            pw  = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0           # W
        except: pw = None
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(h).gpu         # %
        except: util = None
        try:
            mem  = nvml.nvmlDeviceGetMemoryInfo(h)
            mem_mb = mem.used / (1024*1024)
        except: mem_mb = None
        try:
            temp = nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
        except: temp = None
        try:
            name = nvml.nvmlDeviceGetName(h).decode() if hasattr(nvml.nvmlDeviceGetName(h), "decode") else str(nvml.nvmlDeviceGetName(h))
        except: name = None
        return dict(gpu_power_w=pw, gpu_util=util, gpu_mem_mb=mem_mb, gpu_temp=temp, gpu_name=name)

    def _loop(self):
        self.start_time = time.time()
        while not self._stop.is_set():
            ts = time.time()
            cpu = psutil.cpu_percent(interval=None)
            rss = self.proc.memory_info().rss / (1024*1024)          # MB
            gpu = self._gpu_snapshot()
            self.samples.append(dict(ts=ts, cpu_util=cpu, rss_mb=rss, **gpu))
            time.sleep(self.interval)
        self.end_time = time.time()

    def start(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join()
        self.disk1 = psutil.disk_io_counters()
        self.net1  = psutil.net_io_counters()

    def summary(self):
        if not self.samples:
            return {}
        dt = (self.end_time or time.time()) - (self.start_time or time.time())
        # integrate energy: sum(power * delta_t)
        energy_j = 0.0
        prev_ts = None
        for s in self.samples:
            if s["gpu_power_w"] is None:
                energy_j = None
                break
            if prev_ts is None:
                prev_ts = s["ts"]
                continue
            energy_j += s["gpu_power_w"] * (s["ts"] - prev_ts)
            prev_ts = s["ts"]
        # simple reductions
        def vals(key): return [s[key] for s in self.samples if s[key] is not None]
        def agg(key, f): 
            xs = vals(key)
            return f(xs) if xs else None
        import statistics as stats
        out = dict(
            wall_time_s = dt,
            cpu_util_mean = agg("cpu_util", stats.mean),
            cpu_util_p95  = agg("cpu_util", lambda xs: sorted(xs)[int(0.95*len(xs))-1]),
            rss_mb_peak   = agg("rss_mb", max),
            gpu_util_mean = agg("gpu_util", stats.mean),
            gpu_util_peak = agg("gpu_util", max),
            gpu_mem_mb_peak = agg("gpu_mem_mb", max),
            gpu_power_mean_w = agg("gpu_power_w", stats.mean),
            gpu_power_peak_w = agg("gpu_power_w", max),
            gpu_temp_peak_c  = agg("gpu_temp", max),
            gpu_name = self.samples[0].get("gpu_name"),
            energy_j = energy_j,                           # None on CPU
            energy_wh = (energy_j / 3600.0) if energy_j is not None else None,
            disk_read_mb = (self.disk1.read_bytes - self.disk0.read_bytes) / (1024*1024),
            disk_write_mb = (self.disk1.write_bytes - self.disk0.write_bytes) / (1024*1024),
            net_recv_mb = (self.net1.bytes_recv - self.net0.bytes_recv) / (1024*1024),
            net_sent_mb = (self.net1.bytes_sent - self.net0.bytes_sent) / (1024*1024),
        )
        return out
