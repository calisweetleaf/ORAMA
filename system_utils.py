import os
import sys
import time
import json
import uuid
import shutil
import asyncio
import logging
import tempfile
import traceback
import threading
import signal
import socket
import hashlib
import multiprocessing
from enum import Enum, auto
from functools import lru_cache, wraps
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable, TypeVar, Generic, Coroutine, Generator, Iterator
from dataclasses import dataclass, field, asdict, fields
from contextlib import contextmanager
from collections import deque

# Import psutil for system monitoring if available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import platform-specific modules
if sys.platform == "win32":
    try:
        import win32api
        import win32con
        import win32process
        import win32security
        import win32service
        import winerror
        import pywintypes
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
        
    try:
        import winreg
        WINREG_AVAILABLE = True
    except ImportError:
        WINREG_AVAILABLE = False
else:
    WIN32_AVAILABLE = False
    WINREG_AVAILABLE = False

# Import aiofiles for async file operations if available
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# Import numpy for advanced metrics if available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Constants
METRIC_HISTORY_SIZE = 3600  # 1 hour at 1 sample/second
DEFAULT_MONITORING_INTERVAL = 1.0  # seconds
DEFAULT_ALERT_THRESHOLD_CPU = 90.0  # percentage
DEFAULT_ALERT_THRESHOLD_MEMORY = 85.0  # percentage
DEFAULT_ALERT_THRESHOLD_DISK = 90.0  # percentage
LOG_ROTATION_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Resource types
class ResourceType(Enum):
    CPU = auto()
    MEMORY = auto()
    GPU = auto()
    DISK = auto()
    NETWORK = auto()
    SYSTEM = auto()

# Security levels
class SecurityLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

# Log levels mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available: int = 0  # bytes
    memory_total: int = 0  # bytes
    disk_percent: Dict[str, float] = field(default_factory=dict)
    disk_io_read: int = 0  # bytes/sec
    disk_io_write: int = 0  # bytes/sec
    network_sent: int = 0  # bytes/sec
    network_received: int = 0  # bytes/sec
    process_count: int = 0
    thread_count: int = 0
    handle_count: int = 0
    boot_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemMetrics':
        """Create metrics from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})

@dataclass
class ProcessMetrics:
    """Process-specific performance metrics."""
    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss: int = 0  # bytes
    io_read: int = 0  # bytes/sec
    io_write: int = 0  # bytes/sec
    open_files: int = 0
    open_connections: int = 0
    threads: int = 0
    status: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessMetrics':
        """Create metrics from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})

@dataclass
class ResourceUsage:
    """Resource usage information."""
    resource_type: ResourceType
    current: float
    limit: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def percentage(self) -> float:
        """Get percentage of resource used."""
        if self.limit <= 0:
            return 0.0
        return (self.current / self.limit) * 100.0
    
    def to_dict(self) -> Dict:
        """Convert resource usage to dictionary."""
        data = asdict(self)
        data["resource_type"] = self.resource_type.name
        data["percentage"] = self.percentage
        return data

@dataclass
class Alert:
    """System alert information."""
    resource_type: ResourceType
    level: SecurityLevel
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        data = asdict(self)
        data["resource_type"] = self.resource_type.name
        data["level"] = self.level.name
        return data

class MetricsBuffer:
    """Buffer for storing time-series metrics with efficient querying."""
    
    def __init__(self, max_size: int = METRIC_HISTORY_SIZE):
        """Initialize metrics buffer."""
        self.max_size = max_size
        self.buffer: Dict[str, deque] = {}
        self.lock = threading.RLock()
    
    def add(self, metric_name: str, value: float, timestamp: Optional[float] = None) -> None:
        """Add a metric value to the buffer."""
        with self.lock:
            if metric_name not in self.buffer:
                self.buffer[metric_name] = deque(maxlen=self.max_size)
            
            ts = timestamp or time.time()
            self.buffer[metric_name].append((ts, value))
    
    def get_latest(self, metric_name: str) -> Optional[Tuple[float, float]]:
        """Get the latest value for a metric."""
        with self.lock:
            if metric_name not in self.buffer or not self.buffer[metric_name]:
                return None
            
            return self.buffer[metric_name][-1]
    
    def get_range(self, metric_name: str, 
                 start_time: Optional[float] = None, 
                 end_time: Optional[float] = None) -> List[Tuple[float, float]]:
        """Get metrics within a time range."""
        with self.lock:
            if metric_name not in self.buffer:
                return []
            
            # Set default time range if not provided
            if end_time is None:
                end_time = time.time()
            if start_time is None:
                start_time = end_time - 3600  # Default to last hour
            
            # Filter metrics by time range
            return [
                (ts, value) for ts, value in self.buffer[metric_name]
                if start_time <= ts <= end_time
            ]
    
    def get_statistics(self, metric_name: str, 
                      start_time: Optional[float] = None, 
                      end_time: Optional[float] = None) -> Dict[str, float]:
        """Get statistical information for a metric in the given time range."""
        data = [value for _, value in self.get_range(metric_name, start_time, end_time)]
        
        if not data:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0
            }
        
        if NUMPY_AVAILABLE:
            # Use numpy for accurate statistics
            array = np.array(data)
            return {
                "count": len(data),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "std": float(np.std(array)),
                "percentile_95": float(np.percentile(array, 95)),
                "percentile_99": float(np.percentile(array, 99))
            }
        else:
            # Fallback to basic statistics
            data_sorted = sorted(data)
            return {
                "count": len(data),
                "min": min(data),
                "max": max(data),
                "mean": sum(data) / len(data),
                "median": data_sorted[len(data_sorted) // 2]
            }
    
    def get_metric_names(self) -> List[str]:
        """Get all metric names in the buffer."""
        with self.lock:
            return list(self.buffer.keys())
    
    def clear(self, metric_name: Optional[str] = None) -> None:
        """Clear metrics buffer for a specific metric or all metrics."""
        with self.lock:
            if metric_name:
                if metric_name in self.buffer:
                    self.buffer[metric_name].clear()
            else:
                for name in self.buffer:
                    self.buffer[name].clear()
    
    def to_dict(self) -> Dict[str, List[Tuple[float, float]]]:
        """Convert buffer to dictionary."""
        with self.lock:
            return {name: list(values) for name, values in self.buffer.items()}

class SystemMonitor:
    """
    System monitoring and metrics collection.
    
    Provides:
    - CPU, memory, disk, and network usage monitoring
    - Process-specific metrics collection
    - Alerting for threshold violations
    - Historical metrics with statistical analysis
    - Resource limitation enforcement
    """
    
    def __init__(self, config: Dict, logger=None):
        """Initialize system monitor with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger("orama.system.monitor")
        
        # Monitoring configuration
        self.interval = config.get("interval", DEFAULT_MONITORING_INTERVAL)
        self.watch_processes = config.get("watch_processes", True)
        self.watch_network = config.get("watch_network", True)
        self.watch_filesystem = config.get("watch_filesystem", True)
        self.watch_hardware = config.get("watch_hardware", True)
        
        # Alert thresholds
        self.thresholds = config.get("alert_thresholds", {
            "cpu_percent": DEFAULT_ALERT_THRESHOLD_CPU,
            "memory_percent": DEFAULT_ALERT_THRESHOLD_MEMORY,
            "disk_percent": DEFAULT_ALERT_THRESHOLD_DISK
        })
        
        # Metrics storage
        self.metrics_buffer = MetricsBuffer(config.get("metrics_history", METRIC_HISTORY_SIZE))
        self.system_metrics = SystemMetrics()
        self.process_metrics: Dict[int, ProcessMetrics] = {}
        self.disk_io_prev = (0, 0)  # (read_bytes, write_bytes)
        self.net_io_prev = (0, 0)  # (bytes_sent, bytes_recv)
        self.prev_time = time.time()
        
        # Alert tracking
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.max_alert_history = config.get("max_alert_history", 100)
        
        # State tracking
        self._running = False
        self._monitor_task = None
        
        # Update timestamp
        self.last_update = 0.0
        
        # Resource paths to monitor
        self.monitored_paths = config.get("monitored_paths", [])
        if not self.monitored_paths and sys.platform == "win32":
            # Default to system drives on Windows
            drives = [f"{d}:\\" for d in "CDEF" if os.path.exists(f"{d}:\\")]
            self.monitored_paths = drives
        elif not self.monitored_paths:
            # Default to root and home on Unix-like systems
            self.monitored_paths = ["/", os.path.expanduser("~")]
        
        # File system watching
        self.watched_directories = config.get("watch_directories", [])
        self.file_states: Dict[str, Dict[str, float]] = {}
        
        # Process watching
        self.watched_processes = config.get("watch_processes", [])
        
        self.logger.info("System monitor initialized")
    
    async def start(self) -> None:
        """Start monitoring system resources."""
        self.logger.info("Starting system monitor...")
        self._running = True
        
        try:
            # Initial system check
            await self._update_metrics()
            
            # Start monitoring task
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("System monitor started successfully")
        except Exception as e:
            self._running = False
            self.logger.error(f"Failed to start system monitor: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop monitoring system resources."""
        self.logger.info("Stopping system monitor...")
        self._running = False
        
        try:
            # Cancel monitoring task
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("System monitor stopped")
        except Exception as e:
            self.logger.error(f"Error during system monitor shutdown: {e}", exc_info=True)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop to collect system metrics periodically."""
        self.logger.info("Starting monitoring loop")
        
        while self._running:
            try:
                # Update metrics
                await self._update_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.interval)
                
            except asyncio.CancelledError:
                self.logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.interval * 2)  # Longer sleep on error
    
    async def _update_metrics(self) -> None:
        """Update system and process metrics."""
        if not PSUTIL_AVAILABLE:
            # No psutil, use basic metrics
            await self._update_basic_metrics()
            return
            
        try:
            current_time = time.time()
            time_diff = current_time - self.prev_time
            
            # Update system metrics
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics_buffer.add("cpu_percent", cpu_percent)
            
            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            self.metrics_buffer.add("memory_percent", memory_percent)
            self.metrics_buffer.add("memory_available", memory_available)
            
            # Disk usage
            disk_percent = {}
            for path in self.monitored_paths:
                try:
                    usage = psutil.disk_usage(path)
                    disk_percent[path] = usage.percent
                    self.metrics_buffer.add(f"disk_percent_{path}", usage.percent)
                except (FileNotFoundError, PermissionError):
                    pass
            
            # Disk I/O
            if time_diff > 0:
                try:
                    disk_io = psutil.disk_io_counters()
                    if disk_io and self.disk_io_prev != (0, 0):
                        read_diff = disk_io.read_bytes - self.disk_io_prev[0]
                        write_diff = disk_io.write_bytes - self.disk_io_prev[1]
                        
                        disk_io_read = read_diff / time_diff
                        disk_io_write = write_diff / time_diff
                        
                        self.metrics_buffer.add("disk_io_read", disk_io_read)
                        self.metrics_buffer.add("disk_io_write", disk_io_write)
                    
                    if disk_io:
                        self.disk_io_prev = (disk_io.read_bytes, disk_io.write_bytes)
                except (AttributeError, FileNotFoundError):
                    pass
            
            # Network I/O
            if self.watch_network and time_diff > 0:
                try:
                    net_io = psutil.net_io_counters()
                    if net_io and self.net_io_prev != (0, 0):
                        sent_diff = net_io.bytes_sent - self.net_io_prev[0]
                        recv_diff = net_io.bytes_recv - self.net_io_prev[1]
                        
                        net_sent = sent_diff / time_diff
                        net_recv = recv_diff / time_diff
                        
                        self.metrics_buffer.add("network_sent", net_sent)
                        self.metrics_buffer.add("network_received", net_recv)
                    
                    if net_io:
                        self.net_io_prev = (net_io.bytes_sent, net_io.bytes_recv)
                except (AttributeError, FileNotFoundError):
                    pass
            
            # Process information
            process_count = len(psutil.pids())
            self.metrics_buffer.add("process_count", process_count)
            
            # System-wide metrics
            try:
                boot_time = psutil.boot_time()
                thread_count = 0
                handle_count = 0
                
                # Watch specific processes if configured
                if self.watch_processes:
                    # Clear old process metrics
                    self.process_metrics = {}
                    
                    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 
                                                    'memory_info', 'num_threads', 'status']):
                        try:
                            # Skip processes not in watched list if specified
                            if self.watched_processes and proc.info['name'] not in self.watched_processes:
                                continue
                                
                            pid = proc.info['pid']
                            
                            # Get detailed process info
                            with proc.oneshot():
                                name = proc.info['name']
                                cpu_percent = proc.info['cpu_percent'] or 0.0
                                memory_percent = proc.info['memory_percent'] or 0.0
                                memory_rss = proc.info['memory_info'].rss if proc.info['memory_info'] else 0
                                threads = proc.info['num_threads'] or 0
                                status = proc.info['status'] or "unknown"
                                
                                try:
                                    io_counters = proc.io_counters()
                                    io_read = io_counters.read_bytes if io_counters else 0
                                    io_write = io_counters.write_bytes if io_counters else 0
                                except (psutil.AccessDenied, AttributeError):
                                    io_read = 0
                                    io_write = 0
                                
                                try:
                                    open_files = len(proc.open_files())
                                except (psutil.AccessDenied, AttributeError):
                                    open_files = 0
                                
                                try:
                                    open_connections = len(proc.connections())
                                except (psutil.AccessDenied, AttributeError):
                                    open_connections = 0
                            
                            # Update process metrics
                            self.process_metrics[pid] = ProcessMetrics(
                                pid=pid,
                                name=name,
                                cpu_percent=cpu_percent,
                                memory_percent=memory_percent,
                                memory_rss=memory_rss,
                                io_read=io_read,
                                io_write=io_write,
                                open_files=open_files,
                                open_connections=open_connections,
                                threads=threads,
                                status=status,
                                timestamp=current_time
                            )
                            
                            # Update thread count
                            thread_count += threads
                            
                            # Add process-specific metrics to buffer
                            self.metrics_buffer.add(f"process_{pid}_cpu", cpu_percent)
                            self.metrics_buffer.add(f"process_{pid}_memory", memory_percent)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            pass
                
                # Add thread count to metrics
                self.metrics_buffer.add("thread_count", thread_count)
                
                # Get handle count on Windows
                if sys.platform == "win32" and WIN32_AVAILABLE:
                    try:
                        # This is Windows-specific
                        import ctypes
                        handle_count = ctypes.windll.kernel32.GetProcessHandleCount(ctypes.c_ulong(-1), ctypes.byref(ctypes.c_ulong()))
                        self.metrics_buffer.add("handle_count", handle_count)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning(f"Error collecting process metrics: {e}")
            
            # Check file system changes if configured
            if self.watch_filesystem and self.watched_directories:
                await self._check_file_changes()
            
            # Update system metrics object
            self.system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                memory_total=memory_total,
                disk_percent=disk_percent,
                disk_io_read=self.metrics_buffer.get_latest("disk_io_read")[1] if self.metrics_buffer.get_latest("disk_io_read") else 0,
                disk_io_write=self.metrics_buffer.get_latest("disk_io_write")[1] if self.metrics_buffer.get_latest("disk_io_write") else 0,
                network_sent=self.metrics_buffer.get_latest("network_sent")[1] if self.metrics_buffer.get_latest("network_sent") else 0,
                network_received=self.metrics_buffer.get_latest("network_received")[1] if self.metrics_buffer.get_latest("network_received") else 0,
                process_count=process_count,
                thread_count=thread_count,
                handle_count=handle_count,
                boot_time=boot_time,
                timestamp=current_time
            )
            
            # Check for alerts
            await self._check_alerts()
            
            # Update timestamps
            self.prev_time = current_time
            self.last_update = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}", exc_info=True)
    
    async def _update_basic_metrics(self) -> None:
        """Update basic system metrics without psutil."""
        try:
            current_time = time.time()
            
            # Basic CPU usage (platform dependent)
            cpu_percent = 0.0
            if sys.platform == "win32" and WIN32_AVAILABLE:
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    idle_time = ctypes.c_ulonglong()
                    kernel_time = ctypes.c_ulonglong()
                    user_time = ctypes.c_ulonglong()
                    
                    kernel32.GetSystemTimes(ctypes.byref(idle_time), ctypes.byref(kernel_time), ctypes.byref(user_time))
                    
                    idle = idle_time.value
                    total = kernel_time.value + user_time.value
                    
                    if hasattr(self, 'prev_idle') and hasattr(self, 'prev_total'):
                        idle_diff = idle - self.prev_idle
                        total_diff = total - self.prev_total
                        if total_diff > 0:
                            cpu_percent = 100.0 * (1.0 - idle_diff / total_diff)
                    
                    self.prev_idle = idle
                    self.prev_total = total
                except Exception as e:
                    self.logger.warning(f"Error getting CPU usage: {e}")
            
            # Basic memory usage
            memory_percent = 0.0
            memory_available = 0
            memory_total = 0
            
            if sys.platform == "win32" and WIN32_AVAILABLE:
                try:
                    import ctypes
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                        
                    memory_info = MEMORYSTATUSEX()
                    memory_info.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_info))
                    
                    memory_percent = memory_info.dwMemoryLoad
                    memory_total = memory_info.ullTotalPhys
                    memory_available = memory_info.ullAvailPhys
                except Exception as e:
                    self.logger.warning(f"Error getting memory info: {e}")
            else:
                # Try to parse from /proc/meminfo on Linux
                try:
                    if os.path.exists("/proc/meminfo"):
                        mem_info = {}
                        with open("/proc/meminfo", "r") as f:
                            for line in f:
                                fields = line.split()
                                if len(fields) >= 3:
                                    key, value = fields[0].rstrip(':'), int(fields[1])
                                    mem_info[key] = value
                        
                        if "MemTotal" in mem_info and "MemFree" in mem_info:
                            memory_total = mem_info["MemTotal"] * 1024  # Convert from KB to bytes
                            memory_free = mem_info["MemFree"] * 1024
                            memory_available = memory_free
                            if "MemAvailable" in mem_info:
                                memory_available = mem_info["MemAvailable"] * 1024
                            
                            memory_percent = 100.0 * (1.0 - memory_available / memory_total)
                except Exception as e:
                    self.logger.warning(f"Error reading /proc/meminfo: {e}")
            
            # Basic disk usage
            disk_percent = {}
            for path in self.monitored_paths:
                try:
                    # This is cross-platform
                    usage = shutil.disk_usage(path)
                    percent = 100.0 * (usage.used / usage.total) if usage.total > 0 else 0.0
                    disk_percent[path] = percent
                    
                    self.metrics_buffer.add(f"disk_percent_{path}", percent)
                except (FileNotFoundError, PermissionError):
                    pass
            
            # Update system metrics object
            self.system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                memory_total=memory_total,
                disk_percent=disk_percent,
                timestamp=current_time
            )
            
            # Add to metrics buffer
            self.metrics_buffer.add("cpu_percent", cpu_percent)
            self.metrics_buffer.add("memory_percent", memory_percent)
            self.metrics_buffer.add("memory_available", memory_available)
            
            # Check for alerts
            await self._check_alerts()
            
            # Update timestamps
            self.prev_time = current_time
            self.last_update = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating basic metrics: {e}", exc_info=True)
    
    async def _check_file_changes(self) -> None:
        """Check for file system changes in watched directories."""
        for directory in self.watched_directories:
            try:
                # Skip if directory doesn't exist
                if not os.path.exists(directory) or not os.path.isdir(directory):
                    continue
                    
                # Get current file states
                current_state = {}
                for entry in os.scandir(directory):
                    try:
                        current_state[entry.name] = entry.stat().st_mtime
                    except (FileNotFoundError, PermissionError):
                        pass
                
                # Compare with previous state
                if directory in self.file_states:
                    prev_state = self.file_states[directory]
                    
                    # Check for new or modified files
                    for name, mtime in current_state.items():
                        if name not in prev_state:
                            # New file
                            self.logger.debug(f"New file detected: {os.path.join(directory, name)}")
                        elif mtime > prev_state[name]:
                            # Modified file
                            self.logger.debug(f"File modified: {os.path.join(directory, name)}")
                    
                    # Check for deleted files
                    for name in prev_state:
                        if name not in current_state:
                            # Deleted file
                            self.logger.debug(f"File deleted: {os.path.join(directory, name)}")
                
                # Update file states
                self.file_states[directory] = current_state
                
            except Exception as e:
                self.logger.warning(f"Error checking directory {directory}: {e}")
    
    async def _check_alerts(self) -> None:
        """Check for resource usage alerts."""
        # CPU usage alert
        if self.system_metrics.cpu_percent >= self.thresholds.get("cpu_percent", DEFAULT_ALERT_THRESHOLD_CPU):
            await self._add_alert(
                ResourceType.CPU,
                SecurityLevel.MEDIUM if self.system_metrics.cpu_percent >= 95 else SecurityLevel.LOW,
                f"CPU usage is high: {self.system_metrics.cpu_percent:.1f}%",
                self.system_metrics.cpu_percent,
                self.thresholds.get("cpu_percent", DEFAULT_ALERT_THRESHOLD_CPU)
            )
        
        # Memory usage alert
        if self.system_metrics.memory_percent >= self.thresholds.get("memory_percent", DEFAULT_ALERT_THRESHOLD_MEMORY):
            await self._add_alert(
                ResourceType.MEMORY,
                SecurityLevel.MEDIUM if self.system_metrics.memory_percent >= 95 else SecurityLevel.LOW,
                f"Memory usage is high: {self.system_metrics.memory_percent:.1f}%",
                self.system_metrics.memory_percent,
                self.thresholds.get("memory_percent", DEFAULT_ALERT_THRESHOLD_MEMORY)
            )
        
        # Disk usage alerts
        for path, percent in self.system_metrics.disk_percent.items():
            if percent >= self.thresholds.get("disk_percent", DEFAULT_ALERT_THRESHOLD_DISK):
                await self._add_alert(
                    ResourceType.DISK,
                    SecurityLevel.MEDIUM if percent >= 95 else SecurityLevel.LOW,
                    f"Disk usage is high on {path}: {percent:.1f}%",
                    percent,
                    self.thresholds.get("disk_percent", DEFAULT_ALERT_THRESHOLD_DISK)
                )
    
    async def _add_alert(self, resource_type: ResourceType, level: SecurityLevel, 
                       message: str, value: float, threshold: float) -> None:
        """Add a new alert and manage alert history."""
        # Create alert
        alert = Alert(
            resource_type=resource_type,
            level=level,
            message=message,
            value=value,
            threshold=threshold
        )
        
        # Check if similar alert is already active
        for existing in self.active_alerts:
            if (existing.resource_type == resource_type and 
                existing.level == level and
                abs(existing.value - value) < 5.0):  # Allow small variations
                return
        
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Add to history
        self.alert_history.append(alert)
        
        # Limit history size
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history:]
        
        # Log alert
        log_method = self.logger.warning if level == SecurityLevel.LOW else self.logger.error
        log_method(f"ALERT: {message}")
    
    async def clear_alert(self, alert_id: int) -> bool:
        """Clear an active alert by index."""
        if 0 <= alert_id < len(self.active_alerts):
            self.active_alerts.pop(alert_id)
            return True
        return False
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self.system_metrics
    
    def get_process_metrics(self, pid: Optional[int] = None) -> Union[Dict[int, ProcessMetrics], Optional[ProcessMetrics]]:
        """Get process metrics for a specific process or all processes."""
        if pid is not None:
            return self.process_metrics.get(pid)
        return self.process_metrics
    
    def get_metric_history(self, metric_name: str, 
                          start_time: Optional[float] = None, 
                          end_time: Optional[float] = None) -> List[Tuple[float, float]]:
        """Get historical values for a specific metric."""
        return self.metrics_buffer.get_range(metric_name, start_time, end_time)
    
    def get_metric_statistics(self, metric_name: str, 
                             start_time: Optional[float] = None, 
                             end_time: Optional[float] = None) -> Dict[str, float]:
        """Get statistical information for a metric."""
        return self.metrics_buffer.get_statistics(metric_name, start_time, end_time)
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        return self.metrics_buffer.get_metric_names()
    
    def get_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return self.active_alerts
    
    def get_alert_history(self) -> List[Alert]:
        """Get alert history."""
        return self.alert_history

class ResourceManager:
    """
    Resource allocation and management.
    
    Provides:
    - Resource allocation tracking
    - Constraint enforcement
    - Resource usage optimization
    - Priority-based allocation
    """
    
    def __init__(self, config: Dict, system_monitor: Optional[SystemMonitor] = None, logger=None):
        """Initialize resource manager with configuration."""
        self.config = config
        self.system_monitor = system_monitor
        self.logger = logger or logging.getLogger("orama.system.resources")
        
        # Resource configuration
        self.allocation_strategy = config.get("allocation_strategy", "priority")  # priority, fair, first-come
        
        # CPU configuration
        self.cpu_config = config.get("cpu", {})
        self.cpu_limit = self.cpu_config.get("limit_percent", 80)
        self.cpu_priority = self.cpu_config.get("priority", "normal")  # high, normal, low
        
        # Memory configuration
        self.memory_config = config.get("memory", {})
        memory_info = self._get_memory_info()
        self.memory_total = memory_info["total"]
        self.memory_limit_mb = self.memory_config.get("limit_mb", int(self.memory_total / (1024 * 1024) * 0.8))
        self.memory_reserve_mb = self.memory_config.get("reserve_mb", int(self.memory_total / (1024 * 1024) * 0.1))
        
        # GPU configuration
        self.gpu_config = config.get("gpu", {})
        self.gpu_enable = self.gpu_config.get("enable", False)
        self.gpu_memory_limit_mb = self.gpu_config.get("memory_limit_mb", 0)
        
        # Resource tracking
        self.allocated_resources: Dict[str, Dict[ResourceType, float]] = {}
        self.resource_locks: Dict[ResourceType, asyncio.Lock] = {
            ResourceType.CPU: asyncio.Lock(),
            ResourceType.MEMORY: asyncio.Lock(),
            ResourceType.GPU: asyncio.Lock(),
            ResourceType.DISK: asyncio.Lock(),
            ResourceType.NETWORK: asyncio.Lock()
        }
        
        # Set process priority if configured
        if sys.platform == "win32" and WIN32_AVAILABLE:
            self._set_process_priority()
        
        # State tracking
        self._running = False
        self._resource_task = None
        
        self.logger.info("Resource manager initialized")
    
    def _get_memory_info(self) -> Dict[str, int]:
        """Get system memory information."""
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.total - memory.available
            }
        
        # Fallback to basic info
        if sys.platform == "win32" and WIN32_AVAILABLE:
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                    
                memory_info = MEMORYSTATUSEX()
                memory_info.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_info))
                
                return {
                    "total": memory_info.ullTotalPhys,
                    "available": memory_info.ullAvailPhys,
                    "used": memory_info.ullTotalPhys - memory_info.ullAvailPhys
                }
            except Exception:
                pass
        
        # Default fallback
        return {
            "total": 8 * 1024 * 1024 * 1024,  # Assume 8GB
            "available": 4 * 1024 * 1024 * 1024,
            "used": 4 * 1024 * 1024 * 1024
        }
    
    def _set_process_priority(self) -> None:
        """Set process priority based on configuration."""
        if not WIN32_AVAILABLE:
            return
            
        try:
            process = win32process.GetCurrentProcess()
            priority_class = None
            
            if self.cpu_priority == "high":
                priority_class = win32process.HIGH_PRIORITY_CLASS
            elif self.cpu_priority == "normal":
                priority_class = win32process.NORMAL_PRIORITY_CLASS
            elif self.cpu_priority == "low":
                priority_class = win32process.BELOW_NORMAL_PRIORITY_CLASS
            
            if priority_class:
                win32process.SetPriorityClass(process, priority_class)
                self.logger.info(f"Set process priority to {self.cpu_priority}")
        except Exception as e:
            self.logger.warning(f"Failed to set process priority: {e}")
    
    async def start(self) -> None:
        """Start resource manager."""
        self.logger.info("Starting resource manager...")
        self._running = True
        
        try:
            # Start resource monitoring task
            self._resource_task = asyncio.create_task(self._resource_monitor_loop())
            
            self.logger.info("Resource manager started successfully")
        except Exception as e:
            self._running = False
            self.logger.error(f"Failed to start resource manager: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop resource manager."""
        self.logger.info("Stopping resource manager...")
        self._running = False
        
        try:
            # Cancel resource monitoring task
            if self._resource_task:
                self._resource_task.cancel()
                try:
                    await self._resource_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Resource manager stopped")
        except Exception as e:
            self.logger.error(f"Error during resource manager shutdown: {e}", exc_info=True)
    
    async def _resource_monitor_loop(self) -> None:
        """Monitor resource usage and enforce limits."""
        self.logger.info("Starting resource monitor loop")
        
        while self._running:
            try:
                # Get current resource usage
                if self.system_monitor:
                    metrics = self.system_monitor.get_system_metrics()
                    
                    # Check CPU usage
                    if metrics.cpu_percent > self.cpu_limit:
                        self.logger.warning(f"CPU usage exceeding limit: {metrics.cpu_percent:.1f}% > {self.cpu_limit}%")
                        
                        # Implement throttling if needed
                        await self._enforce_cpu_limit()
                    
                    # Check memory usage
                    memory_used_mb = (metrics.memory_total - metrics.memory_available) / (1024 * 1024)
                    if memory_used_mb > self.memory_limit_mb:
                        self.logger.warning(f"Memory usage exceeding limit: {memory_used_mb:.1f}MB > {self.memory_limit_mb}MB")
                        
                        # Implement memory control if needed
                        await self._enforce_memory_limit()
                
                # Sleep for monitoring interval
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                self.logger.info("Resource monitor loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in resource monitor loop: {e}", exc_info=True)
                await asyncio.sleep(10.0)  # Longer sleep on error
    
    async def _enforce_cpu_limit(self) -> None:
        """Enforce CPU usage limit."""
        # This is a placeholder for actual CPU throttling implementation
        # In a real implementation, you might want to:
        # 1. Pause or slow down non-critical tasks
        # 2. Adjust thread priorities
        # 3. Use OS-specific CPU affinity settings
        pass
    
    async def _enforce_memory_limit(self) -> None:
        """Enforce memory usage limit."""
        # This is a placeholder for actual memory limit enforcement
        # In a real implementation, you might want to:
        # 1. Trigger garbage collection
        # 2. Release cached data
        # 3. Terminate or suspend memory-intensive operations
        if hasattr(gc, "collect"):
            gc.collect()
    
    async def allocate_resource(self, resource_type: ResourceType, amount: float, 
                               owner_id: str, priority: int = 1) -> bool:
        """Allocate a resource."""
        async with self.resource_locks[resource_type]:
            try:
                # Check if owner already has allocations
                if owner_id not in self.allocated_resources:
                    self.allocated_resources[owner_id] = {}
                
                # Get current allocation
                current = self.allocated_resources[owner_id].get(resource_type, 0.0)
                
                # Check if allocation is possible based on resource type
                if resource_type == ResourceType.CPU:
                    # CPU allocation is percentage-based
                    total_allocated = sum(alloc.get(resource_type, 0.0) for alloc in self.allocated_resources.values())
                    if total_allocated + amount > self.cpu_limit:
                        self.logger.warning(f"CPU allocation denied: {total_allocated + amount:.1f}% > {self.cpu_limit}%")
                        return False
                
                elif resource_type == ResourceType.MEMORY:
                    # Memory allocation is MB-based
                    total_allocated = sum(alloc.get(resource_type, 0.0) for alloc in self.allocated_resources.values())
                    if total_allocated + amount > self.memory_limit_mb:
                        self.logger.warning(f"Memory allocation denied: {total_allocated + amount:.1f}MB > {self.memory_limit_mb}MB")
                        return False
                
                elif resource_type == ResourceType.GPU and self.gpu_enable:
                    # GPU allocation is MB-based
                    total_allocated = sum(alloc.get(resource_type, 0.0) for alloc in self.allocated_resources.values())
                    if total_allocated + amount > self.gpu_memory_limit_mb:
                        self.logger.warning(f"GPU allocation denied: {total_allocated + amount:.1f}MB > {self.gpu_memory_limit_mb}MB")
                        return False
                
                # Update allocation
                self.allocated_resources[owner_id][resource_type] = current + amount
                
                self.logger.debug(f"Resource allocated: {resource_type.name} +{amount} to {owner_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error allocating resource: {e}", exc_info=True)
                return False
    
    async def release_resource(self, resource_type: ResourceType, amount: float, 
                              owner_id: str) -> bool:
        """Release an allocated resource."""
        async with self.resource_locks[resource_type]:
            try:
                # Check if owner has allocations
                if owner_id not in self.allocated_resources:
                    return False
                
                # Check if resource is allocated
                if resource_type not in self.allocated_resources[owner_id]:
                    return False
                
                # Get current allocation
                current = self.allocated_resources[owner_id].get(resource_type, 0.0)
                
                # Update allocation
                new_amount = max(0.0, current - amount)
                if new_amount > 0:
                    self.allocated_resources[owner_id][resource_type] = new_amount
                else:
                    del self.allocated_resources[owner_id][resource_type]
                    if not self.allocated_resources[owner_id]:
                        del self.allocated_resources[owner_id]
                
                self.logger.debug(f"Resource released: {resource_type.name} -{amount} from {owner_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error releasing resource: {e}", exc_info=True)
                return False
    
    async def get_resource_usage(self, resource_type: ResourceType, owner_id: Optional[str] = None) -> Dict:
        """Get current resource usage."""
        async with self.resource_locks[resource_type]:
            try:
                if owner_id:
                    # Get usage for specific owner
                    if owner_id not in self.allocated_resources:
                        return {"allocated": 0.0, "limit": self._get_resource_limit(resource_type)}
                    
                    return {
                        "allocated": self.allocated_resources[owner_id].get(resource_type, 0.0),
                        "limit": self._get_resource_limit(resource_type)
                    }
                else:
                    # Get total usage
                    total_allocated = sum(
                        alloc.get(resource_type, 0.0) 
                        for alloc in self.allocated_resources.values()
                    )
                    
                    return {
                        "allocated": total_allocated,
                        "limit": self._get_resource_limit(resource_type),
                        "available": max(0.0, self._get_resource_limit(resource_type) - total_allocated)
                    }
            except Exception as e:
                self.logger.error(f"Error getting resource usage: {e}", exc_info=True)
                return {"error": str(e)}
    
    def _get_resource_limit(self, resource_type: ResourceType) -> float:
        """Get resource limit based on type."""
        if resource_type == ResourceType.CPU:
            return self.cpu_limit
        elif resource_type == ResourceType.MEMORY:
            return self.memory_limit_mb
        elif resource_type == ResourceType.GPU and self.gpu_enable:
            return self.gpu_memory_limit_mb
        return 0.0
    
    def get_all_allocations(self) -> Dict[str, Dict[str, float]]:
        """Get all current resource allocations."""
        result = {}
        for owner_id, allocations in self.allocated_resources.items():
            result[owner_id] = {res_type.name: amount for res_type, amount in allocations.items()}
        return result

class SecurityManager:
    """
    Security controls and data protection.
    
    Provides:
    - Permission management
    - Access control
    - Data encryption
    - Secure operations
    """
    
    def __init__(self, config: Dict, logger=None):
        """Initialize security manager with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger("orama.system.security")
        
        # Access control configuration
        self.access_config = config.get("access", {})
        self.permission_model = self.access_config.get("permission_model", "allow_with_exceptions")
        self.require_confirmation = set(self.access_config.get("require_confirmation", [
            "file_delete", "system_change", "network_access"
        ]))
        
        # Operation validation
        self.operation_config = config.get("operation", {})
        self.risk_assessment = self.operation_config.get("risk_assessment", True)
        self.dangerous_confirmation = self.operation_config.get("dangerous_operation_confirmation", True)
        
        # Data protection
        self.data_config = config.get("data", {})
        self.encrypt_sensitive = self.data_config.get("encrypt_sensitive", True)
        self.encryption_method = self.data_config.get("encryption_method", "AES-256")
        self.secure_delete = self.data_config.get("secure_delete", True)
        
        # Communication security
        self.comm_config = config.get("communication", {})
        self.encrypt_channels = self.comm_config.get("encrypt_channels", True)
        self.verify_endpoints = self.comm_config.get("verify_endpoints", True)
        
        # Permission tracking
        self.granted_permissions: Dict[str, Set[str]] = {}
        self.permission_history: List[Dict] = []
        self.max_permission_history = 100
        
        # Security analysis
        self.risk_profiles: Dict[str, Dict] = {}
        
        self.logger.info("Security manager initialized")
    
    def check_permission(self, operation: str, context: Dict = None) -> bool:
        """Check if an operation is permitted based on permission model."""
        try:
            # Default context
            if context is None:
                context = {}
            
            # For 'allow_all' model
            if self.permission_model == "allow_all":
                return True
            
            # For 'deny_with_exceptions' model
            if self.permission_model == "deny_with_exceptions":
                # Check if operation has been explicitly granted
                operation_key = self._format_operation_key(operation, context)
                
                # Check if operation key is in any granted permissions
                for perms in self.granted_permissions.values():
                    if operation_key in perms:
                        return True
                    
                    # Check for wildcard permissions
                    parts = operation_key.split('.')
                    for i in range(len(parts)):
                        wildcard = '.'.join(parts[:i]) + '.*'
                        if wildcard in perms:
                            return True
                
                return False
            
            # For 'allow_with_exceptions' model (default)
            if self.permission_model == "allow_with_exceptions":
                # Check if operation is in confirmation list
                if operation in self.require_confirmation:
                    # Operation requires explicit confirmation
                    operation_key = self._format_operation_key(operation, context)
                    
                    # Check if operation key is in any granted permissions
                    for perms in self.granted_permissions.values():
                        if operation_key in perms:
                            return True
                    
                    return False
                    
                # Not in confirmation list, so allowed by default
                return True
            
            # Unknown model, default to cautious approach
            self.logger.warning(f"Unknown permission model: {self.permission_model}, defaulting to deny")
            return False
        except Exception as e:
            self.logger.error(f"Error checking permission: {e}", exc_info=True)
            return False
    
    def grant_permission(self, operation: str, context: Dict, 
                        session_id: str, duration: Optional[float] = None) -> bool:
        """Grant permission for an operation."""
        try:
            # Format operation key
            operation_key = self._format_operation_key(operation, context)
            
            # Create session if it doesn't exist
            if session_id not in self.granted_permissions:
                self.granted_permissions[session_id] = set()
            
            # Add permission
            self.granted_permissions[session_id].add(operation_key)
            
            # Add to history
            self.permission_history.append({
                "operation": operation,
                "context": context,
                "session_id": session_id,
                "granted_at": time.time(),
                "expires_at": time.time() + duration if duration else None
            })
            
            # Limit history size
            if len(self.permission_history) > self.max_permission_history:
                self.permission_history = self.permission_history[-self.max_permission_history:]
            
            self.logger.info(f"Permission granted: {operation_key} to session {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error granting permission: {e}", exc_info=True)
            return False
    
    def revoke_permission(self, operation: str, context: Dict, session_id: str) -> bool:
        """Revoke permission for an operation."""
        try:
            # Format operation key
            operation_key = self._format_operation_key(operation, context)
            
            # Check if session exists
            if session_id not in self.granted_permissions:
                return False
            
            # Remove permission
            if operation_key in self.granted_permissions[session_id]:
                self.granted_permissions[session_id].remove(operation_key)
                
                # Remove session if empty
                if not self.granted_permissions[session_id]:
                    del self.granted_permissions[session_id]
                
                self.logger.info(f"Permission revoked: {operation_key} from session {session_id}")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error revoking permission: {e}", exc_info=True)
            return False
    
    def _format_operation_key(self, operation: str, context: Dict) -> str:
        """Format operation key for permission tracking."""
        # Simple key format: operation.param1_value.param2_value
        key_parts = [operation]
        
        # Add relevant context values
        if context:
            for key in sorted(context.keys()):
                value = context[key]
                if isinstance(value, (str, int, float, bool)):
                    # Only use simple types in key
                    key_parts.append(f"{key}_{value}")
        
        return '.'.join(key_parts)
    
    def expire_permissions(self) -> None:
        """Expire temporary permissions based on duration."""
        current_time = time.time()
        
        # Check permission history for expirations
        for entry in self.permission_history:
            if entry.get("expires_at") and entry["expires_at"] <= current_time:
                # Permission has expired
                session_id = entry["session_id"]
                operation = entry["operation"]
                context = entry["context"]
                
                # Revoke permission
                self.revoke_permission(operation, context, session_id)
    
    def assess_risk(self, operation: str, context: Dict) -> Tuple[float, str]:
        """Assess risk level of an operation."""
        try:
            # Default risk score (0.0 to 1.0)
            risk_score = 0.0
            risk_factors = []
            
            # Risk assessment based on operation type
            if operation == "file_delete":
                # Deleting files is inherently risky
                risk_score += 0.3
                risk_factors.append("File deletion")
                
                # Check for system paths
                path = context.get("path", "")
                if path:
                    if sys.platform == "win32":
                        system_paths = ["C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)"]
                        for sys_path in system_paths:
                            if path.startswith(sys_path):
                                risk_score += 0.5
                                risk_factors.append(f"System directory: {sys_path}")
                    else:
                        system_paths = ["/bin", "/sbin", "/usr/bin", "/usr/sbin", "/etc", "/var"]
                        for sys_path in system_paths:
                            if path.startswith(sys_path):
                                risk_score += 0.5
                                risk_factors.append(f"System directory: {sys_path}")
            
            elif operation == "command_execution":
                # Command execution is risky
                risk_score += 0.4
                risk_factors.append("Command execution")
                
                # Check for dangerous commands
                command = context.get("command", "")
                if command:
                    dangerous_patterns = ["rm -rf", "deltree", "format", "shutdown", "reboot", "sudo", "chown", "chmod"]
                    for pattern in dangerous_patterns:
                        if pattern in command:
                            risk_score += 0.3
                            risk_factors.append(f"Dangerous command pattern: {pattern}")
            
            elif operation == "network_access":
                # Network access has some risk
                risk_score += 0.2
                risk_factors.append("Network access")
                
                # Check for external vs internal
                url = context.get("url", "")
                if url:
                    if not url.startswith(("http://localhost", "https://localhost", "http://127.0.0.1")):
                        risk_score += 0.1
                        risk_factors.append("External network access")
            
            # Clamp risk score
            risk_score = min(1.0, risk_score)
            
            # Format risk assessment
            if risk_score >= 0.8:
                risk_level = "Critical"
            elif risk_score >= 0.6:
                risk_level = "High"
            elif risk_score >= 0.3:
                risk_level = "Medium"
            elif risk_score >= 0.1:
                risk_level = "Low"
            else:
                risk_level = "Minimal"
            
            # Format risk description
            if risk_score >= 0.8:
                risk_level = "Critical"
            elif risk_score >= 0.6:
                risk_level = "High"
            elif risk_score >= 0.3:
                risk_level = "Medium"
            elif risk_score >= 0.1:
                risk_level = "Low"
            else:
                risk_level = "Minimal"
            
            # Format risk description
            risk_description = f"{risk_level} risk ({risk_score:.2f})"
            if risk_factors:
                risk_description += f": {', '.join(risk_factors)}"
            
            return (risk_score, risk_description)
        except Exception as e:
            self.logger.error(f"Error assessing risk: {e}", exc_info=True)
            return (0.5, f"Risk assessment error: {str(e)}")
    
    async def encrypt_data(self, data: Union[str, bytes], key: Optional[str] = None) -> bytes:
        """Encrypt sensitive data."""
        if not self.encrypt_sensitive:
            # Return data as-is if encryption is disabled
            if isinstance(data, str):
                return data.encode('utf-8')
            return data
            
        try:
            # Simple encryption using key derivation and AES
            if key is None:
                # Generate a key from system-specific information
                system_key = platform.node() + platform.platform()
                key = hashlib.sha256(system_key.encode()).hexdigest()
            
            # Convert data to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Initialize encryption
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            import os
            
            # Generate a random salt and IV
            salt = os.urandom(16)
            iv = os.urandom(16)
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            derived_key = kdf.derive(key.encode())
            
            # Encrypt data
            cipher = Cipher(algorithms.AES(derived_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad data to AES block size
            block_size = 16
            padding_length = block_size - (len(data_bytes) % block_size)
            padded_data = bytes(data_bytes) + bytes([padding_length]) * padding_length
            
            # Encrypt
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine salt, IV, and ciphertext
            return salt + iv + ciphertext
        except Exception as e:
            self.logger.error(f"Error encrypting data: {e}", exc_info=True)
            # Return data as-is on error
            if isinstance(data, str):
                return data.encode('utf-8')
            return data
    
    async def decrypt_data(self, data: bytes, key: Optional[str] = None) -> bytes:
        """Decrypt sensitive data."""
        if not self.encrypt_sensitive:
            # Return data as-is if encryption is disabled
            return data
            
        try:
            # Simple decryption using key derivation and AES
            if key is None:
                # Generate a key from system-specific information
                system_key = platform.node() + platform.platform()
                key = hashlib.sha256(system_key.encode()).hexdigest()
            
            # Initialize decryption
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            # Extract salt, IV, and ciphertext
            salt = data[:16]
            iv = data[16:32]
            ciphertext = data[32:]
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            derived_key = kdf.derive(key.encode())
            
            # Decrypt data
            cipher = Cipher(algorithms.AES(derived_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            
            # Decrypt
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            padding_length = padded_data[-1]
            return padded_data[:-padding_length]
        except Exception as e:
            self.logger.error(f"Error decrypting data: {e}", exc_info=True)
            # Return data as-is on error
            return data
    
    async def secure_delete_file(self, path: str, passes: int = 3) -> bool:
        """Securely delete a file by overwriting before deletion."""
        if not self.secure_delete:
            # Use normal deletion if secure delete is disabled
            try:
                os.unlink(path)
                return True
            except Exception as e:
                self.logger.error(f"Error deleting file: {e}", exc_info=True)
                return False
            
        try:
            # Check if file exists
            if not os.path.exists(path) or not os.path.isfile(path):
                return False
                
            # Get file size
            file_size = os.path.getsize(path)
            
            # Open file for writing
            with open(path, "r+b") as f:
                # Multiple overwrite passes with different patterns
                for _ in range(passes):
                    # Seek to beginning
                    f.seek(0)
                    
                    # Use different patterns for each pass
                    if _ == 0:
                        pattern = b'\x00'  # Zeros
                    elif _ == 1:
                        pattern = b'\xFF'  # Ones
                    else:
                        # Random data for additional passes
                        pattern = os.urandom(1)
                    
                    # Write pattern in chunks
                    chunk_size = 1024 * 1024  # 1MB chunks
                    remaining = file_size
                    
                    while remaining > 0:
                        write_size = min(chunk_size, remaining)
                        f.write(pattern * write_size)
                        remaining -= write_size
                    
                    # Flush to disk
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally delete the file
            os.unlink(path)
            return True
        except Exception as e:
            self.logger.error(f"Error securely deleting file: {e}", exc_info=True)
            return False

class LogManager:
    """
    Advanced logging and diagnostics.
    
    Provides:
    - Centralized logging with rotation
    - Structured logging with context
    - Log filtering and analysis
    - Performance metrics integration
    """
    
    def __init__(self, config: Dict):
        """Initialize log manager with configuration."""
        self.config = config
        
        # Logging configuration
        self.log_level = config.get("level", "info").lower()
        self.log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.log_file = config.get("file")
        self.log_max_size = config.get("max_size", LOG_ROTATION_SIZE)
        self.log_backup_count = config.get("backup_count", LOG_BACKUP_COUNT)
        self.console_logging = config.get("console", True)
        
        # Additional configuration
        self.structured_logging = config.get("structured", False)
        self.log_metrics = config.get("metrics", False)
        
        # Initialize logging
        self._setup_logging()
        
        # Create main logger
        self.logger = self.get_logger("orama.system.log")
        
        # Context tracking
        self.context_stack: Dict[str, List[Dict]] = {}
        
        self.logger.info("Log manager initialized")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Reset logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Set log level
        log_level = LOG_LEVELS.get(self.log_level, logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        
        # Add console handler if enabled
        if self.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logging.root.addHandler(console_handler)
        
        # Add file handler if configured
        if self.log_file:
            try:
                # Ensure directory exists
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                # Use rotating file handler for log rotation
                file_handler = logging.handlers.RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.log_max_size,
                    backupCount=self.log_backup_count
                )
                file_handler.setFormatter(formatter)
                logging.root.addHandler(file_handler)
            except Exception as e:
                print(f"Error setting up file logging: {e}")
        
        # Set root logger level
        logging.root.setLevel(log_level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)
    
    @contextmanager
    def context(self, logger_name: str, **context) -> Generator[logging.Logger, None, None]:
        """Context manager for adding context to logs."""
        logger = self.get_logger(logger_name)
        
        # Add context
        if logger_name not in self.context_stack:
            self.context_stack[logger_name] = []
            
        self.context_stack[logger_name].append(context)
        
        try:
            yield logger
        finally:
            # Remove context
            if logger_name in self.context_stack and self.context_stack[logger_name]:
                self.context_stack[logger_name].pop()
    
    def _format_message_with_context(self, logger_name: str, message: str) -> str:
        """Format log message with context if available."""
        if logger_name not in self.context_stack or not self.context_stack[logger_name]:
            return message
            
        # Get current context
        context = self.context_stack[logger_name][-1]
        
        # Format context
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        
        return f"{message} [{context_str}]"
    
    async def archive_logs(self, archive_dir: str) -> bool:
        """Archive log files to a specified directory."""
        if not self.log_file:
            return False
            
        try:
            # Ensure archive directory exists
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir, exist_ok=True)
            
            # Get log files
            log_dir = os.path.dirname(self.log_file)
            log_name = os.path.basename(self.log_file)
            
            # Create archive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{os.path.splitext(log_name)[0]}_{timestamp}.log"
            archive_path = os.path.join(archive_dir, archive_name)
            
            # Copy current log to archive
            shutil.copy2(self.log_file, archive_path)
            
            # Rotate current log
            with open(self.log_file, "w") as f:
                f.write(f"Log rotated at {datetime.now().isoformat()}\n")
            
            return True
        except Exception as e:
            print(f"Error archiving logs: {e}")
            return False
    
    async def search_logs(self, pattern: str, log_file: Optional[str] = None) -> List[str]:
        """Search logs for a pattern."""
        file_to_search = log_file or self.log_file
        
        if not file_to_search or not os.path.exists(file_to_search):
            return []
            
        try:
            matching_lines = []
            
            # Use async file reading if available
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(file_to_search, "r") as f:
                    async for line in f:
                        if pattern in line:
                            matching_lines.append(line.strip())
            else:
                # Fallback to synchronous reading
                with open(file_to_search, "r") as f:
                    for line in f:
                        if pattern in line:
                            matching_lines.append(line.strip())
            
            return matching_lines
        except Exception as e:
            print(f"Error searching logs: {e}")
            return []

class PerformanceTracker:
    """
    Performance tracking and analysis.
    
    Provides:
    - Operation timing and profiling
    - Resource usage tracking
    - Performance optimization suggestions
    - Historical performance analysis
    """
    
    def __init__(self, config: Dict, logger=None):
        """Initialize performance tracker with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger("orama.system.performance")
        
        # Performance tracking configuration
        self.tracking_interval = config.get("tracking_interval", 1.0)  # seconds
        self.metrics_history = config.get("metrics_history", 3600)  # number of data points
        
        # Alert thresholds
        self.alert_thresholds = config.get("alert_thresholds", {
            "cpu_percent": 90,
            "memory_percent": 85,
            "response_time": 5000  # ms
        })
        
        # Operation tracking
        self.operation_timings: Dict[str, List[Tuple[float, float]]] = {}
        self.current_operations: Dict[str, float] = {}
        
        # Metrics storage
        self.metrics_buffer = MetricsBuffer(self.metrics_history)
        
        # State tracking
        self._running = False
        self._tracking_task = None
        
        # Performance report cache
        self.last_report_time = 0
        self.cached_report: Dict = {}
        
        self.logger.info("Performance tracker initialized")
    
    async def start_tracking(self) -> None:
        """Start performance tracking."""
        self.logger.info("Starting performance tracking...")
        self._running = True
        
        try:
            # Start tracking task
            self._tracking_task = asyncio.create_task(self._tracking_loop())
            
            self.logger.info("Performance tracking started successfully")
        except Exception as e:
            self._running = False
            self.logger.error(f"Failed to start performance tracking: {e}", exc_info=True)
            raise
    
    async def stop_tracking(self) -> None:
        """Stop performance tracking."""
        self.logger.info("Stopping performance tracking...")
        self._running = False
        
        try:
            # Cancel tracking task
            if self._tracking_task:
                self._tracking_task.cancel()
                try:
                    await self._tracking_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Performance tracking stopped")
        except Exception as e:
            self.logger.error(f"Error during performance tracking shutdown: {e}", exc_info=True)
    
    async def _tracking_loop(self) -> None:
        """Background task for performance tracking."""
        self.logger.info("Starting tracking loop")
        
        while self._running:
            try:
                # Track system metrics
                await self._track_system_metrics()
                
                # Sleep for tracking interval
                await asyncio.sleep(self.tracking_interval)
                
            except asyncio.CancelledError:
                self.logger.info("Tracking loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in tracking loop: {e}", exc_info=True)
                await asyncio.sleep(self.tracking_interval * 2)  # Longer sleep on error
    
    async def _track_system_metrics(self) -> None:
        """Track system performance metrics."""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics_buffer.add("cpu_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_buffer.add("memory_percent", memory.percent)
            self.metrics_buffer.add("memory_available", memory.available)
            
            # Disk I/O
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics_buffer.add("disk_read_count", disk_io.read_count)
                    self.metrics_buffer.add("disk_write_count", disk_io.write_count)
                    self.metrics_buffer.add("disk_read_bytes", disk_io.read_bytes)
                    self.metrics_buffer.add("disk_write_bytes", disk_io.write_bytes)
            except (AttributeError, FileNotFoundError):
                pass
            
            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                if net_io:
                    self.metrics_buffer.add("net_bytes_sent", net_io.bytes_sent)
                    self.metrics_buffer.add("net_bytes_recv", net_io.bytes_recv)
                    self.metrics_buffer.add("net_packets_sent", net_io.packets_sent)
                    self.metrics_buffer.add("net_packets_recv", net_io.packets_recv)
            except (AttributeError, FileNotFoundError):
                pass
            
            # Check for performance alerts
            await self._check_performance_alerts()
            
        except Exception as e:
            self.logger.error(f"Error tracking system metrics: {e}", exc_info=True)
    
    async def _check_performance_alerts(self) -> None:
        """Check for performance issues that require alerts."""
        try:
            # Check CPU usage
            cpu_percent = self.metrics_buffer.get_latest("cpu_percent")
            if cpu_percent and cpu_percent[1] >= self.alert_thresholds.get("cpu_percent", 90):
                self.logger.warning(f"High CPU usage detected: {cpu_percent[1]:.1f}%")
            
            # Check memory usage
            memory_percent = self.metrics_buffer.get_latest("memory_percent")
            if memory_percent and memory_percent[1] >= self.alert_thresholds.get("memory_percent", 85):
                self.logger.warning(f"High memory usage detected: {memory_percent[1]:.1f}%")
            
            # Check operation response times
            for operation, timings in self.operation_timings.items():
                if not timings:
                    continue
                
                # Get recent timings (last 10)
                recent_timings = timings[-10:]
                avg_time_ms = sum((end - start) * 1000 for start, end in recent_timings) / len(recent_timings)
                
                if avg_time_ms >= self.alert_thresholds.get("response_time", 5000):
                    self.logger.warning(f"Slow operation detected: {operation} ({avg_time_ms:.1f}ms)")
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}", exc_info=True)
    
    @contextmanager
    def track_operation(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager for tracking operation timing."""
        start_time = time.time()
        operation_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        
        self.current_operations[operation_id] = start_time
        
        try:
            yield
        finally:
            # Record timing
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Remove from current operations
            if operation_id in self.current_operations:
                del self.current_operations[operation_id]
            
            # Add to operation timings
            if operation_name not in self.operation_timings:
                self.operation_timings[operation_name] = []
                
            self.operation_timings[operation_name].append((start_time, end_time))
            
            # Limit history size
            max_history = 1000
            if len(self.operation_timings[operation_name]) > max_history:
                self.operation_timings[operation_name] = self.operation_timings[operation_name][-max_history:]
            
            # Log slow operations
            if duration_ms > self.alert_thresholds.get("response_time", 5000):
                self.logger.warning(f"Slow operation: {operation_name} ({duration_ms:.1f}ms)")
            elif duration_ms > 1000:
                self.logger.info(f"Operation timing: {operation_name} ({duration_ms:.1f}ms)")
    
    async def track_operation_async(self, operation_name: str) -> Callable:
        """Decorator for tracking async operation timing."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                operation_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
                
                self.current_operations[operation_id] = start_time
                
                try:
                    return await func(*args, **kwargs)
                finally:
                    # Record timing
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000
                    
                    # Remove from current operations
                    if operation_id in self.current_operations:
                        del self.current_operations[operation_id]
                    
                    # Add to operation timings
                    if operation_name not in self.operation_timings:
                        self.operation_timings[operation_name] = []
                        
                    self.operation_timings[operation_name].append((start_time, end_time))
                    
                    # Limit history size
                    max_history = 1000
                    if len(self.operation_timings[operation_name]) > max_history:
                        self.operation_timings[operation_name] = self.operation_timings[operation_name][-max_history:]
                    
                    # Log slow operations
                    if duration_ms > self.alert_thresholds.get("response_time", 5000):
                        self.logger.warning(f"Slow operation: {operation_name} ({duration_ms:.1f}ms)")
                    elif duration_ms > 1000:
                        self.logger.info(f"Operation timing: {operation_name} ({duration_ms:.1f}ms)")
            
            return wrapper
        
        return decorator
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict:
        """Get statistics for operations."""
        if operation_name:
            # Get stats for specific operation
            if operation_name not in self.operation_timings:
                return {"error": f"Operation not found: {operation_name}"}
                
            timings = self.operation_timings[operation_name]
            durations_ms = [(end - start) * 1000 for start, end in timings]
            
            if not durations_ms:
                return {"count": 0}
                
            return {
                "count": len(durations_ms),
                "min_ms": min(durations_ms),
                "max_ms": max(durations_ms),
                "avg_ms": sum(durations_ms) / len(durations_ms),
                "recent_avg_ms": sum(durations_ms[-10:]) / min(10, len(durations_ms)) if durations_ms else 0,
                "total_time_ms": sum(durations_ms)
            }
        else:
            # Get stats for all operations
            result = {}
            for op_name in self.operation_timings:
                result[op_name] = self.get_operation_stats(op_name)
            return result
    
    def get_system_metrics(self, metric_name: Optional[str] = None, 
                           start_time: Optional[float] = None, 
                           end_time: Optional[float] = None) -> Dict:
        """Get system performance metrics."""
        if metric_name:
            # Get specific metric
            data = self.metrics_buffer.get_range(metric_name, start_time, end_time)
            stats = self.metrics_buffer.get_statistics(metric_name, start_time, end_time)
            
            return {
                "metric": metric_name,
                "data": data,
                "statistics": stats
            }
        else:
            # Get all metrics
            metrics = self.metrics_buffer.get_metric_names()
            result = {}
            
            for metric in metrics:
                latest = self.metrics_buffer.get_latest(metric)
                if latest:
                    result[metric] = latest[1]
            
            return result
    
    async def generate_performance_report(self, force_refresh: bool = False) -> Dict:
        """Generate a comprehensive performance report."""
        current_time = time.time()
        
        # Use cached report if available and recent
        if not force_refresh and self.cached_report and current_time - self.last_report_time < 60:
            return self.cached_report
            
        try:
            report = {}
            
            # System metrics
            system_metrics = {}
            for metric in ["cpu_percent", "memory_percent", "memory_available"]:
                latest = self.metrics_buffer.get_latest(metric)
                if latest:
                    system_metrics[metric] = latest[1]
                    
                    # Add trend (last 10 minutes)
                    start_time = current_time - 600
                    data = self.metrics_buffer.get_range(metric, start_time)
                    if data:
                        values = [v for _, v in data]
                        if len(values) >= 2:
                            trend = values[-1] - values[0]
                            system_metrics[f"{metric}_trend"] = trend
            
            report["system_metrics"] = system_metrics
            
            # Disk metrics
            disk_metrics = {}
            for metric in ["disk_read_bytes", "disk_write_bytes"]:
                latest = self.metrics_buffer.get_latest(metric)
                if latest:
                    disk_metrics[metric] = latest[1]
            
            report["disk_metrics"] = disk_metrics
            
            # Network metrics
            network_metrics = {}
            for metric in ["net_bytes_sent", "net_bytes_recv"]:
                latest = self.metrics_buffer.get_latest(metric)
                if latest:
                    network_metrics[metric] = latest[1]
            
            report["network_metrics"] = network_metrics
            
            # Operation statistics
            operation_stats = self.get_operation_stats()
            
            # Filter for slowest operations
            slow_operations = {}
            for op_name, stats in operation_stats.items():
                if isinstance(stats, dict) and "avg_ms" in stats and stats["avg_ms"] > 100:
                    slow_operations[op_name] = stats
            
            report["slow_operations"] = dict(sorted(
                slow_operations.items(),
                key=lambda x: x[1].get("avg_ms", 0),
                reverse=True
            )[:10])  # Top 10 slowest
            
            # Performance issues
            issues = []
            
            # Check CPU usage
            if system_metrics.get("cpu_percent", 0) > 80:
                issues.append({
                    "type": "high_cpu_usage",
                    "severity": "warning",
                    "message": f"High CPU usage detected: {system_metrics.get('cpu_percent', 0):.1f}%"
                })
            
            # Check memory usage
            if system_metrics.get("memory_percent", 0) > 80:
                issues.append({
                    "type": "high_memory_usage",
                    "severity": "warning",
                    "message": f"High memory usage detected: {system_metrics.get('memory_percent', 0):.1f}%"
                })
            
            # Check for slow operations
            for op_name, stats in slow_operations.items():
                if stats.get("avg_ms", 0) > 1000:
                    issues.append({
                        "type": "slow_operation",
                        "severity": "warning",
                        "message": f"Slow operation detected: {op_name} ({stats.get('avg_ms', 0):.1f}ms)"
                    })
            
            report["issues"] = issues
            
            # Optimization suggestions
            suggestions = []
            
            # CPU optimization
            if system_metrics.get("cpu_percent", 0) > 70:
                suggestions.append({
                    "type": "cpu_optimization",
                    "message": "Consider optimizing CPU-intensive operations or increasing CPU limit"
                })
            
            # Memory optimization
            if system_metrics.get("memory_percent", 0) > 70:
                suggestions.append({
                    "type": "memory_optimization",
                    "message": "Consider optimizing memory usage or increasing memory limit"
                })
            
            # Operation optimization
            if slow_operations:
                suggestions.append({
                    "type": "operation_optimization",
                    "message": f"Consider optimizing the {len(slow_operations)} slowest operations"
                })
            
            report["optimization_suggestions"] = suggestions
            
            # Update cache
            self.cached_report = report
            self.last_report_time = current_time
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}", exc_info=True)
            return {"error": str(e)}

def get_system_info() -> Dict:
    """Get comprehensive system information."""
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "interpreter": sys.executable,
        "hostname": socket.gethostname()
    }
    
    # Add more detailed info if psutil is available
    if PSUTIL_AVAILABLE:
        try:
            cpu_count = psutil.cpu_count(logical=True)
            info["cpu_count"] = str(cpu_count) if cpu_count is not None else "N/A"
            
            cpu_physical = psutil.cpu_count(logical=False)
            info["cpu_physical_count"] = str(cpu_physical) if cpu_physical is not None else "N/A"
            
            memory = psutil.virtual_memory()
            info["memory_total"] = str(memory.total)
            info["memory_available"] = str(memory.available)
        except Exception as e:
            info["error"] = f"Failed to get detailed system info: {str(e)}"
