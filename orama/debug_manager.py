"""
ORAMA Debug and Profiling System

This module provides advanced debugging, profiling, and logging capabilities including:
- Structured logging with multiple handlers
- Performance profiling
- Debug mode with detailed tracing
- Remote logging support
"""

import os
import sys
import time
import json
import logging
import logging.handlers
import asyncio
import threading
import traceback
import cProfile
import pstats
import io
import socket
import aiohttp
from typing import Callable, Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager

@dataclass
class DebugConfig:
    """Debug configuration."""
    enabled: bool = False
    trace_all: bool = False
    profile_enabled: bool = False
    log_level: str = "INFO"
    file_logging: bool = True
    console_logging: bool = True
    remote_logging: bool = False
    remote_endpoint: str = ""
    log_dir: str = "logs"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    max_log_backups: int = 5
    structured_logging: bool = True
    include_timestamps: bool = True
    include_thread_info: bool = True

@dataclass
class ProfileResult:
    """Profiling result for a function or code block."""
    name: str
    start_time: float
    end_time: float
    duration: float
    function_calls: int
    stats: Dict
    memory_start: Optional[Dict] = None
    memory_end: Optional[Dict] = None

class DebugManager:
    """
    Manages debugging, profiling, and logging capabilities.
    
    Features:
    - Multiple log handlers (console, file, remote)
    - Structured logging
    - Performance profiling
    - Debug mode with tracing
    - Remote logging support
    """
    
    def __init__(self, config: str, **kwargs) -> None:
        """Initialize the debug manager."""
        self.config = DebugConfig(**json.loads(config)) if config else DebugConfig()

        # Set up logging
        self.logger = self._setup_logging()
        
        # Profiling state
        self._profiler = None
        self._active_profiles: Dict[str, Tuple[cProfile.Profile, float]] = {}
        self.profile_results: List[ProfileResult] = []
        
        # Debug state
        self._debug_contexts: Dict[str, Any] = {}
        self._trace_hooks: Set[Callable] = set()

        # Statistics
        self.stats = {
            "start_time": time.time(),
            "log_entries": 0,
            "profile_runs": 0,
            "trace_events": 0
        }
        
        # Remote logging queue
        self._remote_queue: asyncio.Queue = asyncio.Queue()
        self._remote_task = None
        
        self.logger.info("Debug manager initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging with configured handlers."""
        # Create logger
        logger = logging.getLogger("orama")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        if self.config.structured_logging:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"thread": "%(threadName)s", "message": %(message)s}'
            )
        else:
            parts = []
            if self.config.include_timestamps:
                parts.append("%(asctime)s")
            if self.config.include_thread_info:
                parts.append("%(threadName)s")
            parts.extend(["%(levelname)s", "%(message)s"])
            formatter = logging.Formatter(" - ".join(parts))
        
        # Console handler
        if self.config.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_logging:
            os.makedirs(self.config.log_dir, exist_ok=True)
            log_file = os.path.join(self.config.log_dir, "orama.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_log_size,
                backupCount=self.config.max_log_backups
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Remote handler
        if self.config.remote_logging and self.config.remote_endpoint:
            remote_handler = self.RemoteHandler(self._remote_queue)
            remote_handler.setFormatter(formatter)
            logger.addHandler(remote_handler)
        
        return logger
    
    class RemoteHandler(logging.Handler):
        """Handler that sends log records to a remote endpoint."""
        def __init__(self, queue: asyncio.Queue):
            super().__init__()
            self.queue = queue
        
        def emit(self, record):
            try:
                msg = self.format(record)
                asyncio.create_task(self.queue.put(msg))
            except Exception:
                self.handleError(record)
    
    async def start(self) -> None:
        """Start the debug manager's background tasks."""
        if self.config.remote_logging and self.config.remote_endpoint:
            self._remote_task = asyncio.create_task(self._remote_logging_loop())
    
    async def stop(self) -> None:
        """Stop the debug manager's background tasks."""
        if self._remote_task:
            self._remote_task.cancel()
            try:
                await self._remote_task
            except asyncio.CancelledError:
                pass
            self._remote_task = None
    
    async def _remote_logging_loop(self) -> None:
        """Background task that sends logs to remote endpoint."""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Get log entry
                    entry = await self._remote_queue.get()
                    
                    # Send to remote endpoint
                    try:
                        async with session.post(
                            self.config.remote_endpoint,
                            json={"log": entry}
                        ) as response:
                            if response.status >= 400:
                                self.logger.error(
                                    f"Error sending log to remote endpoint: {response.status}"
                                )
                    except Exception as e:
                        self.logger.error(f"Remote logging error: {str(e)}")
                    
                    self._remote_queue.task_done()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in remote logging loop: {str(e)}")
                    await asyncio.sleep(5)  # Wait before retry
    
    @contextmanager
    def profile(self, name: str = None) -> ProfileResult:
        """Context manager for profiling a block of code."""
        if not self.config.profile_enabled:
            yield None
            return
            
        profile_name = name or f"profile_{len(self.profile_results)}"
        profiler = cProfile.Profile()
        start_time = time.time()
        
        # Get initial memory usage if psutil is available
        memory_start = None
        try:
            import psutil
            process = psutil.Process()
            memory_start = {
                "rss": process.memory_info().rss,
                "vms": process.memory_info().vms
            }
        except ImportError:
            pass
        
        try:
            profiler.enable()
            yield profiler
        finally:
            profiler.disable()
            end_time = time.time()
            
            # Get final memory usage
            memory_end = None
            if memory_start:
                try:
                    memory_end = {
                        "rss": process.memory_info().rss,
                        "vms": process.memory_info().vms
                    }
                except:
                    pass
            
            # Get stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats()
            
            # Create profile result
            result = ProfileResult(
                name=profile_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                function_calls=profiler.getstats()[0].callcount,
                stats=s.getvalue(),
                memory_start=memory_start,
                memory_end=memory_end
            )
            
            self.profile_results.append(result)
            self.stats["profile_runs"] += 1
            
            # Log summary
            self.logger.debug(
                f"Profile {profile_name}: {result.duration:.3f}s, "
                f"{result.function_calls} calls"
            )
    
    def profile_decorator(self, name: str = None):
        """Decorator for profiling a function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                profile_name = name or func.__name__
                with self.profile(profile_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_profile_results(self, name: str = None) -> List[ProfileResult]:
        """Get profiling results."""
        if name:
            return [r for r in self.profile_results if r.name == name]
        return self.profile_results
    
    def add_trace_hook(self, hook: callable) -> None:
        """Add a trace hook function."""
        self._trace_hooks.add(hook)
    
    def remove_trace_hook(self, hook: callable) -> None:
        """Remove a trace hook function."""
        self._trace_hooks.discard(hook)
    
    def _trace_callback(self, frame, event, arg):
        """Trace callback for debugging."""
        if not self.config.enabled:
            return
            
        try:
            info = {
                "event": event,
                "function": frame.f_code.co_name,
                "filename": frame.f_code.co_filename,
                "lineno": frame.f_lineno,
                "locals": dict(frame.f_locals) if self.config.trace_all else None
            }
            
            self.stats["trace_events"] += 1
            
            # Notify trace hooks
            for hook in self._trace_hooks:
                try:
                    hook(info)
                except:
                    pass
            
        except:
            pass
        
        return self._trace_callback if self.config.enabled else None
    
    def enable_tracing(self) -> None:
        """Enable debug tracing."""
        if not self.config.enabled:
            self.config.enabled = True
            sys.settrace(self._trace_callback)
            threading.settrace(self._trace_callback)
    
    def disable_tracing(self) -> None:
        """Disable debug tracing."""
        self.config.enabled = False
        sys.settrace(None)
        threading.settrace(None)
    
    def set_debug_context(self, name: str, value: Any) -> None:
        """Set a value in the debug context."""
        self._debug_contexts[name] = value
    
    def get_debug_context(self, name: str, default: Any = None) -> Any:
        """Get a value from the debug context."""
        return self._debug_contexts.get(name, default)
    
    def clear_debug_context(self) -> None:
        """Clear all debug context values."""
        self._debug_contexts.clear()
    
    def get_stats(self) -> Dict:
        """Get debug manager statistics."""
        current_time = time.time()
        return {
            **self.stats,
            "uptime": current_time - self.stats["start_time"],
            "active_profiles": len(self._active_profiles),
            "stored_profiles": len(self.profile_results),
            "debug_contexts": len(self._debug_contexts),
            "trace_hooks": len(self._trace_hooks),
            "remote_queue_size": self._remote_queue.qsize() if self._remote_queue else 0
        }
