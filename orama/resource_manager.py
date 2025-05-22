"""
ORAMA Resource Manager

This module provides resource monitoring and management capabilities including:
- CPU/Memory monitoring
- Network monitoring
- Resource limits and throttling
- Resource usage history and alerts
"""

import os
import sys
import time
import asyncio
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil

@dataclass
class ResourceThreshold:
    """Resource threshold configuration."""
    cpu_percent: float = 80.0  # CPU threshold in percent
    memory_percent: float = 80.0  # Memory threshold in percent
    disk_percent: float = 90.0  # Disk usage threshold in percent
    network_mbps: float = 100.0  # Network threshold in Mbps

@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_total: int
    memory_available: int
    memory_percent: float
    swap_total: int
    swap_used: int
    disk_total: int
    disk_used: int
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    network_mbps_sent: float
    network_mbps_recv: float

@dataclass
class ResourceAlert:
    """Resource usage alert."""
    timestamp: float
    resource_type: str
    value: float
    threshold: float
    message: str

class ResourceManager:
    """
    Manages system resource monitoring and control.
    
    Features:
    - Real-time resource monitoring
    - Resource usage history
    - Resource alerts
    - Resource limits and throttling
    """
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """Initialize the resource manager."""
        self.config = config or {}
        self.logger = logger or logging.getLogger("orama.resources")
        
        # Resource thresholds
        self.thresholds = ResourceThreshold(
            cpu_percent=self.config.get("cpu_threshold", 80.0),
            memory_percent=self.config.get("memory_threshold", 80.0),
            disk_percent=self.config.get("disk_threshold", 90.0),
            network_mbps=self.config.get("network_threshold", 100.0)
        )
        
        # Resource history
        self.history_max_size = self.config.get("history_size", 3600)  # 1 hour at 1 second intervals
        self.history: List[ResourceUsage] = []
        
        # Resource alerts
        self.alerts: List[ResourceAlert] = []
        self.alerts_max_size = self.config.get("alerts_size", 1000)
        self.alert_callbacks: Set[callable] = set()
        
        # Monitoring state
        self._running = False
        self._monitoring_task = None
        self._last_network = None
        
        # Resource control
        self.throttling_enabled = self.config.get("enable_throttling", False)
        self._throttled_processes: Set[int] = set()
        
        # Statistics
        self.stats = {
            "start_time": time.time(),
            "total_alerts": 0,
            "throttle_events": 0
        }
        
        # Initialize intervals
        self.monitoring_interval = self.config.get("monitoring_interval", 1.0)  # seconds
        self.alert_interval = self.config.get("alert_interval", 60.0)  # seconds
        self._last_alert = {}  # Last alert time by resource type
        
        self.logger.info("Resource manager initialized")
    
    async def start(self) -> None:
        """Start resource monitoring."""
        if self._running:
            return
            
        self.logger.info("Starting resource monitoring")
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop(self) -> None:
        """Stop resource monitoring."""
        if not self._running:
            return
            
        self.logger.info("Stopping resource monitoring")
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
    
    async def _monitoring_loop(self) -> None:
        """Background task that monitors system resources."""
        try:
            while self._running:
                # Get resource usage
                usage = await self._get_resource_usage()
                
                # Add to history
                self._add_to_history(usage)
                
                # Check thresholds and generate alerts
                await self._check_thresholds(usage)
                
                # Apply throttling if enabled
                if self.throttling_enabled:
                    await self._apply_throttling(usage)
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Resource monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in resource monitoring loop: {str(e)}", exc_info=True)
            self._running = False
    
    async def _get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network usage
            network = psutil.net_io_counters()
            
            # Calculate network rates
            current_time = time.time()
            network_mbps_sent = 0
            network_mbps_recv = 0
            
            if self._last_network:
                last_net, last_time = self._last_network
                time_diff = current_time - last_time
                if time_diff > 0:
                    bytes_sent_diff = network.bytes_sent - last_net.bytes_sent
                    bytes_recv_diff = network.bytes_recv - last_net.bytes_recv
                    network_mbps_sent = (bytes_sent_diff * 8) / (time_diff * 1_000_000)
                    network_mbps_recv = (bytes_recv_diff * 8) / (time_diff * 1_000_000)
            
            self._last_network = (network, current_time)
            
            # Create usage snapshot
            usage = ResourceUsage(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_total=memory.total,
                memory_available=memory.available,
                memory_percent=memory.percent,
                swap_total=swap.total,
                swap_used=swap.used,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_mbps_sent=network_mbps_sent,
                network_mbps_recv=network_mbps_recv
            )
            
            return usage
            
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {str(e)}", exc_info=True)
            return None
    
    def _add_to_history(self, usage: ResourceUsage) -> None:
        """Add resource usage to history."""
        if usage is None:
            return
            
        self.history.append(usage)
        
        # Limit history size
        if len(self.history) > self.history_max_size:
            self.history = self.history[-self.history_max_size:]
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback to be called when alerts are generated."""
        self.alert_callbacks.add(callback)
    
    def remove_alert_callback(self, callback: callable) -> None:
        """Remove an alert callback."""
        self.alert_callbacks.discard(callback)
    
    async def _check_thresholds(self, usage: ResourceUsage) -> None:
        """Check resource usage against thresholds and generate alerts."""
        if usage is None:
            return
            
        current_time = time.time()
        
        # Check CPU
        if usage.cpu_percent >= self.thresholds.cpu_percent:
            await self._add_alert(
                "cpu",
                usage.cpu_percent,
                self.thresholds.cpu_percent,
                f"CPU usage high: {usage.cpu_percent:.1f}%",
                current_time
            )
        
        # Check memory
        if usage.memory_percent >= self.thresholds.memory_percent:
            await self._add_alert(
                "memory",
                usage.memory_percent,
                self.thresholds.memory_percent,
                f"Memory usage high: {usage.memory_percent:.1f}%",
                current_time
            )
        
        # Check disk
        if usage.disk_percent >= self.thresholds.disk_percent:
            await self._add_alert(
                "disk",
                usage.disk_percent,
                self.thresholds.disk_percent,
                f"Disk usage high: {usage.disk_percent:.1f}%",
                current_time
            )
        
        # Check network
        total_network = usage.network_mbps_sent + usage.network_mbps_recv
        if total_network >= self.thresholds.network_mbps:
            await self._add_alert(
                "network",
                total_network,
                self.thresholds.network_mbps,
                f"Network usage high: {total_network:.1f} Mbps",
                current_time
            )
    
    async def _add_alert(self, resource_type: str, value: float, threshold: float, 
                        message: str, current_time: float) -> None:
        """Add a resource alert if enough time has passed since the last alert."""
        # Check if enough time has passed since last alert
        last_alert_time = self._last_alert.get(resource_type, 0)
        if (current_time - last_alert_time) < self.alert_interval:
            return
            
        # Create alert
        alert = ResourceAlert(
            timestamp=current_time,
            resource_type=resource_type,
            value=value,
            threshold=threshold,
            message=message
        )
        
        # Add to alerts list
        self.alerts.append(alert)
        self._last_alert[resource_type] = current_time
        self.stats["total_alerts"] += 1
        
        # Limit alerts list size
        if len(self.alerts) > self.alerts_max_size:
            self.alerts = self.alerts[-self.alerts_max_size:]
        
        # Log alert
        self.logger.warning(f"Resource Alert: {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")
    
    async def _apply_throttling(self, usage: ResourceUsage) -> None:
        """Apply resource throttling if enabled and thresholds are exceeded."""
        if not self.throttling_enabled or usage is None:
            return
            
        try:
            # Get list of processes sorted by CPU usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            # Check if we need to throttle
            if usage.cpu_percent >= self.thresholds.cpu_percent:
                # Throttle top CPU consuming processes
                for proc in processes[:3]:  # Top 3 processes
                    pid = proc['pid']
                    if pid not in self._throttled_processes:
                        try:
                            p = psutil.Process(pid)
                            p.nice(10)  # Lower priority
                            self._throttled_processes.add(pid)
                            self.stats["throttle_events"] += 1
                            self.logger.info(f"Throttled process {pid} ({proc['name']})")
                        except:
                            pass
            else:
                # Remove throttling if CPU usage is back to normal
                for pid in list(self._throttled_processes):
                    try:
                        p = psutil.Process(pid)
                        p.nice(0)  # Reset priority
                        self._throttled_processes.remove(pid)
                        self.logger.info(f"Unthrottled process {pid}")
                    except:
                        self._throttled_processes.discard(pid)
                        
        except Exception as e:
            self.logger.error(f"Error in throttling: {str(e)}", exc_info=True)
    
    def get_resource_history(self, duration: float = None) -> List[ResourceUsage]:
        """Get resource usage history."""
        if not duration:
            return self.history
            
        cutoff_time = time.time() - duration
        return [usage for usage in self.history if usage.timestamp >= cutoff_time]
    
    def get_alerts(self, duration: float = None) -> List[ResourceAlert]:
        """Get resource alerts."""
        if not duration:
            return self.alerts
            
        cutoff_time = time.time() - duration
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_stats(self) -> Dict:
        """Get resource manager statistics."""
        current_time = time.time()
        return {
            **self.stats,
            "uptime": current_time - self.stats["start_time"],
            "total_history_points": len(self.history),
            "total_alerts_stored": len(self.alerts),
            "throttled_processes": len(self._throttled_processes)
        }
