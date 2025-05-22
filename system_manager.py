"""
ORAMA System Control

This module provides system control and management capabilities including:
- System restart/update
- Power management
- System health checks
- Service management
"""

import os
import sys
import time
import asyncio
import logging
import subprocess
import platform
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil

@dataclass
class SystemConfig:
    """System configuration."""
    allow_restart: bool = False
    allow_shutdown: bool = False
    allow_update: bool = True
    health_check_interval: float = 300.0  # 5 minutes
    service_check_interval: float = 60.0  # 1 minute
    critical_services: List[str] = field(default_factory=list)
    update_command: str = ""
    backup_before_update: bool = True
    backup_path: str = "backups"
    max_backup_size: int = 1024 * 1024 * 1024  # 1GB

@dataclass
class SystemStatus:
    """System status information."""
    timestamp: float
    hostname: str
    platform: str
    python_version: str
    uptime: float
    boot_time: float
    cpu_count: int
    load_average: Tuple[float, float, float]
    services_status: Dict[str, bool]
    power_status: Dict[str, Any]

@dataclass
class UpdateResult:
    """Result of a system update."""
    success: bool
    timestamp: float
    duration: float
    changes: List[str]
    errors: List[str]
    backup_path: Optional[str]
    version_before: str
    version_after: str

class SystemManager:
    """
    Manages system control and monitoring.
    
    Features:
    - System status monitoring
    - Service management
    - System updates
    - Power management
    """
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """Initialize the system manager."""
        self.config = SystemConfig(**config) if config else SystemConfig()
        self.logger = logger or logging.getLogger("orama.system")
        
        # System information
        self.hostname = platform.node()
        self.platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        # Status history
        self.status_history: List[SystemStatus] = []
        self.max_history_size = 1000
        
        # Service status
        self.service_status: Dict[str, bool] = {}
        self.service_restart_attempts: Dict[str, int] = {}
        self.max_restart_attempts = 3
        
        # Update history
        self.update_history: List[UpdateResult] = []
        self.max_update_history = 100
        
        # Monitoring state
        self._running = False
        self._monitoring_task = None
        self._service_check_task = None
        
        # Status callbacks
        self._status_callbacks: Set[callable] = set()
        
        self.logger.info("System manager initialized")
    
    async def start(self) -> None:
        """Start system monitoring."""
        if self._running:
            return
            
        self.logger.info("Starting system monitoring")
        self._running = True
        
        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._service_check_task = asyncio.create_task(self._service_check_loop())
    
    async def stop(self) -> None:
        """Stop system monitoring."""
        if not self._running:
            return
            
        self.logger.info("Stopping system monitoring")
        self._running = False
        
        # Stop monitoring tasks
        for task in [self._monitoring_task, self._service_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._monitoring_task = None
        self._service_check_task = None
    
    async def _monitoring_loop(self) -> None:
        """Background task that monitors system health."""
        try:
            while self._running:
                # Get system status
                status = await self._get_system_status()
                
                # Add to history
                self._add_to_history(status)
                
                # Notify callbacks
                await self._notify_status_callbacks(status)
                
                # Wait for next check
                await asyncio.sleep(self.config.health_check_interval)
                
        except asyncio.CancelledError:
            self.logger.info("System monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in system monitoring loop: {str(e)}", exc_info=True)
            self._running = False
    
    async def _service_check_loop(self) -> None:
        """Background task that monitors critical services."""
        try:
            while self._running:
                # Check critical services
                for service in self.config.critical_services:
                    try:
                        running = await self._check_service(service)
                        
                        # Update status
                        previous_status = self.service_status.get(service)
                        self.service_status[service] = running
                        
                        # Handle service failure
                        if previous_status and not running:
                            self.logger.warning(f"Service {service} stopped running")
                            
                            # Attempt restart if configured
                            if service in self.service_restart_attempts:
                                if self.service_restart_attempts[service] < self.max_restart_attempts:
                                    self.logger.info(f"Attempting to restart {service}")
                                    await self._restart_service(service)
                                    self.service_restart_attempts[service] += 1
                            else:
                                self.service_restart_attempts[service] = 0
                                await self._restart_service(service)
                        
                        # Reset restart attempts if service is running
                        elif running and service in self.service_restart_attempts:
                            self.service_restart_attempts[service] = 0
                            
                    except Exception as e:
                        self.logger.error(f"Error checking service {service}: {str(e)}")
                
                # Wait for next check
                await asyncio.sleep(self.config.service_check_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Service check loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in service check loop: {str(e)}", exc_info=True)
    
    async def _get_system_status(self) -> SystemStatus:
        """Get current system status."""
        try:
            # Get load average
            if hasattr(os, "getloadavg"):
                load_avg = os.getloadavg()
            else:
                load_avg = (0.0, 0.0, 0.0)
            
            # Get power status
            power_status = {}
            try:
                battery = psutil.sensors_battery()
                if battery:
                    power_status = {
                        "battery_percent": battery.percent,
                        "power_plugged": battery.power_plugged,
                        "battery_time_left": battery.secsleft
                    }
            except:
                pass
            
            # Create status
            status = SystemStatus(
                timestamp=time.time(),
                hostname=self.hostname,
                platform=self.platform_info["system"],
                python_version=sys.version,
                uptime=time.time() - psutil.boot_time(),
                boot_time=psutil.boot_time(),
                cpu_count=os.cpu_count() or 0,
                load_average=load_avg,
                services_status=self.service_status.copy(),
                power_status=power_status
            )
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}", exc_info=True)
            return None
    
    def _add_to_history(self, status: SystemStatus) -> None:
        """Add status to history."""
        if status is None:
            return
            
        self.status_history.append(status)
        
        # Limit history size
        if len(self.status_history) > self.max_history_size:
            self.status_history = self.status_history[-self.max_history_size:]
    
    def add_status_callback(self, callback: callable) -> None:
        """Add a callback to be called when system status changes."""
        self._status_callbacks.add(callback)
    
    def remove_status_callback(self, callback: callable) -> None:
        """Remove a status callback."""
        self._status_callbacks.discard(callback)
    
    async def _notify_status_callbacks(self, status: SystemStatus) -> None:
        """Notify all status callbacks."""
        for callback in self._status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status)
                else:
                    callback(status)
            except Exception as e:
                self.logger.error(f"Error in status callback: {str(e)}")
    
    async def _check_service(self, service: str) -> bool:
        """Check if a service is running."""
        try:
            if sys.platform == "win32":
                # Windows service check
                result = await asyncio.create_subprocess_exec(
                    "sc", "query", service,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                return b"RUNNING" in stdout
            else:
                # Unix service check
                result = await asyncio.create_subprocess_exec(
                    "systemctl", "is-active", service,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                return b"active" in stdout
                
        except Exception as e:
            self.logger.error(f"Service check error for {service}: {str(e)}")
            return False
    
    async def _restart_service(self, service: str) -> bool:
        """Restart a system service."""
        try:
            if sys.platform == "win32":
                # Windows service restart
                result = await asyncio.create_subprocess_exec(
                    "sc", "stop", service,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await result.communicate()
                
                await asyncio.sleep(2)  # Wait for service to stop
                
                result = await asyncio.create_subprocess_exec(
                    "sc", "start", service,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await result.communicate()
            else:
                # Unix service restart
                result = await asyncio.create_subprocess_exec(
                    "systemctl", "restart", service,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await result.communicate()
            
            # Verify service is running
            await asyncio.sleep(5)  # Wait for service to start
            return await self._check_service(service)
            
        except Exception as e:
            self.logger.error(f"Service restart error for {service}: {str(e)}")
            return False
    
    async def update_system(self) -> UpdateResult:
        """Perform system update."""
        if not self.config.allow_update:
            raise PermissionError("System updates are not allowed")
        
        start_time = time.time()
        version_before = self._get_system_version()
        changes = []
        errors = []
        backup_path = None
        
        try:
            # Create backup if configured
            if self.config.backup_before_update:
                backup_path = await self._create_backup()
            
            # Run update command
            if self.config.update_command:
                self.logger.info("Running system update")
                
                result = await asyncio.create_subprocess_exec(
                    *self.config.update_command.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                stdout, stderr = await result.communicate()
                
                # Process output
                if stdout:
                    changes.extend(stdout.decode().splitlines())
                if stderr:
                    errors.extend(stderr.decode().splitlines())
                
                success = result.returncode == 0
            else:
                raise ValueError("No update command configured")
            
            # Get new version
            version_after = self._get_system_version()
            
            # Create result
            result = UpdateResult(
                success=success,
                timestamp=time.time(),
                duration=time.time() - start_time,
                changes=changes,
                errors=errors,
                backup_path=backup_path,
                version_before=version_before,
                version_after=version_after
            )
            
            # Add to history
            self.update_history.append(result)
            if len(self.update_history) > self.max_update_history:
                self.update_history = self.update_history[-self.max_update_history:]
            
            # Log result
            if success:
                self.logger.info("System update completed successfully")
            else:
                self.logger.error("System update failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"System update error: {str(e)}", exc_info=True)
            
            # Create error result
            return UpdateResult(
                success=False,
                timestamp=time.time(),
                duration=time.time() - start_time,
                changes=changes,
                errors=[str(e)] + errors,
                backup_path=backup_path,
                version_before=version_before,
                version_after=version_before
            )
    
    def _get_system_version(self) -> str:
        """Get current system version."""
        try:
            if sys.platform == "win32":
                return platform.version()
            else:
                # Try to get Linux distribution version
                try:
                    import distro
                    return distro.version()
                except ImportError:
                    return platform.version()
        except:
            return "unknown"
    
    async def _create_backup(self) -> Optional[str]:
        """Create system backup."""
        try:
            # Create backup directory
            os.makedirs(self.config.backup_path, exist_ok=True)
            
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.config.backup_path, f"backup_{timestamp}.tar.gz")
            
            # Create backup command
            if sys.platform == "win32":
                # Windows backup using wbadmin
                result = await asyncio.create_subprocess_exec(
                    "wbadmin", "start", "backup",
                    "-backupTarget:"+self.config.backup_path,
                    "-quiet",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                # Unix backup using tar
                result = await asyncio.create_subprocess_exec(
                    "tar", "czf", backup_file,
                    "--exclude=/proc", "--exclude=/sys",
                    "--exclude=/dev", "--exclude=/run",
                    "--exclude=/mnt", "--exclude=/media",
                    "--exclude=/tmp", "--exclude=/var/cache",
                    "/",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            await result.communicate()
            
            if result.returncode == 0:
                self.logger.info(f"System backup created: {backup_file}")
                return backup_file
            else:
                self.logger.error("System backup failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Backup error: {str(e)}", exc_info=True)
            return None
    
    async def restart_system(self, delay: int = 0) -> bool:
        """Restart the system."""
        if not self.config.allow_restart:
            raise PermissionError("System restart is not allowed")
            
        try:
            self.logger.warning(f"System restart initiated (delay: {delay}s)")
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            if sys.platform == "win32":
                os.system("shutdown /r /t 0")
            else:
                os.system("shutdown -r now")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Restart error: {str(e)}", exc_info=True)
            return False
    
    async def shutdown_system(self, delay: int = 0) -> bool:
        """Shut down the system."""
        if not self.config.allow_shutdown:
            raise PermissionError("System shutdown is not allowed")
            
        try:
            self.logger.warning(f"System shutdown initiated (delay: {delay}s)")
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            if sys.platform == "win32":
                os.system("shutdown /s /t 0")
            else:
                os.system("shutdown -h now")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {str(e)}", exc_info=True)
            return False
    
    def get_status_history(self, duration: float = None) -> List[SystemStatus]:
        """Get system status history."""
        if not duration:
            return self.status_history
            
        cutoff_time = time.time() - duration
        return [status for status in self.status_history if status.timestamp >= cutoff_time]
    
    def get_update_history(self) -> List[UpdateResult]:
        """Get system update history."""
        return self.update_history
    
    def get_stats(self) -> Dict:
        """Get system manager statistics."""
        return {
            "status_history_size": len(self.status_history),
            "update_history_size": len(self.update_history),
            "monitored_services": len(self.config.critical_services),
            "active_services": sum(1 for s in self.service_status.values() if s),
            "restart_attempts": sum(self.service_restart_attempts.values())
        }
    
    def get_power_info(self) -> Dict:
        """Get power and battery information."""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "battery_percent": battery.percent,
                    "power_plugged": battery.power_plugged,
                    "battery_time_left": battery.secsleft,
                    "battery_present": True
                }
            return {"battery_present": False}
        except:
            return {"battery_present": False}
