"""
ORAMA Action System

This module provides a comprehensive system for executing and managing various actions,
including command execution, input simulation, browser automation, and file operations.
It includes advanced capabilities for resource management, debugging, and system control.
"""

import os
import sys
import re
import json
import time
import uuid
import asyncio
import logging
import tempfile
import traceback
import shutil
import hashlib
import subprocess
import signal
import base64
import socket
import requests
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable, TypeVar, Coroutine
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urljoin
from io import BytesIO

# Import ORAMA modules
from orama.resource_manager import ResourceManager
from orama.debug_manager import DebugManager
from orama.system_manager import SystemManager

# Input simulation
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Enable failsafe
    INPUT_SIMULATION_AVAILABLE = True
except ImportError:
    INPUT_SIMULATION_AVAILABLE = False

# Win32 API for more advanced Windows interactions
try:
    import win32api
    import win32con
    import win32gui
    import win32process
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

# Browser automation
try:
    import playwright.async_api as playwright
    BROWSER_AUTOMATION_AVAILABLE = True
except ImportError:
    BROWSER_AUTOMATION_AVAILABLE = False

# Clipboard support (use pyperclip for cross-platform clipboard)
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    pyperclip = None

# Audio support
try:
    import winsound  # Windows-specific sound
    import pyttsx3    # Text-to-speech
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Screen recording
try:
    import cv2
    import numpy as np
    SCREEN_RECORDING_AVAILABLE = True
except ImportError:
    SCREEN_RECORDING_AVAILABLE = False

# Network operations (direct HTTP requests)
try:
    import requests
    from requests.exceptions import RequestException
    NETWORK_REQUESTS_AVAILABLE = True
except ImportError:
    NETWORK_REQUESTS_AVAILABLE = False

# Device control
try:
    import serial  # For serial devices (USB, etc.)
    import hid     # For HID devices
    DEVICE_CONTROL_AVAILABLE = True
except ImportError:
    DEVICE_CONTROL_AVAILABLE = False

# Resource monitoring
try:
    import psutil
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False

# Constants
DEFAULT_SHELL = "powershell"  # Default shell for Windows
DEFAULT_COMMAND_TIMEOUT = 30.0  # seconds
DEFAULT_BROWSER_TIMEOUT = 60.0  # seconds
DEFAULT_INPUT_DELAY_FACTOR = 1.0  # Normal human speed
MAX_COMMAND_OUTPUT = 10240  # characters
MAX_RETRIES = 3
DEFAULT_DELAY_BETWEEN_KEYS = 0.05  # seconds
DEFAULT_DELAY_BETWEEN_CLICKS = 0.1  # seconds
DEFAULT_DOWNLOAD_PATH = os.path.join(os.path.expanduser("~"), "Downloads", "orama")
DEFAULT_SCREENSHOT_PATH = os.path.join(os.path.expanduser("~"), "Pictures", "orama")

# Action types
class ActionType(Enum):
    COMMAND_EXECUTION = auto()
    KEYBOARD_INPUT = auto()
    MOUSE_CLICK = auto()
    BROWSER_NAVIGATION = auto()
    FILE_OPERATION = auto()
    SCREENSHOT = auto()
    CLIPBOARD_OPERATION = auto()
    PROCESS_MANAGEMENT = auto()
    WINDOW_MANAGEMENT = auto()
    AUDIO_OPERATION = auto()      # Audio playback and text-to-speech
    SCREEN_RECORDING = auto()     # Video capture of screen
    NETWORK_OPERATION = auto()    # Direct network requests
    DEVICE_CONTROL = auto()       # External device integration
    SYSTEM_CONTROL = auto()       # System-level operations
    PLUGIN_OPERATION = auto()     # Plugin/extension management
    SESSION_MANAGEMENT = auto()   # User session management
    RESOURCE_MANAGEMENT = auto()  # Resource limits and controls
    SECURITY_CONTROL = auto()     # Security and permission management
    CUSTOM = auto()

# Action status
class ActionStatus(Enum):
    PENDING = auto()
    EXECUTING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()

# Action validation result
class ValidationResult(Enum):
    SAFE = auto()
    UNSAFE = auto()
    REQUIRES_CONFIRMATION = auto()

# Browser types
class BrowserType(Enum):
    CHROMIUM = auto()
    FIREFOX = auto()
    WEBKIT = auto()

@dataclass
class Point:
    """2D point for screen coordinates."""
    x: int
    y: int
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def as_tuple(self) -> Tuple[int, int]:
        """Get point as tuple."""
        return (self.x, self.y)

@dataclass
class Rectangle:
    """Rectangle for screen regions."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def left(self) -> int:
        return self.x
    
    @property
    def top(self) -> int:
        return self.y
    
    @property
    def right(self) -> int:
        return self.x + self.width
    
    @property
    def bottom(self) -> int:
        return self.y + self.height
    
    @property
    def center(self) -> Point:
        return Point(self.x + self.width // 2, self.y + self.height // 2)
    
    def contains(self, point: Point) -> bool:
        """Check if point is inside rectangle."""
        return (self.left <= point.x < self.right and 
                self.top <= point.y < self.bottom)
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Get rectangle as tuple."""
        return (self.x, self.y, self.width, self.height)

@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    stdout: Optional[str] = None
    stderr: Optional[str] = None

@dataclass
class Action:
    """Action to be executed by the system."""
    action_type: ActionType
    parameters: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"action_{uuid.uuid4().hex[:8]}")
    description: str = ""
    status: ActionStatus = ActionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[ActionResult] = None
    verification: Optional[Dict] = None
    requires_confirmation: bool = False
    retry_count: int = 0
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert action to dictionary."""
        data = asdict(self)
        data["action_type"] = self.action_type.name
        data["status"] = self.status.name
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        """Create action from dictionary."""
        action_type = ActionType[data["action_type"]] if isinstance(data["action_type"], str) else data["action_type"]
        status = ActionStatus[data["status"]] if isinstance(data["status"], str) else data["status"]
        
        # Convert result if present
        result = None
        if data.get("result"):
            result_data = data["result"]
            result = ActionResult(
                success=result_data.get("success", False),
                data=result_data.get("data"),
                error=result_data.get("error"),
                duration=result_data.get("duration", 0.0),
                stdout=result_data.get("stdout"),
                stderr=result_data.get("stderr")
            )
        
        return cls(
            id=data.get("id", f"action_{uuid.uuid4().hex[:8]}"),
            action_type=action_type,
            parameters=data.get("parameters", {}),
            description=data.get("description", ""),
            status=status,
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=result,
            verification=data.get("verification"),
            requires_confirmation=data.get("requires_confirmation", False),
            retry_count=data.get("retry_count", 0),
            parent_id=data.get("parent_id")
        )

@dataclass
class WindowInfo:
    """Information about a window."""
    handle: int
    title: str
    classname: str
    rect: Rectangle
    process_id: int
    process_name: str
    is_visible: bool
    
    def to_dict(self) -> Dict:
        """Convert window info to dictionary."""
        return {
            "handle": self.handle,
            "title": self.title,
            "classname": self.classname,
            "rect": asdict(self.rect),
            "process_id": self.process_id,
            "process_name": self.process_name,
            "is_visible": self.is_visible
        }

class ActionSystem:
    """
    ORAMA Action System
    
    Provides system access, input simulation, and execution mechanisms for the autonomous agent.
    The action system is responsible for:
    
    1. Command Execution: Safely running system commands with proper validation
    2. Input Simulation: Keyboard and mouse control for application interaction
    3. Browser Automation: Web browsing, content processing, and data extraction
    4. File System Operations: File and directory management with appropriate permissions
    5. Process Management: Starting, monitoring, and controlling processes
    6. Window Management: Finding, focusing, and manipulating application windows
    
    Safety features include:
    - Command validation and sanitization
    - Confirmation for dangerous operations
    - Resource usage monitoring
    - Automatic error recovery
    - Comprehensive logging and auditing
    """
    
    def __init__(self, config: Dict, memory_engine=None, cognitive_engine=None, logger=None):
        """Initialize the action system with configuration."""
        self.config = config
        self.memory = memory_engine
        self.cognitive = cognitive_engine
        self.logger = logger or logging.getLogger("orama.action")
        
        # Initialize the advanced modules
        self.resource_manager = ResourceManager(
            config=config.get("resources", {}),
            logger=logging.getLogger("orama.resources")
        )
        
        self.debug_manager = DebugManager(
            config=config.get("debug", {})
        )
        
        self.system_manager = SystemManager(
            config=config.get("system", {}),
            logger=logging.getLogger("orama.system")
        )
        
        # Command execution configuration
        self.command_config = config.get("command", {})
        self.shell = self.command_config.get("shell", DEFAULT_SHELL)
        self.command_timeout = self.command_config.get("timeout", DEFAULT_COMMAND_TIMEOUT)
        self.max_output = self.command_config.get("max_output", MAX_COMMAND_OUTPUT)
        self.dangerous_commands = set(self.command_config.get("dangerous_commands", [
            "format", "del /s", "rm -rf", "shutdown", "taskkill", "netsh firewall"
        ]))
        
        # Input simulation configuration
        self.input_config = config.get("input", {})
        self.delay_factor = self.input_config.get("delay_factor", DEFAULT_INPUT_DELAY_FACTOR)
        self.verification = self.input_config.get("verification", True)
        self.retry_count = self.input_config.get("retry_count", MAX_RETRIES)
        self.click_verification = self.input_config.get("click_verification", True)
        
        # Browser automation configuration
        self.browser_config = config.get("browser", {})
        self.browser_engine = self.browser_config.get("engine", "playwright")
        self.browser_type = self.browser_config.get("browser_type", "chromium")
        self.headless = self.browser_config.get("headless", False)
        self.user_agent = self.browser_config.get("user_agent", None)
        self.browser_timeout = self.browser_config.get("timeout", DEFAULT_BROWSER_TIMEOUT)
        self.allow_downloads = self.browser_config.get("allow_downloads", True)
        self.download_path = self.browser_config.get("download_path", DEFAULT_DOWNLOAD_PATH)
        
        # Initialize components
        self.browser = None
        self.browser_context = None
        self.current_page = None
        self.previous_clipboard = None
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize recording state
        self._recording = None
        self._recording_task = None
        self._setup_recording()
        
        # Recent actions tracking
        self.recent_actions: List[Action] = []
        self.max_recent_actions = 100
        
        # Command history
        self.command_history: List[Dict] = []
        self.max_command_history = 100
        
        # Browser history
        self.browser_history: List[Dict] = []
        self.max_browser_history = 100
        
        # Windows mapping
        self.windows_cache: Dict[int, WindowInfo] = {}
        self.windows_cache_time = 0
        self.windows_cache_ttl = 5.0  # seconds
        
        # Thread pool for background operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Tasks
        self._running = False
        self._browser_init_task = None
        
        # Locks
        self._command_lock = asyncio.Lock()
        self._input_lock = asyncio.Lock()
        self._browser_lock = asyncio.Lock()
        self._windows_lock = asyncio.Lock()
        
        self.logger.info("Action system initialized")
    
    async def start(self) -> None:
        """Start the action system and initialize resources."""
        self.logger.info("Starting action system...")
        self._running = True
        
        try:
            # Ensure necessary directories exist
            self._create_directories()
            
            # Start advanced modules
            await self.resource_manager.start()
            await self.debug_manager.start()
            await self.system_manager.start()
            
            # Initialize input simulation if available
            if not INPUT_SIMULATION_AVAILABLE:
                self.logger.warning("PyAutoGUI not available, input simulation will be limited")
            else:
                # Configure PyAutoGUI
                pyautogui.PAUSE = 0.1 * self.delay_factor
                self.logger.info("Input simulation initialized")
            
            # Initialize browser in background if configured to auto-start
            if self.browser_config.get("auto_start", False) and BROWSER_AUTOMATION_AVAILABLE:
                self._browser_init_task = asyncio.create_task(self._init_browser())
            
            # Initialize recording state
            self._setup_recording()
            
            # Record startup in history
            action = Action(
                action_type=ActionType.CUSTOM,
                description="Action system startup",
                status=ActionStatus.SUCCEEDED,
                result=ActionResult(success=True, data={
                    "message": "Action system started",
                    "resource_manager": "active",
                    "debug_manager": "active",
                    "system_manager": "active"
                })
            )
            action.started_at = time.time()
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info("Action system started successfully")
        except Exception as e:
            self._running = False
            self.logger.error(f"Failed to start action system: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the action system and clean up resources."""
        self.logger.info("Stopping action system...")
        self._running = False
        
        try:
            # Stop advanced modules
            try:
                await self.resource_manager.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping resource manager: {e}")
                
            try:
                await self.debug_manager.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping debug manager: {e}")
                
            try:
                await self.system_manager.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping system manager: {e}")
            
            # Close browser if open
            if self.browser:
                try:
                    await self._close_browser()
                except Exception as e:
                    self.logger.warning(f"Error closing browser: {e}")
            
            # Cancel browser init task if running
            if self._browser_init_task and not self._browser_init_task.done():
                self._browser_init_task.cancel()
                try:
                    await self._browser_init_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.warning(f"Error cancelling browser init task: {e}")
            
            # Stop recording if active
            if self._recording and self._recording['running']:
                try:
                    await self.stop_screen_recording()
                except Exception as e:
                    self.logger.warning(f"Error stopping screen recording: {e}")
            
            # Clean up temporary files
            try:
                temp_dir = tempfile.gettempdir()
                orama_temp_pattern = os.path.join(temp_dir, "orama_*")
                for temp_file in Path(temp_dir).glob("orama_*"):
                    if temp_file.is_file():
                        temp_file.unlink()
            except Exception as e:
                self.logger.warning(f"Error cleaning up temporary files: {e}")
            
            # Clean up thread pool
            self.executor.shutdown(wait=False)
            
            self.logger.info("Action system stopped")
        except Exception as e:
            self.logger.error(f"Error during action system shutdown: {e}", exc_info=True)
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        try:
            # Create download directory
            download_path = Path(self.download_path)
            download_path.mkdir(parents=True, exist_ok=True)
            
            # Create screenshot directory
            screenshot_path = Path(DEFAULT_SCREENSHOT_PATH)
            screenshot_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Error creating directories: {e}")
    
    def _setup_recording(self):
        """Setup recording defaults."""
        if self._recording is None:
            self._recording = {
                'writer': None,
                'fps': 20,
                'region': None,
                'output_path': None,
                'frames': 0,
                'start_time': None,
                'action_id': None,
                'running': False
            }
    
    #--------------------------------------------------------------------
    # Command Execution Methods
    #--------------------------------------------------------------------
    
    async def execute_command(self, command: str, shell: bool = True, 
                             cwd: Optional[str] = None, env: Optional[Dict] = None, 
                             timeout: Optional[float] = None) -> ActionResult:
        """Execute a system command with validation and safety checks."""
        async with self._command_lock:
            start_time = time.time()
            action_id = f"cmd_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.COMMAND_EXECUTION,
                description=f"Execute command: {self._truncate_command(command)}",
                parameters={
                    "command": command,
                    "shell": shell,
                    "cwd": cwd,
                    "env": env,
                    "timeout": timeout
                }
            )
            action.started_at = start_time
            
            self.logger.info(f"Executing command: {self._truncate_command(command)}")
            
            try:
                # Validate command
                validation_result, reason = self._validate_command(command)
                
                # Handle unsafe commands
                if validation_result == ValidationResult.UNSAFE:
                    error_msg = f"Command validation failed: {reason}"
                    self.logger.warning(error_msg)
                    
                    # Create result
                    result = ActionResult(
                        success=False,
                        error=error_msg,
                        duration=time.time() - start_time
                    )
                    
                    # Update action
                    action.status = ActionStatus.FAILED
                    action.result = result
                    action.completed_at = time.time()
                    self._add_to_recent_actions(action)
                    self._add_to_command_history(command, False, error_msg)
                    
                    return result
                
                # For commands that require confirmation, mark as such
                if validation_result == ValidationResult.REQUIRES_CONFIRMATION:
                    action.requires_confirmation = True
                    # In a real system, this would wait for user confirmation
                    # For now, we'll allow it to proceed with a warning
                    self.logger.warning(f"Command requires confirmation: {self._truncate_command(command)}")
                
                # Use provided timeout or default
                cmd_timeout = timeout or self.command_timeout
                
                # Prepare environment
                command_env = os.environ.copy()
                if env:
                    command_env.update(env)
                
                # Run command in thread to avoid blocking
                stdout_data, stderr_data, return_code = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._run_command_sync,
                    command, shell, cwd, command_env, cmd_timeout
                )
                
                # Truncate output if too long
                if stdout_data and len(stdout_data) > self.max_output:
                    stdout_data = stdout_data[:self.max_output] + f"\n... [Output truncated, {len(stdout_data)} total bytes]"
                
                if stderr_data and len(stderr_data) > self.max_output:
                    stderr_data = stderr_data[:self.max_output] + f"\n... [Output truncated, {len(stderr_data)} total bytes]"
                
                # Determine success based on return code
                success = return_code == 0
                
                # Create result
                result = ActionResult(
                    success=success,
                    data={"return_code": return_code},
                    error=f"Command failed with return code {return_code}" if not success else None,
                    duration=time.time() - start_time,
                    stdout=stdout_data,
                    stderr=stderr_data
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED if success else ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                # Add to history
                self._add_to_command_history(command, success, stderr_data if not success else None)
                
                # Log result
                if success:
                    self.logger.info(f"Command executed successfully: {self._truncate_command(command)}")
                else:
                    self.logger.warning(f"Command failed: {self._truncate_command(command)}, Return code: {return_code}")
                
                return result
            except asyncio.TimeoutError:
                # Handle command timeout
                error_msg = f"Command timed out after {cmd_timeout:.1f} seconds"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                self._add_to_command_history(command, False, error_msg)
                
                return result
            except Exception as e:
                # Handle other errors
                error_msg = f"Command execution error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                self._add_to_command_history(command, False, error_msg)
                
                return result
    
    def _run_command_sync(self, command: str, shell: bool, cwd: Optional[str], 
                        env: Dict, timeout: float) -> Tuple[str, str, int]:
        """Run command synchronously in a separate thread."""
        try:
            # Start the process
            process = subprocess.Popen(
                command if shell else command.split(),
                shell=shell,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return stdout, stderr, process.returncode
            except subprocess.TimeoutExpired:
                # Kill process if it times out
                process.kill()
                process.wait()
                raise asyncio.TimeoutError(f"Command timed out after {timeout:.1f} seconds")
        except Exception as e:
            return "", str(e), -1
    
    def _validate_command(self, command: str) -> Tuple[ValidationResult, str]:
        """Validate a command for safety concerns."""
        # Check for empty command
        if not command or not command.strip():
            return ValidationResult.UNSAFE, "Empty command"
        
        # Check for dangerous commands
        cmd_lower = command.lower()
        for dangerous_cmd in self.dangerous_commands:
            if dangerous_cmd.lower() in cmd_lower:
                return ValidationResult.REQUIRES_CONFIRMATION, f"Command contains dangerous pattern: {dangerous_cmd}"
        
        # Check for potentially unsafe shell characters in non-shell mode
        if "|" in command or ">" in command or "<" in command or "&" in command:
            return ValidationResult.REQUIRES_CONFIRMATION, "Command contains shell operators"
        
        # Check for system directories
        system_paths = [
            os.environ.get("WINDIR", "C:\\Windows"),
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "System32"),
            "/etc", "/var", "/usr", "/boot"
        ]
        
        for path in system_paths:
            if path.lower() in cmd_lower:
                return ValidationResult.REQUIRES_CONFIRMATION, f"Command references system directory: {path}"
        
        return ValidationResult.SAFE, ""
    
    def _truncate_command(self, command: str, max_length: int = 100) -> str:
        """Truncate command for logging."""
        if len(command) <= max_length:
            return command
        return command[:max_length] + "..."
    
    def _add_to_command_history(self, command: str, success: bool, error: Optional[str]) -> None:
        """Add command to history."""
        # Create history entry
        entry = {
            "command": command,
            "timestamp": time.time(),
            "success": success,
            "error": error
        }
        
        # Add to history
        self.command_history.append(entry)
        
        # Limit history size
        if len(self.command_history) > self.max_command_history:
            self.command_history = self.command_history[-self.max_command_history:]
    
    #--------------------------------------------------------------------
    # Input Simulation Methods
    #--------------------------------------------------------------------
    
    async def type_text(self, text: str, interval: Optional[float] = None) -> ActionResult:
        """Type text using keyboard simulation."""
        async with self._input_lock:
            start_time = time.time()
            action_id = f"input_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.KEYBOARD_INPUT,
                description=f"Type text: {text[:20] + '...' if len(text) > 20 else text}",
                parameters={
                    "text": text,
                    "interval": interval
                }
            )
            action.started_at = start_time
            
            if not INPUT_SIMULATION_AVAILABLE:
                error_msg = "Input simulation not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Calculate typing interval
                typing_interval = interval if interval is not None else DEFAULT_DELAY_BETWEEN_KEYS * self.delay_factor
                
                # Type text in thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: pyautogui.write(text, interval=typing_interval)
                )
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={"characters_typed": len(text)},
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Typed text: {text[:20] + '...' if len(text) > 20 else text}")
                return result
            except Exception as e:
                error_msg = f"Text typing error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def press_key(self, key: str, presses: int = 1, interval: Optional[float] = None) -> ActionResult:
        """Press a keyboard key."""
        async with self._input_lock:
            start_time = time.time()
            action_id = f"key_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.KEYBOARD_INPUT,
                description=f"Press key: {key} ({presses} times)",
                parameters={
                    "key": key,
                    "presses": presses,
                    "interval": interval
                }
            )
            action.started_at = start_time
            
            if not INPUT_SIMULATION_AVAILABLE:
                error_msg = "Input simulation not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Calculate press interval
                press_interval = interval if interval is not None else DEFAULT_DELAY_BETWEEN_KEYS * self.delay_factor
                
                # Press key in thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: pyautogui.press(key, presses=presses, interval=press_interval)
                )
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={"key": key, "presses": presses},
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Pressed key: {key} ({presses} times)")
                return result
            except Exception as e:
                error_msg = f"Key press error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def press_keys(self, keys: List[str]) -> ActionResult:
        """Press a combination of keys."""
        async with self._input_lock:
            start_time = time.time()
            action_id = f"keys_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.KEYBOARD_INPUT,
                description=f"Press keys: {'+'.join(keys)}",
                parameters={
                    "keys": keys
                }
            )
            action.started_at = start_time
            
            if not INPUT_SIMULATION_AVAILABLE:
                error_msg = "Input simulation not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Press key combination in thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: pyautogui.hotkey(*keys)
                )
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={"keys": keys},
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Pressed key combination: {'+'.join(keys)}")
                return result
            except Exception as e:
                error_msg = f"Key combination error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def mouse_move(self, x: int, y: int, duration: Optional[float] = None) -> ActionResult:
        """Move the mouse cursor to specific coordinates."""
        async with self._input_lock:
            start_time = time.time()
            action_id = f"mouse_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.MOUSE_CLICK,
                description=f"Move mouse to: ({x}, {y})",
                parameters={
                    "x": x,
                    "y": y,
                    "duration": duration
                }
            )
            action.started_at = start_time
            
            if not INPUT_SIMULATION_AVAILABLE:
                error_msg = "Input simulation not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Calculate move duration
                move_duration = duration if duration is not None else DEFAULT_DELAY_BETWEEN_KEYS * self.delay_factor
                
                # Move mouse in thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: pyautogui.moveTo(x, y, duration=move_duration)
                )
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={"x": x, "y": y},
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Moved mouse to: ({x}, {y})")
                return result
            except Exception as e:
                error_msg = f"Mouse move error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def mouse_click(self, x: Optional[int] = None, y: Optional[int] = None, 
                        button: str = "left", clicks: int = 1, 
                        interval: Optional[float] = None) -> ActionResult:
        """Click the mouse at the specified coordinates."""
        async with self._input_lock:
            start_time = time.time()
            action_id = f"click_{uuid.uuid4().hex[:8]}"
            
            # If no coordinates provided, use current position
            if x is None or y is None:
                if INPUT_SIMULATION_AVAILABLE:
                    current_pos = pyautogui.position()
                    x = x if x is not None else current_pos.x
                    y = y if y is not None else current_pos.y
                else:
                    x = x if x is not None else 0
                    y = y if y is not None else 0
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.MOUSE_CLICK,
                description=f"Click {button} button at: ({x}, {y}) ({clicks} times)",
                parameters={
                    "x": x,
                    "y": y,
                    "button": button,
                    "clicks": clicks,
                    "interval": interval
                }
            )
            action.started_at = start_time
            
            if not INPUT_SIMULATION_AVAILABLE:
                error_msg = "Input simulation not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Calculate click interval
                click_interval = interval if interval is not None else DEFAULT_DELAY_BETWEEN_CLICKS * self.delay_factor
                
                # Click in thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: pyautogui.click(x, y, clicks=clicks, interval=click_interval, button=button)
                )
                
                # Verify click if enabled
                click_verified = True
                if self.click_verification and WIN32_AVAILABLE:
                    # Get window at click position
                    window_handle = win32gui.WindowFromPoint((x, y))
                    if window_handle:
                        # Check if window is responsive
                        try:
                            window_title = win32gui.GetWindowText(window_handle)
                            # If title is empty but handle exists, try to get class name
                            if not window_title:
                                window_class = win32gui.GetClassName(window_handle)
                                if not window_class:
                                    click_verified = False
                        except Exception:
                            click_verified = False
                    else:
                        click_verified = False
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={
                        "x": x,
                        "y": y,
                        "button": button,
                        "clicks": clicks,
                        "verified": click_verified
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Clicked {button} button at: ({x}, {y}) ({clicks} times)")
                return result
            except Exception as e:
                error_msg = f"Mouse click error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def mouse_drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                       duration: Optional[float] = None, button: str = "left") -> ActionResult:
        """Drag the mouse from start to end coordinates."""
        async with self._input_lock:
            start_time = time.time()
            action_id = f"drag_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.MOUSE_CLICK,
                description=f"Drag mouse from: ({start_x}, {start_y}) to ({end_x}, {end_y})",
                parameters={
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "duration": duration,
                    "button": button
                }
            )
            action.started_at = start_time
            
            if not INPUT_SIMULATION_AVAILABLE:
                error_msg = "Input simulation not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Calculate drag duration
                drag_duration = duration if duration is not None else 0.5 * self.delay_factor
                
                # Drag in thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: pyautogui.dragTo(end_x, end_y, duration=drag_duration, button=button, mouseDownUp=True)
                )
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={
                        "start_x": start_x,
                        "start_y": start_y,
                        "end_x": end_x,
                        "end_y": end_y,
                        "button": button
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Dragged mouse from: ({start_x}, {start_y}) to ({end_x}, {end_y})")
                return result
            except Exception as e:
                error_msg = f"Mouse drag error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def mouse_scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> ActionResult:
        """Scroll the mouse wheel at the specified coordinates."""
        async with self._input_lock:
            start_time = time.time()
            action_id = f"scroll_{uuid.uuid4().hex[:8]}"
            
            # If no coordinates provided, use current position
            if x is None or y is None:
                if INPUT_SIMULATION_AVAILABLE:
                    current_pos = pyautogui.position()
                    x = x if x is not None else current_pos.x
                    y = y if y is not None else current_pos.y
                else:
                    x = x if x is not None else 0
                    y = y if y is not None else 0
            
            # Create action record
            direction = "down" if clicks < 0 else "up"
            action = Action(
                id=action_id,
                action_type=ActionType.MOUSE_CLICK,
                description=f"Scroll {direction} ({abs(clicks)} clicks) at: ({x}, {y})",
                parameters={
                    "x": x,
                    "y": y,
                    "clicks": clicks
                }
            )
            action.started_at = start_time
            
            if not INPUT_SIMULATION_AVAILABLE:
                error_msg = "Input simulation not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Move to position if provided
                if x is not None and y is not None:
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: pyautogui.moveTo(x, y)
                    )
                
                # Scroll in thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: pyautogui.scroll(clicks)
                )
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={
                        "x": x,
                        "y": y,
                        "clicks": clicks,
                        "direction": direction
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Scrolled {direction} ({abs(clicks)} clicks) at: ({x}, {y})")
                return result
            except Exception as e:
                error_msg = f"Mouse scroll error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def take_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None,
                           filename: Optional[str] = None) -> ActionResult:
        """Take a screenshot of the screen or region."""
        async with self._input_lock:
            start_time = time.time()
            action_id = f"screenshot_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.SCREENSHOT,
                description="Take screenshot" + (f" of region: {region}" if region else ""),
                parameters={
                    "region": region,
                    "filename": filename
                }
            )
            action.started_at = start_time
            
            if not INPUT_SIMULATION_AVAILABLE:
                error_msg = "Screenshot functionality not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Generate filename if not provided
                if not filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(DEFAULT_SCREENSHOT_PATH, f"screenshot_{timestamp}.png")
                elif not os.path.isabs(filename):
                    filename = os.path.join(DEFAULT_SCREENSHOT_PATH, filename)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Take screenshot in thread to avoid blocking
                if region:
                    screenshot = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: pyautogui.screenshot(region=region)
                    )
                else:
                    screenshot = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        pyautogui.screenshot
                    )
                
                # Save screenshot
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: screenshot.save(filename)
                )
                
                # Get file size
                file_size = os.path.getsize(filename)
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={
                        "filename": filename,
                        "region": region,
                        "size": file_size,
                        "resolution": screenshot.size
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Took screenshot: {filename}")
                return result
            except Exception as e:
                error_msg = f"Screenshot error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    #--------------------------------------------------------------------
    # Browser Automation Methods
    #--------------------------------------------------------------------
    
    async def _init_browser(self) -> bool:
        """Initialize the browser automation."""
        async with self._browser_lock:
            if not BROWSER_AUTOMATION_AVAILABLE:
                self.logger.warning("Browser automation not available")
                return False
                
            if self.browser:
                # Browser already initialized
                return True
                
            try:
                self.logger.info("Initializing browser automation...")
                
                # Launch Playwright
                playwright_obj = await playwright.async_playwright().start()
                
                # Determine browser type
                browser_type_name = self.browser_type.lower() if isinstance(self.browser_type, str) else self.browser_type.name.lower()
                if browser_type_name == "chromium" or browser_type_name == BrowserType.CHROMIUM.name.lower():
                    browser_class = playwright_obj.chromium
                elif browser_type_name == "firefox" or browser_type_name == BrowserType.FIREFOX.name.lower():
                    browser_class = playwright_obj.firefox
                elif browser_type_name == "webkit" or browser_type_name == BrowserType.WEBKIT.name.lower():
                    browser_class = playwright_obj.webkit
                else:
                    browser_class = playwright_obj.chromium
                
                # Configure browser options
                browser_args = []
                
                # Launch browser
                self.browser = await browser_class.launch(
                    headless=self.headless,
                    args=browser_args if browser_args else None
                )
                
                # Configure browser context
                context_options = {}
                
                # Set user agent if provided
                if self.user_agent:
                    context_options["user_agent"] = self.user_agent
                
                # Configure downloads if enabled
                if self.allow_downloads:
                    os.makedirs(self.download_path, exist_ok=True)
                    context_options["accept_downloads"] = True
                    context_options["downloads_path"] = self.download_path
                
                # Create browser context
                self.browser_context = await self.browser.new_context(**context_options)
                
                # Create initial page
                self.current_page = await self.browser_context.new_page()
                
                # Set default timeout
                self.current_page.set_default_timeout(self.browser_timeout * 1000)  # ms
                
                self.logger.info(f"Browser initialized: {browser_type_name}")
                return True
            except Exception as e:
                self.logger.error(f"Browser initialization error: {str(e)}", exc_info=True)
                
                # Clean up resources
                if hasattr(self, "current_page") and self.current_page:
                    try:
                        await self.current_page.close()
                    except:
                        pass
                    self.current_page = None
                
                if hasattr(self, "browser_context") and self.browser_context:
                    try:
                        await self.browser_context.close()
                    except:
                        pass
                    self.browser_context = None
                
                if hasattr(self, "browser") and self.browser:
                    try:
                        await self.browser.close()
                    except:
                        pass
                    self.browser = None
                
                return False
    
    async def _close_browser(self) -> bool:
        """Close the browser."""
        async with self._browser_lock:
            if not self.browser:
                return True
                
            try:
                # Close current page
                if self.current_page:
                    try:
                        await self.current_page.close()
                    except:
                        pass
                    self.current_page = None
                
                # Close browser context
                if self.browser_context:
                    try:
                        await self.browser_context.close()
                    except:
                        pass
                    self.browser_context = None
                
                # Close browser
                if self.browser:
                    try:
                        await self.browser.close()
                    except:
                        pass
                    self.browser = None
                
                self.logger.info("Browser closed")
                return True
            except Exception as e:
                self.logger.error(f"Error closing browser: {str(e)}", exc_info=True)
                return False
    
    async def navigate_to(self, url: str, wait_until: str = "load") -> ActionResult:
        """Navigate to a URL in the browser."""
        async with self._browser_lock:
            start_time = time.time()
            action_id = f"nav_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.BROWSER_NAVIGATION,
                description=f"Navigate to: {url}",
                parameters={
                    "url": url,
                    "wait_until": wait_until
                }
            )
            action.started_at = start_time
            
            if not BROWSER_AUTOMATION_AVAILABLE:
                error_msg = "Browser automation not available"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Initialize browser if not already done
                if not self.browser or not self.current_page:
                    initialized = await self._init_browser()
                    if not initialized or not self.current_page:
                        raise Exception("Browser initialization failed")
                
                # Ensure URL is valid
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                
                # Navigate to URL
                response = await self.current_page.goto(url, wait_until=wait_until, timeout=self.browser_timeout * 1000)
                
                # Get status code
                status_code = response.status if response else 0
                
                # Add to browser history
                self._add_to_browser_history(url, status_code == 200)
                
                # Create result
                result = ActionResult(
                    success=status_code >= 200 and status_code < 400,
                    data={
                        "url": url,
                        "status_code": status_code,
                        "title": await self.current_page.title() if self.current_page else None
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED if result.success else ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Navigated to: {url} (Status: {status_code})")
                return result
            except Exception as e:
                error_msg = f"Navigation error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def get_page_content(self) -> ActionResult:
        """Get the HTML content of the current page."""
        async with self._browser_lock:
            start_time = time.time()
            action_id = f"content_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.BROWSER_NAVIGATION,
                description="Get page content",
                parameters={}
            )
            action.started_at = start_time
            
            if not BROWSER_AUTOMATION_AVAILABLE or not self.browser or not self.current_page:
                error_msg = "Browser not available or not initialized"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Get page content
                content = await self.current_page.content()
                
                # Get page title
                title = await self.current_page.title()
                
                # Get page URL
                url = self.current_page.url
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={
                        "content": content,
                        "title": title,
                        "url": url,
                        "length": len(content)
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Got page content: {title} ({len(content)} bytes)")
                return result
            except Exception as e:
                error_msg = f"Error getting page content: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def execute_javascript(self, script: str, arg: Any = None) -> ActionResult:
        """Execute JavaScript on the current page."""
        async with self._browser_lock:
            start_time = time.time()
            action_id = f"js_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.BROWSER_NAVIGATION,
                description=f"Execute JavaScript: {script[:30] + '...' if len(script) > 30 else script}",
                parameters={
                    "script": script,
                    "arg": arg
                }
            )
            action.started_at = start_time
            
            if not BROWSER_AUTOMATION_AVAILABLE or not self.browser or not self.current_page:
                error_msg = "Browser not available or not initialized"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Execute JavaScript
                result_data = await self.current_page.evaluate(script, arg)
                
                # Convert result to JSON-serializable format if possible
                try:
                    result_json = json.dumps(result_data)
                    result_data = json.loads(result_json)
                except:
                    # If not JSON-serializable, convert to string
                    result_data = str(result_data)
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={"result": result_data},
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Executed JavaScript on page")
                return result
            except Exception as e:
                error_msg = f"JavaScript execution error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def click_element(self, selector: str, timeout: Optional[float] = None) -> ActionResult:
        """Click an element on the page using a CSS selector."""
        async with self._browser_lock:
            start_time = time.time()
            action_id = f"click_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.BROWSER_NAVIGATION,
                description=f"Click element: {selector}",
                parameters={
                    "selector": selector,
                    "timeout": timeout
                }
            )
            action.started_at = start_time
            
            if not BROWSER_AUTOMATION_AVAILABLE or not self.browser or not self.current_page:
                error_msg = "Browser not available or not initialized"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Use provided timeout or default
                click_timeout = (timeout or self.browser_timeout) * 1000  # ms
                
                # Wait for selector to be available
                await self.current_page.wait_for_selector(selector, timeout=click_timeout)
                
                # Click element
                await self.current_page.click(selector, timeout=click_timeout)
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={"selector": selector},
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Clicked element: {selector}")
                return result
            except Exception as e:
                error_msg = f"Element click error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def fill_form(self, selector: str, value: str, timeout: Optional[float] = None) -> ActionResult:
        """Fill a form field with the specified value."""
        async with self._browser_lock:
            start_time = time.time()
            action_id = f"form_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.BROWSER_NAVIGATION,
                description=f"Fill form field: {selector} = {value[:20] + '...' if len(value) > 20 else value}",
                parameters={
                    "selector": selector,
                    "value": value,
                    "timeout": timeout
                }
            )
            action.started_at = start_time
            
            if not BROWSER_AUTOMATION_AVAILABLE or not self.browser or not self.current_page:
                error_msg = "Browser not available or not initialized"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Use provided timeout or default
                form_timeout = (timeout or self.browser_timeout) * 1000  # ms
                
                # Wait for selector to be available
                await self.current_page.wait_for_selector(selector, timeout=form_timeout)
                
                # Fill form field
                await self.current_page.fill(selector, value, timeout=form_timeout)
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={"selector": selector, "value_length": len(value)},
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Filled form field: {selector}")
                return result
            except Exception as e:
                error_msg = f"Form fill error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def take_screenshot_browser(self, selector: Optional[str] = None, 
                                    filename: Optional[str] = None,
                                    full_page: bool = False) -> ActionResult:
        """Take a screenshot of the current page or a specific element."""
        async with self._browser_lock:
            start_time = time.time()
            action_id = f"screenshot_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.SCREENSHOT,
                description="Take browser screenshot" + (f" of element: {selector}" if selector else ""),
                parameters={
                    "selector": selector,
                    "filename": filename,
                    "full_page": full_page
                }
            )
            action.started_at = start_time
            
            if not BROWSER_AUTOMATION_AVAILABLE or not self.browser or not self.current_page:
                error_msg = "Browser not available or not initialized"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Generate filename if not provided
                if not filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(DEFAULT_SCREENSHOT_PATH, f"browser_screenshot_{timestamp}.png")
                elif not os.path.isabs(filename):
                    filename = os.path.join(DEFAULT_SCREENSHOT_PATH, filename)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Take screenshot
                if selector:
                    # Screenshot of specific element
                    element = await self.current_page.wait_for_selector(selector)
                    await element.screenshot(path=filename)
                else:
                    # Screenshot of entire page
                    await self.current_page.screenshot(path=filename, full_page=full_page)
                
                # Get file size
                file_size = os.path.getsize(filename)
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={
                        "filename": filename,
                        "selector": selector,
                        "full_page": full_page,
                        "size": file_size
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Took browser screenshot: {filename}")
                return result
            except Exception as e:
                error_msg = f"Browser screenshot error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    async def download_file(self, url: str, filename: Optional[str] = None) -> ActionResult:
        """Download a file from a URL."""
        async with self._browser_lock:
            start_time = time.time()
            action_id = f"download_{uuid.uuid4().hex[:8]}"
            
            # Create action record
            action = Action(
                id=action_id,
                action_type=ActionType.BROWSER_NAVIGATION,
                description=f"Download file: {url}",
                parameters={
                    "url": url,
                    "filename": filename
                }
            )
            action.started_at = start_time
            
            if not BROWSER_AUTOMATION_AVAILABLE or not self.browser or not self.current_page:
                error_msg = "Browser not available or not initialized"
                self.logger.warning(error_msg)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
            
            try:
                # Ensure downloads are enabled
                if not self.allow_downloads:
                    raise Exception("Downloads are not enabled in browser configuration")
                
                # Ensure URL is valid
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                
                # Extract filename from URL if not provided
                if not filename:
                    parsed_url = urlparse(url)
                    path_filename = os.path.basename(parsed_url.path)
                    if path_filename:
                        filename = path_filename
                    else:
                        # Generate a filename if one can't be extracted
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"download_{timestamp}"
                
                # Ensure the download directory exists
                os.makedirs(self.download_path, exist_ok=True)
                
                # Generate full path
                full_path = os.path.join(self.download_path, filename)
                
                # Create a new page for download to avoid navigation issues
                if not self.browser_context:
                    raise Exception("Browser context not initialized")
                download_page = await self.browser_context.new_page()
                
                # Set up download handler
                download_event = asyncio.Event()
                download_info = {}
                
                async def handle_download(download):
                    nonlocal download_info
                    # Save download info
                    download_info["suggested_filename"] = download.suggested_filename
                    
                    # Save to specified path
                    await download.save_as(full_path)
                    
                    # Signal completion
                    download_event.set()
                
                # Set up download handler
                download_page.on("download", handle_download)
                
                # Navigate to URL
                await download_page.goto(url, timeout=self.browser_timeout * 1000)
                
                # Wait for download to complete or timeout
                try:
                    await asyncio.wait_for(download_event.wait(), timeout=self.browser_timeout)
                except asyncio.TimeoutError:
                    raise Exception(f"Download timed out after {self.browser_timeout} seconds")
                
                # Close download page
                await download_page.close()
                
                # Verify file was downloaded
                if not os.path.exists(full_path):
                    raise Exception("File download failed: File not found")
                
                # Get file size
                file_size = os.path.getsize(full_path)
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={
                        "url": url,
                        "filename": filename,
                        "path": full_path,
                        "size": file_size,
                        "suggested_filename": download_info.get("suggested_filename", filename)
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Downloaded file: {url} -> {full_path} ({file_size} bytes)")
                return result
            except Exception as e:
                error_msg = f"File download error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # Create result
                result = ActionResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                return result
    
    def _add_to_browser_history(self, url: str, success: bool) -> None:
        """Add URL to browser history."""
        # Create history entry
        entry = {
            "url": url,
            "timestamp": time.time(),
            "success": success,
            "title": ""  # Will be updated asynchronously
        }
        
        # Add to history
        self.browser_history.append(entry)
        
        # Limit history size
        if len(self.browser_history) > self.max_browser_history:
            self.browser_history = self.browser_history[-self.max_browser_history:]
        
        # Update title asynchronously
        if self.browser and self.current_page:
            asyncio.create_task(self._update_history_title(len(self.browser_history) - 1))
    
    async def _update_history_title(self, index: int) -> None:
        """Update browser history entry with page title."""
        if index < 0 or index >= len(self.browser_history):
            return
            
        try:
            if self.browser and self.current_page:
                title = await self.current_page.title()
                if title:
                    self.browser_history[index]["title"] = title
        except Exception:
            pass
    
    #--------------------------------------------------------------------
    # File System Methods
    #--------------------------------------------------------------------
    
    async def list_directory(self, path: str) -> ActionResult:
        """List contents of a directory."""
        start_time = time.time()
        action_id = f"ls_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.FILE_OPERATION,
            description=f"List directory: {path}",
            parameters={
                "path": path
            }
        )
        action.started_at = start_time
        
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(path))
            
            # Validate path existence
            if not os.path.exists(norm_path):
                raise FileNotFoundError(f"Directory not found: {norm_path}")
                
            # Validate it's a directory
            if not os.path.isdir(norm_path):
                raise NotADirectoryError(f"Not a directory: {norm_path}")
            
            # List directory contents
            entries = []
            
            for entry in os.scandir(norm_path):
                try:
                    stat_info = entry.stat()
                    entry_info = {
                        "name": entry.name,
                        "path": entry.path,
                        "is_dir": entry.is_dir(),
                        "is_file": entry.is_file(),
                        "size": stat_info.st_size if entry.is_file() else 0,
                        "created": stat_info.st_ctime,
                        "modified": stat_info.st_mtime,
                        "accessed": stat_info.st_atime
                    }
                    entries.append(entry_info)
                except Exception as e:
                    # Skip entries that can't be accessed
                    self.logger.warning(f"Error accessing {entry.path}: {str(e)}")
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "path": norm_path,
                    "entries": entries,
                    "count": len(entries)
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Listed directory: {norm_path} ({len(entries)} entries)")
            return result
        except Exception as e:
            error_msg = f"Directory listing error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def read_file(self, path: str, binary: bool = False, max_size: int = 10 * 1024 * 1024) -> ActionResult:
        """Read contents of a file."""
        start_time = time.time()
        action_id = f"read_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.FILE_OPERATION,
            description=f"Read file: {path}",
            parameters={
                "path": path,
                "binary": binary,
                "max_size": max_size
            }
        )
        action.started_at = start_time
        
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(path))
            
            # Validate path existence
            if not os.path.exists(norm_path):
                raise FileNotFoundError(f"File not found: {norm_path}")
                
            # Validate it's a file
            if not os.path.isfile(norm_path):
                raise IsADirectoryError(f"Not a file: {norm_path}")
            
            # Check file size
            file_size = os.path.getsize(norm_path)
            if file_size > max_size:
                raise ValueError(f"File too large: {file_size} bytes (max: {max_size} bytes)")
            
            # Read file content
            mode = "rb" if binary else "r"
            encoding = None if binary else "utf-8"
            
            content = await asyncio.to_thread(
                self._read_file_sync,
                norm_path,
                mode,
                encoding
            )
            
            # For binary data, return base64-encoded string
            if binary:
                import base64
                if isinstance(content, str):
                    content = content.encode('utf-8')
                content_b64 = base64.b64encode(content).decode('ascii')
                content_to_return = content_b64
            else:
                content_to_return = content
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "path": norm_path,
                    "size": file_size,
                    "binary": binary,
                    "content": content_to_return
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Read file: {norm_path} ({file_size} bytes)")
            return result
        except Exception as e:
            error_msg = f"File read error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    def _read_file_sync(self, path: str, mode: str, encoding: Optional[str]) -> Union[str, bytes]:
        """Read file synchronously."""
        with open(path, mode=mode, encoding=encoding) as f:
            return f.read()
    
    async def write_file(self, path: str, content: Union[str, bytes], 
                       append: bool = False, create_dirs: bool = True) -> ActionResult:
        """Write content to a file."""
        start_time = time.time()
        action_id = f"write_{uuid.uuid4().hex[:8]}"
        
        # Determine if content is binary
        is_binary = isinstance(content, bytes)
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.FILE_OPERATION,
            description=f"{'Append to' if append else 'Write'} file: {path}",
            parameters={
                "path": path,
                "content_length": len(content),
                "append": append,
                "create_dirs": create_dirs,
                "binary": is_binary
            }
        )
        action.started_at = start_time
        
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(path))
            
            # Create parent directories if needed
            if create_dirs:
                parent_dir = os.path.dirname(norm_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
            
            # Determine write mode
            if is_binary:
                mode = "ab" if append else "wb"
            else:
                mode = "a" if append else "w"
            
            # Write content
            await asyncio.to_thread(
                self._write_file_sync,
                norm_path,
                content,
                mode
            )
            
            # Get file size
            file_size = os.path.getsize(norm_path)
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "path": norm_path,
                    "size": file_size,
                    "append": append,
                    "binary": is_binary
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"{'Appended to' if append else 'Wrote'} file: {norm_path} ({file_size} bytes)")
            return result
        except Exception as e:
            error_msg = f"File write error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    def _write_file_sync(self, path: str, content: Union[str, bytes], mode: str) -> None:
        """Write file synchronously."""
        with open(path, mode=mode) as f:
            f.write(content)
    
    async def copy_file(self, source: str, destination: str, overwrite: bool = False) -> ActionResult:
        """Copy a file from source to destination."""
        start_time = time.time()
        action_id = f"cp_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.FILE_OPERATION,
            description=f"Copy file: {source} -> {destination}",
            parameters={
                "source": source,
                "destination": destination,
                "overwrite": overwrite
            }
        )
        action.started_at = start_time
        
        try:
            norm_source = os.path.normpath(os.path.expanduser(source))
            norm_dest = os.path.normpath(os.path.expanduser(destination))
            
            # Validate source existence
            if not os.path.exists(norm_source):
                raise FileNotFoundError(f"Source file not found: {norm_source}")
                
            # Validate source is a file
            if not os.path.isfile(norm_source):
                raise IsADirectoryError(f"Source is not a file: {norm_source}")
            
            # Check if destination exists
            if os.path.exists(norm_dest):
                if not overwrite:
                    raise FileExistsError(f"Destination file already exists: {norm_dest}")
                
                # Check if destination is a directory
                if os.path.isdir(norm_dest):
                    norm_dest = os.path.join(norm_dest, os.path.basename(norm_source))
            
            # Create parent directories if needed
            parent_dir = os.path.dirname(norm_dest)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Copy file
            await asyncio.to_thread(shutil.copy2, norm_source, norm_dest)
            
            # Get file sizes
            source_size = os.path.getsize(norm_source)
            dest_size = os.path.getsize(norm_dest)
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "source": norm_source,
                    "destination": norm_dest,
                    "source_size": source_size,
                    "destination_size": dest_size
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Copied file: {norm_source} -> {norm_dest} ({source_size} bytes)")
            return result
        except Exception as e:
            error_msg = f"File copy error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def move_file(self, source: str, destination: str, overwrite: bool = False) -> ActionResult:
        """Move a file from source to destination."""
        start_time = time.time()
        action_id = f"mv_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.FILE_OPERATION,
            description=f"Move file: {source} -> {destination}",
            parameters={
                "source": source,
                "destination": destination,
                "overwrite": overwrite
            }
        )
        action.started_at = start_time
        
        try:
            norm_source = os.path.normpath(os.path.expanduser(source))
            norm_dest = os.path.normpath(os.path.expanduser(destination))
            
            # Validate source existence
            if not os.path.exists(norm_source):
                raise FileNotFoundError(f"Source file not found: {norm_source}")
                
            # Validate source is a file
            if not os.path.isfile(norm_source):
                raise IsADirectoryError(f"Source is not a file: {norm_source}")
            
            # Check if destination exists
            if os.path.exists(norm_dest):
                if not overwrite:
                    raise FileExistsError(f"Destination file already exists: {norm_dest}")
                
                # Check if destination is a directory
                if os.path.isdir(norm_dest):
                    norm_dest = os.path.join(norm_dest, os.path.basename(norm_source))
            
            # Create parent directories if needed
            parent_dir = os.path.dirname(norm_dest)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Move file
            await asyncio.to_thread(shutil.move, norm_source, norm_dest)
            
            # Get file size
            dest_size = os.path.getsize(norm_dest)
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "source": norm_source,
                    "destination": norm_dest,
                    "size": dest_size
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Moved file: {norm_source} -> {norm_dest} ({dest_size} bytes)")
            return result
        except Exception as e:
            error_msg = f"File move error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def delete_file(self, path: str, force: bool = False) -> ActionResult:
        """Delete a file."""
        start_time = time.time()
        action_id = f"rm_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.FILE_OPERATION,
            description=f"Delete file: {path}",
            parameters={
                "path": path,
                "force": force
            }
        )
        action.started_at = start_time
        
        # Mark as potentially dangerous operation
        action.requires_confirmation = True
        
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(path))
            
            # Validate path existence
            if not os.path.exists(norm_path):
                raise FileNotFoundError(f"File not found: {norm_path}")
                
            # Validate it's a file
            if not os.path.isfile(norm_path):
                raise IsADirectoryError(f"Not a file: {norm_path}")
            
            # Get file size before deletion
            file_size = os.path.getsize(norm_path)
            
            # Delete file
            await asyncio.to_thread(os.unlink, norm_path)
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "path": norm_path,
                    "size": file_size
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Deleted file: {norm_path} ({file_size} bytes)")
            return result
        except Exception as e:
            error_msg = f"File deletion error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def create_directory(self, path: str, exist_ok: bool = True) -> ActionResult:
        """Create a directory."""
        start_time = time.time()
        action_id = f"mkdir_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.FILE_OPERATION,
            description=f"Create directory: {path}",
            parameters={
                "path": path,
                "exist_ok": exist_ok
            }
        )
        action.started_at = start_time
        
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(path))
            
            # Create directory
            await asyncio.to_thread(os.makedirs, norm_path, exist_ok=exist_ok)
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "path": norm_path,
                    "already_existed": os.path.exists(norm_path) and not (os.path.getmtime(norm_path) >= start_time)
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Created directory: {norm_path}")
            return result
        except Exception as e:
            error_msg = f"Directory creation error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    #--------------------------------------------------------------------
    # Process and Window Management Methods
    #--------------------------------------------------------------------
    
    async def start_process(self, command: str, arguments: Optional[List[str]] = None, 
                          cwd: Optional[str] = None, env: Optional[Dict] = None,
                          shell: bool = False, detach: bool = False) -> ActionResult:
        """Start a new process."""
        start_time = time.time()
        action_id = f"proc_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.PROCESS_MANAGEMENT,
            description=f"Start process: {command}",
            parameters={
                "command": command,
                "arguments": arguments,
                "cwd": cwd,
                "env": env,
                "shell": shell,
                "detach": detach
            }
        )
        action.started_at = start_time
        
        try:
            # Validate command
            if not command:
                raise ValueError("Command cannot be empty")
                
            # Normalize command path if it's a file path
            if os.path.exists(command):
                command = os.path.normpath(os.path.expanduser(command))
                
            # Normalize working directory if provided
            if cwd:
                cwd = os.path.normpath(os.path.expanduser(cwd))
                if not os.path.exists(cwd) or not os.path.isdir(cwd):
                    raise NotADirectoryError(f"Working directory not found: {cwd}")
            
            # Prepare environment variables
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # Prepare command and arguments
            full_command = [command]
            if arguments:
                full_command.extend(arguments)
            else:
                arguments = []
            
            # Determine process creation flags and settings
            creation_flags = 0
            
            if detach:
                if sys.platform == "win32":
                    # On Windows, use DETACHED_PROCESS flag
                    creation_flags |= subprocess.DETACHED_PROCESS
                    
                    # Use Popen directly with specific flags
                    process = await asyncio.to_thread(
                        lambda: subprocess.Popen(
                            command if shell else full_command,
                            shell=shell,
                            cwd=cwd,
                            env=process_env,
                            creationflags=creation_flags,
                            start_new_session=True,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    )
                else:
                    # On Unix-like systems, use start_new_session
                    process = await asyncio.to_thread(
                        lambda: subprocess.Popen(
                            command if shell else full_command,
                            shell=shell,
                            cwd=cwd,
                            env=process_env,
                            start_new_session=True,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    )
            else:
                # Non-detached process with stdout/stderr capture
                process = await asyncio.to_thread(
                    lambda: subprocess.Popen(
                        command if shell else full_command,
                        shell=shell,
                        cwd=cwd,
                        env=process_env,
                        creationflags=creation_flags,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                )
                
                # For non-detached processes, wait briefly to capture initial output
                if not detach:
                    try:
                        # Wait up to 0.5 seconds for initial output
                        stdout_data, stderr_data = "", ""
                        if process.stdout and process.stderr:
                            for _ in range(5):  # Try up to 5 times
                                if process.stdout.readable():
                                    stdout_line = process.stdout.readline()
                                    if stdout_line:
                                        stdout_data += stdout_line
                                
                                if process.stderr.readable():
                                    stderr_line = process.stderr.readline()
                                    if stderr_line:
                                        stderr_data += stderr_line
                                
                                # Check if process has already completed
                                if process.poll() is not None:
                                    break
                                    
                                # Short sleep between checks
                                await asyncio.sleep(0.1)
                    except Exception as e:
                        self.logger.warning(f"Error capturing initial output: {str(e)}")
            
            # Get process info
            pid = process.pid
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "pid": pid,
                    "command": command,
                    "arguments": arguments,
                    "detached": detach,
                    "initial_stdout": stdout_data if not detach and 'stdout_data' in locals() else None,
                    "initial_stderr": stderr_data if not detach and 'stderr_data' in locals() else None
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Started process: {command} (PID: {pid})")
            return result
        except Exception as e:
            error_msg = f"Process start error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def get_windows(self, refresh: bool = False) -> ActionResult:
        """Get list of all visible windows."""
        start_time = time.time()
        action_id = f"win_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.WINDOW_MANAGEMENT,
            description="Get window list",
            parameters={
                "refresh": refresh
            }
        )
        action.started_at = start_time
        
        try:
            if not WIN32_AVAILABLE:
                raise ImportError("Win32 API not available")
                
            async with self._windows_lock:
                # Check if we need to refresh the cache
                current_time = time.time()
                if refresh or not self.windows_cache or (current_time - self.windows_cache_time) > self.windows_cache_ttl:
                    # Get all top-level windows
                    windows = []
                    
                    def enum_windows_callback(hwnd, _):
                        # Skip invisible windows
                        if not win32gui.IsWindowVisible(hwnd):
                            return True
                        
                        # Skip windows with empty titles
                        title = win32gui.GetWindowText(hwnd)
                        if not title:
                            return True
                        
                        # Get window info
                        try:
                            # Get class name
                            classname = win32gui.GetClassName(hwnd) or ""
                            
                            # Get window rect
                            rect = win32gui.GetWindowRect(hwnd)
                            window_rect = Rectangle(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
                            
                            # Get process ID
                            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
                            
                            # Get process name
                            process_name = "unknown"
                            try:
                                import psutil
                                process = psutil.Process(process_id)
                                process_name = process.name()
                            except:
                                pass
                                
                            # Create window info
                            window = WindowInfo(
                                handle=hwnd,
                                title=title,
                                classname=classname,
                                rect=window_rect,
                                process_id=process_id,
                                process_name=process_name,
                                is_visible=True
                            )
                            
                            windows.append(window)
                        except Exception as e:
                            # Skip windows that throw errors
                            pass
                        
                        return True
                    
                    # Enumerate windows
                    win32gui.EnumWindows(enum_windows_callback, None)
                    
                    # Update cache
                    self.windows_cache = {w.handle: w for w in windows}
                    self.windows_cache_time = current_time
                
                # Convert window cache to list
                windows_list = list(self.windows_cache.values())
                
                # Create result
                result = ActionResult(
                    success=True,
                    data={
                        "windows": [w.to_dict() for w in windows_list],
                        "count": len(windows_list),
                        "timestamp": self.windows_cache_time
                    },
                    duration=time.time() - start_time
                )
                
                # Update action
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                self.logger.info(f"Retrieved window list: {len(windows_list)} windows")
                return result
        except Exception as e:
            error_msg = f"Window list error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def focus_window(self, window_handle: Optional[int] = None, window_title: Optional[str] = None) -> ActionResult:
        """Focus a specific window by handle or title."""
        start_time = time.time()
        action_id = f"focus_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.WINDOW_MANAGEMENT,
            description=f"Focus window: {window_handle if window_handle else window_title}",
            parameters={
                "window_handle": window_handle,
                "window_title": window_title
            }
        )
        action.started_at = start_time
        
        try:
            if not WIN32_AVAILABLE:
                raise ImportError("Win32 API not available")
                
            # Find window handle if title is provided
            if window_handle is None and window_title:
                # First try exact match
                window_handle = win32gui.FindWindow(None, window_title)
                
                # If not found, try partial match
                if not window_handle:
                    # Get all windows first
                    result = await self.get_windows(refresh=True)
                    if not result.success:
                        raise Exception("Failed to get window list")
                        
                    # Find window with matching title
                    windows = result.data["windows"]
                    for window in windows:
                        if window_title.lower() in window["title"].lower():
                            window_handle = window["handle"]
                            break
            
            # Verify window handle
            if not window_handle:
                raise ValueError("Window not found")
            
            # Check if window exists
            if not win32gui.IsWindow(window_handle):
                raise ValueError(f"Invalid window handle: {window_handle}")
            
            # Focus window
            win32gui.SetForegroundWindow(window_handle)
            
            # Get window title
            window_title = win32gui.GetWindowText(window_handle)
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "handle": window_handle,
                    "title": window_title
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Focused window: {window_title} (Handle: {window_handle})")
            return result
        except Exception as e:
            error_msg = f"Window focus error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    #--------------------------------------------------------------------
    # Clipboard Methods
    #--------------------------------------------------------------------
    
    async def get_clipboard(self) -> ActionResult:
        """Get text from clipboard."""
        start_time = time.time()
        action_id = f"clip_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.CLIPBOARD_OPERATION,
            description="Get clipboard content",
            parameters={}
        )
        action.started_at = start_time
        
        try:
            if not CLIPBOARD_AVAILABLE or pyperclip is None or not hasattr(pyperclip, 'paste'):
                raise ImportError("pyperclip not available for clipboard operations")
            clipboard_content = await asyncio.to_thread(pyperclip.paste)
            self.previous_clipboard = clipboard_content
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "content": clipboard_content,
                    "length": len(clipboard_content) if clipboard_content else 0
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Got clipboard content: {len(clipboard_content) if clipboard_content else 0} chars")
            return result
        except Exception as e:
            error_msg = f"Clipboard get error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def set_clipboard(self, text: str) -> ActionResult:
        """Set text to clipboard."""
        start_time = time.time()
        action_id = f"clip_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.CLIPBOARD_OPERATION,
            description=f"Set clipboard content: {text[:20] + '...' if len(text) > 20 else text}",
            parameters={
                "text": text
            }
        )
        action.started_at = start_time
        
        try:
            if not CLIPBOARD_AVAILABLE or pyperclip is None or not hasattr(pyperclip, 'paste') or not hasattr(pyperclip, 'copy'):
                raise ImportError("pyperclip not available for clipboard operations")
            previous = await asyncio.to_thread(pyperclip.paste)
            self.previous_clipboard = previous
            await asyncio.to_thread(pyperclip.copy, text)
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "content": text,
                    "length": len(text),
                    "previous_length": len(previous) if previous else 0
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Set clipboard content: {len(text)} chars")
            return result
        except Exception as e:
            error_msg = f"Clipboard set error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def restore_clipboard(self) -> ActionResult:
        """Restore previous clipboard content."""
        start_time = time.time()
        action_id = f"clip_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.CLIPBOARD_OPERATION,
            description="Restore previous clipboard content",
            parameters={}
        )
        action.started_at = start_time
        
        try:
            if not CLIPBOARD_AVAILABLE or pyperclip is None or not hasattr(pyperclip, 'copy'):
                raise ImportError("pyperclip not available for clipboard operations")
            if self.previous_clipboard is None or not isinstance(self.previous_clipboard, str):
                raise ValueError("No previous clipboard content to restore")
            await asyncio.to_thread(pyperclip.copy, self.previous_clipboard)
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "content": self.previous_clipboard,
                    "length": len(self.previous_clipboard) if self.previous_clipboard else 0
                },
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Restored clipboard content: {len(self.previous_clipboard) if self.previous_clipboard else 0} chars")
            return result
        except Exception as e:
            error_msg = f"Clipboard restore error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    #--------------------------------------------------------------------
    # Action Management Methods
    #--------------------------------------------------------------------
    
    def _add_to_recent_actions(self, action: Action) -> None:
        """Add action to recent actions list."""
        self.recent_actions.append(action)
        
        # Limit list size
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[-self.max_recent_actions:]
    
    async def get_recent_actions(self, count: int = 10, action_type: Optional[ActionType] = None) -> List[Dict]:
        """Get list of recent actions."""
        # Filter by type if specified
        if action_type:
            filtered = [a for a in self.recent_actions if a.action_type == action_type]
        else:
            filtered = self.recent_actions
            
        # Sort by completion time (newest first)
        sorted_actions = sorted(
            filtered,
            key=lambda a: a.completed_at if a.completed_at else a.created_at,
            reverse=True
        )
        
        # Limit to requested count
        return [a.to_dict() for a in sorted_actions[:count]]
    
    async def get_action(self, action_id: str) -> Optional[Dict]:
        """Get details of a specific action."""
        for action in self.recent_actions:
            if action.id == action_id:
                return action.to_dict()
        return None
    
    async def execute_action_sequence(self, actions: List[Dict]) -> ActionResult:
        """Execute a sequence of actions."""
        start_time = time.time()
        sequence_id = f"seq_{uuid.uuid4().hex[:8]}"
        
        # Create master action for the sequence
        master_action = Action(
            id=sequence_id,
            action_type=ActionType.CUSTOM,
            description=f"Execute action sequence ({len(actions)} actions)",
            parameters={
                "action_count": len(actions)
            }
        )
        master_action.started_at = start_time
        
        results = []
        success = True
        error = None
        
        try:
            for i, action_data in enumerate(actions):
                # Get action type
                action_type_name = action_data.get("action_type", "CUSTOM")
                try:
                    action_type = ActionType[action_type_name] if isinstance(action_type_name, str) else action_type_name
                except (KeyError, ValueError):
                    action_type = ActionType.CUSTOM
                
                # Create action
                action = Action(
                    action_type=action_type,
                    description=action_data.get("description", f"Action #{i+1}"),
                    parameters=action_data.get("parameters", {}),
                    parent_id=sequence_id
                )
                
                # Execute action based on type
                result = None
                
                if action_type == ActionType.COMMAND_EXECUTION:
                    command = action.parameters.get("command", "")
                    if command:
                        result = await self.execute_command(
                            command=command,
                            shell=action.parameters.get("shell", True),
                            cwd=action.parameters.get("cwd"),
                            env=action.parameters.get("env"),
                            timeout=action.parameters.get("timeout")
                        )
                
                elif action_type == ActionType.KEYBOARD_INPUT:
                    # Determine keyboard action type
                    input_type = action.parameters.get("input_type", "text")
                    
                    if input_type == "text":
                        text = action.parameters.get("text", "")
                        interval = action.parameters.get("interval")
                        result = await self.type_text(text, interval)
                    
                    elif input_type == "key":
                        key = action.parameters.get("key", "")
                        presses = action.parameters.get("presses", 1)
                        interval = action.parameters.get("interval")
                        result = await self.press_key(key, presses, interval)
                    
                    elif input_type == "hotkey":
                        keys = action.parameters.get("keys", [])
                        result = await self.press_keys(keys)
                
                elif action_type == ActionType.MOUSE_CLICK:
                    # Determine mouse action type
                    mouse_type = action.parameters.get("mouse_type", "click")
                    
                    if mouse_type == "move":
                        x = action.parameters.get("x", 0)
                        y = action.parameters.get("y", 0)
                        duration = action.parameters.get("duration")
                        result = await self.mouse_move(x, y, duration)
                    
                    elif mouse_type == "click":
                        x = action.parameters.get("x")
                        y = action.parameters.get("y")
                        button = action.parameters.get("button", "left")
                        clicks = action.parameters.get("clicks", 1)
                        interval = action.parameters.get("interval")
                        result = await self.mouse_click(x, y, button, clicks, interval)
                    
                    elif mouse_type == "drag":
                        start_x = action.parameters.get("start_x", 0)
                        start_y = action.parameters.get("start_y", 0)
                        end_x = action.parameters.get("end_x", 0)
                        end_y = action.parameters.get("end_y", 0)
                        duration = action.parameters.get("duration")
                        button = action.parameters.get("button", "left")
                        result = await self.mouse_drag(start_x, start_y, end_x, end_y, duration, button)
                    
                    elif mouse_type == "scroll":
                        clicks = action.parameters.get("clicks", 0)
                        x = action.parameters.get("x")
                        y = action.parameters.get("y")
                        result = await self.mouse_scroll(clicks, x, y)
                
                elif action_type == ActionType.BROWSER_NAVIGATION:
                    # Determine browser action type
                    browser_type = action.parameters.get("browser_type", "navigate")
                    
                    if browser_type == "navigate":
                        url = action.parameters.get("url", "")
                        wait_until = action.parameters.get("wait_until", "load")
                        result = await self.navigate_to(url, wait_until)
                    
                    elif browser_type == "get_content":
                        result = await self.get_page_content()
                    
                    elif browser_type == "execute_js":
                        script = action.parameters.get("script", "")
                        arg = action.parameters.get("arg")
                        result = await self.execute_javascript(script, arg)
                    
                    elif browser_type == "click_element":
                        selector = action.parameters.get("selector", "")
                        timeout = action.parameters.get("timeout")
                        result = await self.click_element(selector, timeout)
                    
                    elif browser_type == "fill_form":
                        selector = action.parameters.get("selector", "")
                        value = action.parameters.get("value", "")
                        timeout = action.parameters.get("timeout")
                        result = await self.fill_form(selector, value, timeout)
                    
                    elif browser_type == "take_screenshot":
                        selector = action.parameters.get("selector")
                        filename = action.parameters.get("filename")
                        full_page = action.parameters.get("full_page", False)
                        result = await self.take_screenshot_browser(selector, filename, full_page)
                    
                    elif browser_type == "download_file":
                        url = action.parameters.get("url", "")
                        filename = action.parameters.get("filename")
                        result = await self.download_file(url, filename)
                
                elif action_type == ActionType.FILE_OPERATION:
                    # Determine file operation type
                    file_type = action.parameters.get("file_type", "read")
                    
                    if file_type == "list":
                        path = action.parameters.get("path", "")
                        result = await self.list_directory(path)
                    
                    elif file_type == "read":
                        path = action.parameters.get("path", "")
                        binary = action.parameters.get("binary", False)
                        max_size = action.parameters.get("max_size", 10 * 1024 * 1024)
                        result = await self.read_file(path, binary, max_size)
                    
                    elif file_type == "write":
                        path = action.parameters.get("path", "")
                        content = action.parameters.get("content", "")
                        append = action.parameters.get("append", False)
                        create_dirs = action.parameters.get("create_dirs", True)
                        result = await self.write_file(path, content, append, create_dirs)
                    
                    elif file_type == "copy":
                        source = action.parameters.get("source", "")
                        destination = action.parameters.get("destination", "")
                        overwrite = action.parameters.get("overwrite", False)
                        result = await self.copy_file(source, destination, overwrite)
                    
                    elif file_type == "move":
                        source = action.parameters.get("source", "")
                        destination = action.parameters.get("destination", "")
                        overwrite = action.parameters.get("overwrite", False)
                        result = await self.move_file(source, destination, overwrite)
                    
                    elif file_type == "delete":
                        path = action.parameters.get("path", "")
                        force = action.parameters.get("force", False)
                        result = await self.delete_file(path, force)
                    
                    elif file_type == "create_directory":
                        path = action.parameters.get("path", "")
                        exist_ok = action.parameters.get("exist_ok", True)
                        result = await self.create_directory(path, exist_ok)
                
                elif action_type == ActionType.PROCESS_MANAGEMENT:
                    # Determine process action type
                    process_type = action.parameters.get("process_type", "start")
                    
                    if process_type == "start":
                        command = action.parameters.get("command", "")
                        arguments = action.parameters.get("arguments", [])
                        cwd = action.parameters.get("cwd")
                        env = action.parameters.get("env")
                        shell = action.parameters.get("shell", False)
                        detach = action.parameters.get("detach", False)
                        result = await self.start_process(command, arguments, cwd, env, shell, detach)
                
                elif action_type == ActionType.WINDOW_MANAGEMENT:
                    # Determine window action type
                    window_type = action.parameters.get("window_type", "get_windows")
                    
                    if window_type == "get_windows":
                        refresh = action.parameters.get("refresh", False)
                        result = await self.get_windows(refresh)
                    
                    elif window_type == "focus_window":
                        window_handle = action.parameters.get("window_handle")
                        window_title = action.parameters.get("window_title")
                        result = await self.focus_window(window_handle, window_title)
                
                elif action_type == ActionType.CLIPBOARD_OPERATION:
                    # Determine clipboard action type
                    clipboard_type = action.parameters.get("clipboard_type", "get")
                    
                    if clipboard_type == "get":
                        result = await self.get_clipboard()
                    
                    elif clipboard_type == "set":
                        text = action.parameters.get("text", "")
                        result = await self.set_clipboard(text)
                    
                    elif clipboard_type == "restore":
                        result = await self.restore_clipboard()
                
                # Handle custom actions
                if result is None:
                    result = ActionResult(
                        success=False,
                        error=f"Unsupported action type: {action_type.name}",
                        duration=0.0
                    )
                
                # Update action with result
                action.status = ActionStatus.SUCCEEDED if result.success else ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                self._add_to_recent_actions(action)
                
                # Add result to sequence results
                results.append(result)
                
                # Check for failure
                if not result.success:
                    success = False
                    error = result.error
                    break
            
            # Create master result
            master_result = ActionResult(
                success=success,
                data={
                    "results": [r.to_dict() for r in results]
                },
                error=error,
                duration=time.time() - start_time
            )
            
            # Update master action
            master_action.status = ActionStatus.SUCCEEDED if success else ActionStatus.FAILED
            master_action.result = master_result
            master_action.completed_at = time.time()
            self._add_to_recent_actions(master_action)
            
            return master_result
        except Exception as e:
            error_msg = f"Action sequence execution error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create result
            master_result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update master action
            master_action.status = ActionStatus.FAILED
            master_action.result = master_result
            master_action.completed_at = time.time()
            self._add_to_recent_actions(master_action)
            
            return master_result
    
    #--------------------------------------------------------------------
    # Audio Operations
    #--------------------------------------------------------------------
    
    async def play_sound(self, sound_type: str = "beep", frequency: int = 1000, 
                        duration: int = 500, filename: Optional[str] = None) -> ActionResult:
        """Play a sound or audio file."""
        start_time = time.time()
        action_id = f"audio_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.AUDIO_OPERATION,
            description=f"Play sound: {sound_type}" + (f" from {filename}" if filename else ""),
            parameters={
                "sound_type": sound_type,
                "frequency": frequency,
                "duration": duration,
                "filename": filename
            }
        )
        action.started_at = start_time
        
        if not AUDIO_AVAILABLE:
            error_msg = "Audio playback not available"
            self.logger.warning(error_msg)
            
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
        
        try:
            if sound_type == "beep":
                # Use winsound for beeps
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: winsound.Beep(frequency, duration)
                )
                
            elif sound_type == "file" and filename:
                # Play from file
                if os.path.exists(filename):
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: winsound.PlaySound(filename, winsound.SND_FILENAME)
                    )
                else:
                    raise FileNotFoundError(f"Audio file not found: {filename}")
                    
            elif sound_type == "text":
                # Text-to-speech
                text = action.parameters.get("text", "")
                if not text:
                    raise ValueError("No text provided for text-to-speech")
                
                def speak_text():
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    speak_text
                )
            
            else:
                raise ValueError(f"Unsupported sound type: {sound_type}")
            
            result = ActionResult(
                success=True,
                data={
                    "sound_type": sound_type,
                    "duration": duration
                },
                duration=time.time() - start_time
            )
            
            action.status = ActionStatus.SUCCEEDED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Played sound: {sound_type}")
            return result
            
        except Exception as e:
            error_msg = f"Audio playback error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    #--------------------------------------------------------------------
    # Screen Recording Methods
    #--------------------------------------------------------------------
    
    async def start_screen_recording(self, region: Optional[Tuple[int, int, int, int]] = None,
                                   fps: int = 20) -> ActionResult:
        """Start recording the screen or a specific region."""
        start_time = time.time()
        action_id = f"record_{uuid.uuid4().hex[:8]}"
        
        # Create action record
        action = Action(
            id=action_id,
            action_type=ActionType.SCREEN_RECORDING,
            description=f"Start screen recording" + (f" of region: {region}" if region else ""),
            parameters={
                "region": region,
                "fps": fps
            }
        )
        action.started_at = start_time
        
        if not SCREEN_RECORDING_AVAILABLE:
            error_msg = "Screen recording not available"
            self.logger.warning(error_msg)
            
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
            
        try:
            # Check if already recording
            if hasattr(self, '_recording') and self._recording:
                raise RuntimeError("Screen recording already in progress")
            
            # Create recording directory if it doesn't exist
            recording_dir = os.path.join(DEFAULT_SCREENSHOT_PATH, "recordings")
            os.makedirs(recording_dir, exist_ok=True)
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(recording_dir, f"recording_{timestamp}.mp4")
            
            # Setup video writer
            if region:
                width, height = region[2], region[3]
            else:
                # Get screen size
                screen_size = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: pyautogui.size()
                )
                width, height = screen_size
                
            # Define codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Store recording state
            self._recording = {
                'writer': writer,
                'fps': fps,
                'region': region,
                'output_path': output_path,
                'frames': 0,
                'start_time': start_time,
                'action_id': action_id,
                'running': True
            }
            
            # Start recording in background
            self._recording_task = asyncio.create_task(self._recording_loop())
            
            result = ActionResult(
                success=True,
                data={
                    "action_id": action_id,
                    "fps": fps,
                    "region": region,
                    "output_path": output_path
                },
                duration=time.time() - start_time
            )
            
            action.status = ActionStatus.EXECUTING  # Still recording
            action.result = result
            self._add_to_recent_actions(action)
            
            self.logger.info(f"Started screen recording: {output_path}")
            return result
            
        except Exception as e:
            error_msg = f"Screen recording error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            action.status = ActionStatus.FAILED
            action.result = result
            action.completed_at = time.time()
            self._add_to_recent_actions(action)
            
            return result
    
    async def _recording_loop(self) -> None:
        """Background task that captures frames for recording."""
        try:
            recording = self._recording
            interval = 1.0 / recording['fps']
            
            while recording['running']:
                # Capture screenshot
                if recording['region']:
                    screenshot = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: pyautogui.screenshot(region=recording['region'])
                    )
                else:
                    screenshot = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        pyautogui.screenshot
                    )
                
                # Convert PIL image to OpenCV format
                frame = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                )
                
                # Write frame
                recording['writer'].write(frame)
                recording['frames'] += 1
                
                # Sleep until next frame
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            self.logger.info("Recording loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in recording loop: {str(e)}", exc_info=True)
            self._recording['running'] = False
    
    async def stop_screen_recording(self) -> ActionResult:
        """Stop the current screen recording and save the video."""
        start_time = time.time()
        
        if not hasattr(self, '_recording') or not self._recording:
            error_msg = "No screen recording in progress"
            self.logger.warning(error_msg)
            
            return ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
        # Get recording info
        recording = self._recording
        action_id = recording['action_id']
        
        try:
            # Stop recording loop
            recording['running'] = False
            
            if hasattr(self, '_recording_task') and self._recording_task:
                self._recording_task.cancel()
                try:
                    await self._recording_task
                except asyncio.CancelledError:
                    pass
                
            # Release video writer
            writer = recording['writer']
            
            # This must be done in a thread as it's blocking
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                writer.release
            )
            
            # Calculate duration
            duration = time.time() - recording['start_time']
            
            # Get action and update
            action = None
            for a in self.recent_actions:
                if a.id == action_id:
                    action = a
                    break
            
            output_path = recording['output_path']
            frames = recording['frames']
            
            # Clean up recording state
            self._recording = None
            self._recording_task = None
            
            # Create result
            result = ActionResult(
                success=True,
                data={
                    "output_path": output_path,
                    "frames": frames,
                    "duration": duration,
                    "fps": recording['fps']
                },
                duration=time.time() - start_time
            )
            
            # Update action if found
            if action:
                action.status = ActionStatus.SUCCEEDED
                action.result = result
                action.completed_at = time.time()
                
            self.logger.info(f"Stopped screen recording: {output_path}, {frames} frames in {duration:.1f} seconds")
            return result
            
        except Exception as e:
            error_msg = f"Error stopping screen recording: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Try to clean up
            self._recording = None
            
            if hasattr(self, '_recording_task'):
                self._recording_task = None
                
            # Create result
            result = ActionResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
            
            # Update action if found
            if action:
                action.status = ActionStatus.FAILED
                action.result = result
                action.completed_at = time.time()
                
            return result
    
    #--------------------------------------------------------------------
    # Advanced Capabilities
    #--------------------------------------------------------------------
    
    def get_resource_usage(self) -> Optional[Dict]:
        """Get current resource usage information."""
        if hasattr(self.resource_manager, 'history') and self.resource_manager.history:
            return asdict(self.resource_manager.history[-1])
        return None
    
    def get_resource_alerts(self, duration: Optional[float] = None) -> List[Dict]:
        """Get recent resource alerts."""
        alerts = self.resource_manager.get_alerts(duration)
        return [asdict(alert) for alert in alerts]
    
    def enable_debug_mode(self, profile: bool = True, trace: bool = False) -> None:
        """Enable debug mode with optional profiling and tracing."""
        self.debug_manager.config.enabled = True
        self.debug_manager.config.profile_enabled = profile
        self.debug_manager.config.trace_all = trace
        if trace:
            self.debug_manager.enable_tracing()
    
    def disable_debug_mode(self) -> None:
        """Disable debug mode."""
        self.debug_manager.config.enabled = False
        self.debug_manager.config.profile_enabled = False
        self.debug_manager.disable_tracing()
    
    def get_profile_results(self, name: Optional[str] = None) -> List[Dict]:
        """Get profiling results."""
        results = self.debug_manager.get_profile_results(name)
        return [asdict(result) for result in results]
    
    def get_system_status(self) -> Optional[Dict]:
        """Get current system status."""
        if hasattr(self.system_manager, 'status_history') and self.system_manager.status_history:
            return asdict(self.system_manager.status_history[-1])
        return None
    
    def get_service_status(self) -> Dict[str, bool]:
        """Get status of monitored services."""
        return self.system_manager.service_status.copy()
    
    async def update_system(self) -> Dict:
        """Perform system update."""
        result = await self.system_manager.update_system()
        return asdict(result)
    
    def get_power_info(self) -> Dict:
        """Get power and battery information."""
        return self.system_manager.get_power_info()
    
    def get_advanced_stats(self) -> Dict:
        """Get combined statistics from all managers."""
        return {
            "resources": self.resource_manager.get_stats(),
            "debug": self.debug_manager.get_stats(),
            "system": self.system_manager.get_stats(),
            "action": {
                "recent_actions": len(self.recent_actions),
                "command_history": len(self.command_history),
                "browser_history": len(getattr(self, 'browser_history', [])),
                "running": self._running
            }
        }
