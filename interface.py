#!/usr/bin/env python3
# ORAMA System - Interface Layer
# User interaction and command processing

import os
import re
import sys
import json
import time
import uuid
import asyncio
import logging
import traceback
import inspect
import argparse
from enum import Enum, auto
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable, Coroutine
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Import colorama for terminal colors if available
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Define dummy color constants if colorama is not available
    class DummyFore:
        RED = YELLOW = GREEN = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class DummyStyle:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""
    Fore = DummyFore()
    Style = DummyStyle()

# Import rich for advanced terminal features if available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import prompt_toolkit for advanced input handling if available
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter, FuzzyCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style as PromptStyle
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Constants
DEFAULT_COMMAND_TIMEOUT = 60.0  # seconds
DEFAULT_HISTORY_FILE = "~/.orama_history"
DEFAULT_CONFIG_FILE = "~/.orama_config.json"
MAX_RESPONSE_HISTORY = 100
DEFAULT_RESPONSE_FORMAT = "rich"  # "plain", "rich", "markdown"

# Command categories
class CommandCategory(Enum):
    SYSTEM = auto()      # System control and information
    TASK = auto()        # Task management
    MEMORY = auto()      # Memory operations
    ACTION = auto()      # Action execution
    COGNITIVE = auto()   # Cognitive operations
    FILE = auto()        # File operations
    CONFIG = auto()      # Configuration management
    HELP = auto()        # Help and documentation
    CUSTOM = auto()      # Custom/user-defined commands

@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration": self.duration
        }

@dataclass
class Command:
    """Command definition."""
    name: str
    description: str
    category: CommandCategory
    handler: Callable
    usage: str = ""
    examples: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    
    def to_dict(self) -> Dict:
        """Convert command to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.name,
            "usage": self.usage,
            "examples": self.examples,
            "aliases": self.aliases,
            "requires_confirmation": self.requires_confirmation
        }

@dataclass
class UserResponse:
    """User response to a system prompt or question."""
    text: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary."""
        return {
            "text": self.text,
            "timestamp": self.timestamp
        }

@dataclass
class SystemResponse:
    """System response to a user command or query."""
    content: str
    format: str = "text"  # text, json, table, error
    timestamp: float = field(default_factory=time.time)
    command: Optional[str] = None
    duration: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "format": self.format,
            "timestamp": self.timestamp,
            "command": self.command,
            "duration": self.duration
        }

class Interface:
    """
    ORAMA Interface Layer
    
    Provides user interaction and command processing capabilities for the autonomous agent.
    The interface layer is responsible for:
    
    1. Command Parsing: Interpret user commands and queries
    2. Response Formatting: Present system responses in appropriate formats
    3. System Status Display: Show system state and activity
    4. Configuration Management: Handle user preferences and settings
    5. Interactive Workflows: Guide users through multi-step processes
    
    The interface supports multiple interaction models:
    - Command-line interface (CLI)
    - Programmatic API for integration with other systems
    - Rich interactive terminal UI (when supported libraries are available)
    """
    
    def __init__(self, config: Dict, memory_engine=None, orchestrator=None, cognitive_engine=None, 
                 action_system=None, logger=None):
        """Initialize the interface with configuration."""
        self.config = config
        self.memory = memory_engine
        self.orchestrator = orchestrator
        self.cognitive = cognitive_engine
        self.action = action_system
        self.logger = logger or logging.getLogger("orama.interface")
        
        # Interface configuration
        self.command_config = config.get("command", {})
        self.command_timeout = self.command_config.get("timeout", DEFAULT_COMMAND_TIMEOUT)
        self.command_history_file = self.command_config.get("history_file", DEFAULT_HISTORY_FILE)
        
        # Status display configuration
        self.status_config = config.get("status", {})
        self.update_interval = self.status_config.get("update_interval", 0.5)  # seconds
        self.metrics_displayed = self.status_config.get("metrics_displayed", ["cpu", "memory", "task_count", "active_tasks"])
        
        # Results configuration
        self.results_config = config.get("results", {})
        self.max_displayed = self.results_config.get("max_displayed", 10)
        self.response_format = self.results_config.get("format", DEFAULT_RESPONSE_FORMAT)
        
        # Configuration management
        self.config_mgmt = config.get("config", {})
        self.allow_runtime_changes = self.config_mgmt.get("allow_runtime_changes", True)
        self.save_on_exit = self.config_mgmt.get("save_on_exit", True)
        self.config_file = self.config_mgmt.get("config_file", DEFAULT_CONFIG_FILE)
        
        # Command registry
        self.commands: Dict[str, Command] = {}
        self.command_aliases: Dict[str, str] = {}
        
        # Response history
        self.response_history: List[SystemResponse] = []
        self.max_responses = MAX_RESPONSE_HISTORY
        
        # Interactive components
        self.console = Console() if RICH_AVAILABLE else None
        self.prompt_session = None
        if PROMPT_TOOLKIT_AVAILABLE:
            # Set up prompt session with history
            history_file = os.path.expanduser(self.command_history_file)
            self.prompt_session = PromptSession(
                history=FileHistory(history_file),
                auto_suggest=AutoSuggestFromHistory(),
                style=PromptStyle.from_dict({
                    'prompt': 'ansicyan bold',
                    'user-input': 'ansiwhite',
                })
            )
        
        # State tracking
        self._running = False
        self._status_task = None
        self._status_live = None
        
        # Register built-in commands
        self._register_commands()
        
        self.logger.info("Interface initialized")
    
    async def start(self) -> None:
        """Start the interface and initialize resources."""
        self.logger.info("Starting interface...")
        self._running = True
        
        try:
            # Load configuration if exists
            config_file = os.path.expanduser(self.config_file)
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Update config with user preferences
                    self._update_config(user_config)
            
            # Start status display if interactive and rich is available
            if RICH_AVAILABLE and self.orchestrator:
                self._status_task = asyncio.create_task(self._status_display_loop())
            
            self.logger.info("Interface started successfully")
        except Exception as e:
            self._running = False
            self.logger.error(f"Failed to start interface: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the interface and perform cleanup."""
        self.logger.info("Stopping interface...")
        self._running = False
        
        try:
            # Cancel status display task
            if self._status_task:
                self._status_task.cancel()
                try:
                    await self._status_task
                except asyncio.CancelledError:
                    pass
            
            # Save configuration if enabled
            if self.save_on_exit:
                await self._save_config()
                
            self.logger.info("Interface stopped")
        except Exception as e:
            self.logger.error(f"Error during interface shutdown: {e}", exc_info=True)
    
    #--------------------------------------------------------------------
    # Command Registration and Execution
    #--------------------------------------------------------------------
    
    def _register_commands(self) -> None:
        """Register all built-in commands."""
        # System commands
        self.register_command(
            Command(
                name="status",
                description="Display system status",
                category=CommandCategory.SYSTEM,
                handler=self._cmd_status,
                usage="status [component]",
                examples=["status", "status cognitive", "status memory"],
                aliases=["stat", "st"]
            )
        )
        
        self.register_command(
            Command(
                name="shutdown",
                description="Shut down the system",
                category=CommandCategory.SYSTEM,
                handler=self._cmd_shutdown,
                usage="shutdown",
                examples=["shutdown"],
                aliases=["exit", "quit"],
                requires_confirmation=True
            )
        )
        
        self.register_command(
            Command(
                name="version",
                description="Display system version",
                category=CommandCategory.SYSTEM,
                handler=self._cmd_version,
                usage="version",
                examples=["version"],
                aliases=["ver"]
            )
        )
        
        # Task commands
        self.register_command(
            Command(
                name="task",
                description="Task management commands",
                category=CommandCategory.TASK,
                handler=self._cmd_task,
                usage="task <create|list|status|cancel> [args...]",
                examples=[
                    "task create 'Find large files'",
                    "task list",
                    "task status abc123",
                    "task cancel xyz789"
                ],
                aliases=["t"]
            )
        )
        
        # Memory commands
        self.register_command(
            Command(
                name="memory",
                description="Memory management commands",
                category=CommandCategory.MEMORY,
                handler=self._cmd_memory,
                usage="memory <store|retrieve|list|delete> [args...]",
                examples=[
                    "memory store 'Important note about the project'",
                    "memory retrieve 'project'",
                    "memory list semantic",
                    "memory delete abc123"
                ],
                aliases=["mem", "m"]
            )
        )
        
        # Action commands
        self.register_command(
            Command(
                name="action",
                description="Action execution commands",
                category=CommandCategory.ACTION,
                handler=self._cmd_action,
                usage="action <type> [args...]",
                examples=[
                    "action command 'dir'",
                    "action type 'Hello world'",
                    "action click 100 200",
                    "action navigate https://example.com"
                ],
                aliases=["act", "a"]
            )
        )
        
        # Cognitive commands
        self.register_command(
            Command(
                name="think",
                description="Perform reasoning about a topic",
                category=CommandCategory.COGNITIVE,
                handler=self._cmd_think,
                usage="think <question or topic>",
                examples=[
                    "think How should I organize my project files?",
                    "think What's the best approach for data processing?"
                ],
                aliases=["reason", "analyze"]
            )
        )
        
        # File commands
        self.register_command(
            Command(
                name="file",
                description="File operation commands",
                category=CommandCategory.FILE,
                handler=self._cmd_file,
                usage="file <list|read|write|copy|move|delete> [args...]",
                examples=[
                    "file list .",
                    "file read config.txt",
                    "file write notes.txt 'This is a note'",
                    "file copy source.txt destination.txt",
                    "file delete temp.txt"
                ],
                aliases=["f"]
            )
        )
        
        # Configuration commands
        self.register_command(
            Command(
                name="config",
                description="Configuration management commands",
                category=CommandCategory.CONFIG,
                handler=self._cmd_config,
                usage="config <get|set|list|save|load> [args...]",
                examples=[
                    "config list",
                    "config get interface.response_format",
                    "config set interface.response_format markdown",
                    "config save",
                    "config load"
                ],
                aliases=["cfg"]
            )
        )
        
        # Help command
        self.register_command(
            Command(
                name="help",
                description="Display help information",
                category=CommandCategory.HELP,
                handler=self._cmd_help,
                usage="help [command]",
                examples=["help", "help task", "help file read"],
                aliases=["?", "h"]
            )
        )
    
    def register_command(self, command: Command) -> None:
        """Register a command in the command registry."""
        # Register main command
        self.commands[command.name] = command
        
        # Register aliases
        for alias in command.aliases:
            self.command_aliases[alias] = command.name
    
    def get_command(self, name: str) -> Optional[Command]:
        """Get a command by name or alias."""
        # Check if it's a direct command name
        if name in self.commands:
            return self.commands[name]
            
        # Check if it's an alias
        if name in self.command_aliases:
            command_name = self.command_aliases[name]
            return self.commands[command_name]
            
        return None
    
    async def execute_command(self, command_line: str) -> CommandResult:
        """Parse and execute a command line."""
        if not command_line or not command_line.strip():
            return CommandResult(success=False, error="Empty command")
            
        start_time = time.time()
        
        try:
            # Parse command line
            parts = self._parse_command_line(command_line)
            if not parts:
                return CommandResult(success=False, error="Empty command")
                
            command_name = parts[0].lower()
            args = parts[1:]
            
            # Get command
            command = self.get_command(command_name)
            if not command:
                return CommandResult(success=False, error=f"Unknown command: {command_name}")
            
            # Check if command requires confirmation
            if command.requires_confirmation:
                confirmed = await self._confirm_command(command, args)
                if not confirmed:
                    return CommandResult(success=False, error="Command cancelled by user")
            
            # Execute command with timeout
            try:
                result = await asyncio.wait_for(
                    command.handler(args),
                    timeout=self.command_timeout
                )
                
                # Add duration
                result.duration = time.time() - start_time
                
                return result
            except asyncio.TimeoutError:
                return CommandResult(
                    success=False,
                    error=f"Command timed out after {self.command_timeout} seconds",
                    duration=time.time() - start_time
                )
        except Exception as e:
            error_msg = f"Command execution error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return CommandResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
    
    def _parse_command_line(self, command_line: str) -> List[str]:
        """Parse a command line into command and arguments."""
        parts = []
        current_part = ""
        in_quotes = False
        quote_char = None
        
        for char in command_line:
            if char in ["'", '"'] and (not in_quotes or quote_char == char):
                # Toggle quote state
                in_quotes = not in_quotes
                if in_quotes:
                    quote_char = char
                else:
                    quote_char = None
            elif char.isspace() and not in_quotes:
                # Space outside quotes, end of part
                if current_part:
                    parts.append(current_part)
                    current_part = ""
            else:
                # Add character to current part
                current_part += char
        
        # Add the last part if any
        if current_part:
            parts.append(current_part)
        
        return parts
    
    async def _confirm_command(self, command: Command, args: List[str]) -> bool:
        """Ask for confirmation before executing a command."""
        # Format command and args for display
        command_str = command.name
        args_str = " ".join(args)
        full_command = f"{command_str} {args_str}".strip()
        
        # Ask for confirmation
        message = f"{Fore.YELLOW}Confirm execution of: {Fore.WHITE}{full_command}{Fore.YELLOW} (y/n):{Style.RESET_ALL} "
        
        if self.prompt_session:
            # Use prompt_toolkit if available
            response = await asyncio.to_thread(
                self.prompt_session.prompt,
                message
            )
        else:
            # Use standard input
            print(message, end="", flush=True)
            response = input().strip().lower()
        
        return response.lower() in ["y", "yes"]
    
    #--------------------------------------------------------------------
    # Command Handlers
    #--------------------------------------------------------------------
    
    async def _cmd_status(self, args: List[str]) -> CommandResult:
        """Handle the status command."""
        if not self.orchestrator:
            return CommandResult(success=False, error="Orchestrator not available")
            
        try:
            # Get system status
            status = await self.orchestrator.get_system_status()
            
            # If a component is specified, filter the results
            if args and args[0]:
                component = args[0].lower()
                
                if component in status["components"]:
                    return CommandResult(
                        success=True,
                        data={
                            "component": component,
                            "status": status["components"][component]
                        }
                    )
                else:
                    return CommandResult(
                        success=False,
                        error=f"Unknown component: {component}"
                    )
            
            # Return full status
            return CommandResult(
                success=True,
                data=status
            )
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to get status: {str(e)}"
            )
    
    async def _cmd_shutdown(self, args: List[str]) -> CommandResult:
        """Handle the shutdown command."""
        # This will just return success, actual shutdown is handled by caller
        return CommandResult(
            success=True,
            data={
                "message": "System shutdown initiated"
            }
        )
    
    async def _cmd_version(self, args: List[str]) -> CommandResult:
        """Handle the version command."""
        version_info = {
            "name": "ORAMA System",
            "version": "1.0.0",
            "build_date": "2025-05-20",
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        return CommandResult(
            success=True,
            data=version_info
        )
    
    async def _cmd_task(self, args: List[str]) -> CommandResult:
        """Handle the task command."""
        if not self.orchestrator:
            return CommandResult(success=False, error="Orchestrator not available")
            
        if not args:
            return CommandResult(success=False, error="Missing task subcommand")
            
        subcommand = args[0].lower()
        
        try:
            if subcommand == "create":
                # task create <description> [priority]
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing task description")
                    
                description = args[1]
                priority = int(args[2]) if len(args) > 2 and args[2].isdigit() else 1
                
                # Create task
                task_id = await self.orchestrator.schedule_task(
                    task_type="cognitive",
                    description=description,
                    priority=priority
                )
                
                return CommandResult(
                    success=True,
                    data={
                        "task_id": task_id,
                        "description": description
                    }
                )
                
            elif subcommand == "list":
                # Get all tasks
                tasks = await self.orchestrator.task_manager.get_all_tasks()
                
                return CommandResult(
                    success=True,
                    data={
                        "tasks": tasks
                    }
                )
                
            elif subcommand == "status":
                # task status <task_id>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing task ID")
                    
                task_id = args[1]
                
                # Get task status
                task = await self.orchestrator.get_task_status(task_id)
                
                if not task:
                    return CommandResult(success=False, error=f"Task not found: {task_id}")
                
                return CommandResult(
                    success=True,
                    data=task
                )
                
            elif subcommand == "cancel":
                # task cancel <task_id>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing task ID")
                    
                task_id = args[1]
                
                # Cancel task
                success = await self.orchestrator.cancel_task(task_id)
                
                if not success:
                    return CommandResult(success=False, error=f"Failed to cancel task: {task_id}")
                
                return CommandResult(
                    success=True,
                    data={
                        "task_id": task_id,
                        "message": "Task cancelled"
                    }
                )
                
            else:
                return CommandResult(success=False, error=f"Unknown task subcommand: {subcommand}")
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Task command error: {str(e)}"
            )
    
    async def _cmd_memory(self, args: List[str]) -> CommandResult:
        """Handle the memory command."""
        if not self.memory:
            return CommandResult(success=False, error="Memory engine not available")
            
        if not args:
            return CommandResult(success=False, error="Missing memory subcommand")
            
        subcommand = args[0].lower()
        
        try:
            if subcommand == "store":
                # memory store <content> [memory_type] [importance]
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing memory content")
                    
                content = args[1]
                memory_type = args[2] if len(args) > 2 else "semantic"
                importance = float(args[3]) if len(args) > 3 and args[3].replace(".", "").isdigit() else 0.5
                
                # Store memory
                memory_id = await self.memory.create_memory(
                    content=content,
                    memory_type=memory_type,
                    importance=importance
                )
                
                return CommandResult(
                    success=True,
                    data={
                        "memory_id": memory_id,
                        "content": content,
                        "memory_type": memory_type,
                        "importance": importance
                    }
                )
                
            elif subcommand == "retrieve":
                # memory retrieve <query> [memory_types] [limit]
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing query")
                    
                query = args[1]
                memory_types = args[2].split(",") if len(args) > 2 else None
                limit = int(args[3]) if len(args) > 3 and args[3].isdigit() else 5
                
                # Retrieve memories
                results = await self.memory.remember(
                    query=query,
                    memory_types=memory_types,
                    limit=limit
                )
                
                # Format results
                formatted_results = []
                for result in results.results:
                    formatted_results.append({
                        "content": result.content,
                        "score": result.score,
                        "memory_type": result.memory_type,
                        "created_at": result.metadata.created_at if hasattr(result.metadata, "created_at") else None
                    })
                
                return CommandResult(
                    success=True,
                    data={
                        "results": formatted_results,
                        "total_found": results.total_found,
                        "query_time_ms": results.query_time_ms
                    }
                )
                
            elif subcommand == "delete":
                # memory delete <memory_id>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing memory ID")
                    
                memory_id = args[1]
                
                # Delete memory
                success = await self.memory.forget(memory_id)
                
                if not success:
                    return CommandResult(success=False, error=f"Failed to delete memory: {memory_id}")
                
                return CommandResult(
                    success=True,
                    data={
                        "memory_id": memory_id,
                        "message": "Memory deleted"
                    }
                )
                
            else:
                return CommandResult(success=False, error=f"Unknown memory subcommand: {subcommand}")
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Memory command error: {str(e)}"
            )
    
    async def _cmd_action(self, args: List[str]) -> CommandResult:
        """Handle the action command."""
        if not self.action:
            return CommandResult(success=False, error="Action system not available")
            
        if not args:
            return CommandResult(success=False, error="Missing action type")
            
        action_type = args[0].lower()
        
        try:
            if action_type == "command":
                # action command <command>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing command")
                    
                command = args[1]
                
                # Execute command
                result = await self.action.execute_command(command)
                
                return CommandResult(
                    success=result.success,
                    data={
                        "command": command,
                        "return_code": result.data["return_code"] if result.success and "return_code" in result.data else None,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    },
                    error=result.error
                )
                
            elif action_type == "type":
                # action type <text>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing text")
                    
                text = args[1]
                
                # Type text
                result = await self.action.type_text(text)
                
                return CommandResult(
                    success=result.success,
                    data={
                        "text": text,
                        "characters_typed": result.data["characters_typed"] if result.success and "characters_typed" in result.data else None
                    },
                    error=result.error
                )
                
            elif action_type == "click":
                # action click <x> <y>
                if len(args) < 3:
                    return CommandResult(success=False, error="Missing coordinates")
                    
                try:
                    x = int(args[1])
                    y = int(args[2])
                except ValueError:
                    return CommandResult(success=False, error="Invalid coordinates")
                
                # Click
                result = await self.action.mouse_click(x, y)
                
                return CommandResult(
                    success=result.success,
                    data={
                        "x": x,
                        "y": y,
                        "verified": result.data["verified"] if result.success and "verified" in result.data else None
                    },
                    error=result.error
                )
                
            elif action_type == "navigate":
                # action navigate <url>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing URL")
                    
                url = args[1]
                
                # Navigate
                result = await self.action.navigate_to(url)
                
                return CommandResult(
                    success=result.success,
                    data={
                        "url": url,
                        "status_code": result.data["status_code"] if result.success and "status_code" in result.data else None,
                        "title": result.data["title"] if result.success and "title" in result.data else None
                    },
                    error=result.error
                )
                
            elif action_type == "screenshot":
                # action screenshot [filename]
                filename = args[1] if len(args) > 1 else None
                
                # Take screenshot
                result = await self.action.take_screenshot(filename=filename)
                
                return CommandResult(
                    success=result.success,
                    data={
                        "filename": result.data["filename"] if result.success and "filename" in result.data else None,
                        "size": result.data["size"] if result.success and "size" in result.data else None,
                        "resolution": result.data["resolution"] if result.success and "resolution" in result.data else None
                    },
                    error=result.error
                )
                
            else:
                return CommandResult(success=False, error=f"Unknown action type: {action_type}")
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Action command error: {str(e)}"
            )
    
    async def _cmd_think(self, args: List[str]) -> CommandResult:
        """Handle the think command."""
        if not self.cognitive:
            return CommandResult(success=False, error="Cognitive engine not available")
            
        if not args:
            return CommandResult(success=False, error="Missing question or topic")
            
        # Join all arguments as the question
        question = " ".join(args)
        
        try:
            # Perform reasoning
            response = await self.cognitive.reason_about(question)
            
            return CommandResult(
                success=True,
                data={
                    "question": question,
                    "reasoning": response
                }
            )
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Reasoning error: {str(e)}"
            )
    
    async def _cmd_file(self, args: List[str]) -> CommandResult:
        """Handle the file command."""
        if not self.action:
            return CommandResult(success=False, error="Action system not available")
            
        if not args:
            return CommandResult(success=False, error="Missing file subcommand")
            
        subcommand = args[0].lower()
        
        try:
            if subcommand == "list":
                # file list <path>
                if len(args) < 2:
                    path = "."  # Current directory
                else:
                    path = args[1]
                
                # List directory
                result = await self.action.list_directory(path)
                
                return CommandResult(
                    success=result.success,
                    data=result.data if result.success else None,
                    error=result.error
                )
                
            elif subcommand == "read":
                # file read <path>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing file path")
                    
                path = args[1]
                
                # Read file
                result = await self.action.read_file(path)
                
                return CommandResult(
                    success=result.success,
                    data=result.data if result.success else None,
                    error=result.error
                )
                
            elif subcommand == "write":
                # file write <path> <content>
                if len(args) < 3:
                    return CommandResult(success=False, error="Missing file path or content")
                    
                path = args[1]
                content = args[2]
                
                # Write file
                result = await self.action.write_file(path, content)
                
                return CommandResult(
                    success=result.success,
                    data=result.data if result.success else None,
                    error=result.error
                )
                
            elif subcommand == "copy":
                # file copy <source> <destination>
                if len(args) < 3:
                    return CommandResult(success=False, error="Missing source or destination path")
                    
                source = args[1]
                destination = args[2]
                
                # Copy file
                result = await self.action.copy_file(source, destination)
                
                return CommandResult(
                    success=result.success,
                    data=result.data if result.success else None,
                    error=result.error
                )
                
            elif subcommand == "move":
                # file move <source> <destination>
                if len(args) < 3:
                    return CommandResult(success=False, error="Missing source or destination path")
                    
                source = args[1]
                destination = args[2]
                
                # Move file
                result = await self.action.move_file(source, destination)
                
                return CommandResult(
                    success=result.success,
                    data=result.data if result.success else None,
                    error=result.error
                )
                
            elif subcommand == "delete":
                # file delete <path>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing file path")
                    
                path = args[1]
                
                # Delete file
                result = await self.action.delete_file(path)
                
                return CommandResult(
                    success=result.success,
                    data=result.data if result.success else None,
                    error=result.error
                )
                
            else:
                return CommandResult(success=False, error=f"Unknown file subcommand: {subcommand}")
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"File command error: {str(e)}"
            )
    
    async def _cmd_config(self, args: List[str]) -> CommandResult:
        """Handle the config command."""
        if not args:
            return CommandResult(success=False, error="Missing config subcommand")
            
        subcommand = args[0].lower()
        
        try:
            if subcommand == "list":
                # Return full configuration
                return CommandResult(
                    success=True,
                    data=self.config
                )
                
            elif subcommand == "get":
                # config get <key>
                if len(args) < 2:
                    return CommandResult(success=False, error="Missing configuration key")
                    
                key = args[1]
                
                # Get configuration value
                value = self._get_config_value(key)
                
                if value is None:
                    return CommandResult(success=False, error=f"Configuration key not found: {key}")
                
                return CommandResult(
                    success=True,
                    data={
                        "key": key,
                        "value": value
                    }
                )
                
            elif subcommand == "set":
                # config set <key> <value>
                if len(args) < 3:
                    return CommandResult(success=False, error="Missing configuration key or value")
                    
                key = args[1]
                value = args[2]
                
                # Check if runtime changes are allowed
                if not self.allow_runtime_changes:
                    return CommandResult(success=False, error="Runtime configuration changes are disabled")
                
                # Set configuration value
                success = self._set_config_value(key, value)
                
                if not success:
                    return CommandResult(success=False, error=f"Failed to set configuration: {key}")
                
                return CommandResult(
                    success=True,
                    data={
                        "key": key,
                        "value": value,
                        "message": "Configuration updated"
                    }
                )
                
            elif subcommand == "save":
                # Save configuration to file
                await self._save_config()
                
                return CommandResult(
                    success=True,
                    data={
                        "message": "Configuration saved to file",
                        "file": self.config_file
                    }
                )
                
            elif subcommand == "load":
                # Load configuration from file
                success = await self._load_config()
                
                if not success:
                    return CommandResult(success=False, error="Failed to load configuration")
                
                return CommandResult(
                    success=True,
                    data={
                        "message": "Configuration loaded from file",
                        "file": self.config_file
                    }
                )
                
            else:
                return CommandResult(success=False, error=f"Unknown config subcommand: {subcommand}")
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Config command error: {str(e)}"
            )
    
    async def _cmd_help(self, args: List[str]) -> CommandResult:
        """Handle the help command."""
        if not args:
            # Show general help
            commands_by_category = {}
            
            for command in self.commands.values():
                category = command.category.name
                if category not in commands_by_category:
                    commands_by_category[category] = []
                    
                commands_by_category[category].append({
                    "name": command.name,
                    "description": command.description,
                    "usage": command.usage
                })
            
            return CommandResult(
                success=True,
                data={
                    "categories": commands_by_category
                }
            )
        else:
            # Show help for specific command
            command_name = args[0].lower()
            command = self.get_command(command_name)
            
            if not command:
                return CommandResult(success=False, error=f"Unknown command: {command_name}")
            
            return CommandResult(
                success=True,
                data=command.to_dict()
            )
    
    #--------------------------------------------------------------------
    # Configuration Management
    #--------------------------------------------------------------------
    
    def _get_config_value(self, key: str) -> Any:
        """Get a configuration value by key path."""
        # Split key path
        parts = key.split(".")
        
        # Traverse config
        current = self.config
        for part in parts:
            if part not in current:
                return None
            current = current[part]
        
        return current
    
    def _set_config_value(self, key: str, value: str) -> bool:
        """Set a configuration value by key path."""
        # Split key path
        parts = key.split(".")
        
        # Traverse config
        current = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Can't go further
                return False
            current = current[part]
        
        # Set value
        try:
            # Try to convert to appropriate type
            if value.lower() == "true":
                converted_value = True
            elif value.lower() == "false":
                converted_value = False
            elif value.lower() == "null" or value.lower() == "none":
                converted_value = None
            elif value.replace(".", "").replace("-", "").isdigit():
                if "." in value:
                    converted_value = float(value)
                else:
                    converted_value = int(value)
            else:
                converted_value = value
                
            current[parts[-1]] = converted_value
            return True
        except Exception as e:
            self.logger.error(f"Error setting configuration value: {e}")
            return False
    
    def _update_config(self, new_config: Dict) -> None:
        """Update configuration with new values."""
        # Recursively update config
        def update_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    update_dict(target[key], value)
                else:
                    target[key] = value
        
        update_dict(self.config, new_config)
    
    async def _save_config(self) -> bool:
        """Save configuration to file."""
        try:
            # Expand path
            config_file = os.path.expanduser(self.config_file)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            # Save config
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            self.logger.info(f"Configuration saved to {config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}", exc_info=True)
            return False
    
    async def _load_config(self) -> bool:
        """Load configuration from file."""
        try:
            # Expand path
            config_file = os.path.expanduser(self.config_file)
            
            # Check if file exists
            if not os.path.exists(config_file):
                self.logger.warning(f"Configuration file not found: {config_file}")
                return False
            
            # Load config
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                
            # Update config
            self._update_config(user_config)
            
            self.logger.info(f"Configuration loaded from {config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            return False
    
    #--------------------------------------------------------------------
    # Response Formatting
    #--------------------------------------------------------------------
    
    def _format_response(self, result: CommandResult, format: Optional[str] = None) -> SystemResponse:
        """Format a command result for display."""
        # Use specified format or default
        response_format = format or self.response_format
        
        if not result.success:
            # Error response
            if COLORS_AVAILABLE:
                content = f"{Fore.RED}Error: {result.error}{Style.RESET_ALL}"
            else:
                content = f"Error: {result.error}"
                
            return SystemResponse(
                content=content,
                format="error",
                command=None,
                duration=result.duration
            )
        
        # Success response
        if response_format == "plain":
            # Simple text format
            if isinstance(result.data, str):
                content = result.data
            elif isinstance(result.data, dict):
                # Format dictionary as key-value pairs
                lines = []
                for key, value in result.data.items():
                    lines.append(f"{key}: {value}")
                content = "\n".join(lines)
            elif isinstance(result.data, list):
                # Format list as bullet points
                lines = []
                for item in result.data:
                    lines.append(f"- {item}")
                content = "\n".join(lines)
            else:
                # Just convert to string
                content = str(result.data)
        
        elif response_format == "rich" and RICH_AVAILABLE:
            # Rich text format using rich library
            if isinstance(result.data, str):
                content = result.data
            elif isinstance(result.data, dict):
                # Create table for dictionary
                table = Table(show_header=True, header_style="bold")
                table.add_column("Key")
                table.add_column("Value")
                
                # Add rows
                self._add_dict_to_table(table, result.data)
                
                # Render table to string
                content = table.__str__()
            elif isinstance(result.data, list):
                # Create table for list
                table = Table(show_header=False)
                table.add_column("")
                
                # Add rows
                for item in result.data:
                    table.add_row(str(item))
                
                # Render table to string
                content = table.__str__()
        
        elif response_format == "markdown":
            # Markdown format
            if isinstance(result.data, str):
                content = result.data
            elif isinstance(result.data, dict):
                # Format dictionary as markdown
                lines = ["| Key | Value |", "|-----|-------|"]
                
                self._add_dict_to_markdown(lines, result.data)
                
                content = "\n".join(lines)
            elif isinstance(result.data, list):
                # Format list as markdown bullet points
                lines = []
                for item in result.data:
                    lines.append(f"- {item}")
                content = "\n".join(lines)
            else:
                # Just convert to string
                content = f"```\n{result.data}\n```"
        
        else:
            # Fallback to basic text
            if isinstance(result.data, (dict, list)):
                try:
                    content = json.dumps(result.data, indent=2)
                except:
                    content = str(result.data)
            else:
                content = str(result.data)
        
        return SystemResponse(
            content=content,
            format=response_format,
            command=None,
            duration=result.duration
        )
    
    def _add_dict_to_table(self, table: 'Table', data: Dict, prefix: str = "") -> None:
        """Add dictionary items to a rich table."""
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively add nested dictionary
                self._add_dict_to_table(table, value, f"{full_key}.")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Add list of dictionaries
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._add_dict_to_table(table, item, f"{full_key}[{i}].")
                    else:
                        table.add_row(f"{full_key}[{i}]", str(item))
            else:
                # Add simple value
                table.add_row(str(full_key), str(value))
    
    def _add_dict_to_markdown(self, lines: List[str], data: Dict, prefix: str = "") -> None:
        """Add dictionary items to markdown table."""
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively add nested dictionary
                self._add_dict_to_markdown(lines, value, f"{full_key}.")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Add list of dictionaries
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._add_dict_to_markdown(lines, item, f"{full_key}[{i}].")
                    else:
                        lines.append(f"| {full_key}[{i}] | {item} |")
            else:
                # Add simple value
                lines.append(f"| {full_key} | {value} |")
    
    #--------------------------------------------------------------------
    # Status Display
    #--------------------------------------------------------------------
    
    async def _status_display_loop(self) -> None:
        """Background task for status display."""
        self.logger.info("Starting status display loop")
        
        if not RICH_AVAILABLE or not self.orchestrator:
            self.logger.warning("Rich library or orchestrator not available, status display disabled")
            return
            
        try:
            # Create progress table
            table = Table(show_header=True, header_style="bold")
            table.add_column("Component")
            table.add_column("Status")
            table.add_column("Details")
            
            # Create live display
            self._status_live = Live(table, refresh_per_second=4, transient=True)
            self._status_live.start()
            
            while self._running:
                # Get system status
                status = await self.orchestrator.get_system_status()
                
                # Update table
                table = Table(show_header=True, header_style="bold")
                table.add_column("Component")
                table.add_column("Status")
                table.add_column("Details")
                
                # Add component status
                for component, details in status["components"].items():
                    component_status = details["status"]
                    
                    # Format status with color
                    if component_status == "ready":
                        status_display = "[green]Ready[/green]"
                    elif component_status == "error":
                        status_display = f"[red]Error: {details.get('error', 'Unknown error')}[/red]"
                    else:
                        status_display = component_status.capitalize()
                    
                    # Format details
                    detail_text = ""
                    if "metrics" in details and details["metrics"]:
                        metrics = []
                        for key, value in details["metrics"].items():
                            metrics.append(f"{key}: {value}")
                        detail_text = ", ".join(metrics)
                    
                    table.add_row(component.capitalize(), status_display, detail_text)
                
                # Add system metrics
                table.add_row("", "", "")
                table.add_row("[bold]System Metrics[/bold]", "", "")
                
                for resource, details in status["resources"].items():
                    usage = details.get("percentage_used", 0)
                    
                    # Format usage with color
                    if usage > 90:
                        usage_display = f"[red]{usage:.1f}%[/red]"
                    elif usage > 70:
                        usage_display = f"[yellow]{usage:.1f}%[/yellow]"
                    else:
                        usage_display = f"[green]{usage:.1f}%[/green]"
                    
                    table.add_row(
                        resource.capitalize(),
                        usage_display,
                        f"Allocated: {details.get('allocated', 0)}, Limit: {details.get('limit', 0)}"
                    )
                
                # Add task status
                table.add_row("", "", "")
                table.add_row("[bold]Task Status[/bold]", "", "")
                table.add_row(
                    "Tasks",
                    f"Active: {status['tasks']['active']}",
                    f"Pending: {status['tasks']['pending']}, Completed: {status['tasks']['completed']}"
                )
                
                # Update live display
                self._status_live.update(table)
                
                # Sleep for update interval
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Status display loop cancelled")
            if self._status_live:
                self._status_live.stop()
        except Exception as e:
            self.logger.error(f"Error in status display loop: {e}", exc_info=True)
            if self._status_live:
                self._status_live.stop()
    
    #--------------------------------------------------------------------
    # Interactive Interface
    #--------------------------------------------------------------------
    
    async def interactive_session(self) -> None:
        """Run an interactive command session."""
        # Print welcome message
        self._print_welcome()
        
        # Command loop
        while self._running:
            try:
                # Get command
                command_line = await self._get_command_input()
                
                if not command_line or not command_line.strip():
                    continue
                    
                # Handle special command
                if command_line.lower() in ["exit", "quit", "shutdown"]:
                    # Confirm exit
                    shutdown_command = self.get_command("shutdown")
                    if shutdown_command:
                        confirmed = await self._confirm_command(
                            shutdown_command,
                            []
                        )
                        
                        if confirmed:
                            print("Shutting down...")
                            await self.stop()
                            break
                        else:
                            continue
                    else:
                        # Fallback if shutdown command is not registered
                        print("Shutting down...")
                        await self.stop()
                        break
                
                # Execute command
                result = await self.execute_command(command_line)
                
                # Format and display result
                response = self._format_response(result)
                
                # Add to history
                self.response_history.append(response)
                if len(self.response_history) > self.max_responses:
                    self.response_history = self.response_history[-self.max_responses:]
                
                # Display response
                self._display_response(response)
                
            except KeyboardInterrupt:
                print("\nCommand interrupted")
            except Exception as e:
                if COLORS_AVAILABLE:
                    print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                else:
                    print(f"Error: {str(e)}")
    
    async def _get_command_input(self) -> str:
        """Get command input from user."""
        if self.prompt_session:
            # Use prompt_toolkit if available
            prompt = f"{Fore.CYAN}orama> {Style.RESET_ALL}" if COLORS_AVAILABLE else "orama> "
            
            # Create command completer
            command_names = list(self.commands.keys()) + list(self.command_aliases.keys())
            completer = FuzzyCompleter(WordCompleter(command_names))
            
            # Get input
            return await asyncio.to_thread(
                self.prompt_session.prompt,
                prompt,
                completer=completer
            )
        else:
            # Use standard input
            prompt = f"{Fore.CYAN}orama> {Style.RESET_ALL}" if COLORS_AVAILABLE else "orama> "
            print(prompt, end="", flush=True)
            return input().strip()
    
    def _print_welcome(self) -> None:
        """Print welcome message."""
        if COLORS_AVAILABLE:
            print(f"{Fore.GREEN}=============================================")
            print(f"{Fore.CYAN}           ORAMA SYSTEM {Style.BRIGHT}v1.0.0{Style.NORMAL}")
            print(f"{Fore.BLUE}     Autonomous Agent Architecture")
            print(f"{Fore.GREEN}=============================================")
            print(f"{Fore.WHITE}Type '{Fore.YELLOW}help{Fore.WHITE}' for a list of commands.")
            print(f"Type '{Fore.YELLOW}exit{Fore.WHITE}' to quit.")
            print()
        else:
            print("=============================================")
            print("           ORAMA SYSTEM v1.0.0")
            print("     Autonomous Agent Architecture")
            print("=============================================")
            print("Type 'help' for a list of commands.")
            print("Type 'exit' to quit.")
            print()
    
    def _display_response(self, response: SystemResponse) -> None:
        """Display a system response."""
        # Print response with appropriate formatting
        if response.format == "error":
            # Error response
            print(response.content)
        elif response.format == "rich" and RICH_AVAILABLE:
            # Rich formatted response
            try:
                # Try to parse as JSON first
                data = json.loads(response.content)
                self.console.print_json(json.dumps(data))
            except:
                # Not JSON, try to parse as table
                if response.content.startswith("|") and "\n|" in response.content:
                    # Looks like a table, create rich table
                    self.console.print(response.content)
                else:
                    # Just print as text
                    self.console.print(response.content)
        elif response.format == "markdown" and RICH_AVAILABLE:
            # Markdown formatted response
            self.console.print(Markdown(response.content))
        else:
            # Plain text response
            print(response.content)
        
        if response.duration:
            if COLORS_AVAILABLE:
                print(f"{Fore.BLUE}Command completed in {response.duration:.3f}s{Style.RESET_ALL}")
            else:
                print(f"Command completed in {response.duration:.3f}s")
        
        print()  # Empty line after response

# Process command-line arguments if run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORAMA Interface")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("command", nargs="*", help="Command to execute")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create test config
    config = {
        "command": {
            "timeout": 60.0,
            "history_file": "~/.orama_history"
        },
        "status": {
            "update_interval": 0.5,
            "metrics_displayed": ["cpu", "memory", "task_count", "active_tasks"]
        },
        "results": {
            "max_displayed": 10,
            "format": "rich"
        },
        "config": {
            "allow_runtime_changes": True,
            "save_on_exit": True,
            "config_file": "~/.orama_config.json"
        }
    }
    
    async def main():
        # Create interface
        interface = Interface(config)
        await interface.start()
        
        try:
            if args.interactive:
                # Run interactive session
                await interface.interactive_session()
            elif args.command:
                # Execute command
                command_line = " ".join(args.command)
                result = await interface.execute_command(command_line)
                
                # Format and display result
                response = interface._format_response(result)
                interface._display_response(response)
                
                # Exit with success/failure code
                sys.exit(0 if result.success else 1)
            else:
                # No command, print help
                parser.print_help()
        finally:
            # Shutdown
            await interface.stop()
    
    # Run main
    asyncio.run(main())
