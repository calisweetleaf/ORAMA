#!/usr/bin/env python3
# ORAMA System - Orchestration Engine
# Event-driven coordination for cognitive, memory, and action subsystems

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import traceback
import psutil
import signal
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable, TypeVar, Generic, Coroutine
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Type variables for generics
T = TypeVar('T')

# Constants for orchestration
DEFAULT_EVENT_BUFFER_SIZE = 1000
DEFAULT_TASK_QUEUE_SIZE = 100
MAX_CONCURRENT_TASKS = 3
DEFAULT_CYCLE_INTERVAL = 0.01  # seconds
MAX_EVENT_BATCH_SIZE = 10
DEFAULT_BACKPRESSURE_THRESHOLD = 0.8  # 80% capacity
DEFAULT_SHUTDOWN_TIMEOUT = 15.0  # seconds

# Task priorities aligned with cognitive_engine.py
class TaskPriority(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

# Task execution status aligned with cognitive_engine.py
class TaskStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()

# Event types for the orchestration system
class EventType(Enum):
    # System lifecycle events
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    COMPONENT_READY = auto()
    COMPONENT_ERROR = auto()
    
    # Task management events
    TASK_CREATED = auto()
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TASK_CANCELLED = auto()
    
    # Perception events
    SCREEN_CHANGE = auto()
    TEXT_DETECTED = auto()
    UI_ELEMENT_DETECTED = auto()
    PROCESS_STARTED = auto()
    PROCESS_ENDED = auto()
    FILE_CHANGED = auto()
    ERROR_DETECTED = auto()
    
    # Action events
    ACTION_REQUESTED = auto()
    ACTION_STARTED = auto()
    ACTION_COMPLETED = auto()
    ACTION_FAILED = auto()
    
    # Memory events
    MEMORY_STORED = auto()
    MEMORY_RETRIEVED = auto()
    MEMORY_UPDATED = auto()
    MEMORY_CONSOLIDATED = auto()
    
    # User interaction events
    USER_COMMAND = auto()
    USER_FEEDBACK = auto()
    USER_INTERRUPT = auto()
    
    # Resource management events
    RESOURCE_PRESSURE = auto()
    RESOURCE_RELEASED = auto()
    
    # Custom event
    CUSTOM = auto()

# Component types in the system
class ComponentType(Enum):
    ORCHESTRATOR = auto()
    COGNITIVE = auto()
    MEMORY = auto()
    ACTION = auto()
    INTERFACE = auto()
    SYSTEM = auto()

@dataclass
class Event:
    """Event for inter-component communication."""
    event_type: EventType
    source: ComponentType
    target: Optional[ComponentType] = None  # None means broadcast
    data: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher values = higher priority
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.name,
            "source": self.source.name,
            "target": self.target.name if self.target else None,
            "data": self.data,
            "timestamp": self.timestamp,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        """Create event from dictionary."""
        return cls(
            id=data.get("id", f"evt_{uuid.uuid4().hex[:8]}"),
            event_type=EventType[data["event_type"]],
            source=ComponentType[data["source"]],
            target=ComponentType[data["target"]] if data.get("target") else None,
            data=data.get("data", {}),
            timestamp=data.get("timestamp", time.time()),
            priority=data.get("priority", 0)
        )

@dataclass
class Resource:
    """Resource allocation information."""
    name: str
    type: str  # cpu, memory, gpu, disk, network, etc.
    allocated: float
    available: float
    limit: float
    utilization: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    @property
    def percentage_used(self) -> float:
        """Get percentage of resource used."""
        if self.limit == 0:
            return 0
        return (self.allocated / self.limit) * 100
    
    @property
    def is_critical(self) -> bool:
        """Check if resource usage is critical."""
        return self.percentage_used > 90

@dataclass
class ComponentStatus:
    """Status information for a system component."""
    component_type: ComponentType
    status: str  # ready, starting, error, etc.
    last_updated: float = field(default_factory=time.time)
    metrics: Dict = field(default_factory=dict)
    error: Optional[str] = None
    
    def update(self, status: str, metrics: Optional[Dict] = None, error: Optional[str] = None) -> None:
        """Update component status."""
        self.status = status
        self.last_updated = time.time()
        if metrics:
            self.metrics.update(metrics)
        if error:
            self.error = error

class PriorityQueue(Generic[T]):
    """Priority queue implementation for events and tasks."""
    
    def __init__(self, maxsize: int = 0):
        """Initialize the priority queue."""
        self.maxsize = maxsize
        self.queue: List[Tuple[int, int, T]] = []  # (priority, sequence, item)
        self.sequence = 0  # For FIFO ordering of same priority items
        self.lock = asyncio.Lock()
    
    async def put(self, item: T, priority: int = 0) -> None:
        """Put an item in the queue with given priority."""
        async with self.lock:
            if self.maxsize > 0 and len(self.queue) >= self.maxsize:
                raise asyncio.QueueFull()
            
            self.queue.append((priority, self.sequence, item))
            self.sequence += 1
            self.queue.sort(reverse=True)  # Higher priority first
    
    async def get(self) -> T:
        """Get highest priority item from the queue."""
        async with self.lock:
            if not self.queue:
                raise asyncio.QueueEmpty()
            
            _, _, item = self.queue.pop(0)
            return item
    
    async def remove(self, predicate: Callable[[T], bool]) -> Optional[T]:
        """Remove an item that matches the predicate."""
        async with self.lock:
            for i, (_, _, item) in enumerate(self.queue):
                if predicate(item):
                    return self.queue.pop(i)[2]
            return None
    
    def __len__(self) -> int:
        """Get queue size."""
        return len(self.queue)
    
    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0
    
    @property
    def full(self) -> bool:
        """Check if queue is full."""
        return self.maxsize > 0 and len(self.queue) >= self.maxsize
    
    @property
    def usage_ratio(self) -> float:
        """Get queue usage ratio."""
        if self.maxsize == 0:
            return 0
        return len(self.queue) / self.maxsize

class EventBus:
    """Event bus for inter-component communication."""
    
    def __init__(self, buffer_size: int = DEFAULT_EVENT_BUFFER_SIZE):
        """Initialize the event bus."""
        self.buffer_size = buffer_size
        self.event_queue = PriorityQueue[Event](maxsize=buffer_size)
        self.listeners: Dict[EventType, List[Callable[[Event], Coroutine]]] = {}
        self.broadcast_listeners: List[Callable[[Event], Coroutine]] = []
        self.logger = logging.getLogger("orama.orchestrator.eventbus")
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        try:
            await self.event_queue.put(event, priority=event.priority)
        except asyncio.QueueFull:
            self.logger.warning(f"Event queue full, dropping event: {event.event_type.name}")
    
    async def subscribe(self, event_type: EventType, callback: Callable[[Event], Coroutine]) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        
        self.listeners[event_type].append(callback)
    
    async def subscribe_broadcast(self, callback: Callable[[Event], Coroutine]) -> None:
        """Subscribe to all events (broadcast)."""
        self.broadcast_listeners.append(callback)
    
    async def unsubscribe(self, event_type: EventType, callback: Callable[[Event], Coroutine]) -> None:
        """Unsubscribe from a specific event type."""
        if event_type in self.listeners and callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)
    
    async def unsubscribe_broadcast(self, callback: Callable[[Event], Coroutine]) -> None:
        """Unsubscribe from broadcast events."""
        if callback in self.broadcast_listeners:
            self.broadcast_listeners.remove(callback)
    
    async def process_events(self, batch_size: int = MAX_EVENT_BATCH_SIZE) -> int:
        """Process a batch of events from the queue."""
        processed = 0
        
        for _ in range(batch_size):
            if self.event_queue.empty:
                break
                
            try:
                event = await self.event_queue.get()
                
                # Deliver to specific listeners
                callbacks = self.listeners.get(event.event_type, [])
                for callback in callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        self.logger.error(f"Error in event handler for {event.event_type.name}: {e}", exc_info=True)
                
                # Deliver to broadcast listeners
                for callback in self.broadcast_listeners:
                    try:
                        await callback(event)
                    except Exception as e:
                        self.logger.error(f"Error in broadcast handler for {event.event_type.name}: {e}", exc_info=True)
                
                processed += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.error(f"Error processing event: {e}", exc_info=True)
        
        return processed
    
    @property
    def backpressure(self) -> float:
        """Get backpressure level (0.0 to 1.0)."""
        return self.event_queue.usage_ratio

class TaskManager:
    """Manager for task scheduling and execution."""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_TASKS, queue_size: int = DEFAULT_TASK_QUEUE_SIZE):
        """Initialize the task manager."""
        self.max_concurrent = max_concurrent
        self.task_queue = PriorityQueue[Dict](maxsize=queue_size)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.tasks_history: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger("orama.orchestrator.taskmanager")
    
    async def schedule(self, task_data: Dict, priority: int = 0) -> str:
        """Schedule a task for execution."""
        # Generate task ID if not provided
        if "id" not in task_data:
            task_data["id"] = f"task_{uuid.uuid4().hex[:8]}"
        
        # Add metadata
        task_data["scheduled_at"] = time.time()
        task_data["priority"] = priority
        task_data["status"] = "pending"
        
        # Add to queue
        try:
            await self.task_queue.put(task_data, priority)
            self.logger.info(f"Scheduled task: {task_data['id']}")
            return task_data["id"]
        except asyncio.QueueFull:
            self.logger.warning(f"Task queue full, rejecting task: {task_data.get('description', 'unknown')}")
            return ""
    
    async def execute(self, task_executor: Callable[[Dict], Coroutine]) -> Optional[Dict]:
        """Execute next task if slots available."""
        async with self.lock:
            # Check if we can execute more tasks
            if len(self.active_tasks) >= self.max_concurrent:
                return None
                
            # Check if queue is empty
            if self.task_queue.empty:
                return None
                
            # Get next task
            task_data = await self.task_queue.get()
            task_id = task_data["id"]
            
            # Update task status
            task_data["status"] = "executing"
            task_data["started_at"] = time.time()
            
            # Create asyncio task
            task = asyncio.create_task(self._execute_task(task_executor, task_data))
            self.active_tasks[task_id] = task
            
            self.logger.info(f"Started execution of task: {task_id}")
            return task_data
    
    async def _execute_task(self, task_executor: Callable[[Dict], Coroutine], task_data: Dict) -> None:
        """Execute a task and handle completion."""
        task_id = task_data["id"]
        
        try:
            # Execute task
            result = await task_executor(task_data)
            
            # Update task data
            task_data["status"] = "completed"
            task_data["completed_at"] = time.time()
            task_data["result"] = result
            
            self.logger.info(f"Task completed: {task_id}")
        except Exception as e:
            # Handle failure
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            
            task_data["status"] = "failed"
            task_data["completed_at"] = time.time()
            task_data["error"] = error_msg
            task_data["stack_trace"] = stack_trace
            
            self.logger.error(f"Task failed: {task_id}. Error: {error_msg}")
        finally:
            # Remove from active tasks
            async with self.lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                
                # Add to history
                self.tasks_history[task_id] = task_data
                
                # Limit history size
                if len(self.tasks_history) > 1000:
                    oldest_key = min(self.tasks_history.keys(), key=lambda k: self.tasks_history[k].get("completed_at", 0))
                    del self.tasks_history[oldest_key]
    
    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending or active task."""
        async with self.lock:
            # Check if task is in queue
            task_data = await self.task_queue.remove(lambda t: t["id"] == task_id)
            
            if task_data:
                # Update task status
                task_data["status"] = "cancelled"
                task_data["completed_at"] = time.time()
                
                # Add to history
                self.tasks_history[task_id] = task_data
                
                self.logger.info(f"Cancelled queued task: {task_id}")
                return True
                
            # Check if task is active
            if task_id in self.active_tasks:
                # Cancel running task
                self.active_tasks[task_id].cancel()
                
                # Create history entry if not already present
                if task_id not in self.tasks_history:
                    self.tasks_history[task_id] = {
                        "id": task_id,
                        "status": "cancelled",
                        "completed_at": time.time()
                    }
                else:
                    self.tasks_history[task_id]["status"] = "cancelled"
                    self.tasks_history[task_id]["completed_at"] = time.time()
                
                self.logger.info(f"Cancelled active task: {task_id}")
                return True
            
            return False
    
    async def get_status(self, task_id: str) -> Optional[Dict]:
        """Get the status of a task."""
        # Check history first
        if task_id in self.tasks_history:
            return self.tasks_history[task_id]
            
        # Check active tasks
        if task_id in self.active_tasks:
            # Find task data
            for active_id, task in self.active_tasks.items():
                if active_id == task_id:
                    return {
                        "id": task_id,
                        "status": "executing",
                        "started_at": time.time()  # Approximate if not stored
                    }
        
        # Check queue
        async with self.lock:
            # We don't have a direct way to check the queue without removing
            # This is a limitation of our current implementation
            return None
    
    async def get_all_tasks(self) -> Dict[str, Dict]:
        """Get all active and historical tasks."""
        result = {}
        
        # Add historical tasks
        result.update(self.tasks_history)
        
        # Add active tasks that aren't in history yet
        for task_id in self.active_tasks:
            if task_id not in result:
                result[task_id] = {
                    "id": task_id,
                    "status": "executing"
                }
        
        return result
    
    @property
    def queue_size(self) -> int:
        """Get number of queued tasks."""
        return len(self.task_queue)
    
    @property
    def active_count(self) -> int:
        """Get number of active tasks."""
        return len(self.active_tasks)
    
    @property
    def total_tasks(self) -> int:
        """Get total number of tasks (queued + active)."""
        return self.queue_size + self.active_count
    
    @property
    def backpressure(self) -> float:
        """Get backpressure level (0.0 to 1.0)."""
        if self.max_concurrent == 0:
            return 0
        
        # Combine queue and active tasks pressure
        queue_pressure = self.task_queue.usage_ratio
        active_pressure = min(1.0, len(self.active_tasks) / self.max_concurrent)
        
        # Weighted combination
        return (queue_pressure * 0.7) + (active_pressure * 0.3)

class ResourceManager:
    """Manager for system resources."""
    
    def __init__(self, config: Dict):
        """Initialize the resource manager."""
        self.config = config
        self.resources: Dict[str, Resource] = {}
        self.history: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, value)
        self.history_max_points = config.get("history_max_points", 1000)
        self.logger = logging.getLogger("orama.orchestrator.resourcemanager")
        
        # Initialize resources
        self._init_resources()
    
    def _init_resources(self) -> None:
        """Initialize resource tracking."""
        # CPU resource
        cpu_limit = self.config.get("cpu", {}).get("limit_percent", 80)
        self.resources["cpu"] = Resource(
            name="CPU",
            type="cpu",
            allocated=0,
            available=float(psutil.cpu_count(logical=True) or 0),
            limit=cpu_limit,
            utilization=0.0
        )
        self.history["cpu"] = []
        
        # Memory resource
        memory = psutil.virtual_memory()
        memory_limit_mb = self.config.get("memory", {}).get("limit_mb", memory.total // (1024 * 1024) * 0.8)
        self.resources["memory"] = Resource(
            name="Memory",
            type="memory",
            allocated=0,
            available=memory.available // (1024 * 1024),  # MB
            limit=memory_limit_mb,
            utilization=0.0
        )
        self.history["memory"] = []
        
        # GPU resource if enabled
        if self.config.get("gpu", {}).get("enable", False):
            gpu_memory_limit_mb = self.config.get("gpu", {}).get("memory_limit_mb", 4096)
            self.resources["gpu"] = Resource(
                name="GPU",
                type="gpu",
                allocated=0,
                available=gpu_memory_limit_mb,  # This is a placeholder
                limit=gpu_memory_limit_mb,
                utilization=0.0
            )
            self.history["gpu"] = []
    
    async def update(self) -> Dict[str, bool]:
        """Update resource utilization metrics."""
        critical_resources = {}
        
        try:
            # Update CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if "cpu" in self.resources:
                self.resources["cpu"].utilization = cpu_percent
                self.resources["cpu"].last_updated = time.time()
                self.resources["cpu"].allocated = cpu_percent
                
                # Add to history
                self.history["cpu"].append((time.time(), cpu_percent))
                if len(self.history["cpu"]) > self.history_max_points:
                    self.history["cpu"] = self.history["cpu"][-self.history_max_points:]
                
                # Check if critical
                critical_resources["cpu"] = self.resources["cpu"].is_critical
            
            # Update memory utilization
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            if "memory" in self.resources:
                self.resources["memory"].utilization = memory_percent
                self.resources["memory"].last_updated = time.time()
                self.resources["memory"].allocated = (memory.total - memory.available) // (1024 * 1024)  # MB
                self.resources["memory"].available = memory.available // (1024 * 1024)  # MB
                
                # Add to history
                self.history["memory"].append((time.time(), memory_percent))
                if len(self.history["memory"]) > self.history_max_points:
                    self.history["memory"] = self.history["memory"][-self.history_max_points:]
                
                # Check if critical
                critical_resources["memory"] = self.resources["memory"].is_critical
            
            # Update GPU utilization if available
            # This would require a GPU monitoring library like pynvml
            if "gpu" in self.resources:
                # Placeholder for actual GPU monitoring
                critical_resources["gpu"] = self.resources["gpu"].is_critical
        except Exception as e:
            self.logger.error(f"Error updating resource utilization: {e}", exc_info=True)
        
        return critical_resources
    
    def allocate(self, resource_type: str, amount: float) -> bool:
        """Allocate resources for a task."""
        if resource_type not in self.resources:
            return False
            
        resource = self.resources[resource_type]
        
        # Check if allocation is possible
        if resource.allocated + amount > resource.limit:
            return False
            
        # Update allocation
        resource.allocated += amount
        resource.last_updated = time.time()
        
        return True
    
    def release(self, resource_type: str, amount: float) -> None:
        """Release allocated resources."""
        if resource_type not in self.resources:
            return
            
        resource = self.resources[resource_type]
        
        # Update allocation
        resource.allocated = max(0, resource.allocated - amount)
        resource.last_updated = time.time()
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all resources."""
        return {
            name: {
                "name": res.name,
                "type": res.type,
                "allocated": res.allocated,
                "available": res.available,
                "limit": res.limit,
                "utilization": res.utilization,
                "percentage_used": res.percentage_used,
                "is_critical": res.is_critical,
                "last_updated": res.last_updated
            }
            for name, res in self.resources.items()
        }
    
    def get_history(self, resource_type: str, points: int = 0) -> List[Tuple[float, float]]:
        """Get utilization history for a resource."""
        if resource_type not in self.history:
            return []
            
        if points <= 0 or points > len(self.history[resource_type]):
            return self.history[resource_type]
            
        return self.history[resource_type][-points:]

class Orchestrator:
    """
    ORAMA Orchestration Engine
    
    Central coordination system that manages the interaction between all ORAMA subsystems:
    - Cognitive Engine: Perception and reasoning
    - Memory Engine: Persistent knowledge storage
    - Action System: Physical interaction with the environment
    - Interface: User interaction
    
    The orchestrator implements:
    - Event-driven communication between components
    - Task scheduling and execution
    - Resource management and allocation
    - System state monitoring and control
    - Error handling and recovery
    """
    
    def __init__(self, config: Dict, cognitive_engine=None, action_system=None, 
                 memory_engine=None, interface=None, logger=None):
        """Initialize the orchestrator with configuration and subsystem references."""
        self.config = config
        self.logger = logger or logging.getLogger("orama.orchestrator")
        
        # Store subsystem references
        self.cognitive = cognitive_engine
        self.action = action_system
        self.memory = memory_engine
        self.interface = interface
        
        # Initialize event bus
        event_buffer_size = config.get("events", {}).get("buffer_size", DEFAULT_EVENT_BUFFER_SIZE)
        self.event_bus = EventBus(buffer_size=event_buffer_size)
        
        # Initialize task manager
        max_concurrent = config.get("tasks", {}).get("max_concurrent", MAX_CONCURRENT_TASKS)
        queue_size = config.get("tasks", {}).get("queue_size", DEFAULT_TASK_QUEUE_SIZE)
        self.task_manager = TaskManager(max_concurrent=max_concurrent, queue_size=queue_size)
        
        # Initialize resource manager
        self.resource_manager = ResourceManager(config.get("resources", {}))
        
        # Component status tracking
        self.component_status: Dict[ComponentType, ComponentStatus] = {}
        
        # Control variables
        self.cycle_interval = config.get("cycle_interval", DEFAULT_CYCLE_INTERVAL)
        self.backpressure_threshold = config.get("backpressure_threshold", DEFAULT_BACKPRESSURE_THRESHOLD)
        self._running = False
        self._main_task = None
        self._resource_task = None
        
        # Performance metrics
        self.metrics = {
            "events_processed": 0,
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "cycles_executed": 0,
            "backpressure": 0.0,
            "cycle_time_ms": 0.0,
            "event_processing_time_ms": 0.0,
            "resource_update_time_ms": 0.0
        }
        
        # Locks
        self._metrics_lock = asyncio.Lock()
        
        self.logger.info("Orchestrator initialized")
    
    async def start(self) -> None:
        """Start the orchestrator and initialize event handling."""
        self.logger.info("Starting orchestrator...")
        self._running = True
        
        try:
            # Initialize component status
            self._init_component_status()
            
            # Set up event subscriptions
            await self._setup_event_subscriptions()
            
            # Start background tasks
            self._main_task = asyncio.create_task(self._main_loop())
            self._resource_task = asyncio.create_task(self._resource_monitor_loop())
            
            # Publish system startup event
            await self.event_bus.publish(Event(
                event_type=EventType.SYSTEM_STARTUP,
                source=ComponentType.ORCHESTRATOR,
                data={"timestamp": time.time()}
            ))
            
            self.logger.info("Orchestrator started successfully")
        except Exception as e:
            self._running = False
            self.logger.error(f"Failed to start orchestrator: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the orchestrator and perform cleanup."""
        self.logger.info("Stopping orchestrator...")
        self._running = False
        
        try:
            # Publish system shutdown event
            await self.event_bus.publish(Event(
                event_type=EventType.SYSTEM_SHUTDOWN,
                source=ComponentType.ORCHESTRATOR,
                data={"timestamp": time.time()}
            ))
            
            # Wait for tasks to complete
            shutdown_timeout = self.config.get("shutdown_timeout", DEFAULT_SHUTDOWN_TIMEOUT)
            
            # Cancel main task
            if self._main_task:
                self._main_task.cancel()
                try:
                    await asyncio.wait_for(self._main_task, timeout=shutdown_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Cancel resource task
            if self._resource_task:
                self._resource_task.cancel()
                try:
                    await asyncio.wait_for(self._resource_task, timeout=shutdown_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Cancel all active tasks
            active_tasks = await self.task_manager.get_all_tasks()
            for task_id, task_data in active_tasks.items():
                if task_data.get("status") == "executing":
                    await self.task_manager.cancel(task_id)
            
            self.logger.info("Orchestrator stopped")
        except Exception as e:
            self.logger.error(f"Error during orchestrator shutdown: {e}", exc_info=True)
    
    def _init_component_status(self) -> None:
        """Initialize component status tracking."""
        for component_type in ComponentType:
            self.component_status[component_type] = ComponentStatus(
                component_type=component_type,
                status="initializing"
            )
        
        # Set orchestrator as ready
        self.component_status[ComponentType.ORCHESTRATOR].status = "ready"
    
    async def _setup_event_subscriptions(self) -> None:
        """Set up subscriptions for event handling."""
        # System events
        await self.event_bus.subscribe(EventType.COMPONENT_READY, self._handle_component_ready)
        await self.event_bus.subscribe(EventType.COMPONENT_ERROR, self._handle_component_error)
        
        # Task events
        await self.event_bus.subscribe(EventType.TASK_CREATED, self._handle_task_created)
        await self.event_bus.subscribe(EventType.TASK_COMPLETED, self._handle_task_completed)
        await self.event_bus.subscribe(EventType.TASK_FAILED, self._handle_task_failed)
        
        # Resource events
        await self.event_bus.subscribe(EventType.RESOURCE_PRESSURE, self._handle_resource_pressure)
        
        # User events
        await self.event_bus.subscribe(EventType.USER_COMMAND, self._handle_user_command)
        await self.event_bus.subscribe(EventType.USER_INTERRUPT, self._handle_user_interrupt)
        
        # Log all events if in debug mode
        if self.logger.level <= logging.DEBUG:
            await self.event_bus.subscribe_broadcast(self._log_event)
    
    async def _main_loop(self) -> None:
        """Main orchestration loop."""
        self.logger.info("Starting main orchestration loop")
        
        while self._running:
            try:
                cycle_start = time.time()
                
                # Process events
                event_start = time.time()
                events_processed = await self.event_bus.process_events()
                event_end = time.time()
                
                # Update metrics
                async with self._metrics_lock:
                    self.metrics["events_processed"] += events_processed
                    self.metrics["event_processing_time_ms"] = (event_end - event_start) * 1000
                
                # Execute tasks if resources available
                if self.all_components_ready() and not self.is_backpressure_critical():
                    # Execute up to N tasks per cycle
                    for _ in range(self.task_manager.max_concurrent - self.task_manager.active_count):
                        task_data = await self.task_manager.execute(self._execute_task)
                        if not task_data:
                            break
                
                # Update system metrics
                cycle_end = time.time()
                cycle_time = (cycle_end - cycle_start) * 1000  # ms
                
                async with self._metrics_lock:
                    self.metrics["cycles_executed"] += 1
                    self.metrics["cycle_time_ms"] = cycle_time
                    self.metrics["backpressure"] = self.calculate_system_backpressure()
                
                # Dynamic sleep to maintain target cycle rate
                sleep_time = max(0, self.cycle_interval - (cycle_end - cycle_start))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    # Yield to allow other tasks to run
                    await asyncio.sleep(0)
                
            except asyncio.CancelledError:
                self.logger.info("Main orchestration loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in main orchestration loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Longer sleep on error
    
    async def _resource_monitor_loop(self) -> None:
        """Resource monitoring loop."""
        self.logger.info("Starting resource monitor loop")
        
        while self._running:
            try:
                # Update resource metrics
                resource_start = time.time()
                critical_resources = await self.resource_manager.update()
                resource_end = time.time()
                
                # Update metrics
                async with self._metrics_lock:
                    self.metrics["resource_update_time_ms"] = (resource_end - resource_start) * 1000
                
                # Check for critical resources
                for resource_type, is_critical in critical_resources.items():
                    if is_critical:
                        # Emit resource pressure event
                        await self.event_bus.publish(Event(
                            event_type=EventType.RESOURCE_PRESSURE,
                            source=ComponentType.ORCHESTRATOR,
                            data={
                                "resource_type": resource_type,
                                "utilization": self.resource_manager.resources[resource_type].utilization,
                                "allocated": self.resource_manager.resources[resource_type].allocated,
                                "limit": self.resource_manager.resources[resource_type].limit
                            },
                            priority=2  # Higher priority
                        ))
                
                # Sleep interval
                await asyncio.sleep(1.0)  # 1 second update interval
                
            except asyncio.CancelledError:
                self.logger.info("Resource monitor loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in resource monitor loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Longer sleep on error
    
    #--------------------------------------------------------------------
    # Event Handlers
    #--------------------------------------------------------------------
    
    async def _log_event(self, event: Event) -> None:
        """Log all events (for debugging)."""
        self.logger.debug(f"Event: {event.event_type.name} from {event.source.name} to {event.target.name if event.target else 'all'}")
    
    async def _handle_component_ready(self, event: Event) -> None:
        """Handle component ready event."""
        component_type = event.source
        
        if component_type in self.component_status:
            self.component_status[component_type].update(
                status="ready",
                metrics=event.data.get("metrics", {})
            )
            
            self.logger.info(f"Component ready: {component_type.name}")
    
    async def _handle_component_error(self, event: Event) -> None:
        """Handle component error event."""
        component_type = event.source
        error = event.data.get("error", "Unknown error")
        
        if component_type in self.component_status:
            self.component_status[component_type].update(
                status="error",
                error=error
            )
            
            self.logger.error(f"Component error: {component_type.name}. Error: {error}")
    
    async def _handle_task_created(self, event: Event) -> None:
        """Handle task created event."""
        task_data = event.data
        
        # Schedule task
        priority = task_data.get("priority", 0)
        await self.task_manager.schedule(task_data, priority)
        
        # Update metrics
        async with self._metrics_lock:
            self.metrics["tasks_scheduled"] += 1
    
    async def _handle_task_completed(self, event: Event) -> None:
        """Handle task completed event."""
        task_data = event.data
        
        # Update metrics
        async with self._metrics_lock:
            self.metrics["tasks_completed"] += 1
            
        self.logger.info(f"Task completed: {task_data.get('id', 'unknown')}")
    
    async def _handle_task_failed(self, event: Event) -> None:
        """Handle task failed event."""
        task_data = event.data
        error = task_data.get("error", "Unknown error")
        
        # Update metrics
        async with self._metrics_lock:
            self.metrics["tasks_failed"] += 1
            
        self.logger.warning(f"Task failed: {task_data.get('id', 'unknown')}. Error: {error}")
    
    async def _handle_resource_pressure(self, event: Event) -> None:
        """Handle resource pressure event."""
        resource_type = event.data.get("resource_type", "unknown")
        utilization = event.data.get("utilization", 0)
        
        self.logger.warning(f"Resource pressure: {resource_type} at {utilization:.1f}% utilization")
        
        # Implement resource pressure mitigation strategies
        if resource_type == "memory" and utilization > 90:
            # Example: Cancel low priority tasks
            tasks = await self.task_manager.get_all_tasks()
            for task_id, task in tasks.items():
                if task.get("status") == "pending" and task.get("priority", 0) < 1:
                    await self.task_manager.cancel(task_id)
                    self.logger.info(f"Cancelled low priority task {task_id} due to memory pressure")
    
    async def _handle_user_command(self, event: Event) -> None:
        """Handle user command event."""
        command = event.data.get("command", "")
        parameters = event.data.get("parameters", {})
        
        self.logger.info(f"User command: {command}")
        
        # Process different types of commands
        if command == "status":
            # Return system status
            status = await self.get_system_status()
            
            # Publish status as event for interface
            await self.event_bus.publish(Event(
                event_type=EventType.CUSTOM,
                source=ComponentType.ORCHESTRATOR,
                target=ComponentType.INTERFACE,
                data={
                    "type": "status_response",
                    "status": status
                }
            ))
        elif command == "cancel_task":
            # Cancel a task
            task_id = parameters.get("task_id", "")
            if task_id:
                success = await self.task_manager.cancel(task_id)
                
                await self.event_bus.publish(Event(
                    event_type=EventType.CUSTOM,
                    source=ComponentType.ORCHESTRATOR,
                    target=ComponentType.INTERFACE,
                    data={
                        "type": "cancel_response",
                        "success": success,
                        "task_id": task_id
                    }
                ))
        elif command == "shutdown":
            # Graceful shutdown request
            self.logger.info("User requested system shutdown")
            
            # Acknowledge shutdown request
            await self.event_bus.publish(Event(
                event_type=EventType.CUSTOM,
                source=ComponentType.ORCHESTRATOR,
                target=ComponentType.INTERFACE,
                data={
                    "type": "shutdown_initiated",
                    "message": "System shutdown initiated"
                }
            ))
            
            # Initiate shutdown sequence
            # In a real implementation, this would trigger a graceful shutdown of all components
            # For now, we'll just stop the orchestrator
            await self.stop()
    
    async def _handle_user_interrupt(self, event: Event) -> None:
        """Handle user interrupt event."""
        interrupt_type = event.data.get("type", "unknown")
        
        self.logger.info(f"User interrupt: {interrupt_type}")
        
        if interrupt_type == "cancel_all":
            # Cancel all pending and active tasks
            tasks = await self.task_manager.get_all_tasks()
            for task_id, task in tasks.items():
                if task.get("status") in ["pending", "executing"]:
                    await self.task_manager.cancel(task_id)
            
            # Acknowledge interrupt
            await self.event_bus.publish(Event(
                event_type=EventType.CUSTOM,
                source=ComponentType.ORCHESTRATOR,
                target=ComponentType.INTERFACE,
                data={
                    "type": "interrupt_response",
                    "message": "All tasks cancelled"
                }
            ))
    
    #--------------------------------------------------------------------
    # Task Execution
    #--------------------------------------------------------------------
    
    async def _execute_task(self, task_data: Dict) -> Dict:
        """Execute a task by dispatching to appropriate subsystem."""
        task_id = task_data["id"]
        task_type = task_data.get("type", "cognitive")
        description = task_data.get("description", "")
        
        self.logger.info(f"Executing task {task_id}: {description}")
        
        try:
            # Publish task started event
            await self.event_bus.publish(Event(
                event_type=EventType.TASK_STARTED,
                source=ComponentType.ORCHESTRATOR,
                data={
                    "id": task_id,
                    "type": task_type,
                    "description": description
                }
            ))
            
            # Allocate resources
            # In a real implementation, we would allocate specific resources based on task type
            # For now, we'll just track the execution
            
            # Dispatch task based on type
            result = None
            if task_type == "cognitive" and self.cognitive:
                # Task for cognitive engine
                result = await self._dispatch_cognitive_task(task_data)
            elif task_type == "action" and self.action:
                # Task for action system
                result = await self._dispatch_action_task(task_data)
            elif task_type == "memory" and self.memory:
                # Task for memory engine
                result = await self._dispatch_memory_task(task_data)
            else:
                # Generic task handling
                # In a real implementation, this would be more sophisticated
                result = {
                    "status": "simulated",
                    "message": f"Simulated execution of task: {description}"
                }
                
                # Add some delay to simulate work
                await asyncio.sleep(1.0)
            
            # Publish task completed event
            await self.event_bus.publish(Event(
                event_type=EventType.TASK_COMPLETED,
                source=ComponentType.ORCHESTRATOR,
                data={
                    "id": task_id,
                    "type": task_type,
                    "description": description,
                    "result": result
                }
            ))
            
            return result
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Task execution failed: {error_msg}", exc_info=True)
            
            # Publish task failed event
            await self.event_bus.publish(Event(
                event_type=EventType.TASK_FAILED,
                source=ComponentType.ORCHESTRATOR,
                data={
                    "id": task_id,
                    "type": task_type,
                    "description": description,
                    "error": error_msg
                }
            ))
            
            # Re-raise to let task manager handle it
            raise
    
    async def _dispatch_cognitive_task(self, task_data: Dict) -> Dict:
        """Dispatch a task to the cognitive engine."""
        if not self.cognitive:
            raise RuntimeError("Cognitive engine not available")
            
        task_id = task_data["id"]
        description = task_data.get("description", "")
        
        # In a real implementation, this would use actual cognitive engine API
        # For now, we'll simulate the cognitive task
        
        # Create a cognitive task
        cognitive_task_id = await self.cognitive.create_task(
            description=description,
            priority=TaskPriority.NORMAL,
            context=task_data.get("context", {})
        )
        
        # Wait for task completion (in a real implementation, this would be asynchronous)
        await asyncio.sleep(2.0)  # Simulate processing time
        
        # Get task result
        task_result = await self.cognitive.get_task(cognitive_task_id)
        
        return {
            "cognitive_task_id": cognitive_task_id,
            "status": "completed",
            "result": task_result
        }
    
    async def _dispatch_action_task(self, task_data: Dict) -> Dict:
        """Dispatch a task to the action system."""
        if not self.action:
            raise RuntimeError("Action system not available")
            
        action_type = task_data.get("action_type", "")
        parameters = task_data.get("parameters", {})
        
        # In a real implementation, this would use actual action system API
        # For now, we'll simulate the action
        
        # Simulate action execution
        await asyncio.sleep(1.0)
        
        return {
            "action_type": action_type,
            "status": "executed",
            "result": f"Simulated action: {action_type}"
        }
    
    async def _dispatch_memory_task(self, task_data: Dict) -> Dict:
        """Dispatch a task to the memory engine."""
        if not self.memory:
            raise RuntimeError("Memory engine not available")
            
        operation = task_data.get("operation", "")
        parameters = task_data.get("parameters", {})
        
        # In a real implementation, this would use actual memory engine API
        # For now, we'll simulate the memory operation
        
        if operation == "store":
            # Store memory
            content = parameters.get("content", "")
            memory_type = parameters.get("memory_type", "semantic")
            
            memory_id = await self.memory.create_memory(
                content=content,
                memory_type=memory_type,
                metadata=parameters.get("metadata", {}),
                importance=parameters.get("importance", 0.5)
            )
            
            return {
                "operation": "store",
                "memory_id": memory_id,
                "status": "success"
            }
        elif operation == "retrieve":
            # Retrieve memory
            query = parameters.get("query", "")
            
            memory_results = await self.memory.remember(
                query=query,
                memory_types=parameters.get("memory_types", None),
                limit=parameters.get("limit", 5)
            )
            
            return {
                "operation": "retrieve",
                "query": query,
                "results": memory_results,
                "status": "success"
            }
        else:
            return {
                "operation": operation,
                "status": "simulated",
                "message": f"Simulated memory operation: {operation}"
            }
    
    #--------------------------------------------------------------------
    # System Status Methods
    #--------------------------------------------------------------------
    
    def all_components_ready(self) -> bool:
        """Check if all required components are ready."""
        required_components = [
            ComponentType.ORCHESTRATOR,
            ComponentType.COGNITIVE,
            ComponentType.MEMORY,
            ComponentType.ACTION
        ]
        
        for component in required_components:
            if component not in self.component_status:
                return False
                
            if self.component_status[component].status != "ready":
                return False
                
        return True
    
    def calculate_system_backpressure(self) -> float:
        """Calculate overall system backpressure level."""
        # Combine event and task backpressure
        event_pressure = self.event_bus.backpressure
        task_pressure = self.task_manager.backpressure
        
        # Get resource pressure
        resource_status = self.resource_manager.get_status()
        cpu_pressure = resource_status.get("cpu", {}).get("percentage_used", 0) / 100
        memory_pressure = resource_status.get("memory", {}).get("percentage_used", 0) / 100
        
        # Weighted combination
        return (
            (event_pressure * 0.3) + 
            (task_pressure * 0.3) + 
            (cpu_pressure * 0.2) + 
            (memory_pressure * 0.2)
        )
    
    def is_backpressure_critical(self) -> bool:
        """Check if system backpressure exceeds threshold."""
        return self.calculate_system_backpressure() > self.backpressure_threshold
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        # Get component status
        components = {
            component_type.name: {
                "status": status.status,
                "last_updated": status.last_updated,
                "metrics": status.metrics,
                "error": status.error
            }
            for component_type, status in self.component_status.items()
        }
        
        # Get resource status
        resources = self.resource_manager.get_status()
        
        # Get task status
        tasks = await self.task_manager.get_all_tasks()
        active_tasks = sum(1 for t in tasks.values() if t.get("status") == "executing")
        pending_tasks = sum(1 for t in tasks.values() if t.get("status") == "pending")
        completed_tasks = sum(1 for t in tasks.values() if t.get("status") in ["completed", "failed", "cancelled"])
        
        # Get metrics
        metrics = dict(self.metrics)
        
        # Build status report
        return {
            "status": "running" if self._running else "stopped",
            "uptime": time.time() - self.component_status[ComponentType.ORCHESTRATOR].last_updated,
            "backpressure": self.calculate_system_backpressure(),
            "components": components,
            "resources": resources,
            "tasks": {
                "active": active_tasks,
                "pending": pending_tasks,
                "completed": completed_tasks,
                "queue_size": self.task_manager.queue_size,
                "recent": {tid: task for tid, task in list(tasks.items())[-10:]}
            },
            "metrics": metrics,
            "timestamp": time.time()
        }
    
    #--------------------------------------------------------------------
    # External API Methods
    #--------------------------------------------------------------------
    
    async def schedule_task(self, task_type: str, description: str, 
                          parameters: Dict = None, priority: int = 0) -> str:
        """Schedule a task for execution."""
        task_data = {
            "id": f"task_{uuid.uuid4().hex[:8]}",
            "type": task_type,
            "description": description,
            "parameters": parameters or {},
            "created_at": time.time()
        }
        
        await self.event_bus.publish(Event(
            event_type=EventType.TASK_CREATED,
            source=ComponentType.ORCHESTRATOR,
            data=task_data,
            priority=priority
        ))
        
        return task_data["id"]
    
    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get the status of a task."""
        return await self.task_manager.get_status(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        return await self.task_manager.cancel(task_id)
    
    async def emit_event(self, event_type: EventType, data: Dict, 
                       source: Optional[ComponentType] = None, target: Optional[ComponentType] = None) -> None:
        """Emit an event into the event bus."""
        source = source or ComponentType.SYSTEM
        
        await self.event_bus.publish(Event(
            event_type=event_type,
            source=source,
            target=target,
            data=data
        ))
    
    async def get_resources_history(self, resource_type: str = "cpu", points: int = 60) -> List[Tuple[float, float]]:
        """Get resource utilization history."""
        return self.resource_manager.get_history(resource_type, points)
    
    async def get_metrics(self) -> Dict:
        """Get system metrics."""
        async with self._metrics_lock:
            return dict(self.metrics)

    async def publish_event(self, event: Event) -> None:
        """Publish an event to the event bus. Wrapper around event_bus.publish."""
        if not hasattr(self, 'event_bus') or self.event_bus is None:
            self.logger.error("Event bus not initialized in Orchestrator.")
            return
        await self.event_bus.publish(event)

    async def process_event_queue(self, batch_size: int = MAX_EVENT_BATCH_SIZE) -> int:
        """Process a batch of events from the event queue. Wrapper around event_bus.process_events."""
        if not hasattr(self, 'event_bus') or self.event_bus is None:
            self.logger.error("Event bus not initialized in Orchestrator.")
            return 0
        return await self.event_bus.process_events(batch_size)

# Simple test code when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the ORAMA Orchestration Engine")
    args = parser.parse_args()
    
    async def test_orchestrator():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Create test config
        config = {
            "events": {
                "buffer_size": 1000,
                "processing_threads": 2
            },
            "tasks": {
                "max_concurrent": 3,
                "queue_size": 100,
                "priority_levels": 5,
                "default_priority": 2
            },
            "resources": {
                "allocation_strategy": "priority",
                "cpu": {
                    "limit_percent": 80
                },
                "memory": {
                    "limit_mb": 8192
                },
                "gpu": {
                    "enable": False
                }
            }
        }
        
        # Create mock components for testing
        class MockComponent:
            def __init__(self, name):
                self.name = name
            
            async def create_task(self, description, priority=None, context=None):
                return f"mock_{uuid.uuid4().hex[:8]}"
                
            async def get_task(self, task_id):
                return {
                    "id": task_id,
                    "status": "completed",
                    "result": f"Mock result for {self.name}"
                }
                
            async def create_memory(self, content, memory_type, metadata=None, importance=0.5):
                return f"memory_{uuid.uuid4().hex[:8]}"
                
            async def remember(self, query, memory_types=None, limit=5):
                class MockResult:
                    def __init__(self):
                        self.results = []
                        self.total_found = 0
                        self.query_time_ms = 10.0
                
                return MockResult()
        
        # Create mock components
        mock_cognitive = MockComponent("cognitive")
        mock_action = MockComponent("action")
        mock_memory = MockComponent("memory")
        mock_interface = MockComponent("interface")
        
        # Create orchestrator
        orchestrator = Orchestrator(
            config=config,
            cognitive_engine=mock_cognitive,
            action_system=mock_action,
            memory_engine=mock_memory,
            interface=mock_interface
        )
        
        # Start orchestrator
        await orchestrator.start()
        
        try:
            # Simulate component ready events
            await orchestrator.emit_event(
                event_type=EventType.COMPONENT_READY,
                source=ComponentType.COGNITIVE,
                data={"metrics": {"initialization_time_ms": 150}}
            )
            
            await orchestrator.emit_event(
                event_type=EventType.COMPONENT_READY,
                source=ComponentType.MEMORY,
                data={"metrics": {"initialization_time_ms": 120}}
            )
            
            await orchestrator.emit_event(
                event_type=EventType.COMPONENT_READY,
                source=ComponentType.ACTION,
                data={"metrics": {"initialization_time_ms": 100}}
            )
            
            await orchestrator.emit_event(
                event_type=EventType.COMPONENT_READY,
                source=ComponentType.INTERFACE,
                data={"metrics": {"initialization_time_ms": 80}}
            )
            
            # Let system stabilize
            await asyncio.sleep(0.5)
            
            # Schedule some tasks
            print("Scheduling tasks...")
            tasks = []
            for i in range(5):
                task_id = await orchestrator.schedule_task(
                    task_type="cognitive",
                    description=f"Test task {i}",
                    parameters={"param1": f"value{i}"},
                    priority=i % 3  # Mix of priorities
                )
                tasks.append(task_id)
                print(f"Scheduled task: {task_id}")
            
            # Let tasks process
            print("\nLetting tasks process...")
            for i in range(5):
                await asyncio.sleep(0.5)
                metrics = await orchestrator.get_metrics()
                status = await orchestrator.get_system_status()
                
                print(f"Cycle {i+1}:")
                print(f"  Events: {metrics['events_processed']}")
                print(f"  Tasks completed: {metrics['tasks_completed']}")
                print(f"  Backpressure: {metrics['backpressure']:.2f}")
                print(f"  Active tasks: {status['tasks']['active']}")
                print(f"  Pending tasks: {status['tasks']['pending']}")
                
            # Check final status
            print("\nFinal system status:")
            status = await orchestrator.get_system_status()
            
            print(f"Components status:")
            for name, info in status["components"].items():
                print(f"  {name}: {info['status']}")
                
            print(f"Resource utilization:")
            for name, info in status["resources"].items():
                print(f"  {name}: {info['utilization']:.1f}% used")
                
            print(f"Tasks summary:")
            print(f"  Active: {status['tasks']['active']}")
            print(f"  Pending: {status['tasks']['pending']}")
            print(f"  Completed: {status['tasks']['completed']}")
            
            # Test task cancellation
            if tasks:
                print("\nTesting task cancellation...")
                task_to_cancel = tasks[0]
                cancelled = await orchestrator.cancel_task(task_to_cancel)
                print(f"Task {task_to_cancel} cancellation result: {cancelled}")
            
            # Test user command event
            print("\nTesting user command...")
            await orchestrator.emit_event(
                event_type=EventType.USER_COMMAND,
                source=ComponentType.INTERFACE,
                data={"command": "status"}
            )
            
            # Let command process
            await asyncio.sleep(0.5)
        finally:
            # Stop orchestrator
            print("\nStopping orchestrator...")
            await orchestrator.stop()
            print("Orchestrator stopped")
    
    # Run test
    asyncio.run(test_orchestrator())
