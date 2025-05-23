#!/usr/bin/env python3
# ORAMA System - Cognitive Engine
# Unified cognitive architecture integrating perception and reasoning capabilities

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import tempfile
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# CV and image processing
try:
    import cv2
    import PIL
    from PIL import Image
    import pytesseract
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# Machine learning and inference
try:
    import onnxruntime as ort
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# LLM integration
try:
    import httpx
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Memory system type annotations
try:
    from memory_engine import MemoryEngine, MemoryMetadata, MemoryResult, MemoryQueryResult
    MEMORY_TYPE_SEMANTIC = "semantic"
    MEMORY_TYPE_EPISODIC = "episodic"
    MEMORY_TYPE_PROCEDURAL = "procedural"
    MEMORY_TYPE_REFLECTIVE = "reflective"
except ImportError:
    # Define placeholder types if memory_engine is not available
    MemoryEngine = Any
    MemoryMetadata = Any
    MemoryResult = Any
    MemoryQueryResult = Any
    MEMORY_TYPE_SEMANTIC = "semantic"
    MEMORY_TYPE_EPISODIC = "episodic"
    MEMORY_TYPE_PROCEDURAL = "procedural"
    MEMORY_TYPE_REFLECTIVE = "reflective"

# Constants
DEFAULT_LLM_TIMEOUT = 30.0  # seconds
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
MAX_RETRIES = 3
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_PERCEPTION_INTERVAL = 0.1  # seconds
DEFAULT_REASONING_INTERVAL = 1.0  # seconds
MAX_PLAN_STEPS = 20
MAX_REASONING_DEPTH = 5
CONFIDENCE_THRESHOLD = 0.7

# Task priority levels
class TaskPriority(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

# Task execution status
class TaskStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()

# Perception event types
class EventType(Enum):
    SCREEN_CHANGE = auto()
    TEXT_DETECTED = auto()
    UI_ELEMENT_DETECTED = auto()
    PROCESS_STARTED = auto()
    PROCESS_ENDED = auto()
    FILE_CHANGED = auto()
    ERROR_DETECTED = auto()
    CUSTOM = auto()

@dataclass
class PerceptionEvent:
    """Event detected by the perception system."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    source: str = "perception"
    data: Dict = field(default_factory=dict)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "source": self.source,
            "data": self.data,
            "confidence": self.confidence
        }

@dataclass
class UIElement:
    """Detected UI element on screen."""
    element_type: str  # button, input, dropdown, etc.
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    text: Optional[str] = None
    confidence: float = 1.0
    attributes: Dict = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates of the element."""
        x, y, w, h = self.bounding_box
        return (x + w // 2, y + h // 2)
    
    @property
    def area(self) -> int:
        """Get area of the element in pixels."""
        _, _, w, h = self.bounding_box
        return w * h

@dataclass
class TaskPlan:
    """Plan for executing a task."""
    steps: List[Dict]
    goal: str
    context: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    current_step: int = 0
    verification_steps: List[Dict] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if plan execution is complete."""
        return self.current_step >= len(self.steps)
    
    @property
    def current(self) -> Optional[Dict]:
        """Get current step."""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def advance(self) -> None:
        """Advance to next step."""
        self.current_step += 1

@dataclass
class Task:
    """Task for the cognitive system to execute."""
    id: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    plan: Optional[TaskPlan] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    parent_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary."""
        result = asdict(self)
        result["priority"] = self.priority.name
        result["status"] = self.status.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create task from dictionary."""
        priority = TaskPriority[data["priority"]] if isinstance(data["priority"], str) else data["priority"]
        status = TaskStatus[data["status"]] if isinstance(data["status"], str) else data["status"]
        
        return cls(
            id=data["id"],
            description=data["description"],
            priority=priority,
            status=status,
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            plan=data.get("plan"),
            result=data.get("result"),
            error=data.get("error"),
            parent_id=data.get("parent_id"),
            subtasks=data.get("subtasks", []),
            context=data.get("context", {})
        )

@dataclass
class ReasoningContext:
    """Context for reasoning operations."""
    task: Task
    relevant_memories: List[MemoryResult] = field(default_factory=list)
    perception_events: List[PerceptionEvent] = field(default_factory=list)
    reasoning_history: List[Dict] = field(default_factory=list)
    system_state: Dict = field(default_factory=dict)
    variables: Dict = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    depth: int = 0
    
    def to_dict(self) -> Dict:
        """Convert context to dictionary for LLM prompting."""
        return {
            "task": self.task.to_dict(),
            "relevant_memories": [m.content for m in self.relevant_memories],
            "perception_events": [e.to_dict() for e in self.perception_events],
            "reasoning_history": self.reasoning_history,
            "system_state": self.system_state,
            "variables": self.variables,
            "constraints": self.constraints,
            "depth": self.depth
        }

class CognitiveEngine:
    """
    ORAMA Cognitive Engine - Unified architecture for perception and reasoning
    
    Integrates LLM-based reasoning with perception capabilities to create a cohesive
    cognitive system for autonomous agents. The engine combines computer vision,
    system state monitoring, and language model inference to perceive, reason about,
    and plan actions in the computing environment.
    
    Key capabilities:
    - Screen analysis and UI element detection
    - System state monitoring and event detection
    - Context-aware reasoning using local LLMs
    - Hierarchical task planning and execution
    - Integration with memory for knowledge persistence
    - Learning from experience through reflective processes
    """
    
    def __init__(self, config: Dict, memory_engine: MemoryEngine, logger=None):
        """Initialize the cognitive engine with configuration."""
        self.config = config
        self.memory = memory_engine
        self.logger = logger or logging.getLogger("orama.cognitive")
        
        # LLM configuration
        self.llm_config = config.get("llm", {})
        self.model = self.llm_config.get("model", "deepseek-coder-7b-instruct-q5_K_M")
        self.context_size = self.llm_config.get("context_size", DEFAULT_CONTEXT_SIZE)
        self.temperature = self.llm_config.get("temperature", DEFAULT_TEMPERATURE)
        self.top_p = self.llm_config.get("top_p", DEFAULT_TOP_P)
        self.timeout = self.llm_config.get("timeout", DEFAULT_LLM_TIMEOUT)
        self.streaming = self.llm_config.get("streaming", True)
        self.fallback_model = self.llm_config.get("fallback_model", "gemma-2b-instruct-q5_K_M")
        
        # Perception configuration
        self.perception_config = config.get("perception", {})
        self.perception_interval = self.perception_config.get("interval", DEFAULT_PERCEPTION_INTERVAL)
        self.screen_capture_config = self.perception_config.get("screen_capture", {})
        self.ocr_config = self.perception_config.get("ocr", {})
        self.ui_detection_config = self.perception_config.get("ui_detection", {})
        self.system_monitor_config = self.perception_config.get("system_monitor", {})
        
        # Reasoning configuration
        self.reasoning_config = config.get("reasoning", {})
        self.reasoning_interval = self.reasoning_config.get("interval", DEFAULT_REASONING_INTERVAL)
        self.planning_config = self.reasoning_config.get("planning", {})
        self.decision_config = self.reasoning_config.get("decision", {})
        
        # Initialize perception components
        self.perception_system = None
        self.ui_detector = None
        self.ocr_engine = None
        self.system_monitor = None
        
        # Initialize reasoning components
        self.llm_client = None
        self.planning_engine = None
        self.decision_engine = None
        
        # State variables
        self.current_screen = None
        self.ui_elements = []
        self.detected_text = []
        self.recent_events = []
        self.system_state = {}
        
        # Event buffer
        self.event_buffer = asyncio.Queue(maxsize=1000)
        
        # Task storage
        self.tasks = {}
        self.active_task_id = None
        
        # Thread pool for background operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Task for background perception
        self._perception_task = None
        self._reasoning_task = None
        self._running = False
        
        # Locks for thread safety
        self._perception_lock = asyncio.Lock()
        self._reasoning_lock = asyncio.Lock()
        self._llm_lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()
        
        # Tracking metrics
        self.metrics = {
            "perception_fps": 0,
            "llm_calls": 0,
            "llm_tokens_in": 0,
            "llm_tokens_out": 0,
            "llm_latency_ms": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "reasoning_cycles": 0
        }
        
        self.logger.info("Cognitive engine initialized")
    
    async def start(self) -> None:
        """Start the cognitive engine and initialize all subsystems."""
        self.logger.info("Starting cognitive engine...")
        self._running = True
        
        try:
            # Initialize LLM client
            await self._init_llm_client()
            
            # Initialize perception components
            await self._init_perception_system()
            
            # Initialize reasoning components
            await self._init_reasoning_components()
            
            # Start background tasks
            self._perception_task = asyncio.create_task(self._perception_loop())
            self._reasoning_task = asyncio.create_task(self._reasoning_loop())
            
            self.logger.info("Cognitive engine started successfully")
        except Exception as e:
            self._running = False
            self.logger.error(f"Failed to start cognitive engine: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the cognitive engine and perform cleanup."""
        self.logger.info("Stopping cognitive engine...")
        self._running = False
        
        # Cancel background tasks
        if self._perception_task:
            self._perception_task.cancel()
            try:
                await self._perception_task
            except asyncio.CancelledError:
                pass
            
        if self._reasoning_task:
            self._reasoning_task.cancel()
            try:
                await self._reasoning_task
            except asyncio.CancelledError:
                pass
        
        # Release resources
        if hasattr(self, 'ui_detector') and self.ui_detector:
            # Explicitly delete the model if it exists to free resources
            if "model" in self.ui_detector and self.ui_detector["model"] is not None:
                # Assuming self.ui_detector["model"] is the ONNX session
                # If ONNXRuntime sessions have a close/release method, it should be called here.
                # For now, just deleting the reference as per current pattern.
                del self.ui_detector["model"]
                self.logger.info("UI detection model reference deleted from ui_detector")
            
            # Now delete the ui_detector dictionary itself
            del self.ui_detector
            self.logger.info("UI detector dictionary deleted")
            
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("Cognitive engine stopped")
    
    #--------------------------------------------------------------------
    # Initialization Methods
    #--------------------------------------------------------------------
    
    async def _init_llm_client(self) -> None:
        """Initialize the LLM client for inference."""
        if not LLM_AVAILABLE:
            self.logger.warning("HTTPX not available, LLM functionality will be limited")
            return
            
        try:
            # Setup httpx client for Ollama API
            self.llm_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10)
            )
            
            # Test connection to Ollama
            ollama_host = self.llm_config.get("host", "http://localhost:11434")
            response = await self.llm_client.get(f"{ollama_host}/api/version")
            
            if response.status_code == 200:
                version_info = response.json()
                self.logger.info(f"Connected to Ollama version {version_info.get('version', 'unknown')}")
                
                # Check if our models are available
                response = await self.llm_client.get(f"{ollama_host}/api/tags")
                models = response.json().get("models", [])
                
                model_names = [m.get("name", "") for m in models]
                
                if self.model not in model_names:
                    self.logger.warning(f"Primary model '{self.model}' not found in Ollama, it may need to be downloaded")
                
                if self.fallback_model not in model_names and self.fallback_model != self.model:
                    self.logger.warning(f"Fallback model '{self.fallback_model}' not found in Ollama")
            else:
                self.logger.warning(f"Failed to connect to Ollama: {response.status_code} {response.text}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}", exc_info=True)
            self.logger.warning("LLM functionality will be limited")
    
    async def _init_perception_system(self) -> None:
        """Initialize perception system components."""
        if not CV_AVAILABLE:
            self.logger.warning("OpenCV not available, perception capabilities will be limited")
            return
            
        try:
            # Initialize screen capture
            self.logger.info("Initializing screen capture...")
            self.perception_system = {
                "initialized": True,
                "last_screen_hash": None,
                "screen_change_threshold": int(self.screen_capture_config.get("change_threshold", 5)),
                "capture_regions": self.screen_capture_config.get("regions", ["full"]),
                "resolution": self.screen_capture_config.get("resolution", "native"),
                "rate": self.screen_capture_config.get("rate", 10)  # frames per second
            }
            
            # Initialize OCR if available
            if not hasattr(pytesseract, 'image_to_string'):
                self.logger.warning("Tesseract OCR not properly configured, text recognition will be limited")
            else:
                self.ocr_engine = {
                    "initialized": True,
                    "languages": self.ocr_config.get("languages", ["eng"]),
                    "confidence": self.ocr_config.get("confidence", 0.7),
                    "custom_config": self.ocr_config.get("config", "")
                }
                self.logger.info("OCR engine initialized")
                
            # Initialize UI element detector if ONNX is available
            if ML_AVAILABLE:
                ui_model_path = self.ui_detection_config.get("model")
                
                if ui_model_path and os.path.exists(ui_model_path):
                    # Initialize ONNX session with the model
                    self.logger.info(f"Loading UI detection model from {ui_model_path}")
                    
                    # Use ONNX Runtime for efficient inference
                    self.ui_detector = {
                        "initialized": True,
                        "model": ort.InferenceSession(
                            ui_model_path, 
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        ),
                        "confidence": self.ui_detection_config.get("confidence", 0.75),
                        "classes": self.ui_detection_config.get("classes", 
                                                             ["button", "input", "dropdown", 
                                                              "checkbox", "radio", "text", 
                                                              "image", "icon"])
                    }
                    self.logger.info("UI detector initialized with model")
                else:
                    self.logger.warning("UI detection model not found, using simplified detection")
                    self.ui_detector = {
                        "initialized": True,
                        "model": None,
                        "confidence": self.ui_detection_config.get("confidence", 0.75),
                        "classes": self.ui_detection_config.get("classes", 
                                                             ["button", "input", "dropdown", 
                                                              "checkbox", "radio", "text", 
                                                              "image", "icon"])
                    }
            else:
                self.logger.warning("ONNX Runtime not available, UI detection will be limited")
            
            # Initialize system monitor
            self.system_monitor = {
                "initialized": True,
                "process_check_interval": self.system_monitor_config.get("process_check_interval", 1.0),
                "watched_directories": self.system_monitor_config.get("watch_directories", []),
                "last_check_time": time.time(),
                "processes": {},
                "file_states": {}
            }
            
            self.logger.info("Perception system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize perception system: {e}", exc_info=True)
            raise
    
    async def _init_reasoning_components(self) -> None:
        """Initialize reasoning components."""
        try:
            # Initialize planning engine
            self.planning_engine = {
                "initialized": True,
                "max_steps": self.planning_config.get("max_steps", MAX_PLAN_STEPS),
                "validation": self.planning_config.get("validation", True),
                "timeout": self.planning_config.get("timeout", 60.0)
            }
            
            # Initialize decision engine
            self.decision_engine = {
                "initialized": True,
                "risk_threshold": self.decision_config.get("risk_threshold", 0.3),
                "confidence_threshold": self.decision_config.get("confidence_threshold", CONFIDENCE_THRESHOLD),
                "verification_required": self.decision_config.get("verification_required", True)
            }
            
            self.logger.info("Reasoning components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize reasoning components: {e}", exc_info=True)
            raise
    
    #--------------------------------------------------------------------
    # Perception Methods
    #--------------------------------------------------------------------
    
    async def _perception_loop(self) -> None:
        """Main perception loop to monitor environment continuously."""
        self.logger.info("Starting perception loop")
        last_capture_time = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # Control perception rate
                if current_time - last_capture_time < self.perception_interval:
                    await asyncio.sleep(0.01)  # Short sleep to prevent CPU spinning
                    continue
                
                last_capture_time = current_time
                
                # Run perception cycle
                async with self._perception_lock:
                    await self._perception_cycle()
                
                # Calculate FPS
                cycle_time = time.time() - current_time
                if cycle_time > 0:
                    self.metrics["perception_fps"] = 1.0 / cycle_time
                
                # Brief sleep to yield CPU
                await asyncio.sleep(0)
                
            except asyncio.CancelledError:
                self.logger.info("Perception loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in perception loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Longer sleep on error
    
    async def _perception_cycle(self) -> None:
        """Run a single perception cycle to update environment state."""
        try:
            # Capture screen
            new_screen = await self._capture_screen()
            if new_screen is not None:
                screen_changed = await self._process_screen(new_screen)
                
                # If screen changed significantly, analyze content
                if screen_changed:
                    # Detect UI elements
                    ui_elements = await self._detect_ui_elements(new_screen)
                    if ui_elements:
                        self.ui_elements = ui_elements
                        await self._emit_event(
                            EventType.UI_ELEMENT_DETECTED,
                            {"elements": [asdict(elem) for elem in ui_elements]}
                        )
                    
                    # Perform OCR
                    text_regions = await self._extract_text(new_screen)
                    if text_regions:
                        self.detected_text = text_regions
                        await self._emit_event(
                            EventType.TEXT_DETECTED,
                            {"text": text_regions}
                        )
            
            # Monitor system state
            await self._check_system_state()
            
        except Exception as e:
            self.logger.error(f"Error in perception cycle: {e}", exc_info=True)
    
    async def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture current screen state."""
        self.logger.warning("Using SIMULATED screen capture. Implement actual screen capture for real functionality.")
        if not CV_AVAILABLE:
            return None
            
        try:
            # Use mss or other screen capture method based on platform
            # This is a simplified implementation
            
            # For demo purposes, let's create a simulated screen
            # In a real implementation, this would use platform-specific
            # screen capture APIs
            
            # Create a blank image representing the screen
            screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Draw some UI elements for testing
            # In a real implementation, this would capture the actual screen
            cv2.rectangle(screen, (100, 100), (300, 150), (0, 120, 255), -1)  # Button
            cv2.rectangle(screen, (100, 200), (500, 240), (255, 255, 255), -1)  # Input field
            cv2.putText(screen, "Submit", (150, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(screen, "Enter your name", (120, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            
            return screen
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}", exc_info=True)
            return None
    
    async def _process_screen(self, screen: np.ndarray) -> bool:
        """Process captured screen and detect significant changes."""
        try:
            # Calculate screen hash or feature signature to detect changes
            # Using Difference Hash (dHash)
            hash_size = 8 
            # Resize to (hash_size + 1) x hash_size to compare adjacent pixels horizontally
            resized = cv2.resize(screen, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Compute dHash
            dhash_str = []
            for y in range(hash_size):
                for x in range(hash_size): # Iterate hash_size times, comparing x with x+1
                    if gray[y, x] > gray[y, x + 1]:
                        dhash_str.append("1")
                    else:
                        dhash_str.append("0")
            current_dhash = "".join(dhash_str)

            # Check if screen changed significantly using Hamming distance
            significant_change = False
            if self.current_screen is None or self.perception_system.get("last_screen_hash") is None:
                significant_change = True
            else:
                last_dhash = self.perception_system["last_screen_hash"]
                # Calculate Hamming distance
                if len(last_dhash) == len(current_dhash): # Ensure hashes are comparable
                    hamming_distance = sum(c1 != c2 for c1, c2 in zip(last_dhash, current_dhash))
                    # The screen_change_threshold now refers to Hamming distance.
                    # A common threshold for 64-bit dHash might be e.g. 5-10.
                    # The existing default of 0.05 is too low for Hamming distance.
                    # Let's assume a default Hamming distance threshold if not properly set.
                    # For this example, we'll use a hardcoded threshold, but ideally this should be configurable.
                    # If self.perception_system["screen_change_threshold"] was 0.05, it's not suitable for Hamming.
                    # Let's use a more appropriate example threshold like 5 for a 64-bit hash.
                    # This threshold should be adjusted based on empirical testing.
                    # For now, we'll use the configured threshold, but it needs to be set appropriately for dHash.
                    if hamming_distance > self.perception_system["screen_change_threshold"]:
                        significant_change = True
                else: # Hashes are not comparable, assume change
                    significant_change = True

            if significant_change:
                self.current_screen = screen
                self.perception_system["last_screen_hash"] = current_dhash
                
                # Emit screen change event
                await self._emit_event(
                    EventType.SCREEN_CHANGE,
                    {"resolution": screen.shape[:2]}
                )
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Screen processing failed: {e}", exc_info=True)
            return False
    
    async def _detect_ui_elements(self, screen: np.ndarray) -> List[UIElement]:
        """Detect UI elements in the screen."""
        if not self.ui_detector or not self.ui_detector.get("initialized", False):
            return []
            
        try:
            elements = []
            
            if self.ui_detector["model"] is not None:
                # Use ONNX model for detection
                # Prepare input
                input_img = cv2.resize(screen, (640, 640))
                input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
                input_img = np.expand_dims(input_img, 0).astype(np.float32) / 255.0
                
                # Run inference
                ort_inputs = {self.ui_detector["model"].get_inputs()[0].name: input_img}
                ort_outputs = self.ui_detector["model"].run(None, ort_inputs)
                
                # Process detections
                detections = ort_outputs[0][0]  # Assuming YOLO-like output format
                
                for detection in detections:
                    if detection[4] < self.ui_detector["confidence"]:
                        continue
                        
                    # Get class with highest confidence
                    class_id = np.argmax(detection[5:])
                    class_name = self.ui_detector["classes"][class_id]
                    confidence = detection[5 + class_id]
                    
                    if confidence < self.ui_detector["confidence"]:
                        continue
                    
                    # Convert normalized coordinates to pixel coordinates
                    h, w = screen.shape[:2]
                    x, y, box_w, box_h = detection[:4]
                    x1 = int((x - box_w / 2) * w)
                    y1 = int((y - box_h / 2) * h)
                    x2 = int((x + box_w / 2) * w)
                    y2 = int((y + box_h / 2) * h)
                    
                    # Create UI element
                    element = UIElement(
                        element_type=class_name,
                        bounding_box=(x1, y1, x2 - x1, y2 - y1),
                        confidence=float(confidence),
                        attributes={}
                    )
                    elements.append(element)
            else:
                # Simplified detection using basic OpenCV methods
                # Convert to grayscale
                gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                
                # Find potential UI elements using contour detection
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Filter out small contours
                    if cv2.contourArea(contour) < 1000:
                        continue
                        
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Simple heuristic for element type based on shape
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if aspect_ratio > 3:
                        element_type = "input"
                    elif 0.8 < aspect_ratio < 1.2:
                        element_type = "button"
                    else:
                        element_type = "text"
                    
                    # Create UI element
                    element = UIElement(
                        element_type=element_type,
                        bounding_box=(x, y, w, h),
                        confidence=0.8,
                        attributes={}
                    )
                    elements.append(element)
            
            # Extract text for elements if OCR is available
            if self.ocr_engine and self.ocr_engine.get("initialized", False):
                for element in elements:
                    x, y, w, h = element.bounding_box
                    if w > 20 and h > 10:  # Minimum size for OCR
                        roi = screen[y:y+h, x:x+w]
                        text = await self._ocr_region(roi)
                        if text:
                            element.text = text
            
            return elements
        except Exception as e:
            self.logger.error(f"UI element detection failed: {e}", exc_info=True)
            return []
    
    async def _extract_text(self, screen: np.ndarray) -> List[Dict]:
        """Extract text from screen using OCR."""
        if not self.ocr_engine or not self.ocr_engine.get("initialized", False):
            return []
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing to improve OCR
            # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # Original line
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find potential text regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                # Filter out small contours
                if cv2.contourArea(contour) < 500:
                    continue
                    
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract region from original image
                roi = screen[y:y+h, x:x+w]
                
                # Perform OCR on region
                text = await self._ocr_region(roi)
                if text:
                    text_regions.append({
                        "text": text,
                        "position": (x, y, w, h)
                    })
            
            return text_regions
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}", exc_info=True)
            return []
    
    async def _ocr_region(self, image: np.ndarray) -> str:
        """Perform OCR on specific image region."""
        if not hasattr(pytesseract, 'image_to_string'):
            return ""
            
        try:
            # Convert to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Run OCR in a thread to avoid blocking
            config = f"-l {'+'.join(self.ocr_engine['languages'])} --oem 1 --psm 6"
            if self.ocr_engine["custom_config"]:
                config += f" {self.ocr_engine['custom_config']}"
                
            text = await asyncio.to_thread(
                pytesseract.image_to_string,
                pil_img,
                config=config
            )
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"OCR failed: {e}", exc_info=True)
            return ""
    
    async def _check_system_state(self) -> None:
        """Monitor system state and detect changes."""
        if not self.system_monitor or not self.system_monitor.get("initialized", False):
            return
            
        try:
            current_time = time.time()
            
            # Check if it's time to monitor processes
            if current_time - self.system_monitor["last_check_time"] >= self.system_monitor["process_check_interval"]:
                self.system_monitor["last_check_time"] = current_time
                
                # This is a simplified implementation
                # In a real implementation, we would use psutil or similar
                # to monitor running processes and system resources
                
                # For demo purposes, let's simulate some system state changes
                # In a real implementation, this would track actual processes
                
                # Simulate process start/stop events occasionally
                if random.random() < 0.05:  # 5% chance of process change
                    if random.random() < 0.5:  # Start new process
                        process_id = f"process_{uuid.uuid4().hex[:8]}"
                        process_name = random.choice(["notepad.exe", "explorer.exe", "chrome.exe"])
                        
                        self.system_monitor["processes"][process_id] = {
                            "name": process_name,
                            "started_at": current_time
                        }
                        
                        await self._emit_event(
                            EventType.PROCESS_STARTED,
                            {"process_id": process_id, "name": process_name}
                        )
                    elif self.system_monitor["processes"]:  # Stop existing process
                        process_id = random.choice(list(self.system_monitor["processes"].keys()))
                        process_info = self.system_monitor["processes"].pop(process_id)
                        
                        await self._emit_event(
                            EventType.PROCESS_ENDED,
                            {"process_id": process_id, "name": process_info["name"]}
                        )
                
                # Check watched directories for changes (now non-blocking in the loop)
                for directory in self.system_monitor["watched_directories"]:
                    previous_file_states_for_dir = self.system_monitor["file_states"].get(directory, {})
                    
                    # Offload blocking file operations to a thread
                    new_file_states_for_dir, file_change_events = await asyncio.to_thread(
                        self._perform_file_check_sync, directory, previous_file_states_for_dir
                    )
                    
                    # Process detected events
                    for event_data in file_change_events:
                        await self._emit_event(EventType.FILE_CHANGED, event_data)
                    
                    # Update state if new states were successfully fetched
                    if new_file_states_for_dir is not None:
                         self.system_monitor["file_states"][directory] = new_file_states_for_dir

        except Exception as e:
            self.logger.error(f"System state monitoring failed: {e}", exc_info=True)

    def _perform_file_check_sync(self, directory_path: str, previous_states: Dict[str, float]) -> Tuple[Optional[Dict[str, float]], List[Dict]]:
        """
        Synchronous helper to check a single directory for file changes.
        Returns new file states and a list of change events.
        """
        change_events = []
        current_states = {}

        try:
            if not os.path.exists(directory_path):
                # If directory doesn't exist (e.g., was deleted), mark all previous files as deleted
                for filename in previous_states:
                    file_path_abs = os.path.join(directory_path, filename) # Construct full path for event
                    change_events.append({"path": file_path_abs, "type": "deleted"})
                return None, change_events # Return None for states as dir is gone

            # Expand user-specific paths like ~
            expanded_directory_path = Path(directory_path).expanduser()
            if not expanded_directory_path.exists():
                 # Log or handle if path doesn't exist after expansion (similar to above)
                if previous_states: # If there were previous states, they are now effectively deleted
                    for filename in previous_states:
                        change_events.append({"path": os.path.join(str(expanded_directory_path), filename), "type": "deleted"})
                return None, change_events


            current_states = {f.name: f.stat().st_mtime for f in expanded_directory_path.glob("*") if f.is_file()}

            # Detect new or modified files
            for filename, mtime in current_states.items():
                if filename not in previous_states:
                    change_events.append({"path": os.path.join(str(expanded_directory_path), filename), "type": "created"})
                elif mtime > previous_states[filename]:
                    change_events.append({"path": os.path.join(str(expanded_directory_path), filename), "type": "modified"})

            # Detect deletions
            for filename in previous_states:
                if filename not in current_states:
                    change_events.append({"path": os.path.join(str(expanded_directory_path), filename), "type": "deleted"})
            
            return current_states, change_events

        except Exception as e:
            # Log error from within the sync function for clarity, but _check_system_state will also log.
            # self.logger.error(f"Error in _perform_file_check_sync for {directory_path}: {e}", exc_info=True) 
            # Cannot use self.logger directly in a static/non-instance method if we make it static.
            # For now, it's an instance method, so self.logger is fine.
            # Or, pass logger as an argument, or let the caller handle all logging.
            # For simplicity here, we'll rely on the caller's logger.
            # Return previous_states to avoid losing state on temporary error, and empty events list.
            # Alternatively, return (None, []) to signal error and let caller decide. Let's signal error.
            return None, [] # Signal error by returning None for states

    async def _emit_event(self, event_type: EventType, data: Dict) -> None:
        """Emit a perception event to the event buffer."""
        try:
            event = PerceptionEvent(event_type=event_type, data=data)
            
            # Add to recent events list
            self.recent_events.append(event)
            
            # Keep only most recent events
            max_events = 100
            if len(self.recent_events) > max_events:
                self.recent_events = self.recent_events[-max_events:]
            
            # Add to event buffer for processing
            try:
                self.event_buffer.put_nowait(event)
            except asyncio.QueueFull:
                # If buffer is full, remove oldest event
                _ = await self.event_buffer.get()
                self.event_buffer.put_nowait(event)
                
            # Record to episodic memory if significant
            if event_type in [EventType.PROCESS_STARTED, EventType.PROCESS_ENDED, 
                              EventType.FILE_CHANGED, EventType.ERROR_DETECTED]:
                await self._record_event_to_memory(event)
        except Exception as e:
            self.logger.error(f"Failed to emit event: {e}", exc_info=True)
    
    async def _record_event_to_memory(self, event: PerceptionEvent) -> None:
        """Record significant events to episodic memory."""
        try:
            # Create memory entry
            event_data = event.to_dict()
            memory_text = f"Event {event.event_type.name} at {datetime.fromtimestamp(event.timestamp).isoformat()}: {json.dumps(event.data)}"
            
            # Calculate importance based on event type
            importance = 0.5  # Default importance
            if event.event_type == EventType.ERROR_DETECTED:
                importance = 0.9
            elif event.event_type in [EventType.PROCESS_STARTED, EventType.PROCESS_ENDED]:
                importance = 0.6
            elif event.event_type == EventType.FILE_CHANGED:
                importance = 0.7
            
            # Store in memory
            await self.memory.create_memory(
                content=memory_text,
                memory_type=MEMORY_TYPE_EPISODIC,
                metadata={
                    "event_type": event.event_type.name,
                    "timestamp": event.timestamp,
                    "source": event.source
                },
                importance=importance
            )
        except Exception as e:
            self.logger.error(f"Failed to record event to memory: {e}", exc_info=True)
    
    #--------------------------------------------------------------------
    # Reasoning Methods
    #--------------------------------------------------------------------
    
    async def _reasoning_loop(self) -> None:
        """Main reasoning loop to process tasks and events."""
        self.logger.info("Starting reasoning loop")
        last_reasoning_time = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # Control reasoning rate
                if current_time - last_reasoning_time < self.reasoning_interval:
                    # Process events even when not reasoning
                    while not self.event_buffer.empty():
                        event = await self.event_buffer.get()
                        await self._process_event(event)
                        
                    await asyncio.sleep(0.01)  # Short sleep to prevent CPU spinning
                    continue
                
                last_reasoning_time = current_time
                
                # Process events
                events_processed = 0
                while not self.event_buffer.empty() and events_processed < 10:
                    event = await self.event_buffer.get()
                    await self._process_event(event)
                    events_processed += 1
                
                # Process active task
                async with self._reasoning_lock:
                    await self._reasoning_cycle()
                
                # Update metrics
                self.metrics["reasoning_cycles"] += 1
                
                # Brief sleep to yield CPU
                await asyncio.sleep(0)
                
            except asyncio.CancelledError:
                self.logger.info("Reasoning loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in reasoning loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Longer sleep on error
    
    async def _reasoning_cycle(self) -> None:
        """Run a single reasoning cycle to advance task processing."""
        try:
            # Get current active task
            active_task = await self._get_active_task()
            
            if active_task is None:
                # No active task, select a new one
                await self._select_next_task()
                active_task = await self._get_active_task()
                
                if active_task is None:
                    # Still no active task, nothing to do
                    return
            
            # Process the active task
            if active_task.status == TaskStatus.PENDING:
                # Start task execution
                active_task.status = TaskStatus.IN_PROGRESS
                active_task.started_at = time.time()
                
                # Create a plan for the task
                plan = await self._create_task_plan(active_task)
                active_task.plan = plan
                
                await self._update_task(active_task)
                
                # Log task start
                self.logger.info(f"Started task: {active_task.id} - {active_task.description}")
                
            elif active_task.status == TaskStatus.IN_PROGRESS:
                # Continue executing the task
                if active_task.plan and not active_task.plan.is_complete:
                    # Execute next step in the plan
                    success = await self._execute_plan_step(active_task)
                    
                    if success:
                        # Advance to next step
                        active_task.plan.advance()
                        
                        # Check if plan is complete
                        if active_task.plan.is_complete:
                            # Mark task as succeeded
                            active_task.status = TaskStatus.SUCCEEDED
                            active_task.completed_at = time.time()
                            active_task.result = {"status": "completed", "message": "Task completed successfully"}
                            
                            # Log task completion
                            self.logger.info(f"Completed task: {active_task.id} - {active_task.description}")
                            
                            # Record to memory
                            await self._record_task_completion(active_task)
                            
                            # Update metrics
                            self.metrics["tasks_completed"] += 1
                    else:
                        # Step failed, check for recovery
                        recovery_successful = await self._attempt_recovery(active_task)
                        
                        if not recovery_successful:
                            # Mark task as failed
                            active_task.status = TaskStatus.FAILED
                            active_task.completed_at = time.time()
                            active_task.error = "Failed to execute plan step and recovery failed"
                            
                            # Log task failure
                            self.logger.warning(f"Failed task: {active_task.id} - {active_task.description}")
                            
                            # Record to memory
                            await self._record_task_failure(active_task)
                            
                            # Update metrics
                            self.metrics["tasks_failed"] += 1
                else:
                    # No plan or empty plan, mark as failed
                    active_task.status = TaskStatus.FAILED
                    active_task.completed_at = time.time()
                    active_task.error = "Task has no valid plan"
                    
                    # Log error
                    self.logger.warning(f"Task has no valid plan: {active_task.id}")
                    
                    # Update metrics
                    self.metrics["tasks_failed"] += 1
                
                # Update task in storage
                await self._update_task(active_task)
                
            elif active_task.status in [TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                # Task is complete, clear active task
                self.active_task_id = None
        except Exception as e:
            self.logger.error(f"Error in reasoning cycle: {e}", exc_info=True)
            
            # If there's an active task, mark it as failed
            if self.active_task_id and self.active_task_id in self.tasks:
                active_task = self.tasks[self.active_task_id]
                active_task.status = TaskStatus.FAILED
                active_task.completed_at = time.time()
                active_task.error = f"Internal error: {str(e)}"
                
                await self._update_task(active_task)
                
                # Update metrics
                self.metrics["tasks_failed"] += 1
    
    async def _process_event(self, event: PerceptionEvent) -> None:
        """Process a perception event and update system state."""
        try:
            # Update system state based on event
            if event.event_type == EventType.PROCESS_STARTED:
                process_id = event.data.get("process_id")
                process_name = event.data.get("name")
                
                if "processes" not in self.system_state:
                    self.system_state["processes"] = {}
                    
                self.system_state["processes"][process_id] = {
                    "name": process_name,
                    "started_at": event.timestamp
                }
            elif event.event_type == EventType.PROCESS_ENDED:
                process_id = event.data.get("process_id")
                
                if "processes" in self.system_state and process_id in self.system_state["processes"]:
                    del self.system_state["processes"][process_id]
            elif event.event_type == EventType.FILE_CHANGED:
                file_path = event.data.get("path")
                change_type = event.data.get("type")
                
                if "file_changes" not in self.system_state:
                    self.system_state["file_changes"] = []
                    
                self.system_state["file_changes"].append({
                    "path": file_path,
                    "type": change_type,
                    "timestamp": event.timestamp
                })
                
                # Keep only recent file changes
                max_changes = 100
                if len(self.system_state["file_changes"]) > max_changes:
                    self.system_state["file_changes"] = self.system_state["file_changes"][-max_changes:]
            elif event.event_type == EventType.SCREEN_CHANGE:
                if "screen" not in self.system_state:
                    self.system_state["screen"] = {}
                    
                self.system_state["screen"].update({
                    "last_change": event.timestamp,
                    "resolution": event.data.get("resolution")
                })
            elif event.event_type == EventType.UI_ELEMENT_DETECTED:
                if "ui" not in self.system_state:
                    self.system_state["ui"] = {}
                    
                self.system_state["ui"].update({
                    "last_detection": event.timestamp,
                    "elements": event.data.get("elements", [])
                })
            elif event.event_type == EventType.TEXT_DETECTED:
                if "text" not in self.system_state:
                    self.system_state["text"] = {}
                    
                self.system_state["text"].update({
                    "last_detection": event.timestamp,
                    "regions": event.data.get("text", [])
                })
            elif event.event_type == EventType.ERROR_DETECTED:
                if "errors" not in self.system_state:
                    self.system_state["errors"] = []
                    
                self.system_state["errors"].append({
                    "message": event.data.get("message", "Unknown error"),
                    "source": event.data.get("source", "system"),
                    "timestamp": event.timestamp
                })
                
                # Keep only recent errors
                max_errors = 20
                if len(self.system_state["errors"]) > max_errors:
                    self.system_state["errors"] = self.system_state["errors"][-max_errors:]
        except Exception as e:
            self.logger.error(f"Failed to process event: {e}", exc_info=True)
    
    async def _get_active_task(self) -> Optional[Task]:
        """Get the currently active task."""
        if self.active_task_id and self.active_task_id in self.tasks:
            return self.tasks[self.active_task_id]
        return None
    
    async def _select_next_task(self) -> None:
        """Select the next task to process based on priority."""
        try:
            async with self._task_lock:
                # Find pending tasks
                pending_tasks = [
                    task for task in self.tasks.values()
                    if task.status == TaskStatus.PENDING
                ]
                
                if not pending_tasks:
                    return
                
                # Sort by priority and creation time
                pending_tasks.sort(key=lambda t: (
                    4 - t.priority.value,  # Highest priority first (reversed)
                    t.created_at  # Oldest first
                ))
                
                # Select highest priority task
                if pending_tasks:
                    self.active_task_id = pending_tasks[0].id
        except Exception as e:
            self.logger.error(f"Failed to select next task: {e}", exc_info=True)
    
    async def _create_task_plan(self, task: Task) -> TaskPlan:
        """Create a plan for executing a task."""
        try:
            # Retrieve relevant memories
            memories = await self._retrieve_task_relevant_memories(task)
            
            # Create reasoning context
            context = ReasoningContext(
                task=task,
                relevant_memories=memories,
                perception_events=self.recent_events[-10:],  # Last 10 events
                system_state=self.system_state,
                variables={},
                constraints=[]
            )
            
            # Generate plan using LLM
            plan_result = await self._generate_plan(context)
            
            if not plan_result or "steps" not in plan_result:
                # Fallback to simple plan
                steps = [{"description": task.description, "action": "execute", "params": {}}]
                verification = []
            else:
                steps = plan_result.get("steps", [])
                verification = plan_result.get("verification", [])
            
            # Create plan object
            plan = TaskPlan(
                steps=steps,
                goal=task.description,
                context=context.to_dict(),
                verification_steps=verification
            )
            
            return plan
        except Exception as e:
            self.logger.error(f"Failed to create task plan: {e}", exc_info=True)
            
            # Return minimal plan as fallback
            return TaskPlan(
                steps=[{"description": "Execute task", "action": "execute", "params": {}}],
                goal=task.description
            )
    
    async def _retrieve_task_relevant_memories(self, task: Task) -> List[MemoryResult]:
        """Retrieve memories relevant to the current task."""
        try:
            # Query memory system
            query_result = await self.memory.remember(
                query=task.description,
                memory_types=[MEMORY_TYPE_SEMANTIC, MEMORY_TYPE_EPISODIC, MEMORY_TYPE_PROCEDURAL],
                limit=10
            )
            
            return query_result.results
        except Exception as e:
            self.logger.error(f"Failed to retrieve task memories: {e}", exc_info=True)
            return []
    
    async def _generate_plan(self, context: ReasoningContext) -> Optional[Dict]:
        """Generate a task execution plan using LLM."""
        if not LLM_AVAILABLE or not self.llm_client:
            self.logger.warning("LLM functionality not available for plan generation")
            return None
            
        try:
            # Create prompt for plan generation
            prompt = self._create_planning_prompt(context)
            
            # Generate plan using LLM
            response = await self._llm_inference(prompt, temperature=0.3)
            
            if not response:
                return None
                
            # Parse response to extract plan
            plan_data = self._parse_plan_from_response(response)
            
            return plan_data
        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}", exc_info=True)
            return None
    
    def _create_planning_prompt(self, context: ReasoningContext) -> str:
        """Create a prompt for plan generation."""
        # Convert context to serializable form
        ctx_dict = context.to_dict()
        
        # Add system information
        system_info = {
            "operating_system": "Windows 11",
            "available_tools": [
                "command_execution",
                "keyboard_input",
                "mouse_click",
                "browser_control",
                "file_operations",
                "text_extraction",
                "ui_interaction"
            ],
            "constraints": [
                "Cannot access physical hardware",
                "Cannot install new software without confirmation",
                "Cannot delete system files"
            ]
        }
        
        # Create prompt
        prompt = f"""You are an autonomous agent operating system with perception and action capabilities.
You need to create a detailed plan to accomplish the task described below.

TASK:
{context.task.description}

SYSTEM STATE:
{json.dumps(system_info, indent=2)}

RELEVANT MEMORIES:
{json.dumps([str(m) for m in context.relevant_memories], indent=2)}

RECENT PERCEPTION EVENTS:
{json.dumps([e.to_dict() for e in context.perception_events], indent=2)}

CURRENT SYSTEM STATE:
{json.dumps(context.system_state, indent=2)}

Create a detailed step-by-step plan to accomplish this task. Each step should include:
1. A description of what to do
2. The specific action to take (command_execution, keyboard_input, mouse_click, etc.)
3. Parameters for the action
4. Any conditions or checks

Also include verification steps to validate the success of critical operations.

Respond in JSON format with the following structure:
```json
{{
  "steps": [
    {{
      "description": "Step description",
      "action": "action_name",
      "params": {{
        "param1": "value1",
        "param2": "value2"
      }},
      "conditions": ["condition1", "condition2"]
    }}
  ],
  "verification": [
    {{
      "after_step": 1,
      "check": "What to verify",
      "expected": "Expected outcome",
      "fallback": "What to do if verification fails"
    }}
  ]
}}
```

Ensure your plan is detailed, feasible, and accounts for potential failures."""
        
        return prompt
    
    def _parse_plan_from_response(self, response: str) -> Optional[Dict]:
        """Parse plan data from LLM response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown code blocks
                json_match = re.search(r'(\{[\s\S]*\})', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    self.logger.warning("Could not extract JSON from response")
                    return None
            
            # Parse JSON
            plan_data = json.loads(json_str)
            
            # Validate plan structure
            if "steps" not in plan_data or not isinstance(plan_data["steps"], list):
                self.logger.warning("Invalid plan structure: missing steps array")
                return None
                
            # Ensure each step has required fields
            for i, step in enumerate(plan_data["steps"]):
                if "description" not in step:
                    step["description"] = f"Step {i+1}"
                if "action" not in step:
                    step["action"] = "execute"
                if "params" not in step or not isinstance(step["params"], dict):
                    step["params"] = {}
            
            return plan_data
        except Exception as e:
            self.logger.error(f"Failed to parse plan: {e}", exc_info=True)
            return None
    
    async def _execute_plan_step(self, task: Task) -> bool:
        """Execute a single step in the task plan."""
        if not task.plan or task.plan.is_complete:
            return False
            
        try:
            # Get current step
            step = task.plan.current
            
            if not step:
                return False
                
            self.logger.info(f"Executing step: {step.get('description', 'Unknown step')}")
            
            # Determine action
            action = step.get("action", "")
            params = step.get("params", {})
            
            # This is where we would dispatch to the action system
            # For now, we'll just simulate success/failure
            
            # Simulated execution result
            success = await self._simulate_action_execution(action, params)
            
            if success:
                # Check verification if needed
                verifications = [v for v in task.plan.verification_steps if v.get("after_step") == task.plan.current_step]
                
                if verifications:
                    for verification in verifications:
                        success = await self._verify_step_result(verification, task)
                        if not success:
                            self.logger.warning(f"Verification failed: {verification.get('check', 'Unknown check')}")
                            return False
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to execute plan step: {e}", exc_info=True)
            return False
    
    async def _simulate_action_execution(self, action: str, params: Dict) -> bool:
        """Simulate action execution (this would actually call action system)."""
        # In a real implementation, this would dispatch to action_system.py
        # For now, let's simulate success with occasional failures
        
        self.logger.info(f"Simulating action: {action} with params: {params}")
        
        # Simulate occasional failures
        if random.random() < 0.1:  # 10% chance of failure
            self.logger.warning(f"Action {action} failed")
            return False
            
        # Simulate action execution delay
        await asyncio.sleep(0.5)
        
        return True
    
    async def _verify_step_result(self, verification: Dict, task: Task) -> bool:
        """Verify the result of a plan step."""
        try:
            check = verification.get("check", "")
            expected = verification.get("expected", "")
            
            self.logger.info(f"Verifying: {check}, expecting: {expected}")
            
            # In a real implementation, this would check actual system state
            # For now, simulate verification with occasional failures
            
            # Simulate occasional verification failures
            if random.random() < 0.05:  # 5% chance of verification failure
                self.logger.warning(f"Verification failed: {check}")
                return False
                
            # Simulate verification delay
            await asyncio.sleep(0.2)
            
            return True
        except Exception as e:
            self.logger.error(f"Verification failed with error: {e}", exc_info=True)
            return False
    
    async def _attempt_recovery(self, task: Task) -> bool:
        """Attempt to recover from a failed step."""
        try:
            self.logger.info(f"Attempting recovery for task: {task.id}")
            
            # Create a recovery context
            recovery_context = ReasoningContext(
                task=task,
                relevant_memories=[],  # No need for memories in recovery
                perception_events=self.recent_events[-5:],  # Last 5 events
                reasoning_history=[],
                system_state=self.system_state,
                variables={},
                constraints=[]
            )
            
            # Generate recovery plan using LLM
            recovery_result = await self._generate_recovery_plan(recovery_context, task)
            
            if not recovery_result:
                return False
                
            # Execute recovery action
            recovery_action = recovery_result.get("action", "")
            recovery_params = recovery_result.get("params", {})
            
            success = await self._simulate_action_execution(recovery_action, recovery_params)
            
            return success
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}", exc_info=True)
            return False
    
    async def _generate_recovery_plan(self, context: ReasoningContext, task: Task) -> Optional[Dict]:
        """Generate a recovery plan using LLM."""
        if not LLM_AVAILABLE or not self.llm_client:
            self.logger.warning("LLM functionality not available for recovery planning")
            return None
            
        try:
            # Get failed step
            current_step_idx = task.plan.current_step
            failed_step = task.plan.steps[current_step_idx] if current_step_idx < len(task.plan.steps) else None
            
            # Create prompt for recovery planning
            prompt = f"""You are an autonomous agent operating system with perception and action capabilities.
A step in your task execution plan has failed. You need to generate a recovery strategy.

TASK:
{task.description}

FAILED STEP:
{json.dumps(failed_step, indent=2)}

RECENT PERCEPTION EVENTS:
{json.dumps([e.to_dict() for e in context.perception_events], indent=2)}

CURRENT SYSTEM STATE:
{json.dumps(context.system_state, indent=2)}

Create a recovery strategy for this failure. Focus on either:
1. Trying an alternative approach to achieve the same goal
2. Reverting any partial changes to ensure system stability
3. Preparing to skip this step if it's not critical

Respond in JSON format with the following structure:
```json
{
  "strategy": "alternative|revert|skip",
  "action": "action_name",
  "params": {
    "param1": "value1",
    "param2": "value2"
  },
  "explanation": "Why this recovery approach was chosen"
}
```

Ensure your recovery plan is safe and has a high chance of success."""
            
            # Generate recovery plan using LLM
            response = await self._llm_inference(prompt, temperature=0.2)
            
            if not response:
                return None
                
            # Parse response to extract recovery plan
            recovery_data = self._parse_recovery_from_response(response)
            
            return recovery_data
        except Exception as e:
            self.logger.error(f"Recovery planning failed: {e}", exc_info=True)
            return None
    
    def _parse_recovery_from_response(self, response: str) -> Optional[Dict]:
        """Parse recovery plan from LLM response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown code blocks
                json_match = re.search(r'(\{[\s\S]*\})', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    self.logger.warning("Could not extract JSON from response")
                    return None
            
            # Parse JSON
            recovery_data = json.loads(json_str)
            
            # Validate structure
            if "action" not in recovery_data:
                recovery_data["action"] = "skip"  # Default to skip if no action specified
                
            if "params" not in recovery_data or not isinstance(recovery_data["params"], dict):
                recovery_data["params"] = {}
                
            return recovery_data
        except Exception as e:
            self.logger.error(f"Failed to parse recovery plan: {e}", exc_info=True)
            return None
    
    async def _record_task_completion(self, task: Task) -> None:
        """Record task completion to memory."""
        try:
            # Create memory entry
            task_data = task.to_dict()
            memory_text = f"Successfully completed task: {task.description}"
            
            # Store in memory
            await self.memory.create_memory(
                content=memory_text,
                memory_type=MEMORY_TYPE_EPISODIC,
                metadata={
                    "task_id": task.id,
                    "task_type": "completion",
                    "timestamp": time.time()
                },
                importance=0.7
            )
            
            # Also store as procedural memory if task had multiple steps
            if task.plan and len(task.plan.steps) > 1:
                proc_memory_text = f"Procedure for: {task.description}\n"
                proc_memory_text += "Steps:\n"
                
                for i, step in enumerate(task.plan.steps):
                    proc_memory_text += f"{i+1}. {step.get('description', 'Unknown step')}\n"
                
                await self.memory.create_memory(
                    content=proc_memory_text,
                    memory_type=MEMORY_TYPE_PROCEDURAL,
                    metadata={
                        "task_id": task.id,
                        "task_type": "procedure",
                        "timestamp": time.time()
                    },
                    importance=0.8
                )
        except Exception as e:
            self.logger.error(f"Failed to record task completion: {e}", exc_info=True)
    
    async def _record_task_failure(self, task: Task) -> None:
        """Record task failure to memory."""
        try:
            # Create memory entry
            task_data = task.to_dict()
            memory_text = f"Failed to complete task: {task.description}. Error: {task.error}"
            
            # Store in memory
            await self.memory.create_memory(
                content=memory_text,
                memory_type=MEMORY_TYPE_EPISODIC,
                metadata={
                    "task_id": task.id,
                    "task_type": "failure",
                    "timestamp": time.time()
                },
                importance=0.8  # Higher importance for failures
            )
            
            # Also store as reflective memory for learning
            refl_memory_text = f"Analysis of failed task: {task.description}\n"
            refl_memory_text += f"Error: {task.error}\n"
            
            if task.plan:
                failed_step_idx = task.plan.current_step
                if failed_step_idx < len(task.plan.steps):
                    failed_step = task.plan.steps[failed_step_idx]
                    refl_memory_text += f"Failed at step {failed_step_idx+1}: {failed_step.get('description', 'Unknown step')}\n"
            
            await self.memory.create_memory(
                content=refl_memory_text,
                memory_type=MEMORY_TYPE_REFLECTIVE,
                metadata={
                    "task_id": task.id,
                    "task_type": "reflection",
                    "timestamp": time.time()
                },
                importance=0.9  # Very high importance for learning
            )
        except Exception as e:
            self.logger.error(f"Failed to record task failure: {e}", exc_info=True)
    
    #--------------------------------------------------------------------
    # LLM Inference Methods
    #--------------------------------------------------------------------
    
    async def _llm_inference(self, prompt: str, temperature: float = None) -> Optional[str]:
        """Perform inference using the LLM with retries."""
        if not LLM_AVAILABLE or not self.llm_client:
            self.logger.warning("LLM functionality not available for inference")
            return None
            
        async with self._llm_lock:
            start_time = time.time()
            
            # Set temperature
            temp = temperature if temperature is not None else self.temperature
            
            # Initialize retry counter
            retry_count = 0
            
            while retry_count < MAX_RETRIES:
                try:
                    ollama_host = self.llm_config.get("host", "http://localhost:11434")
                    
                    # Use current model, fallback if specified
                    model_to_use = self.model
                    if retry_count > 0 and self.fallback_model:
                        model_to_use = self.fallback_model
                        self.logger.info(f"Using fallback model: {model_to_use}")
                    
                    # Make request
                    response = await self.llm_client.post(
                        f"{ollama_host}/api/generate",
                        json={
                            "model": model_to_use,
                            "prompt": prompt,
                            "stream": False,
                            "temperature": temp,
                            "top_p": self.top_p,
                            "num_predict": 1024,  # Limit response length
                            "stop": ["<|im_end|>", "<|endoftext|>"]
                        }
                    )
                    
                    # Check response
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get("response", "")
                        
                        # Update metrics
                        end_time = time.time()
                        self.metrics["llm_calls"] += 1
                        self.metrics["llm_latency_ms"] = (end_time - start_time) * 1000
                        
                        # Approximate token counts
                        self.metrics["llm_tokens_in"] += len(prompt.split()) * 0.75
                        self.metrics["llm_tokens_out"] += len(response_text.split()) * 0.75
                        
                        return response_text
                    else:
                        self.logger.warning(f"LLM inference failed with status {response.status_code}: {response.text}")
                except Exception as e:
                    self.logger.error(f"LLM inference error: {e}", exc_info=True)
                
                # Increase retry count and delay before retry
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(1 * retry_count)  # Exponential backoff
            
            self.logger.error(f"LLM inference failed after {MAX_RETRIES} retries")
            return None
    
    async def _streaming_llm_inference(self, prompt: str, callback: Callable[[str], None], temperature: float = None) -> bool:
        """Perform streaming inference using the LLM."""
        if not LLM_AVAILABLE or not self.llm_client:
            self.logger.warning("LLM functionality not available for streaming inference")
            return False
            
        async with self._llm_lock:
            start_time = time.time()
            
            # Set temperature
            temp = temperature if temperature is not None else self.temperature
            
            retry_count = 0
            while retry_count < MAX_RETRIES:
                model_to_use = self.model
                try:
                    if retry_count > 0:
                        # Exponential backoff with a maximum delay cap
                        delay = min(BASE_DELAY * (2 ** (retry_count - 1)), MAX_DELAY)
                        await asyncio.sleep(delay)  # Delay before retry
                        if self.fallback_model:
                            model_to_use = self.fallback_model
                            self.logger.info(f"Streaming LLM: Using fallback model: {model_to_use} after {retry_count} retries.")
                        else:
                            self.logger.info(f"Streaming LLM: Retrying with primary model {model_to_use} (no fallback defined), attempt {retry_count + 1}.")
                    else:
                         self.logger.debug(f"Streaming LLM: Attempt {retry_count + 1} with model {model_to_use}.")


                    ollama_host = self.llm_config.get("host", "http://localhost:11434")
                    
                    # Make request
                    async with self.llm_client.stream(
                        "POST",
                        f"{ollama_host}/api/generate",
                        json={
                            "model": model_to_use,
                            "prompt": prompt,
                            "stream": True,
                            "temperature": temp,
                            "top_p": self.top_p,
                            "num_predict": 1024,  # Limit response length
                            "stop": ["<|im_end|>", "<|endoftext|>"]
                        }
                    ) as response:
                        # Check response status BEFORE attempting to process the stream
                        if response.status_code != 200:
                            self.logger.warning(f"Streaming LLM inference failed with status {response.status_code} for model {model_to_use}. Response: {await response.aread()}")
                            retry_count += 1
                            continue # Go to next retry iteration
                        
                        # Process streaming response
                        full_response = ""
                        async for chunk in response.aiter_bytes():
                            try:
                                # Parse JSON chunk
                                chunk_str = chunk.decode('utf-8')
                                
                                # Handle multiple JSON objects in stream
                                for line in chunk_str.splitlines():
                                    if not line.strip():
                                        continue
                                        
                                    data = json.loads(line)
                                    token = data.get("response", "")
                                    full_response += token
                                    
                                    # Call callback with token
                                    if token and callback:
                                        callback(token)
                            except Exception as e: # Error while processing a chunk
                                self.logger.error(f"Error processing LLM stream chunk for model {model_to_use}: {e}", exc_info=True)
                                # Depending on severity, you might want to break or continue
                                # For now, we log and continue processing other chunks, but this might indicate a larger issue.
                        
                        # If stream completed without HTTP error for this attempt
                        end_time = time.time()
                        self.metrics["llm_calls"] += 1
                        self.metrics["llm_latency_ms"] = (end_time - start_time) * 1000
                        self.metrics["llm_tokens_in"] += len(prompt.split()) * 0.75 # Approximate
                        self.metrics["llm_tokens_out"] += len(full_response.split()) * 0.75 # Approximate
                        return True # Success for this attempt

                except httpx.RequestError as e: # Specific exception for HTTPX request errors
                    self.logger.warning(f"Streaming LLM inference HTTPX RequestError for model {model_to_use} (attempt {retry_count + 1}/{MAX_RETRIES}): {e}", exc_info=True)
                    retry_count += 1
                except Exception as e: # Catch other exceptions during setup or initial connection
                    self.logger.error(f"Streaming LLM inference error for model {model_to_use} (attempt {retry_count + 1}/{MAX_RETRIES}): {e}", exc_info=True)
                    retry_count += 1
            
            self.logger.error(f"Streaming LLM inference failed after {MAX_RETRIES} retries for prompt: {prompt[:100]}...")
            return False
    
    #--------------------------------------------------------------------
    # Task Management Methods
    #--------------------------------------------------------------------
    
    async def create_task(self, description: str, priority: TaskPriority = TaskPriority.NORMAL, context: Dict = None) -> str:
        """Create a new task for the cognitive system to process."""
        async with self._task_lock:
            try:
                # Generate unique ID
                task_id = f"task_{uuid.uuid4().hex[:8]}"
                
                # Create task object
                task = Task(
                    id=task_id,
                    description=description,
                    priority=priority,
                    status=TaskStatus.PENDING,
                    context=context or {}
                )
                
                # Store task
                self.tasks[task_id] = task
                
                self.logger.info(f"Created task: {task_id} - {description}")
                
                # Record to memory
                await self.memory.create_memory(
                    content=f"Created task: {description}",
                    memory_type=MEMORY_TYPE_EPISODIC,
                    metadata={
                        "task_id": task_id,
                        "task_type": "creation",
                        "timestamp": time.time()
                    },
                    importance=0.5
                )
                
                return task_id
            except Exception as e:
                self.logger.error(f"Failed to create task: {e}", exc_info=True)
                return ""
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task details by ID."""
        if task_id in self.tasks:
            return self.tasks[task_id]
        return None
    
    async def update_task(self, task: Task) -> bool:
        """Update task in storage."""
        try:
            self.tasks[task.id] = task
            return True
        except Exception as e:
            self.logger.error(f"Failed to update task: {e}", exc_info=True)
            return False
    
    async def _update_task(self, task: Task) -> bool:
        """Internal method to update task in storage."""
        return await self.update_task(task)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or in-progress task."""
        async with self._task_lock:
            try:
                if task_id not in self.tasks:
                    return False
                    
                task = self.tasks[task_id]
                
                # Can only cancel pending or in-progress tasks
                if task.status not in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                    return False
                    
                # Update task status
                task.status = TaskStatus.CANCELLED
                task.completed_at = time.time()
                
                # Clear active task if this is the active one
                if self.active_task_id == task_id:
                    self.active_task_id = None
                
                self.logger.info(f"Cancelled task: {task_id} - {task.description}")
                
                return True
            except Exception as e:
                self.logger.error(f"Failed to cancel task: {e}", exc_info=True)
                return False
    
    async def get_all_tasks(self, filter_status: Optional[TaskStatus] = None) -> List[Task]:
        """Get all tasks, optionally filtered by status."""
        try:
            if filter_status:
                return [t for t in self.tasks.values() if t.status == filter_status]
            else:
                return list(self.tasks.values())
        except Exception as e:
            self.logger.error(f"Failed to get tasks: {e}", exc_info=True)
            return []
    
    #--------------------------------------------------------------------
    # External API Methods
    #--------------------------------------------------------------------
    
    async def reason_about(self, question: str) -> str:
        """Perform reasoning about a specific question or scenario."""
        if not LLM_AVAILABLE or not self.llm_client:
            return "LLM functionality not available for reasoning"
            
        try:
            # Retrieve relevant memories
            memories = await self.memory.remember(
                query=question,
                limit=5
            )
            
            # Create reasoning context
            context = {
                "question": question,
                "relevant_memories": [m.content for m in memories.results],
                "perception_events": [e.to_dict() for e in self.recent_events[-5:]],
                "system_state": self.system_state
            }
            
            # Create prompt
            prompt = f"""You are an autonomous agent operating system with perception and reasoning capabilities.
Answer the following question based on your knowledge and current system state.

QUESTION:
{question}

RELEVANT MEMORIES:
{json.dumps([str(m) for m in memories.results], indent=2)}

RECENT PERCEPTION EVENTS:
{json.dumps([e.to_dict() for e in self.recent_events[-5:]], indent=2)}

CURRENT SYSTEM STATE:
{json.dumps(self.system_state, indent=2)}

Provide a thoughtful, accurate, and helpful response to the question."""
            
            # Generate reasoning using LLM
            response = await self._llm_inference(prompt)
            
            if not response:
                return "Unable to generate reasoning due to LLM error"
                
            # Record reasoning to memory
            await self.memory.create_memory(
                content=f"Question: {question}\nReasoning: {response}",
                memory_type=MEMORY_TYPE_REFLECTIVE,
                metadata={
                    "question": question,
                    "timestamp": time.time()
                },
                importance=0.6
            )
            
            return response
        except Exception as e:
            self.logger.error(f"Reasoning failed: {e}", exc_info=True)
            return f"Reasoning failed: {str(e)}"
    
    async def understand_screen(self) -> Dict:
        """Analyze the current screen and return a structured understanding."""
        try:
            if self.current_screen is None:
                return {"error": "No screen capture available"}
                
            # Get current UI elements and text
            elements = self.ui_elements
            text_regions = self.detected_text
            
            # Create structured representation
            screen_data = {
                "ui_elements": [
                    {
                        "type": elem.element_type,
                        "position": elem.bounding_box,
                        "text": elem.text,
                        "confidence": elem.confidence,
                        "attributes": elem.attributes
                    }
                    for elem in elements
                ],
                "text_regions": text_regions,
                "timestamp": time.time()
            }
            
            # If LLM is available, get high-level interpretation
            if LLM_AVAILABLE and self.llm_client:
                # Create prompt for screen understanding
                prompt = f"""You are an autonomous agent operating system with perception capabilities.
Analyze and summarize the current screen content based on the detected UI elements and text.

DETECTED UI ELEMENTS:
{json.dumps(screen_data["ui_elements"], indent=2)}

DETECTED TEXT:
{json.dumps(screen_data["text_regions"], indent=2)}

Provide a clear, concise summary of what's currently on the screen.
What application is likely open? What is the user seeing? What interactions are possible?"""
                
                # Generate interpretation using LLM
                interpretation = await self._llm_inference(prompt)
                
                if interpretation:
                    screen_data["interpretation"] = interpretation
            
            return screen_data
        except Exception as e:
            self.logger.error(f"Screen understanding failed: {e}", exc_info=True)
            return {"error": f"Screen understanding failed: {str(e)}"}
    
    async def get_system_state(self) -> Dict:
        """Get the current system state and perception history."""
        try:
            # Create a snapshot of the system state
            state_snapshot = {
                "system_state": self.system_state,
                "recent_events": [e.to_dict() for e in self.recent_events[-20:]],
                "active_task": self.active_task_id,
                "ui_elements_count": len(self.ui_elements),
                "text_regions_count": len(self.detected_text),
                "metrics": self.metrics,
                "timestamp": time.time()
            }
            
            return state_snapshot
        except Exception as e:
            self.logger.error(f"Failed to get system state: {e}", exc_info=True)
            return {"error": f"Failed to get system state: {str(e)}"}

# Simple test code when run directly
if __name__ == "__main__":
    import argparse
    import random
    import re
    
    parser = argparse.ArgumentParser(description="Test the ORAMA Cognitive Engine")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()
    
    async def test_cognitive_engine():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Create test config
        config = {
            "llm": {
                "model": "deepseek-coder-7b-instruct-q5_K_M",
                "context_size": 8192,
                "temperature": 0.7,
                "top_p": 0.9,
                "timeout": 30.0,
                "streaming": True,
                "fallback_model": "gemma-2b-instruct-q5_K_M"
            },
            "perception": {
                "screen_capture": {
                    "rate": 10,
                    "resolution": "native",
                    "regions": ["full"]
                },
                "ocr": {
                    "languages": ["eng"],
                    "confidence": 0.7
                },
                "ui_detection": {
                    "confidence": 0.75,
                    "classes": ["button", "input", "dropdown", "checkbox", "radio", "text", "image", "icon"]
                },
                "system_monitor": {
                    "process_check_interval": 1.0,
                    "watch_directories": ["~/Downloads", "~/Documents"]
                }
            },
            "reasoning": {
                "planning": {
                    "max_steps": 20,
                    "validation": True,
                    "timeout": 60.0
                },
                "decision": {
                    "risk_threshold": 0.3,
                    "confidence_threshold": 0.7,
                    "verification_required": True
                }
            }
        }
        
        # Mock memory engine for testing
        class MockMemoryEngine:
            async def create_memory(self, content, memory_type, metadata=None, importance=0.5):
                return f"memory_{uuid.uuid4().hex[:8]}"
                
            async def remember(self, query, memory_types=None, limit=5, metadata_filter=None):
                from dataclasses import dataclass
                
                @dataclass
                class MockMemoryResult:
                    content: str
                    score: float = 1.0
                    
                @dataclass
                class MockQueryResult:
                    results: list
                    total_found: int
                    query_time_ms: float
                
                # Create mock memory results
                results = [
                    MockMemoryResult(f"Memory about {query}"),
                    MockMemoryResult(f"Another memory related to {query}")
                ]
                
                return MockQueryResult(results=results, total_found=len(results), query_time_ms=10.0)
        
        # Create memory engine
        memory = MockMemoryEngine()
        
        # Create cognitive engine
        cognitive = CognitiveEngine(config, memory)
        await cognitive.start()
        
        try:
            # Test task creation
            task_id = await cognitive.create_task("Test the system by opening a browser")
            print(f"Created task: {task_id}")
            
            # Let the system run for a bit
            print("Letting the system run for 10 seconds...")
            for i in range(10):
                await asyncio.sleep(1)
                print(f"System running... {i+1}/10")
                
                # Get system state
                state = await cognitive.get_system_state()
                print(f"Active task: {state.get('active_task')}")
                print(f"Metrics: {state.get('metrics')}")
                
                # Occasionally add a new task
                if i == 5:
                    task_id2 = await cognitive.create_task("Search for information about AI", 
                                                      TaskPriority.HIGH)
                    print(f"Created high priority task: {task_id2}")
            
            # Test reasoning
            print("\nTesting reasoning capability...")
            reasoning = await cognitive.reason_about("How can I optimize system performance?")
            print(f"Reasoning result: {reasoning[:200]}...")
            
            # Test screen understanding
            print("\nTesting screen understanding...")
            screen_data = await cognitive.understand_screen()
            if "interpretation" in screen_data:
                print(f"Screen interpretation: {screen_data['interpretation'][:200]}...")
            else:
                print(f"Screen data: {screen_data}")
            
            # Get all tasks
            print("\nGetting all tasks...")
            tasks = await cognitive.get_all_tasks()
            for task in tasks:
                print(f"Task {task.id}: {task.description} - Status: {task.status.name}")
        finally:
            # Shutdown
            await cognitive.stop()
            print("\nCognitive engine stopped")
    
    # Run test
    asyncio.run(test_cognitive_engine())
