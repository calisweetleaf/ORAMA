#!/usr/bin/env python3
# ORAMA System - Main Entry Point
# Implements system initialization, event loop, and core orchestration

import os
import sys
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

# Internal imports
from core_engine import CognitiveEngine
from action_system import ActionSystem
from memory_engine import MemoryEngine
from orchestrator import Orchestrator, EventType, ComponentType, Event
from system_utils import SystemMonitor, SystemMetrics, Alert, ResourceType, SecurityLevel

@dataclass
class SystemStatus:
    """Encapsulates system health status."""
    healthy: bool = True
    critical: bool = False
    details: str = ""

class ORAMASystem:
    """Main ORAMA system class implementing the cognitive agent architecture."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the ORAMA system.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.running = False
        
        # Initialize memory first since cognitive engine depends on it
        self.memory = MemoryEngine(self.config["memory"])
        
        # Initialize other core subsystems
        self.cognitive = CognitiveEngine(
            config=self.config["cognitive"],
            memory_engine=self.memory
        )
        self.action = ActionSystem(self.config["action"])
        self.monitor = SystemMonitor(self.config["system"])
        
        # Initialize orchestrator last since it coordinates other subsystems
        self.orchestrator = Orchestrator(
            config=self.config["orchestrator"],
            cognitive_engine=self.cognitive,
            action_system=self.action,
            memory_engine=self.memory
        )

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the system."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("orama.log")
            ]
        )
        return logging.getLogger("ORAMA")

    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_path}")

            # Validate configuration
            required_keys = ["system", "components", "preferences", "orchestrator"]
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing '{key}' configuration in {config_path}")

            required_system_paths = ["paths"]
            for key in required_system_paths:
                if key not in config["system"]:
                    raise ValueError(f"Missing 'system.{key}' configuration in {config_path}")

            required_components_keys = ["perception", "reasoning", "action", "memory"]
            for key in required_components_keys:
                if key not in config["components"]:
                    raise ValueError(f"Missing 'components.{key}' configuration in {config_path}")
            
            # Validate model paths
            models_path = Path(config["system"]["paths"]["models"])
            
            llm_model_name = config.get("components", {}).get("reasoning", {}).get("llm", {}).get("model")
            if llm_model_name and llm_model_name.endswith(".gguf"): # Check if it's a local model
                llm_model_path = models_path / llm_model_name
                if not llm_model_path.exists():
                    raise FileNotFoundError(f"LLM model file not found: {llm_model_path}")

            vision_model_name = config.get("components", {}).get("perception", {}).get("vision", {}).get("model")
            if vision_model_name and vision_model_name.endswith(".onnx"): # Check if it's a local model
                vision_model_path = models_path / vision_model_name
                if not vision_model_path.exists():
                    raise FileNotFoundError(f"Vision model file not found: {vision_model_path}")

            return config
        except Exception as e:
            self.logger.error(f"Failed to load or validate config: {e}")
            raise

    def _evaluate_system_health(self, metrics: SystemMetrics, alerts: List[Alert]) -> SystemStatus:
        """Evaluate overall system health based on metrics and alerts."""
        critical_resources = {
            ResourceType.CPU: 95.0,
            ResourceType.MEMORY: 90.0,
            ResourceType.DISK: 95.0
        }
        
        # Check for critical resource usage
        if metrics.cpu_percent > critical_resources[ResourceType.CPU] or \
           metrics.memory_percent > critical_resources[ResourceType.MEMORY]:
            return SystemStatus(
                healthy=False,
                critical=True,
                details="Critical resource threshold exceeded"
            )
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.level == SecurityLevel.CRITICAL]
        if critical_alerts:
            return SystemStatus(
                healthy=False,
                critical=True,
                details=f"Critical alerts present: {'; '.join(a.message for a in critical_alerts)}"
            )
        
        # Check for high-severity alerts
        high_alerts = [a for a in alerts if a.level == SecurityLevel.HIGH]
        if high_alerts:
            return SystemStatus(
                healthy=False,
                critical=False,
                details=f"High severity alerts present: {'; '.join(a.message for a in high_alerts)}"
            )
        
        return SystemStatus(healthy=True, critical=False, details="System healthy")

    async def start(self):
        """Initialize and start all subsystems."""
        try:
            self.logger.info("Starting ORAMA system...")
            
            # Start subsystems in order - memory first, orchestrator last
            await self.memory.start()
            await self.cognitive.start()
            await self.action.start()
            await self.monitor.start()
            await self.orchestrator.start()
            
            # Send system startup event
            await self.orchestrator.publish_event(Event(
                event_type=EventType.SYSTEM_STARTUP,
                source=ComponentType.ORCHESTRATOR
            ))
            
            self.running = True
            self.logger.info("ORAMA system startup complete")
            
        except Exception as e:
            self.logger.error(f"Startup failed: {e}")
            await self.stop()
            raise

    async def run(self):
        """Main system event loop."""
        try:
            await self.start()
            
            while self.running:
                try:
                    # Get system metrics and alerts
                    metrics = self.monitor.get_metrics() # Corrected: SystemMonitor.get_metrics is not async
                    alerts = self.monitor.get_alerts() # Corrected: SystemMonitor.get_alerts is not async
                    
                    # Evaluate system health
                    status = self._evaluate_system_health(metrics, alerts)
                    if not status.healthy:
                        self.logger.error(f"System health check failed: {status.details}")
                        if status.critical:
                            break
                    
                    # Brief sleep to prevent CPU thrashing
                    await asyncio.sleep(0.01)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received shutdown signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry
            
        finally:
            await self.stop()

    async def stop(self):
        """Gracefully stop all subsystems."""
        self.logger.info("Shutting down ORAMA system...")
        self.running = False
        
        try:
            # Send system shutdown event
            await self.orchestrator.publish_event(Event(
                event_type=EventType.SYSTEM_SHUTDOWN,
                source=ComponentType.ORCHESTRATOR
            ))
            
            # Stop subsystems in reverse order of startup
            await self.orchestrator.stop()
            await self.monitor.stop()
            await self.action.stop()
            await self.cognitive.stop()
            await self.memory.stop()
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self.logger.info("Shutdown complete")

async def main():
    """Entry point for the ORAMA system."""
    system = ORAMASystem()
    await system.run()

if __name__ == "__main__":
    # Run the system using asyncio
    asyncio.run(main())