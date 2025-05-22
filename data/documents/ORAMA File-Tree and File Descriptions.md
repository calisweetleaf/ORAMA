# ORAMA System Project Directory Structure

```
/orama/
├── main.py                 # System entry point and core initialization
├── config.yaml             # Centralized configuration for all subsystems
├── requirements.txt        # Project dependencies
├── LICENSE                 # Open-source license information
│
├── core_engine.py          # Unified cognitive architecture with perception and reasoning
├── action_system.py        # System access, input simulation and execution mechanisms
├── memory_engine.py        # Multi-tier persistent memory architecture
├── orchestrator.py         # Event-driven coordination and task management
├── interface.py            # User interaction and command processing
├── system_utils.py         # System monitoring, resource management, OS integration
│
└── data/                   # System data storage
    ├── vector_store/       # Vector database storage
    ├── knowledge_graph/    # Graph database storage
    ├── documents/          # Document repository
    └── parameters/         # Configuration and parameter storage
```

## File Descriptions

### Main System Files

1. **main.py**
    
    - System entry point, subsystem initialization, and core event loop
    - Manages startup/shutdown sequences and inter-module communication
    - Coordinates all subsystems and handles global exception management
2. **config.yaml**
    
    - Comprehensive configuration for all subsystems
    - Controls resource allocation, model selection, and behavior parameters
    - Defines memory structure, security boundaries, and integration points
3. **core_engine.py**
    
    - Unified cognitive architecture integrating perception and reasoning
    - Manages LLM inference (Ollama/GGUF models), prompt engineering, and context management
    - Implements computer vision pipeline, OCR, UI element detection, and planning capabilities
4. **action_system.py**
    
    - Executes system operations through command execution and input simulation
    - Provides browser automation, file system access, and application control
    - Implements verification mechanisms and safety protocols for operations
5. **memory_engine.py**
    
    - Multi-tier persistent memory architecture (vector store, knowledge graph, document repository)
    - Manages memory operations including encoding, retrieval, consolidation, and optimization
    - Implements caching strategies, embedding generation, and knowledge integration
6. **orchestrator.py**
    
    - Event-driven coordination system for managing subsystem interactions
    - Implements task scheduling, priority management, and resource allocation
    - Manages perception-action cycles and verification loops
7. **interface.py**
    
    - Provides command processing, natural language understanding, and user interaction
    - Implements status reporting, result presentation, and configuration management
    - Handles user commands, notifications, and system control
8. **system_utils.py**
    
    - Implements system monitoring, resource tracking, and diagnostics
    - Provides security enforcement, permission management, and data protection
    - Manages logging, performance analysis, and system health monitoring

### Supporting Files

9. **requirements.txt**
    
    - Lists all Python package dependencies with version specifications
    - Includes core libraries for ML, UI interaction, memory systems, and system access
10. **LICENSE**
    
    - Contains licensing information for the ORAMA System
    - Specifies terms of use, distribution, and modification rights

### Data Directories

11. **data/vector_store/**
    
    - Storage for vector embeddings and semantic search indices
    - Managed by LanceDB or comparable vector database
12. **data/knowledge_graph/**
    
    - Storage for entity-relationship data and causal networks
    - Implemented using SQLite with graph extensions or embedded Neo4j
13. **data/documents/**
    
    - Repository for documents, media assets, and structured records
    - Organized with metadata for efficient content management
14. **data/parameters/**
    
    - Storage for system parameters, learned settings, and configuration state
    - Maintains user preferences and optimization values