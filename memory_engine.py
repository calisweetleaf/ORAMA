#!/usr/bin/env python3
# ORAMA System - Memory Engine
# Multi-tier persistent memory architecture combining vector storage, knowledge graph, and document repository

import os
import json
import time
import asyncio
import logging
import hashlib
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import sqlite3
import aiosqlite
from concurrent.futures import ThreadPoolExecutor

# Vector database imports
try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

# Embedding generation
try:
    import onnxruntime as ort
    import nltk
    from nltk.tokenize import sent_tokenize
    EMBEDDING_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    EMBEDDING_AVAILABLE = False

# Constants for memory management
DEFAULT_VECTOR_DIMENSION = 768
MAX_TRANSACTION_RETRY = 3
DEFAULT_IMPORTANCE_THRESHOLD = 0.3
MEMORY_COMMIT_INTERVAL = 60  # seconds

# Memory entry types
MEMORY_TYPE_EPISODIC = "episodic"  # Experiences and observed events
MEMORY_TYPE_SEMANTIC = "semantic"   # Facts and conceptual knowledge
MEMORY_TYPE_PROCEDURAL = "procedural"  # Skills and action sequences
MEMORY_TYPE_REFLECTIVE = "reflective"  # Self-evaluations and insights

@dataclass
class MemoryMetadata:
    """Metadata for memory entries across all storage types."""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0 with higher values indicating greater importance
    confidence: float = 1.0  # 0.0 to 1.0
    memory_type: str = MEMORY_TYPE_SEMANTIC
    source: str = "system"
    tags: List[str] = field(default_factory=list)
    expiration: Optional[float] = None  # Timestamp for when this memory should expire
    associations: List[str] = field(default_factory=list)  # IDs of related memories
    spatial_context: Optional[str] = None  # Where this memory applies (e.g., application, website)
    temporal_context: Optional[str] = None  # When this memory applies (e.g., time range)

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryMetadata':
        """Create metadata from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in field(cls)]})

    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

@dataclass
class VectorEntry:
    """Vector database entry."""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict

    @classmethod
    def from_dict(cls, data: Dict) -> 'VectorEntry':
        """Create entry from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            embedding=data["embedding"],
            metadata=data["metadata"]
        )

@dataclass
class KnowledgeEntity:
    """Entity in knowledge graph."""
    id: str
    type: str
    properties: Dict
    metadata: Dict

@dataclass
class KnowledgeRelation:
    """Relationship in knowledge graph."""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict
    metadata: Dict

@dataclass
class DocumentEntry:
    """Document store entry."""
    id: str
    filename: str
    content_type: str
    metadata: Dict
    path: str
    size: int

@dataclass
class MemoryResult:
    """Result from memory retrieval operations."""
    content: Any  # Actual memory content (text, entity, document, etc.)
    metadata: MemoryMetadata
    score: float = 1.0  # Relevance or matching score (0.0 to 1.0)
    source: str = ""    # Source storage (vector, graph, document, parameter)
    memory_type: str = ""  # Type of memory

@dataclass
class MemoryQueryResult:
    """Container for query results from memory subsystems."""
    results: List[MemoryResult]
    total_found: int
    query_time_ms: float

class MemoryEngine:
    """
    ORAMA Memory Engine - Multi-tier persistent memory architecture
    
    A unified memory system combining vector storage, knowledge graph, and document repository
    to provide comprehensive memory capabilities for the autonomous agent.
    
    The memory architecture supports:
    - Semantic search through vector embeddings
    - Entity-relationship modeling through knowledge graph
    - Document and media storage
    - Parameter persistence
    - Long-term memory consolidation
    - Hierarchical organization with importance-based retention
    """
    
    def __init__(self, config: Dict, logger=None):
        """Initialize the memory engine with configuration."""
        self.config = config
        self.logger = logger or logging.getLogger("orama.memory")
        
        # Set base paths
        self.base_path = Path(config.get("path", "data"))
        self.vector_path = self.base_path / config.get("vector", {}).get("path", "vector_store")
        self.graph_path = self.base_path / config.get("graph", {}).get("path", "knowledge_graph")
        self.document_path = self.base_path / config.get("document", {}).get("path", "documents")
        self.parameter_path = self.base_path / config.get("parameters", {}).get("path", "parameters")
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Initialize memory subsystems
        self.vector_db = None
        self.graph_db = None
        self.parameter_store = {}
        self.document_index = {}
        
        # Embedding model
        self.embedding_model = None
        self.embedding_dimension = config.get("vector", {}).get("dimension", DEFAULT_VECTOR_DIMENSION)
        
        # Memory management settings
        self.importance_threshold = config.get("management", {}).get(
            "importance_threshold", DEFAULT_IMPORTANCE_THRESHOLD
        )
        self.consolidation_interval = config.get("management", {}).get(
            "consolidation_interval", 86400  # 24 hours in seconds
        )
        self.max_storage_gb = config.get("management", {}).get("max_storage_gb", 10)
        
        # Thread pool for background operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Async lock for memory operations
        self._vector_lock = asyncio.Lock()
        self._graph_lock = asyncio.Lock()
        self._parameter_lock = asyncio.Lock()
        self._document_lock = asyncio.Lock()
        
        # Task for background memory maintenance
        self._maintenance_task = None
        self._running = False
        
        # Memory commit tracking
        self._last_commit_time = time.time()
        self._commit_interval = MEMORY_COMMIT_INTERVAL
        self._pending_changes = False
        
        self.logger.info("Memory engine initialized")
    
    async def start(self) -> None:
        """Start the memory engine and initialize all subsystems."""
        self.logger.info("Starting memory engine...")
        self._running = True
        
        try:
            # Initialize vector database
            await self._init_vector_store()
            
            # Initialize knowledge graph
            await self._init_knowledge_graph()
            
            # Initialize document store
            await self._init_document_store()
            
            # Initialize parameter store
            await self._init_parameter_store()
            
            # Initialize embedding model if available
            await self._init_embedding_model()
            
            # Start background memory maintenance
            self._maintenance_task = asyncio.create_task(self._memory_maintenance_loop())
            
            self.logger.info("Memory engine started successfully")
        except Exception as e:
            self._running = False
            self.logger.error(f"Failed to start memory engine: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the memory engine and perform cleanup."""
        self.logger.info("Stopping memory engine...")
        self._running = False
        
        # Cancel maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            
        # Final memory commit
        await self._commit_memory_changes()
            
        # Close vector database
        if self.vector_db:
            try:
                # Lance DB doesn't need explicit closing
                self.vector_db = None
            except Exception as e:
                self.logger.warning(f"Error closing vector database: {e}")
            
        # Close knowledge graph
        if hasattr(self, 'graph_conn') and self.graph_conn:
            try:
                await self.graph_conn.close()
                self.graph_conn = None
            except Exception as e:
                self.logger.warning(f"Error closing knowledge graph: {e}")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("Memory engine stopped")

    #--------------------------------------------------------------------
    # Vector Store Methods
    #--------------------------------------------------------------------
    
    async def _init_vector_store(self) -> None:
        """Initialize the vector database."""
        if not LANCEDB_AVAILABLE:
            self.logger.warning("LanceDB not available, vector storage will be disabled")
            return
        
        try:
            # Initialize LanceDB
            vector_path = str(self.vector_path)
            self.vector_db = lancedb.connect(vector_path)
            
            # Create or open vector tables
            semantic_schema = {
                "id": "string",
                "text": "string",
                "embedding": f"float32({self.embedding_dimension})",
                "metadata": "json"
            }
            
            # Create tables if they don't exist
            try:
                self.semantic_table = self.vector_db.open_table("semantic_memories")
                self.logger.info("Opened existing semantic memories table")
            except Exception:
                self.semantic_table = self.vector_db.create_table(
                    "semantic_memories",
                    data=[{
                        "id": "init",
                        "text": "Initialization record",
                        "embedding": np.zeros(self.embedding_dimension, dtype=np.float32).tolist(),
                        "metadata": {"system": True}
                    }],
                    schema=semantic_schema
                )
                self.logger.info("Created new semantic memories table")
                
            try:
                self.episodic_table = self.vector_db.open_table("episodic_memories")
                self.logger.info("Opened existing episodic memories table")
            except Exception:
                self.episodic_table = self.vector_db.create_table(
                    "episodic_memories",
                    data=[{
                        "id": "init",
                        "text": "Initialization record",
                        "embedding": np.zeros(self.embedding_dimension, dtype=np.float32).tolist(),
                        "metadata": {"system": True}
                    }],
                    schema=semantic_schema
                )
                self.logger.info("Created new episodic memories table")
                
            self.logger.info("Vector store initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            raise
    
    async def _init_embedding_model(self) -> None:
        """Initialize the embedding model for vector encoding."""
        if not EMBEDDING_AVAILABLE:
            self.logger.warning("ONNX Runtime not available, using random embeddings")
            return
            
        try:
            # Check for embedding model path in config
            model_path = self.config.get("vector", {}).get("embedding_model")
            
            if model_path and os.path.exists(model_path):
                # Initialize ONNX session with the model
                self.logger.info(f"Loading embedding model from {model_path}")
                
                # Use ONNX Runtime for efficient inference
                self.embedding_model = ort.InferenceSession(
                    model_path, 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                
                self.logger.info("Embedding model loaded successfully")
            else:
                self.logger.warning("No embedding model found, using random embeddings")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            self.logger.warning("Falling back to random embeddings")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text input."""
        if not text:
            # Return zero vector for empty text
            return np.zeros(self.embedding_dimension, dtype=np.float32).tolist()
        
        try:
            if self.embedding_model:
                # Use actual embedding model for inference
                # This is a simplified example; real implementation would depend on the specific model
                model_inputs = {
                    'input_ids': np.array([text]), 
                    'attention_mask': np.ones((1, len(text)), dtype=np.int64)
                }
                
                # Run inference using ONNX Runtime
                embedding = await asyncio.to_thread(
                    self.embedding_model.run, 
                    None, 
                    model_inputs
                )
                
                # Normalize embedding
                embedding = embedding[0][0]  # Extract from output
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding.astype(np.float32).tolist()
            else:
                # Use deterministic random embedding for testing/fallback
                # Hash the text to get a deterministic seed
                seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
                np.random.seed(seed)
                
                # Generate random embedding
                embedding = np.random.randn(self.embedding_dimension)
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding.astype(np.float32).tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            # Return random embedding as fallback
            np.random.seed(0)  # Consistent fallback
            embedding = np.random.randn(self.embedding_dimension)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.astype(np.float32).tolist()
    
    async def add_to_vector_store(self, 
                                  text: str, 
                                  metadata: Union[Dict, MemoryMetadata],
                                  memory_type: str = MEMORY_TYPE_SEMANTIC) -> str:
        """Add text entry to vector store with metadata."""
        async with self._vector_lock:
            try:
                if not LANCEDB_AVAILABLE or not self.vector_db:
                    self.logger.warning("Vector database not available, skipping vector storage")
                    return ""
                
                # Convert metadata to dict if needed
                if isinstance(metadata, MemoryMetadata):
                    meta_dict = metadata.to_dict()
                else:
                    meta_dict = metadata
                    
                # Generate unique ID based on content and timestamp
                timestamp = time.time()
                content_hash = hashlib.md5(f"{text}:{timestamp}".encode()).hexdigest()
                entry_id = f"{memory_type}_{content_hash}"
                
                # Generate embedding
                embedding = await self.generate_embedding(text)
                
                # Determine which table to use
                table = self.semantic_table
                if memory_type == MEMORY_TYPE_EPISODIC:
                    table = self.episodic_table
                
                # Create entry
                entry = {
                    "id": entry_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": meta_dict
                }
                
                # Add to vector store
                await asyncio.to_thread(
                    table.add,
                    [entry]
                )
                
                self._pending_changes = True
                
                self.logger.debug(f"Added entry to vector store: {entry_id}")
                return entry_id
            except Exception as e:
                self.logger.error(f"Failed to add to vector store: {e}", exc_info=True)
                return ""
    
    async def search_vector_store(self, 
                                 query: str, 
                                 limit: int = 5,
                                 memory_type: str = MEMORY_TYPE_SEMANTIC,
                                 metadata_filter: Optional[Dict] = None) -> MemoryQueryResult:
        """Search vector store for entries similar to query."""
        start_time = time.time()
        
        if not LANCEDB_AVAILABLE or not self.vector_db:
            self.logger.warning("Vector database not available")
            return MemoryQueryResult(results=[], total_found=0, query_time_ms=0)
        
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Select table based on memory type
            table = self.semantic_table
            if memory_type == MEMORY_TYPE_EPISODIC:
                table = self.episodic_table
            
            # Build query
            vector_query = table.search(query_embedding)
            
            # Add metadata filter if provided
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if isinstance(value, list):
                        # For list values, use "in" predicate
                        filter_str = f"metadata->>'{key}' IN " + str(value).replace('[', '(').replace(']', ')')
                        vector_query = vector_query.where(filter_str)
                    else:
                        # For scalar values, use equality
                        vector_query = vector_query.where(f"metadata->>'{key}'='{value}'")
            
            # Execute search
            results = await asyncio.to_thread(
                vector_query.limit(limit).to_list
            )
            
            # Convert to MemoryResult objects
            memory_results = []
            for item in results:
                # Skip initialization record
                if item["id"] == "init":
                    continue
                    
                metadata = MemoryMetadata.from_dict(item["metadata"])
                
                # Update access info
                metadata.update_access()
                
                # Calculate normalized score (distance to similarity)
                distance = item.get("_distance", 0)
                similarity = 1.0 / (1.0 + distance)
                
                memory_results.append(
                    MemoryResult(
                        content=item["text"],
                        metadata=metadata,
                        score=similarity,
                        source="vector",
                        memory_type=memory_type
                    )
                )
                
                # Update metadata in store
                await self._update_vector_metadata(item["id"], metadata, memory_type)
            
            end_time = time.time()
            query_time_ms = (end_time - start_time) * 1000
            
            return MemoryQueryResult(
                results=memory_results,
                total_found=len(memory_results),
                query_time_ms=query_time_ms
            )
        except Exception as e:
            self.logger.error(f"Vector store search failed: {e}", exc_info=True)
            end_time = time.time()
            query_time_ms = (end_time - start_time) * 1000
            return MemoryQueryResult(results=[], total_found=0, query_time_ms=query_time_ms)
    
    async def _update_vector_metadata(self, 
                                     entry_id: str, 
                                     metadata: MemoryMetadata,
                                     memory_type: str = MEMORY_TYPE_SEMANTIC) -> bool:
        """Update metadata for a vector store entry."""
        if not LANCEDB_AVAILABLE or not self.vector_db:
            return False
            
        try:
            # Select table based on memory type
            table = self.semantic_table
            if memory_type == MEMORY_TYPE_EPISODIC:
                table = self.episodic_table
                
            # Update metadata
            await asyncio.to_thread(
                table.update_with_keys,
                ["id"],
                [{"id": entry_id, "metadata": metadata.to_dict()}]
            )
            
            self._pending_changes = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to update vector metadata: {e}", exc_info=True)
            return False
    
    async def delete_from_vector_store(self, 
                                      entry_id: str,
                                      memory_type: str = MEMORY_TYPE_SEMANTIC) -> bool:
        """Delete entry from vector store."""
        async with self._vector_lock:
            if not LANCEDB_AVAILABLE or not self.vector_db:
                return False
                
            try:
                # Select table based on memory type
                table = self.semantic_table
                if memory_type == MEMORY_TYPE_EPISODIC:
                    table = self.episodic_table
                    
                # Delete entry
                await asyncio.to_thread(
                    table.delete,
                    f"id = '{entry_id}'"
                )
                
                self._pending_changes = True
                self.logger.debug(f"Deleted entry from vector store: {entry_id}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete from vector store: {e}", exc_info=True)
                return False

    #--------------------------------------------------------------------
    # Knowledge Graph Methods
    #--------------------------------------------------------------------
    
    async def _init_knowledge_graph(self) -> None:
        """Initialize the knowledge graph database."""
        try:
            # Create SQLite database for knowledge graph
            graph_db_path = self.graph_path / "graph.db"
            self.graph_path.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.graph_conn = await aiosqlite.connect(str(graph_db_path))
            self.logger.info(f"Connected to knowledge graph at {graph_db_path}")
            
            # Enable foreign keys
            await self.graph_conn.execute("PRAGMA foreign_keys = ON")
            
            # Create tables if they don't exist
            await self.graph_conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL
                )
            """)
            
            await self.graph_conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES entities (id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES entities (id) ON DELETE CASCADE
                )
            """)
            
            # Create indices for better performance
            await self.graph_conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities (type)")
            await self.graph_conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_type ON relations (type)")
            await self.graph_conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_source ON relations (source_id)")
            await self.graph_conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_target ON relations (target_id)")
            
            # Commit changes
            await self.graph_conn.commit()
            
            self.logger.info("Knowledge graph initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge graph: {e}", exc_info=True)
            raise
    
    async def add_entity(self, 
                        entity_type: str, 
                        properties: Dict, 
                        metadata: Union[Dict, MemoryMetadata]) -> str:
        """Add entity to knowledge graph."""
        async with self._graph_lock:
            try:
                if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                    self.logger.error("Knowledge graph connection is not initialized.")
                    return ""
                # Convert metadata to dict if needed
                if isinstance(metadata, MemoryMetadata):
                    meta_dict = metadata.to_dict()
                else:
                    meta_dict = metadata
                
                # Generate unique ID
                timestamp = time.time()
                content_hash = hashlib.md5(f"{entity_type}:{properties}:{timestamp}".encode()).hexdigest()
                entity_id = f"entity_{content_hash}"
                
                # Insert entity
                await self.graph_conn.execute(
                    """
                    INSERT INTO entities (id, type, properties, metadata, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        entity_type,
                        json.dumps(properties),
                        json.dumps(meta_dict),
                        timestamp,
                        timestamp
                    )
                )
                
                # Commit transaction
                await self.graph_conn.commit()
                self._pending_changes = True
                
                self.logger.debug(f"Added entity to knowledge graph: {entity_id}")
                return entity_id
            except Exception as e:
                self.logger.error(f"Failed to add entity: {e}", exc_info=True)
                await self.graph_conn.rollback()
                return ""
    
    async def add_relation(self, 
                          source_id: str, 
                          target_id: str, 
                          relation_type: str,
                          properties: Dict,
                          metadata: Union[Dict, MemoryMetadata]) -> str:
        """Add relationship between entities in knowledge graph."""
        async with self._graph_lock:
            try:
                if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                    self.logger.error("Knowledge graph connection is not initialized.")
                    return ""
                # Convert metadata to dict if needed
                if isinstance(metadata, MemoryMetadata):
                    meta_dict = metadata.to_dict()
                else:
                    meta_dict = metadata
                
                # Generate unique ID
                timestamp = time.time()
                content_hash = hashlib.md5(
                    f"{source_id}:{target_id}:{relation_type}:{timestamp}".encode()
                ).hexdigest()
                relation_id = f"relation_{content_hash}"
                
                # Insert relation
                await self.graph_conn.execute(
                    """
                    INSERT INTO relations 
                    (id, source_id, target_id, type, properties, metadata, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relation_id,
                        source_id,
                        target_id,
                        relation_type,
                        json.dumps(properties),
                        json.dumps(meta_dict),
                        timestamp,
                        timestamp
                    )
                )
                
                # Commit transaction
                await self.graph_conn.commit()
                self._pending_changes = True
                
                self.logger.debug(f"Added relation to knowledge graph: {relation_id}")
                return relation_id
            except Exception as e:
                self.logger.error(f"Failed to add relation: {e}", exc_info=True)
                await self.graph_conn.rollback()
                return ""
    
    async def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Get entity by ID."""
        try:
            if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                self.logger.error("Knowledge graph connection is not initialized.")
                return None
            # Query entity
            async with self.graph_conn.execute(
                "SELECT id, type, properties, metadata FROM entities WHERE id = ?",
                (entity_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                # Update access timestamp
                await self.graph_conn.execute(
                    "UPDATE entities SET last_accessed = ? WHERE id = ?",
                    (time.time(), entity_id)
                )
                await self.graph_conn.commit()
                self._pending_changes = True
                
                # Parse properties and metadata
                properties = json.loads(row[2])
                metadata = json.loads(row[3])
                
                return KnowledgeEntity(
                    id=row[0],
                    type=row[1],
                    properties=properties,
                    metadata=metadata
                )
        except Exception as e:
            self.logger.error(f"Failed to get entity: {e}", exc_info=True)
            return None
    
    async def query_entities(self, 
                            entity_type: Optional[str] = None,
                            property_filters: Optional[Dict] = None,
                            limit: int = 10) -> List[KnowledgeEntity]:
        """Query entities with filters."""
        try:
            if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                self.logger.error("Knowledge graph connection is not initialized.")
                return []
            # Build query
            query = "SELECT id, type, properties, metadata FROM entities"
            params = []
            
            where_clauses = []
            if entity_type:
                where_clauses.append("type = ?")
                params.append(entity_type)
            
            # Add property filters
            if property_filters:
                for key, value in property_filters.items():
                    # SQL for JSON property matching
                    where_clauses.append(f"json_extract(properties, '$.{key}') = ?")
                    params.append(value)
            
            # Combine where clauses
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            # Add limit
            query += " LIMIT ?"
            params.append(limit)
            
            # Execute query
            results = []
            async with self.graph_conn.execute(query, params) as cursor:
                async for row in cursor:
                    # Parse properties and metadata
                    properties = json.loads(row[2])
                    metadata = json.loads(row[3])
                    
                    entity = KnowledgeEntity(
                        id=row[0],
                        type=row[1],
                        properties=properties,
                        metadata=metadata
                    )
                    results.append(entity)
                    
                    # Update access timestamp (in background)
                    asyncio.create_task(self._update_entity_access(row[0]))
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to query entities: {e}", exc_info=True)
            return []
    
    async def _update_entity_access(self, entity_id: str) -> None:
        """Update entity access timestamp."""
        try:
            if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                self.logger.error("Knowledge graph connection is not initialized.")
                return
            await self.graph_conn.execute(
                "UPDATE entities SET last_accessed = ? WHERE id = ?",
                (time.time(), entity_id)
            )
            await self.graph_conn.commit()
            self._pending_changes = True
        except Exception as e:
            self.logger.error(f"Failed to update entity access: {e}", exc_info=True)
    
    async def get_entity_relations(self, 
                                  entity_id: str,
                                  relation_type: Optional[str] = None,
                                  direction: str = "outgoing") -> List[KnowledgeRelation]:
        """Get relationships for an entity."""
        try:
            if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                self.logger.error("Knowledge graph connection is not initialized.")
                return []
            # Build query based on direction
            if direction == "outgoing":
                query = "SELECT id, source_id, target_id, type, properties, metadata FROM relations WHERE source_id = ?"
                params = [entity_id]
            elif direction == "incoming":
                query = "SELECT id, source_id, target_id, type, properties, metadata FROM relations WHERE target_id = ?"
                params = [entity_id]
            else:  # both
                query = "SELECT id, source_id, target_id, type, properties, metadata FROM relations WHERE source_id = ? OR target_id = ?"
                params = [entity_id, entity_id]
            
            # Add relation type filter
            if relation_type:
                query += " AND type = ?"
                params.append(relation_type)
            
            # Execute query
            results = []
            async with self.graph_conn.execute(query, params) as cursor:
                async for row in cursor:
                    # Parse properties and metadata
                    properties = json.loads(row[4])
                    metadata = json.loads(row[5])
                    
                    relation = KnowledgeRelation(
                        id=row[0],
                        source_id=row[1],
                        target_id=row[2],
                        type=row[3],
                        properties=properties,
                        metadata=metadata
                    )
                    results.append(relation)
                    
                    # Update access timestamp (in background)
                    asyncio.create_task(self._update_relation_access(row[0]))
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to get entity relations: {e}", exc_info=True)
            return []
    
    async def _update_relation_access(self, relation_id: str) -> None:
        """Update relation access timestamp."""
        try:
            if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                self.logger.error("Knowledge graph connection is not initialized.")
                return
            await self.graph_conn.execute(
                "UPDATE relations SET last_accessed = ? WHERE id = ?",
                (time.time(), relation_id)
            )
            await self.graph_conn.commit()
            self._pending_changes = True
        except Exception as e:
            self.logger.error(f"Failed to update relation access: {e}", exc_info=True)
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete entity and all its relationships."""
        async with self._graph_lock:
            try:
                if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                    self.logger.error("Knowledge graph connection is not initialized.")
                    return False
                # Delete entity (cascade will remove relations)
                await self.graph_conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
                await self.graph_conn.commit()
                self._pending_changes = True
                
                self.logger.debug(f"Deleted entity from knowledge graph: {entity_id}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete entity: {e}", exc_info=True)
                await self.graph_conn.rollback()
                return False
    
    async def delete_relation(self, relation_id: str) -> bool:
        """Delete relationship."""
        async with self._graph_lock:
            try:
                if not hasattr(self, 'graph_conn') or self.graph_conn is None:
                    self.logger.error("Knowledge graph connection is not initialized.")
                    return False
                # Delete relation
                await self.graph_conn.execute("DELETE FROM relations WHERE id = ?", (relation_id,))
                await self.graph_conn.commit()
                self._pending_changes = True
                
                self.logger.debug(f"Deleted relation from knowledge graph: {relation_id}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete relation: {e}", exc_info=True)
                await self.graph_conn.rollback()
                return False

    #--------------------------------------------------------------------
    # Document Store Methods
    #--------------------------------------------------------------------
    
    async def _init_document_store(self) -> None:
        """Initialize the document store."""
        try:
            # Create document directory
            self.document_path.mkdir(parents=True, exist_ok=True)
            
            # Create document index file
            index_file = self.document_path / "document_index.json"
            if index_file.exists():
                # Load existing index
                with open(index_file, 'r') as f:
                    self.document_index = json.load(f)
                self.logger.info(f"Loaded document index with {len(self.document_index)} entries")
            else:
                # Create new index
                self.document_index = {}
                with open(index_file, 'w') as f:
                    json.dump(self.document_index, f)
                self.logger.info("Created new document index")
            
            self.logger.info("Document store initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize document store: {e}", exc_info=True)
            raise
    
    async def store_document(self, 
                            filename: str, 
                            content: Union[bytes, str], 
                            content_type: str,
                            metadata: Union[Dict, MemoryMetadata]) -> str:
        """Store document with metadata."""
        async with self._document_lock:
            try:
                # Convert metadata to dict if needed
                if isinstance(metadata, MemoryMetadata):
                    meta_dict = metadata.to_dict()
                else:
                    meta_dict = metadata
                
                # Generate unique ID
                timestamp = time.time()
                name_hash = hashlib.md5(f"{filename}:{timestamp}".encode()).hexdigest()
                doc_id = f"doc_{name_hash}"
                
                # Ensure valid filename
                safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
                if not safe_filename:
                    safe_filename = f"document_{doc_id}"
                
                # Create document subdirectory
                doc_dir = self.document_path / doc_id
                doc_dir.mkdir(exist_ok=True)
                
                # Determine file path
                file_path = doc_dir / safe_filename
                
                # Write content
                if isinstance(content, str):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                else:
                    with open(file_path, 'wb') as f:
                        f.write(content)
                
                # Get file size
                file_size = file_path.stat().st_size
                
                # Update document index
                self.document_index[doc_id] = {
                    "id": doc_id,
                    "filename": safe_filename,
                    "content_type": content_type,
                    "path": str(file_path.relative_to(self.document_path)),
                    "size": file_size,
                    "metadata": meta_dict
                }
                
                # Save index
                index_file = self.document_path / "document_index.json"
                with open(index_file, 'w') as f:
                    json.dump(self.document_index, f)
                
                self._pending_changes = True
                self.logger.debug(f"Stored document: {doc_id} ({safe_filename})")
                return doc_id
            except Exception as e:
                self.logger.error(f"Failed to store document: {e}", exc_info=True)
                return ""
    
    async def get_document(self, doc_id: str, as_text: bool = False) -> Optional[Tuple[Any, DocumentEntry]]:
        """Retrieve document and its metadata."""
        try:
            # Check if document exists
            if doc_id not in self.document_index:
                return None
            
            # Get document info
            doc_info = self.document_index[doc_id]
            
            # Create DocumentEntry
            doc_entry = DocumentEntry(
                id=doc_info["id"],
                filename=doc_info["filename"],
                content_type=doc_info["content_type"],
                metadata=doc_info["metadata"],
                path=doc_info["path"],
                size=doc_info["size"]
            )
            
            # Get file path
            file_path = self.document_path / doc_info["path"]
            
            # Read content
            if as_text:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(file_path, 'rb') as f:
                    content = f.read()
            
            # Update access timestamp
            if isinstance(doc_info["metadata"], dict):
                doc_info["metadata"]["last_accessed"] = time.time()
                if "access_count" in doc_info["metadata"]:
                    doc_info["metadata"]["access_count"] += 1
                else:
                    doc_info["metadata"]["access_count"] = 1
                
                # Save index (in background)
                asyncio.create_task(self._save_document_index())
                self._pending_changes = True
            
            return (content, doc_entry)
        except Exception as e:
            self.logger.error(f"Failed to get document: {e}", exc_info=True)
            return None
    
    async def _save_document_index(self) -> None:
        """Save document index to disk."""
        try:
            index_file = self.document_path / "document_index.json"
            with open(index_file, 'w') as f:
                json.dump(self.document_index, f)
        except Exception as e:
            self.logger.error(f"Failed to save document index: {e}", exc_info=True)
    
    async def query_documents(self, 
                             metadata_filter: Optional[Dict] = None,
                             content_type: Optional[str] = None,
                             limit: int = 10) -> List[DocumentEntry]:
        """Query documents with metadata filters."""
        try:
            results = []
            count = 0
            
            # Iterate through document index
            for doc_id, doc_info in self.document_index.items():
                # Apply content type filter
                if content_type and doc_info["content_type"] != content_type:
                    continue
                
                # Apply metadata filters
                if metadata_filter:
                    match = True
                    for key, value in metadata_filter.items():
                        if key not in doc_info["metadata"] or doc_info["metadata"][key] != value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                # Create DocumentEntry
                doc_entry = DocumentEntry(
                    id=doc_info["id"],
                    filename=doc_info["filename"],
                    content_type=doc_info["content_type"],
                    metadata=doc_info["metadata"],
                    path=doc_info["path"],
                    size=doc_info["size"]
                )
                results.append(doc_entry)
                
                count += 1
                if count >= limit:
                    break
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to query documents: {e}", exc_info=True)
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document and its metadata."""
        async with self._document_lock:
            try:
                # Check if document exists
                if doc_id not in self.document_index:
                    return False
                
                # Get document info
                doc_info = self.document_index[doc_id]
                
                # Get file path
                file_path = self.document_path / doc_info["path"]
                
                # Delete file
                if file_path.exists():
                    if file_path.is_file():
                        file_path.unlink()
                    else:
                        shutil.rmtree(file_path)
                
                # Delete parent directory if it exists
                parent_dir = file_path.parent
                if parent_dir.exists() and parent_dir.is_dir():
                    try:
                        shutil.rmtree(parent_dir)
                    except Exception:
                        pass
                
                # Remove from index
                del self.document_index[doc_id]
                
                # Save index
                index_file = self.document_path / "document_index.json"
                with open(index_file, 'w') as f:
                    json.dump(self.document_index, f)
                
                self._pending_changes = True
                self.logger.debug(f"Deleted document: {doc_id}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete document: {e}", exc_info=True)
                return False

    #--------------------------------------------------------------------
    # Parameter Store Methods
    #--------------------------------------------------------------------
    
    async def _init_parameter_store(self) -> None:
        """Initialize the parameter store."""
        try:
            # Create parameter directory
            self.parameter_path.mkdir(parents=True, exist_ok=True)
            
            # Load all parameter files
            for param_file in self.parameter_path.glob("*.json"):
                try:
                    with open(param_file, 'r') as f:
                        params = json.load(f)
                    
                    namespace = param_file.stem
                    self.parameter_store[namespace] = params
                    self.logger.debug(f"Loaded parameter namespace: {namespace}")
                except Exception as e:
                    self.logger.warning(f"Failed to load parameter file {param_file}: {e}")
            
            self.logger.info(f"Parameter store initialized with {len(self.parameter_store)} namespaces")
        except Exception as e:
            self.logger.error(f"Failed to initialize parameter store: {e}", exc_info=True)
            raise
    
    async def set_parameter(self, namespace: str, key: str, value: Any) -> bool:
        """Set parameter value in namespace."""
        async with self._parameter_lock:
            try:
                # Create namespace if it doesn't exist
                if namespace not in self.parameter_store:
                    self.parameter_store[namespace] = {}
                
                # Set parameter
                self.parameter_store[namespace][key] = value
                
                # Save namespace
                await self._save_parameter_namespace(namespace)
                
                self._pending_changes = True
                return True
            except Exception as e:
                self.logger.error(f"Failed to set parameter: {e}", exc_info=True)
                return False
    
    async def get_parameter(self, namespace: str, key: str, default: Any = None) -> Any:
        """Get parameter value from namespace."""
        try:
            # Check if namespace exists
            if namespace not in self.parameter_store:
                return default
            
            # Check if key exists
            if key not in self.parameter_store[namespace]:
                return default
            
            # Return value
            return self.parameter_store[namespace][key]
        except Exception as e:
            self.logger.error(f"Failed to get parameter: {e}", exc_info=True)
            return default
    
    async def get_namespace(self, namespace: str) -> Dict:
        """Get all parameters in namespace."""
        try:
            # Check if namespace exists
            if namespace not in self.parameter_store:
                return {}
            
            # Return namespace
            return self.parameter_store[namespace].copy()
        except Exception as e:
            self.logger.error(f"Failed to get namespace: {e}", exc_info=True)
            return {}
    
    async def delete_parameter(self, namespace: str, key: str) -> bool:
        """Delete parameter from namespace."""
        async with self._parameter_lock:
            try:
                # Check if namespace exists
                if namespace not in self.parameter_store:
                    return False
                
                # Check if key exists
                if key not in self.parameter_store[namespace]:
                    return False
                
                # Delete parameter
                del self.parameter_store[namespace][key]
                
                # Save namespace
                await self._save_parameter_namespace(namespace)
                
                self._pending_changes = True
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete parameter: {e}", exc_info=True)
                return False
    
    async def delete_namespace(self, namespace: str) -> bool:
        """Delete entire parameter namespace."""
        async with self._parameter_lock:
            try:
                # Check if namespace exists
                if namespace not in self.parameter_store:
                    return False
                
                # Delete namespace
                del self.parameter_store[namespace]
                
                # Delete namespace file
                namespace_file = self.parameter_path / f"{namespace}.json"
                if namespace_file.exists():
                    namespace_file.unlink()
                
                self._pending_changes = True
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete namespace: {e}", exc_info=True)
                return False
    
    async def _save_parameter_namespace(self, namespace: str) -> None:
        """Save parameter namespace to disk."""
        try:
            # Get namespace data
            namespace_data = self.parameter_store.get(namespace, {})
            
            # Save to file
            namespace_file = self.parameter_path / f"{namespace}.json"
            with open(namespace_file, 'w') as f:
                json.dump(namespace_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save parameter namespace: {e}", exc_info=True)

    #--------------------------------------------------------------------
    # Memory Management Methods
    #--------------------------------------------------------------------
    
    async def _memory_maintenance_loop(self) -> None:
        """Background task for memory maintenance."""
        self.logger.info("Starting memory maintenance loop")
        
        while self._running:
            try:
                # Check if it's time to commit
                current_time = time.time()
                if self._pending_changes and current_time - self._last_commit_time >= self._commit_interval:
                    await self._commit_memory_changes()
                
                # Check if it's time for consolidation
                last_consolidation = await self.get_parameter("system", "last_memory_consolidation", 0)
                if current_time - last_consolidation >= self.consolidation_interval:
                    await self._consolidate_memory()
                    await self.set_parameter("system", "last_memory_consolidation", current_time)
                
                # Check storage limits
                await self._check_storage_limits()
                
                # Sleep for a while
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.logger.info("Memory maintenance loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in memory maintenance loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def _commit_memory_changes(self) -> None:
        """Commit pending memory changes."""
        if not self._pending_changes:
            return
            
        try:
            self.logger.debug("Committing memory changes")
            
            # Commit knowledge graph changes
            try:
                if hasattr(self, 'graph_conn') and self.graph_conn is not None:
                    await self.graph_conn.commit()
            except Exception as e:
                self.logger.warning(f"Failed to commit knowledge graph: {e}")
            
            # Save document index
            try:
                await self._save_document_index()
            except Exception as e:
                self.logger.warning(f"Failed to save document index: {e}")
            
            # Reset tracking
            self._last_commit_time = time.time()
            self._pending_changes = False
            
            self.logger.debug("Memory changes committed")
        except Exception as e:
            self.logger.error(f"Failed to commit memory changes: {e}", exc_info=True)
    
    async def _consolidate_memory(self) -> None:
        """Consolidate memory for long-term retention."""
        self.logger.info("Starting memory consolidation")
        
        try:
            # Consolidate vector store
            if LANCEDB_AVAILABLE and self.vector_db:
                # Get low importance memories for pruning
                try:
                    # Find semantic memories with low importance
                    low_importance_semantic = await asyncio.to_thread(
                        self.semantic_table.search(np.zeros(self.embedding_dimension).tolist())
                        .where(f"metadata->>'importance' < {self.importance_threshold}")
                        .sort("metadata->>'last_accessed'", "ASC")
                        .limit(100)
                        .to_list
                    )
                    
                    # Find episodic memories with low importance
                    low_importance_episodic = await asyncio.to_thread(
                        self.episodic_table.search(np.zeros(self.embedding_dimension).tolist())
                        .where(f"metadata->>'importance' < {self.importance_threshold}")
                        .sort("metadata->>'last_accessed'", "ASC")
                        .limit(100)
                        .to_list
                    )
                    
                    # Combine and process for pruning
                    for item in low_importance_semantic:
                        if item["id"] == "init":
                            continue
                        
                        # Check if expired
                        metadata = item["metadata"]
                        if metadata.get("expiration") and float(metadata["expiration"]) < time.time():
                            await self.delete_from_vector_store(item["id"], MEMORY_TYPE_SEMANTIC)
                    
                    for item in low_importance_episodic:
                        if item["id"] == "init":
                            continue
                        
                        # Check if expired
                        metadata = item["metadata"]
                        if metadata.get("expiration") and float(metadata["expiration"]) < time.time():
                            await self.delete_from_vector_store(item["id"], MEMORY_TYPE_EPISODIC)
                except Exception as e:
                    self.logger.warning(f"Failed to consolidate vector store: {e}")
            
            # Consolidate knowledge graph
            try:
                # Find old, low-importance entities for pruning
                cutoff_time = time.time() - 60 * 60 * 24 * 30  # 30 days
                
                async with self.graph_conn.execute(
                    f"""
                    SELECT id FROM entities 
                    WHERE last_accessed < ? 
                    AND json_extract(metadata, '$.importance') < ?
                    LIMIT 100
                    """,
                    (cutoff_time, self.importance_threshold)
                ) as cursor:
                    rows = await cursor.fetchall()
                    
                    for row in rows:
                        entity_id = row[0]
                        
                        # Check if this entity has important relations before deleting
                        has_important = False
                        
                        async with self.graph_conn.execute(
                            f"""
                            SELECT COUNT(*) FROM relations 
                            WHERE (source_id = ? OR target_id = ?)
                            AND json_extract(metadata, '$.importance') >= ?
                            """,
                            (entity_id, entity_id, self.importance_threshold)
                        ) as rel_cursor:
                            rel_count = await rel_cursor.fetchone()
                            if rel_count and rel_count[0] > 0:
                                has_important = True
                        
                        if not has_important:
                            await self.delete_entity(entity_id)
            except Exception as e:
                self.logger.warning(f"Failed to consolidate knowledge graph: {e}")
            
            # Consolidate documents
            try:
                to_delete = []
                
                for doc_id, doc_info in self.document_index.items():
                    metadata = doc_info.get("metadata", {})
                    
                    # Check importance and last access
                    importance = metadata.get("importance", 0.5)
                    last_accessed = metadata.get("last_accessed", 0)
                    
                    # Check for expiration
                    if metadata.get("expiration") and float(metadata["expiration"]) < time.time():
                        to_delete.append(doc_id)
                        continue
                    
                    # Check for importance and age
                    if importance < self.importance_threshold:
                        cutoff_time = time.time() - 60 * 60 * 24 * 60  # 60 days
                        if last_accessed < cutoff_time:
                            to_delete.append(doc_id)
                
                # Delete documents
                for doc_id in to_delete:
                    await self.delete_document(doc_id)
            except Exception as e:
                self.logger.warning(f"Failed to consolidate documents: {e}")
            
            self.logger.info("Memory consolidation completed")
        except Exception as e:
            self.logger.error(f"Error during memory consolidation: {e}", exc_info=True)
    
    async def _check_storage_limits(self) -> None:
        """Check storage usage and enforce limits."""
        try:
            # Calculate total size
            total_size_bytes = 0
            
            # Check vector store size
            vector_size = sum(f.stat().st_size for f in self.vector_path.glob('**/*') if f.is_file())
            total_size_bytes += vector_size
            
            # Check knowledge graph size
            graph_size = sum(f.stat().st_size for f in self.graph_path.glob('**/*') if f.is_file())
            total_size_bytes += graph_size
            
            # Check document store size
            doc_size = sum(f.stat().st_size for f in self.document_path.glob('**/*') if f.is_file())
            total_size_bytes += doc_size
            
            # Check parameter store size
            param_size = sum(f.stat().st_size for f in self.parameter_path.glob('**/*') if f.is_file())
            total_size_bytes += param_size
            
            # Convert to GB
            total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
            
            # Check against limit
            if total_size_gb > self.max_storage_gb:
                self.logger.warning(
                    f"Storage limit exceeded: {total_size_gb:.2f}GB > {self.max_storage_gb}GB. "
                    f"Triggering aggressive memory consolidation."
                )
                
                # Lower importance threshold temporarily
                old_threshold = self.importance_threshold
                self.importance_threshold = 0.6  # Higher threshold for more aggressive pruning
                
                # Run consolidation
                await self._consolidate_memory()
                
                # Restore original threshold
                self.importance_threshold = old_threshold
        except Exception as e:
            self.logger.error(f"Error checking storage limits: {e}", exc_info=True)

    #--------------------------------------------------------------------
    # Utility Methods
    #--------------------------------------------------------------------
    
    def _create_directories(self) -> None:
        """Create necessary data directories if they don't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.vector_path.mkdir(parents=True, exist_ok=True)
        self.graph_path.mkdir(parents=True, exist_ok=True)
        self.document_path.mkdir(parents=True, exist_ok=True)
        self.parameter_path.mkdir(parents=True, exist_ok=True)
    
    async def semantic_search(self, 
                             query: str, 
                             limit: int = 5,
                             memory_type: str = MEMORY_TYPE_SEMANTIC,
                             metadata_filter: Optional[Dict] = None) -> MemoryQueryResult:
        """Search semantic memory using vector similarity."""
        # This is just a wrapper around search_vector_store
        return await self.search_vector_store(query, limit, memory_type, metadata_filter)
    
    async def create_memory(self, 
                           content: str, 
                           memory_type: str = MEMORY_TYPE_SEMANTIC,
                           metadata: Optional[Dict] = None,
                           importance: float = 0.5) -> str:
        """Create a new memory entry with appropriate metadata."""
        # Create metadata
        if metadata is None:
            metadata = {}
            
        meta = MemoryMetadata(
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
            importance=importance,
            confidence=1.0,
            memory_type=memory_type,
            source="system",
            tags=metadata.get("tags", []),
            expiration=metadata.get("expiration"),
            associations=metadata.get("associations", []),
            spatial_context=metadata.get("spatial_context"),
            temporal_context=metadata.get("temporal_context")
        )
        
        # Add any additional metadata
        for key, value in metadata.items():
            if not hasattr(meta, key):  # Only add keys not already in MemoryMetadata
                setattr(meta, key, value)
        
        # Store memory based on type
        if memory_type in [MEMORY_TYPE_SEMANTIC, MEMORY_TYPE_EPISODIC, MEMORY_TYPE_REFLECTIVE]:
            return await self.add_to_vector_store(content, meta, memory_type)
        else:
            # For other types, default to semantic
            return await self.add_to_vector_store(content, meta, MEMORY_TYPE_SEMANTIC)
    
    async def remember(self, 
                      query: str, 
                      memory_types: List[str] = None,
                      limit: int = 5,
                      metadata_filter: Optional[Dict] = None) -> MemoryQueryResult:
        """Retrieve memories related to query across multiple memory types."""
        if memory_types is None:
            memory_types = [MEMORY_TYPE_SEMANTIC, MEMORY_TYPE_EPISODIC]
        
        results = []
        total_found = 0
        start_time = time.time()
        
        for memory_type in memory_types:
            # Search vector store for this memory type
            result = await self.search_vector_store(
                query=query,
                limit=limit,
                memory_type=memory_type,
                metadata_filter=metadata_filter
            )
            
            results.extend(result.results)
            total_found += result.total_found
        
        # Sort by relevance score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit results
        results = results[:limit]
        
        end_time = time.time()
        query_time_ms = (end_time - start_time) * 1000
        
        return MemoryQueryResult(
            results=results,
            total_found=total_found,
            query_time_ms=query_time_ms
        )
    
    async def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        # Try to delete from all possible stores
        if memory_id.startswith("semantic_") or memory_id.startswith("episodic_"):
            memory_type = MEMORY_TYPE_SEMANTIC
            if memory_id.startswith("episodic_"):
                memory_type = MEMORY_TYPE_EPISODIC
                
            return await self.delete_from_vector_store(memory_id, memory_type)
        elif memory_id.startswith("entity_"):
            return await self.delete_entity(memory_id)
        elif memory_id.startswith("relation_"):
            return await self.delete_relation(memory_id)
        elif memory_id.startswith("doc_"):
            return await self.delete_document(memory_id)
        else:
            # Unknown ID format
            self.logger.warning(f"Unknown memory ID format: {memory_id}")
            return False
    
    async def search(self, 
                    query: str,
                    search_vectors: bool = True,
                    search_entities: bool = True,
                    search_documents: bool = False,
                    limit: int = 5) -> Dict[str, MemoryQueryResult]:
        """Comprehensive search across all memory stores."""
        results = {}
        start_time = time.time()
        
        # Search vectors if enabled
        if search_vectors:
            # Search semantic memories
            semantic_result = await self.search_vector_store(
                query=query,
                limit=limit,
                memory_type=MEMORY_TYPE_SEMANTIC
            )
            results["semantic"] = semantic_result
            
            # Search episodic memories
            episodic_result = await self.search_vector_store(
                query=query,
                limit=limit,
                memory_type=MEMORY_TYPE_EPISODIC
            )
            results["episodic"] = episodic_result
        
        # Search knowledge graph if enabled
        if search_entities:
            entity_results = []
            
            # Use simple keyword matching for entities
            words = query.lower().split()
            for word in words:
                if len(word) < 3:
                    continue
                    
                # Search for entities with properties containing this word
                entities = await self.query_entities(
                    property_filters=None,
                    limit=limit
                )
                
                for entity in entities:
                    # Check if any property values contain the word
                    match = False
                    for prop, value in entity.properties.items():
                        if isinstance(value, str) and word in value.lower():
                            match = True
                            break
                    
                    if match:
                        metadata = MemoryMetadata.from_dict(entity.metadata)
                        entity_results.append(
                            MemoryResult(
                                content=entity,
                                metadata=metadata,
                                score=0.7,  # Default relevance score for keyword match
                                source="graph",
                                memory_type="entity"
                            )
                        )
            
            # Create result container
            end_time = time.time()
            graph_time_ms = (end_time - start_time) * 1000
            results["entities"] = MemoryQueryResult(
                results=entity_results[:limit],
                total_found=len(entity_results),
                query_time_ms=graph_time_ms
            )
        
        # Search documents if enabled
        if search_documents:
            # This is simplified keyword matching
            # A real implementation would use text extraction and indexing
            doc_results = []
            
            # Query all documents and filter
            documents = await self.query_documents(limit=100)
            
            for doc in documents:
                # Check filename for matches
                if query.lower() in doc.filename.lower():
                    metadata = MemoryMetadata.from_dict(doc.metadata)
                    doc_results.append(
                        MemoryResult(
                            content=doc,
                            metadata=metadata,
                            score=0.6,  # Default relevance score for filename match
                            source="document",
                            memory_type="document"
                        )
                    )
            
            # Create result container
            end_time = time.time()
            doc_time_ms = (end_time - start_time) * 1000
            results["documents"] = MemoryQueryResult(
                results=doc_results[:limit],
                total_found=len(doc_results),
                query_time_ms=doc_time_ms
            )
        
        return results

# Simple test code when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the ORAMA Memory Engine")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()
    
    async def test_memory_engine():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Create test config
        config = {
            "path": args.data_dir,
            "vector": {
                "path": "vector_store",
                "dimension": 768
            },
            "graph": {
                "path": "knowledge_graph"
            },
            "document": {
                "path": "documents"
            },
            "parameters": {
                "path": "parameters"
            },
            "management": {
                "importance_threshold": 0.3,
                "consolidation_interval": 86400,
                "max_storage_gb": 10
            }
        }
        
        # Create memory engine
        memory = MemoryEngine(config)
        await memory.start()
        
        try:
            # Test vector store
            print("Testing vector store...")
            semantic_id = await memory.create_memory(
                "The ORAMA system is an autonomous agent architecture.",
                memory_type=MEMORY_TYPE_SEMANTIC,
                importance=0.8
            )
            print(f"Added semantic memory: {semantic_id}")
            
            episodic_id = await memory.create_memory(
                "Today I learned how to create a persistent memory system.",
                memory_type=MEMORY_TYPE_EPISODIC,
                importance=0.7
            )
            print(f"Added episodic memory: {episodic_id}")
            
            # Test search
            search_result = await memory.remember("memory system")
            print(f"Found {search_result.total_found} memories in {search_result.query_time_ms:.2f}ms")
            for i, result in enumerate(search_result.results):
                print(f"  {i+1}. [{result.score:.2f}] {result.content}")
            
            # Test knowledge graph
            print("\nTesting knowledge graph...")
            entity_id = await memory.add_entity(
                "concept",
                {"name": "Memory", "description": "Storage and retrieval of information"},
                MemoryMetadata(importance=0.9)
            )
            print(f"Added entity: {entity_id}")
            
            entity2_id = await memory.add_entity(
                "concept",
                {"name": "Vector", "description": "Mathematical representation of data"},
                MemoryMetadata(importance=0.85)
            )
            print(f"Added entity: {entity2_id}")
            
            relation_id = await memory.add_relation(
                entity_id,
                entity2_id,
                "uses",
                {"context": "For semantic search"},
                MemoryMetadata(importance=0.8)
            )
            print(f"Added relation: {relation_id}")
            
            # Test entity retrieval
            entity = await memory.get_entity(entity_id)
            if entity:
                print(f"Retrieved entity: {entity.type} - {entity.properties.get('name', '')}")
                
                # Get relations
                relations = await memory.get_entity_relations(entity_id)
                print(f"Entity has {len(relations)} relations")
                for rel in relations:
                    print(f"  {rel.type}: {entity_id} -> {rel.target_id}")
            
            # Test document store
            print("\nTesting document store...")
            doc_id = await memory.store_document(
                "test_document.txt",
                "This is a test document for the ORAMA memory system.",
                "text/plain",
                MemoryMetadata(importance=0.75)
            )
            print(f"Stored document: {doc_id}")
            
            # Retrieve document
            doc_result = await memory.get_document(doc_id, as_text=True)
            if doc_result:
                content, doc_info = doc_result
                print(f"Retrieved document: {doc_info.filename}")
                print(f"Content: {content[:50]}...")
            
            # Test parameter store
            print("\nTesting parameter store...")
            await memory.set_parameter("system", "test_param", "test_value")
            await memory.set_parameter("user", "name", "ORAMA User")
            
            param_value = await memory.get_parameter("system", "test_param")
            print(f"Retrieved parameter: system.test_param = {param_value}")
            
            # Test comprehensive search
            print("\nTesting comprehensive search...")
            search_results = await memory.search("memory", 
                                              search_vectors=True,
                                              search_entities=True,
                                              search_documents=True)
            
            for store_name, result in search_results.items():
                print(f"{store_name}: Found {result.total_found} results")
                for item in result.results:
                    if store_name == "entities":
                        entity_content = item.content
                        print(f"  [{item.score:.2f}] Entity: {entity_content.properties.get('name', '')}")
                    elif store_name == "documents":
                        doc_content = item.content
                        print(f"  [{item.score:.2f}] Document: {doc_content.filename}")
                    else:
                        print(f"  [{item.score:.2f}] {item.content[:50]}...")
        finally:
            # Shutdown
            await memory.stop()
            print("\nMemory engine stopped")
    
    # Run test
    asyncio.run(test_memory_engine())
