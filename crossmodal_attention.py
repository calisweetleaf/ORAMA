# crossmodal_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union, Any
import math
import logging
from contextlib import contextmanager
import time
from dataclasses import dataclass, field
from enum import Enum, auto
import warnings
from functools import lru_cache
import collections
import json
import threading
import numpy as np
from collections import defaultdict, deque
import weakref
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class AttentionError(Exception):
    """Base exception for attention-related errors."""
    def __init__(self, message: str, error_code: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{error_code}] {message}")

class InvalidTensorError(AttentionError):
    """Raised when input tensors have invalid shapes or types."""
    def __init__(self, message: str):
        super().__init__(message, "INVALID_TENSOR")

class MemoryError(AttentionError):
    """Raised when memory constraints are violated."""
    def __init__(self, message: str):
        super().__init__(message, "MEMORY_ERROR")

@dataclass
class AttentionMetrics:
    """Metrics for monitoring attention performance."""
    attention_scores_mean: float
    attention_scores_std: float
    attention_entropy: float
    compute_time_ms: float
    memory_peak_mb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "attention_scores_mean": self.attention_scores_mean,
            "attention_scores_std": self.attention_scores_std,
            "attention_entropy": self.attention_entropy,
            "compute_time_ms": self.compute_time_ms,
            "memory_peak_mb": self.memory_peak_mb
        }

class AttentionStreamType(Enum):
    """Types of attention streams for multi-stream processing."""
    PRIMARY = auto()     # Main task-driving attention
    AUXILIARY = auto()   # Background context augmentation
    MONITORING = auto()  # Passive monitoring of additional streams
    ANALYSIS = auto()    # Deep analysis of specific content
    INTEGRATION = auto() # Cross-stream integration
    TEMPORAL = auto()    # Time-sensitive processing

@dataclass
class StreamConfig:
    """Configuration for an attention stream."""
    stream_type: AttentionStreamType
    priority: float = 1.0  # 0.0 to 1.0, higher = more important
    dynamic_priority: bool = False # If True, priority can change during processing
    decay_rate: float = 0.0  # How quickly stream importance decays
    refresh_factor: float = 0.2  # How much to boost on new content
    modalities: List[str] = field(default_factory=list)  # Which modalities to process
    alpha_depth: float = 1.0  # Attention depth factor (1.0 = normal)

@dataclass
class MemoryItem:
    """An item in persistent memory."""
    content: torch.Tensor
    key: torch.Tensor  # Search key for efficient retrieval
    modality: str
    importance: float = 1.0
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # Time-to-live in seconds, None for permanent
    
    def update_access(self):
        """Update access statistics."""
        self.last_access_time = time.time()
        self.access_count += 1

class TemporalEvent:
    """Represents a specific event across modalities for synchronization."""
    
    def __init__(
        self, 
        name: str, 
        modalities: List[str], 
        features: torch.Tensor, 
        timestamp: float
    ):
        self.name = name
        self.modalities = modalities
        self.features = features
        self.timestamp = timestamp
        self.references: Dict[str, List[str]] = defaultdict(list)
        
    def add_reference(self, modality: str, reference_id: str):
        """Add a reference to this event from a specific modality."""
        self.references[modality].append(reference_id)
        
    def get_modality_overlap(self) -> float:
        """Calculate how many modalities this event spans (0-1)."""
        return len(self.references) / len(self.modalities) if self.modalities else 0

class PersistentMemory:
    """Long-term cross-modal memory with intelligent retrieval."""
    
    def __init__(
        self, 
        capacity: int = 1000,
        dim: int = 4096,
        ttl: Optional[float] = None,  # Default TTL in seconds, None for no expiry
        device: Optional[torch.device] = None
    ):
        self.capacity = capacity
        self.dim = dim
        self.default_ttl = ttl
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Memory storage
        self.items: Dict[str, MemoryItem] = {}  # id -> MemoryItem
        self.modality_items: Dict[str, List[str]] = defaultdict(list)  # modality -> [id, id, ...]
        
        # Indexing for efficient search
        self.keys = torch.zeros((0, dim), device=self.device)
        self.key_to_id: Dict[int, str] = {}  # Row index -> item ID
        
        # Usage statistics
        self.stats = {
            'total_retrievals': 0,
            'total_additions': 0,
            'total_evictions': 0,
            'hit_rate': 0.0,
        }
        
        self._lock = threading.RLock()
    
    def add(
        self, 
        content: torch.Tensor, 
        modality: str,
        key: Optional[torch.Tensor] = None,
        importance: float = 1.0,
        ttl: Optional[float] = None
    ) -> str:
        """
        Add a new item to memory.
        
        Args:
            content: Content tensor to store
            modality: Modality of the content
            key: Optional search key, defaults to content
            importance: Item importance (higher = less likely to be evicted)
            ttl: Time-to-live in seconds, None for default
            
        Returns:
            ID of the stored item
        """
        with self._lock:
            # Generate ID
            item_id = str(uuid.uuid4())
            
            # Use content as key if not provided
            if key is None:
                key = content.detach().float().mean(dim=0) if content.dim() > 1 else content.detach().float()
            
            # Normalize key for cosine similarity
            key = F.normalize(key, p=2, dim=0)
            
            # Create memory item
            item = MemoryItem(
                content=content.detach().to(self.device),
                key=key.to(self.device),
                modality=modality,
                importance=importance,
                ttl=ttl if ttl is not None else self.default_ttl
            )
            
            # If at capacity, evict something
            if len(self.items) >= self.capacity:
                self._evict_item()
            
            # Store the item
            self.items[item_id] = item
            self.modality_items[modality].append(item_id)
            
            # Update index
            self.keys = torch.cat([self.keys, key.unsqueeze(0)], dim=0)
            self.key_to_id[self.keys.size(0) - 1] = item_id
            
            self.stats['total_additions'] += 1
            
            return item_id
    
    def retrieve(
        self, 
        query: torch.Tensor, 
        modality: Optional[str] = None, 
        top_k: int = 5,
        threshold: float = 0.6
    ) -> List[Tuple[str, torch.Tensor, float]]:
        """
        Retrieve items from memory based on query.
        
        Args:
            query: Query vector
            modality: Optional filter by modality
            top_k: Number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (item_id, content, similarity) tuples
        """
        with self._lock:
            self.stats['total_retrievals'] += 1
            
            if len(self.items) == 0:
                return []
            
            # Clean expired items first
            self._clean_expired()
            
            # Filter by modality if specified
            if modality is not None:
                item_ids = self.modality_items.get(modality, [])
                if not item_ids:
                    return []
                    
                # Get keys for items of this modality
                key_indices = [i for i, item_id in self.key_to_id.items() 
                              if item_id in item_ids]
                keys = self.keys[key_indices]
                id_map = {i: self.key_to_id[key_indices[i]] for i in range(len(key_indices))}
            else:
                keys = self.keys
                id_map = self.key_to_id
            
            if keys.size(0) == 0:
                return []
            
            # Normalize query
            query = F.normalize(query.to(self.device), p=2, dim=0)
            
            # Compute similarities
            similarities = torch.matmul(query.unsqueeze(0), keys.T).squeeze(0)
            
            # Filter by threshold
            above_threshold = similarities >= threshold
            
            if not above_threshold.any():
                return []
            
            # Get top-k
            values, indices = torch.topk(
                similarities * above_threshold, 
                min(top_k, int(above_threshold.sum().item()))  # Cast to int
            )
            
            results = []
            for i, idx in enumerate(indices):
                if values[i] < threshold:
                    continue
                    
                item_id = id_map[int(idx.item())]  # Cast to int
                item = self.items[item_id]
                item.update_access()  # Update access statistics
                
                results.append((item_id, item.content, values[i].item()))
            
            # Update hit rate
            if results:
                hits = len(results)
                total = self.stats['total_retrievals']
                self.stats['hit_rate'] = (self.stats['hit_rate'] * (total - 1) + hits / top_k) / total
            
            return results
    
    def _evict_item(self) -> None:
        """Evict least important item from memory."""
        with self._lock:
            if not self.items:
                return
            
            # Compute score for each item
            scores = {}
            curr_time = time.time()
            
            for item_id, item in self.items.items():
                # Base score on importance
                score = item.importance
                
                # Adjust by recency
                time_factor = 1.0 / (1.0 + (curr_time - item.last_access_time) / 3600)  # Hours
                score *= time_factor
                
                # Adjust by access count (more = higher value)
                access_factor = math.log1p(item.access_count) / 10
                score *= (1.0 + access_factor)
                
                scores[item_id] = score
            
            # Find item with lowest score - fix the min function usage
            evict_id = min(scores.keys(), key=lambda k: scores[k])
            
            # Get modality and key index
            item = self.items[evict_id]
            modality = item.modality
            
            # Remove from storage
            del self.items[evict_id]
            self.modality_items[modality].remove(evict_id)
            
            # Rebuild key index (inefficient but simple)
            new_keys = []
            new_key_to_id = {}
            
            for i, (idx, item_id) in enumerate(self.key_to_id.items()):
                if item_id != evict_id:
                    new_keys.append(self.keys[idx].unsqueeze(0))
                    new_key_to_id[i] = item_id
            
            if new_keys:
                self.keys = torch.cat(new_keys, dim=0)
            else:
                self.keys = torch.zeros((0, self.dim), device=self.device)
                
            self.key_to_id = new_key_to_id
            self.stats['total_evictions'] += 1

    def _clean_expired(self) -> int:
        """
        Remove expired items from memory.
        
        Returns:
            Number of items removed
        """
        with self._lock:
            curr_time = time.time()
            expired_ids = []
            
            for item_id, item in self.items.items():
                if item.ttl and curr_time - item.creation_time > item.ttl:
                    expired_ids.append(item_id)
            
            for item_id in expired_ids:
                modality = self.items[item_id].modality
                del self.items[item_id]
                self.modality_items[modality].remove(item_id)
            
            # Rebuild key index if items were expired
            if expired_ids:
                new_keys = []
                new_key_to_id = {}
                
                for i, (idx, item_id) in enumerate(self.key_to_id.items()):
                    if item_id not in expired_ids:
                        new_keys.append(self.keys[idx].unsqueeze(0))
                        new_key_to_id[i] = item_id
                
                if new_keys:
                    self.keys = torch.cat(new_keys, dim=0)
                else:
                    self.keys = torch.zeros((0, self.dim), device=self.device)
                    
                self.key_to_id = new_key_to_id
                self.stats['total_evictions'] += len(expired_ids)
            
            return len(expired_ids)

class ReasoningPathTracker:
    """Tracks attention reasoning paths for explainability."""
    
    def __init__(self, max_paths: int = 10, prune_threshold: float = 0.1):
        self.max_paths = max_paths
        self.prune_threshold = prune_threshold
        self.paths = []
        self.current_path = []
        
    def add_step(self, 
                 query_tokens: List[str], 
                 context_tokens: List[str],
                 attention_weights: torch.Tensor,
                 threshold: float = 0.2):
        """
        Add reasoning step to the current path.
        
        Args:
            query_tokens: List of query token strings
            context_tokens: List of context token strings
            attention_weights: Attention weights [batch, heads, q_len, kv_len]
            threshold: Minimum attention weight to include
        """
        # Average across heads
        avg_weights = attention_weights.mean(dim=1).squeeze(0)  # [q_len, kv_len]
        
        for q_idx in range(min(avg_weights.shape[0], len(query_tokens))):
            q_token = query_tokens[q_idx]
            
            # Get top attended context tokens
            weights = avg_weights[q_idx]
            values, indices = torch.topk(weights, min(5, weights.size(0)))
            
            connections = []
            for i, idx in enumerate(indices):
                if values[i] >= threshold:
                    if idx < len(context_tokens):
                        connections.append((context_tokens[idx], values[i].item()))
            
            if connections:
                self.current_path.append({
                    'query_token': q_token,
                    'connections': connections,
                    'step': len(self.current_path)
                })
    
    def finish_path(self, success: bool = True):
        """
        Finish current reasoning path and store it.
        
        Args:
            success: Whether reasoning was successful
        """
        if self.current_path:
            self.paths.append({
                'steps': self.current_path,
                'success': success,
                'timestamp': time.time()
            })
            
            # Prune if needed
            if len(self.paths) > self.max_paths:
                # Sort by success and recency
                self.paths.sort(key=lambda p: (p['success'], p['timestamp']), reverse=True)
                self.paths = self.paths[:self.max_paths]
            
            self.current_path = []
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualizing reasoning paths.
        
        Returns:
            Dictionary with visualization data
        """
        return {
            'paths': self.paths,
            'stats': {
                'total_paths': len(self.paths),
                'success_rate': sum(1 for p in self.paths if p['success']) / max(1, len(self.paths)),
                'avg_steps': sum(len(p['steps']) for p in self.paths) / max(1, len(self.paths)),
            }
        }

class ModalityConfig:
    """Configuration for modality-specific attention parameterization."""
    
    SUPPORTED_MODALITIES = ["vision", "text", "audio"]
    
    def __init__(
        self,
        name: str,
        dim: int,
        max_seq_len: int,
        head_scale_factor: float = 1.0,
        dropout_rate: Optional[float] = None
    ):
        """
        Initialize modality configuration.
        
        Args:
            name: Modality name (must be in SUPPORTED_MODALITIES)
            dim: Native dimension of the modality
            max_seq_len: Maximum sequence length for this modality
            head_scale_factor: Scaling factor for attention in this modality
            dropout_rate: Optional override for dropout rate
        """
        if name not in self.SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported modality: {name}. Must be one of {self.SUPPORTED_MODALITIES}")
            
        self.name = name
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.head_scale_factor = head_scale_factor
        self.dropout_rate = dropout_rate

class CrossModalAttention3D(nn.Module):
    """
    Production-grade 3D cross-modal attention mechanism with optimized memory usage
    and comprehensive error handling.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        dropout: Dropout probability
        allow_missing: If True, allows missing modalities (routes heads selectively)
        max_sequence_length: Maximum sequence length for memory optimization
        attention_dropout: Dropout specifically for attention weights
        modality_configs: Optional modality-specific configurations
        flash_attention: Whether to use flash attention when available
        cpu_offload: Whether to offload memory-intensive operations to CPU
        checkpoint_attention: Whether to use gradient checkpointing for attention
        use_xformers: Whether to use xFormers for memory-efficient attention
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 16,
        dropout: float = 0.1,
        allow_missing: bool = True,
        max_sequence_length: int = 2048,
        attention_dropout: float = 0.1,
        modality_configs: Optional[Dict[str, ModalityConfig]] = None,
        flash_attention: bool = True,
        cpu_offload: bool = False,
        checkpoint_attention: bool = True,
        use_xformers: bool = False
    ):
        super().__init__()
        
        if not d_model > 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not d_model % num_heads == 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_sequence_length = max_sequence_length
        self.scale = math.sqrt(self.head_dim)
        
        # Performance options
        self.flash_attention = flash_attention
        self.cpu_offload = cpu_offload
        self.checkpoint_attention = checkpoint_attention
        self.use_xformers = use_xformers
        
        # Validate that only one acceleration method is used
        acceleration_methods = sum([flash_attention, use_xformers])
        if acceleration_methods > 1:
            raise ValueError("Only one acceleration method can be enabled at a time")
        
        # Check for flash attention availability
        self._has_flash_attn = False
        if flash_attention:
            try:
                import flash_attn
                self._has_flash_attn = True
            except ImportError:
                logger.warning("flash_attn not available, falling back to standard attention")
                self.flash_attention = False
                
        # Check for xformers availability
        self._has_xformers = False
        if use_xformers:
            try:
                import xformers
                self._has_xformers = True
            except ImportError:
                logger.warning("xformers not available, falling back to standard attention")
                self.use_xformers = False
        
        # Main attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Normalization and regularization
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # State tracking
        self.allow_missing = allow_missing
        self._metrics: Optional[AttentionMetrics] = None
        
        # Modality-specific configurations
        self.modality_configs = modality_configs or {}
        
        # Register modality gates and conditional parameters
        if self.modality_configs:
            # Create a parameter instead of a buffer for better typing
            self.modality_scale_factors = nn.Parameter(
                torch.ones(len(self.modality_configs), num_heads, 1, 1)
            )
            
            for i, (_, config) in enumerate(self.modality_configs.items()):
                # Use multiplication assignment for tensor
                with torch.no_grad():
                    self.modality_scale_factors[i] = self.modality_scale_factors[i] * config.head_scale_factor
        
        # Cache for optimizing repeated computations
        self._cache_size = 8  # Cache for last 8 sequences
        self._key_cache = {}
        self._value_cache = {}
        
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    @torch.jit.ignore
    def _validate_inputs(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """Validate input tensors for shape and type compatibility."""
        if query.dim() != 3:
            raise InvalidTensorError(f"Query must be 3D, got {query.dim()}D")
        if key_value.dim() != 3:
            raise InvalidTensorError(f"Key/Value must be 3D, got {key_value.dim()}D")
            
        batch, seq, dim = query.shape
        if dim != self.d_model:
            raise InvalidTensorError(
                f"Query dimension mismatch: expected {self.d_model}, got {dim}"
            )
            
        if seq > self.max_sequence_length:
            raise MemoryError(
                f"Sequence length {seq} exceeds maximum {self.max_sequence_length}"
            )

        if mask is not None and mask.dtype != torch.bool:
            raise InvalidTensorError(f"Mask must be boolean, got {mask.dtype}")

    @staticmethod
    def _compute_attention_entropy(attn_weights: torch.Tensor) -> float:
        """Compute attention entropy for monitoring attention diversity."""
        entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(-1).mean()
        return entropy.item()

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input tensor for multi-head attention computation."""
        batch, seq, d = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    @contextmanager
    def _monitor_computation(self):
        """Context manager for monitoring computation time and memory usage."""
        start = time.perf_counter()
        initial_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        peak_mem = initial_mem
        
        try:
            yield
        finally:
            end = time.perf_counter()
            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            
            self._metrics = AttentionMetrics(
                attention_scores_mean=getattr(self, '_temp_score_mean', 0.0),
                attention_scores_std=getattr(self, '_temp_score_std', 0.0),
                attention_entropy=getattr(self, '_temp_entropy', 0.0),
                compute_time_ms=(end - start) * 1000,
                memory_peak_mb=(peak_mem - initial_mem) / (1024 * 1024)
            )

    def get_metrics(self) -> Optional[AttentionMetrics]:
        """Get metrics from the last forward pass."""
        return self._metrics

    def _apply_modality_conditioning(
        self, 
        attn_scores: torch.Tensor,
        context_modality: str,
        mod_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply modality-specific conditioning to attention scores.
        
        Args:
            attn_scores: Raw attention scores
            context_modality: Current modality name
            mod_mask: Optional modality mask tensor
            
        Returns:
            Conditioned attention scores
        """
        # If no modality configuration, just return original scores
        if not self.modality_configs or context_modality not in self.modality_configs:
            return attn_scores
            
        # Get modality index for the specified modality
        modality_names = list(self.modality_configs.keys())
        if context_modality not in modality_names:
            return attn_scores
            
        modality_idx = modality_names.index(context_modality)
        
        # Apply modality-specific scaling factor
        scale_factor = self.modality_scale_factors[modality_idx]
        conditioned_scores = attn_scores * scale_factor
        
        return conditioned_scores
        
    def _compute_attention(
        self, 
        Q: torch.Tensor,  # shape: [batch, heads, seq_q, head_dim]
        K: torch.Tensor,  # shape: [batch, heads, seq_k, head_dim]
        V: torch.Tensor,  # shape: [batch, heads, seq_k, head_dim]
        mask: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        mod_mask: Optional[torch.Tensor] = None,
        context_modality: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute attention scores and outputs using the optimal available method.
        
        Args:
            Q, K, V: Query, Key, Value tensors
            mask: Attention mask
            entropy: Optional entropy weighting
            mod_mask: Optional modality mask
            context_modality: Optional modality context name
            
        Returns:
            Attention output tensor
        """
        # Use flash attention if available and enabled
        if self._has_flash_attn and self.flash_attention:
            import flash_attn
            
            # Flash attention expects inputs in shape [batch, seq, heads, head_dim]
            q_flash = Q.transpose(1, 2)  # [batch, seq_q, heads, head_dim]
            k_flash = K.transpose(1, 2)  # [batch, seq_k, heads, head_dim]
            v_flash = V.transpose(1, 2)  # [batch, seq_k, heads, head_dim]
            
            # Create attention mask for flash attention
            flash_mask = None
            if mask is not None:
                # Convert mask to flash_attn compatible format
                if mask.dim() == 4:  # [batch, 1, seq_q, seq_k]
                    flash_mask = ~mask.squeeze(1)  # [batch, seq_q, seq_k]
                elif mask.dim() == 3:  # [batch, seq_q, seq_k]
                    flash_mask = ~mask
            
            # Check which flash attention function exists and use it
            try:
                # Try the newer function name first
                if hasattr(flash_attn, 'flash_attn_func'):
                    # Pack QKV into a single tensor with shape [batch, seq, 3, heads, head_dim]
                    qkv = torch.stack([q_flash, k_flash, v_flash], dim=2)
                    output = flash_attn.flash_attn_func(
                        qkv,
                        dropout_p=self.attention_dropout.p if self.training else 0.0,
                        causal=False,
                        mask=flash_mask
                    )
                # Fall back to older function name if available
                elif hasattr(flash_attn, 'flash_attention_qkvpacked'):
                    qkv = torch.stack([q_flash, k_flash, v_flash], dim=2)
                    output = flash_attn.flash_attention_qkvpacked(
                        qkv,
                        dropout_p=self.attention_dropout.p if self.training else 0.0,
                        causal=False,
                        mask=flash_mask
                    )
                else:
                    # Manual attention as fallback
                    raise AttributeError("No suitable flash attention function found")
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Flash attention failed: {e}. Falling back to standard attention.")
                # Fall through to standard attention calculation below
                return self._compute_standard_attention(Q, K, V, mask, entropy, mod_mask, context_modality)
            
            # Return to original shape [batch, heads, seq_q, head_dim]
            return output.transpose(1, 2)
            
        # Use xFormers memory efficient attention if available
        elif self._has_xformers and self.use_xformers:
            try:
                import xformers.ops as xf_ops
                
                # Create attention mask for xformers
                xf_mask = None
                if mask is not None:
                    # Convert mask to xformers compatible format
                    if mask.dim() == 4:  # [batch, 1, seq_q, seq_k]
                        xf_mask = mask.squeeze(1)  # [batch, seq_q, seq_k]
                    elif mask.dim() == 3:  # [batch, seq_q, seq_k]
                        xf_mask = mask
                
                # Compute xformers attention
                output = xf_ops.memory_efficient_attention(
                    Q.transpose(1, 2),  # [batch, seq_q, heads, head_dim]
                    K.transpose(1, 2),  # [batch, seq_k, heads, head_dim]
                    V.transpose(1, 2),  # [batch, seq_k, heads, head_dim]
                    attn_bias=xf_mask,
                    p=self.attention_dropout.p if self.training else 0.0
                )
                
                # Return to original shape [batch, heads, seq_q, head_dim]
                return output.transpose(1, 2)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to use xformers: {e}. Falling back to standard attention.")
                self._has_xformers = False
                self.use_xformers = False
                # Fall through to standard attention
            
        # Standard attention computation
        else:
            # Compute attention scores
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            # Apply modality conditioning if specified
            if context_modality is not None:
                attn_scores = self._apply_modality_conditioning(
                    attn_scores, context_modality, mod_mask
                )
            
            # Apply attention mask
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # (batch, 1, seq, seq)
                attn_scores.masked_fill_(~mask, float('-inf'))
            
            # Apply modality mask if provided
            if mod_mask is not None:
                attn_scores.masked_fill_(~mod_mask.unsqueeze(1), float('-inf'))
            
            # Apply entropy weighting if provided
            if entropy is not None:
                entropy = entropy.unsqueeze(1) if entropy.dim() == 2 else entropy
                attn_scores = attn_scores + entropy
                
            # Compute attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            
            # Store metrics for monitoring
            with torch.no_grad():
                self._temp_score_mean = attn_weights.mean().item()
                self._temp_score_std = attn_weights.std().item()
                self._temp_entropy = self._compute_attention_entropy(attn_weights)
            
            # Compute weighted sum
            return torch.matmul(attn_weights, V)
    
    def _compute_standard_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        mod_mask: Optional[torch.Tensor] = None,
        context_modality: Optional[str] = None,
    ) -> torch.Tensor:
        """Standard attention computation as fallback."""
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply modality conditioning if specified
        if context_modality is not None:
            attn_scores = self._apply_modality_conditioning(
                attn_scores, context_modality, mod_mask
            )
        
        # Apply attention mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq, seq)
            attn_scores.masked_fill_(~mask, float('-inf'))
        
        # Apply modality mask if provided
        if mod_mask is not None:
            attn_scores.masked_fill_(~mod_mask.unsqueeze(1), float('-inf'))
        
        # Apply entropy weighting if provided
        if entropy is not None:
            entropy = entropy.unsqueeze(1) if entropy.dim() == 2 else entropy
            attn_scores = attn_scores + entropy
            
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Store metrics for monitoring
        with torch.no_grad():
            self._temp_score_mean = attn_weights.mean().item()
            self._temp_score_std = attn_weights.std().item()
            self._temp_entropy = self._compute_attention_entropy(attn_weights)
        
        # Compute weighted sum
        return torch.matmul(attn_weights, V)
    
    def _maybe_offload_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Offload tensor to CPU to save GPU memory if enabled.
        
        Args:
            tensor: Tensor to possibly offload
            
        Returns:
            Tensor, possibly on CPU
        """
        if self.cpu_offload and tensor.device.type == "cuda":
            return tensor.cpu()
        return tensor
    
    def forward(
        self,
        query: torch.Tensor,  # shape: (batch, seq, d_model)
        key_value: torch.Tensor,  # shape: (batch, seq, d_model)
        mask: Optional[torch.Tensor] = None,  # shape: (batch, seq) or (batch, seq, seq)
        entropy: Optional[torch.Tensor] = None,  # shape: (batch, seq) or (batch, seq, 1)
        mod_mask: Optional[torch.Tensor] = None,  # shape: (batch, seq) or (batch, seq, modalities)
        context_modality: Optional[str] = None,
        sequence_id: Optional[str] = None,
        incremental: bool = False
    ) -> torch.Tensor:
        """
        Compute cross-modal attention with comprehensive error handling and monitoring.
        
        Args:
            query: Query tensor (batch, seq, d_model)
            key_value: Key/Value tensor (batch, seq, d_model)
            mask: Optional attention mask
            entropy: Optional per-token entropy/signal
            mod_mask: Optional mask for selective routing
            context_modality: If set, conditions attention on specific modality
            sequence_id: Optional identifier for caching KV tensors
            incremental: If True, perform incremental decoding
            
        Returns:
            Output tensor (batch, seq, d_model)
            
        Raises:
            InvalidTensorError: If input tensors have invalid shapes/types
            MemoryError: If sequence length exceeds maximum
        """
        with self._monitor_computation():
            # Input validation
            self._validate_inputs(query, key_value, mask)
            
            batch_size, seq_len, _ = query.shape
            
            # Project Q, K, V
            Q = self._reshape_for_attention(self.q_proj(query))  # (batch, heads, seq, head_dim)
            
            # Handle KV caching for incremental decoding
            if incremental and sequence_id is not None:
                if sequence_id in self._key_cache:
                    # Append new key/values to the cache
                    K_new = self._reshape_for_attention(self.k_proj(key_value))
                    V_new = self._reshape_for_attention(self.v_proj(key_value))
                    
                    self._key_cache[sequence_id] = torch.cat(
                        [self._key_cache[sequence_id], K_new], dim=2
                    )
                    self._value_cache[sequence_id] = torch.cat(
                        [self._value_cache[sequence_id], V_new], dim=2
                    )
                    
                    # Use the full cached sequence
                    K = self._key_cache[sequence_id]
                    V = self._value_cache[sequence_id]
                else:
                    # Initialize the cache
                    K = self._reshape_for_attention(self.k_proj(key_value))
                    V = self._reshape_for_attention(self.v_proj(key_value))
                    
                    # Store in cache (possibly offload to CPU to save memory)
                    self._key_cache[sequence_id] = self._maybe_offload_to_cpu(K)
                    self._value_cache[sequence_id] = self._maybe_offload_to_cpu(V)
                    
                    # Cleanup old entries if cache is too large
                    if len(self._key_cache) > self._cache_size:
                        oldest_key = next(iter(self._key_cache))
                        del self._key_cache[oldest_key]
                        del self._value_cache[oldest_key]
            else:
                # Standard processing without caching
                K = self._reshape_for_attention(self.k_proj(key_value))
                V = self._reshape_for_attention(self.v_proj(key_value))
            
            # GPU optimization: if K/V were offloaded, bring them back
            if self.cpu_offload:
                K = K.to(query.device)
                V = V.to(query.device)
                
            # Compute attention with the optimal method
            attn_output = self._compute_attention(
                Q, K, V, mask, entropy, mod_mask, context_modality
            )
            
            # Merge heads and project output
            batch_size, _, seq_len, _ = attn_output.shape
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model
            )
            
            # Final projection and residual connection
            out = self.out_proj(attn_output)
            out = self.norm(out + query)  # Pre-norm variant
            
            return out

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        acceleration = "none"
        if self.flash_attention and self._has_flash_attn:
            acceleration = "flash_attention"
        elif self.use_xformers and self._has_xformers:
            acceleration = "xformers"
            
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, acceleration={acceleration}, "
            f"cpu_offload={self.cpu_offload}, "
            f"modalities={list(self.modality_configs.keys()) if self.modality_configs else None}"
        )

    def mod_to_mod_conditioning(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Condition one modality on another with controllable strength.
        
        Args:
            x: The tensor to condition
            cond: The conditioning tensor
            alpha: The conditioning strength (0.0 to 1.0)
            
        Returns:
            Conditioned tensor
        """
        return x + alpha * cond
        
    def multimodal_fusion(
        self,
        text_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform multi-modal fusion across different modality features.
        
        This enables the model to use cross-modal attention to integrate information
        from multiple modalities into a unified representation.
        
        Args:
            text_features: Text embeddings [batch, text_seq, d_model]
            image_features: Optional image embeddings [batch, img_seq, d_model]
            audio_features: Optional audio embeddings [batch, audio_seq, d_model]
            attention_mask: Optional attention mask for the sequence
            
        Returns:
            Fused multi-modal features [batch, total_seq, d_model]
        """
        # Gather available modalities
        features_list = [text_features]
        seq_lengths = [text_features.size(1)]
        modality_names = ["text"]
        
        if image_features is not None:
            features_list.append(image_features)
            seq_lengths.append(image_features.size(1))
            modality_names.append("vision")
            
        if audio_features is not None:
            features_list.append(audio_features)
            seq_lengths.append(audio_features.size(1))
            modality_names.append("audio")
        
        # Concatenate features along sequence dimension
        combined_features = torch.cat(features_list, dim=1)
        
        # Create modality-specific masks
        batch_size = text_features.size(0)
        seq_len = combined_features.size(1)
        
        # Create modality segmentation mask for cross-attention conditioning
        modality_mask = torch.zeros(
            batch_size, seq_len, len(modality_names), 
            dtype=torch.bool, device=text_features.device
        )
        
        # Set each modality's segment in the mask
        start_idx = 0
        for i, length in enumerate(seq_lengths):
            end_idx = start_idx + length
            modality_mask[:, start_idx:end_idx, i] = True
            start_idx = end_idx
            
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, seq_len, 
                dtype=torch.bool, device=text_features.device
            )
            
        # Apply cross-modal attention for each modality
        fused_features = combined_features
        
        # Process each modality as context
        for i, modality in enumerate(modality_names):
            if modality in self.modality_configs:
                # Extract query features for this modality
                mod_mask = modality_mask[:, :, i]  # [batch, seq]
                
                # Apply cross-modal attention using this modality as query context
                fused_features = self(
                    query=fused_features,
                    key_value=fused_features,
                    mask=attention_mask,
                    mod_mask=mod_mask,
                    context_modality=modality
                )
                
        return fused_features

class MambaIntegratedAttention(CrossModalAttention3D):
    """
    Revolutionary attention mechanism combining Neural Mamba's selective state space models 
    with cross-modal attention for more efficient sequence modeling.
    
    This approach integrates selective state space models with traditional attention,
    getting the best of both worlds for cross-modal sequence modeling.
    """
    
    def __init__(
        self, 
        *args, 
        ss_dim: int = 128,
        gating_factor: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # State space parameters
        self.ss_dim = ss_dim
        self.gating_factor = gating_factor
        
        # Parameters for SSM
        self.A = nn.Parameter(torch.randn(self.num_heads, ss_dim, ss_dim) * 0.02)
        self.B = nn.Parameter(torch.randn(self.num_heads, ss_dim, self.head_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(self.num_heads, self.head_dim, ss_dim) * 0.02)
        
        # Dynamic gating mechanism
        self.gate_proj = nn.Linear(self.d_model, self.num_heads)
        
        # For state tracking across sequences
        self.register_buffer("hidden_states", torch.zeros(1, self.num_heads, ss_dim))
        
        # Initialization
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        nn.init.normal_(self.C, mean=0.0, std=0.02)
    
    def _compute_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute selective state space model component.
        
        Args:
            x: Input tensor [batch, heads, seq, head_dim]
            
        Returns:
            SSM output tensor [batch, heads, seq, head_dim]
        """
        batch, heads, seq, dim = x.shape
        
        # Initialize or retrieve hidden state
        h = self.hidden_states.repeat((batch, 1, 1))
        
        outputs = []
        
        # Selective scan operation (simplified)
        for t in range(seq):
            # Update hidden state - equivalent to h = Ah + Bx
            h = torch.bmm(h.reshape(batch * heads, 1, self.ss_dim),
                         self.A.reshape(heads, self.ss_dim, self.ss_dim).expand(batch, -1, -1, -1).reshape(batch * heads, self.ss_dim, self.ss_dim)) + \
                torch.bmm(x[:, :, t].reshape(batch * heads, 1, dim),
                         self.B.reshape(heads, dim, self.ss_dim).expand(batch, -1, -1, -1).reshape(batch * heads, dim, self.ss_dim))
            
            h = h.reshape(batch, heads, self.ss_dim)
            
            # Apply output transformation y = Ch
            y = torch.bmm(h.reshape(batch * heads, 1, self.ss_dim),
                         self.C.reshape(heads, self.ss_dim, dim).expand(batch, -1, -1, -1).reshape(batch * heads, self.ss_dim, dim))
            
            y = y.reshape(batch, heads, 1, dim)
            outputs.append(y)
            
        # Concatenate all timesteps
        return torch.cat(outputs, dim=2)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        mod_mask: Optional[torch.Tensor] = None,
        context_modality: Optional[str] = None,
        sequence_id: Optional[str] = None,
        incremental: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with combined attention and SSM computation.
        
        Args:
            Same as parent CrossModalAttention3D.forward()
            
        Returns:
            Output tensor with blended attention and SSM dynamics
        """
        # Get standard attention output from parent class
        attn_output = super().forward(
            query, key_value, mask, entropy, mod_mask, 
            context_modality, sequence_id, incremental
        )
        
        # Process with SSM pathway
        Q = self._reshape_for_attention(self.q_proj(query))
        ssm_output = self._compute_ssm(Q)
        ssm_output = ssm_output.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.d_model
        )
        
        # Compute dynamic gate
        gate = torch.sigmoid(self.gate_proj(query) * self.gating_factor)
        gate = gate.view(query.shape[0], query.shape[1], self.num_heads, 1)
        gate = gate.mean(dim=2, keepdim=True)
        
        # Blend outputs
        blended_output = gate * attn_output + (1 - gate) * ssm_output
        
        return blended_output


class MultiStreamAttention(CrossModalAttention3D):
    """
    Revolutionary multi-stream attention that can simultaneously process multiple 
    attention streams with dynamic focus control.
    
    This enables the model to simultaneously handle multiple tasks like watching a 
    video while reading text and answering questions, with dynamic focus allocation.
    """
    
    def __init__(
        self,
        *args,
        num_streams: int = 3,
        stream_configs: Optional[List[StreamConfig]] = None,
        stream_mixing: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.num_streams = num_streams
        self.stream_mixing = stream_mixing
        
        # Initialize stream configurations
        if stream_configs:
            self.stream_configs = stream_configs
        else:
            # Default stream configuration
            self.stream_configs = [
                StreamConfig(AttentionStreamType.PRIMARY, priority=1.0),
                StreamConfig(AttentionStreamType.AUXILIARY, priority=0.5),
                StreamConfig(AttentionStreamType.MONITORING, priority=0.3),
            ][:num_streams]
            
        # Stream-specific projections
        self.stream_q_projs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(num_streams)
        ])
        
        self.stream_k_projs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(num_streams)
        ])
        
        self.stream_v_projs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(num_streams)
        ])
        
        # Focus control mechanisms
        self.focus_gate = nn.Linear(self.d_model, num_streams)
        
        # Output integration
        self.stream_outputs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(num_streams)
        ])
        
        self.output_integration = nn.Linear(self.d_model * num_streams, self.d_model)
        
    def forward_stream(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        stream_idx: int,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process a single attention stream.
        
        Args:
            query: Query tensor [batch, seq, dim]
            key_value: Key/Value tensor [batch, seq, dim]
            stream_idx: Stream index
            mask: Attention mask
            **kwargs: Additional arguments for attention
            
        Returns:
            Stream output tensor [batch, seq, dim]
        """
        # Get stream-specific projections
        q_proj = self.stream_q_projs[stream_idx]
        k_proj = self.stream_k_projs[stream_idx]
        v_proj = self.stream_v_projs[stream_idx]
        
        # Original projections for reference
        orig_q_proj = self.q_proj
        orig_k_proj = self.k_proj
        orig_v_proj = self.v_proj
        
        # Temporarily replace projections
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        
        # Process through parent's forward
        stream_output = super().forward(query, key_value, mask, **kwargs)
        
        # Restore original projections
        self.q_proj = orig_q_proj
        self.k_proj = orig_k_proj
        self.v_proj = orig_v_proj
        
        # Apply stream-specific output transformation
        return self.stream_outputs[stream_idx](stream_output)
    
    def compute_focus_distribution(self, query: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic focus distribution across streams.
        
        Args:
            query: Query tensor [batch, seq, dim]
            
        Returns:
            Focus weights [batch, seq, num_streams]
        """
        # Get raw focus logits
        focus_logits = self.focus_gate(query)
        
        # Apply stream priorities
        priorities = torch.tensor(
            [config.priority for config in self.stream_configs],
            device=query.device
        ).view(1, 1, self.num_streams)
        
        focus_logits = focus_logits * priorities
        
        # Convert to distribution
        return F.softmax(focus_logits, dim=-1)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process multiple attention streams simultaneously with dynamic focus.
        
        Args:
            query: Query tensor [batch, seq, dim]
            key_value: Key/Value tensor [batch, seq, dim]
            mask: Attention mask
            **kwargs: Additional arguments for attention
            
        Returns:
            Integrated output tensor [batch, seq, dim]
        """
        # Process each stream
        stream_outputs = []
        for i in range(self.num_streams):
            stream_output = self.forward_stream(query, key_value, i, mask, **kwargs)
            stream_outputs.append(stream_output)
        
        # Compute focus distribution
        focus = self.compute_focus_distribution(query)
        
        if self.stream_mixing:
            # Concatenate and mix streams
            concat_outputs = torch.cat(stream_outputs, dim=2)
            integrated = self.output_integration(concat_outputs)
        else:
            # Weighted sum of streams
            # Replace built-in sum with torch operations to ensure tensor output
            weighted_outputs = torch.stack([
                output * focus[:, :, i:i+1]
                for i, output in enumerate(stream_outputs)
            ], dim=0)
            integrated = torch.sum(weighted_outputs, dim=0)
        
        return integrated


class UncertaintyAwareAttention(CrossModalAttention3D):
    """
    Revolutionary uncertainty-aware attention that explicitly models confidence
    in cross-modal alignment.
    
    This allows the model to express when it's uncertain about connections between
    modalities, enabling more robust decision-making and preventing hallucination.
    """
    
    def __init__(self, *args, mc_samples: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_samples = mc_samples
        
        # Uncertainty modeling
        self.uncertainty_proj = nn.Linear(self.d_model, self.num_heads)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        
    def _compute_attention_with_uncertainty(
        self,
        Q: torch.Tensor,  # shape: [batch, heads, seq_q, head_dim]
        K: torch.Tensor,  # shape: [batch, heads, seq_k, head_dim]
        V: torch.Tensor,  # shape: [batch, heads, seq_k, head_dim]
        mask: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        mod_mask: Optional[torch.Tensor] = None,
        context_modality: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute attention with explicit uncertainty modeling.
        
        Args:
            Q, K, V: Query, Key, Value tensors
            mask: Attention mask
            entropy: Optional entropy weighting
            mod_mask: Optional modality mask
            context_modality: Optional modality context name
            
        Returns:
            Attention output tensor with uncertainty modeling
        """
        # Standard attention score computation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply modality conditioning if specified
        if context_modality is not None and hasattr(self, '_apply_modality_conditioning'):
            attn_scores = self._apply_modality_conditioning(
                attn_scores, context_modality, mod_mask
            )
        
        # Apply attention mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores.masked_fill_(~mask, float('-inf'))
        
        # Get token-specific uncertainty
        token_uncertainty = self.uncertainty_proj(Q.transpose(1, 2).reshape(
            Q.size(0), Q.size(2), -1)).sigmoid()
        
        # Temperature for uncertainty-aware softmax
        temperature = torch.exp(self.log_temperature)
        
        # Higher uncertainty = higher temperature = more uniform attention
        batch_size, num_heads, seq_len, _ = attn_scores.shape
        uncertainty_scale = token_uncertainty.view(batch_size, seq_len, num_heads, 1)
        uncertainty_scale = uncertainty_scale.permute(0, 2, 1, 3)
        
        # Scale attention scores by uncertainty-adjusted temperature
        attn_scores = attn_scores / (temperature * (1 + uncertainty_scale))
        
        # Multiple forward passes with dropout for Monte Carlo estimation
        if self.training:
            ensemble_outputs = []
            for _ in range(self.mc_samples):
                # Sample attention weights with dropout
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.attention_dropout(attn_weights)
                
                # Compute output for this sample
                sample_output = torch.matmul(attn_weights, V)
                ensemble_outputs.append(sample_output)
                
            # Stack and average the ensemble outputs
            stacked_outputs = torch.stack(ensemble_outputs, dim=0)
            output = stacked_outputs.mean(dim=0)
            
            # Compute uncertainty from variance across samples
            uncertainty = stacked_outputs.var(dim=0).mean(dim=-1, keepdim=True)
            
            # Attach uncertainty to output tensor for later use
            self._last_uncertainty = uncertainty
            
            return output
        else:
            # Standard forward pass for inference
            attn_weights = F.softmax(attn_scores, dim=-1)
            return torch.matmul(attn_weights, V)
            
    def forward(self, query, key_value, *args, **kwargs):
        """Forward pass with uncertainty modeling."""
        # Override _compute_attention with uncertainty-aware version
        original_compute_attention = getattr(self, '_compute_attention', None)
        setattr(self, '_compute_attention', self._compute_attention_with_uncertainty)
        
        # Call parent forward with our modified compute_attention
        output = super().forward(query, key_value, *args, **kwargs)
        
        # Restore original method if it existed
        if original_compute_attention is not None:
            setattr(self, '_compute_attention', original_compute_attention)
        
        return output
        
    def get_uncertainty(self):
        """Return token-level uncertainty from last forward pass."""
        if hasattr(self, '_last_uncertainty'):
            return self._last_uncertainty
        return None


class VideoDocumentIntelligence(nn.Module):
    """
    Revolutionary module for simultaneously processing video content and document text,
    with synchronized cross-modal attention between the two streams.
    
    This enables true simultaneous understanding of video and text content, allowing
    the model to answer questions while watching video or referring to documents.
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        video_frames: int = 16,
        max_text_len: int = 2048,
        num_heads: int = 16
    ):
        super().__init__()
        
        # Base cross-modal attention
        self.cross_attn = CrossModalAttention3D(
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Persistent memory for long-term context
        self.memory = PersistentMemory(
            capacity=1000,
            dim=d_model
        )
        
        # Video-specific processing
        self.video_temporal_pos_emb = nn.Parameter(
            torch.zeros(1, video_frames, d_model)
        )
        
        self.video_frame_attention = CrossModalAttention3D(
            d_model=d_model,
            num_heads=8,
            dropout=0.1
        )
        
        # Text-specific processing
        self.text_pos_emb = nn.Parameter(
            torch.zeros(1, max_text_len, d_model)
        )
        
        # Reasoning path tracker for explainability
        self.reasoning_tracker = ReasoningPathTracker()
        
        # Scene change detector
        self.scene_change_detector = nn.Linear(d_model * 2, 1)
        
        # Event synchronization
        self.event_detector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.events = []
        
    def detect_scene_changes(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Detect scene changes between consecutive video frames.
        
        Args:
            video_frames: Video frame features [batch, frames, dim]
            
        Returns:
            Scene change scores [batch, frames-1]
        """
        batch, frames, dim = video_frames.shape
        if frames <= 1:
            return torch.zeros(batch, 0, device=video_frames.device)
            
        # Compare consecutive frames
        frame_pairs = torch.cat([
            video_frames[:, :-1],
            video_frames[:, 1:]
        ], dim=2)
        
        # Compute change score
        change_scores = torch.sigmoid(self.scene_change_detector(frame_pairs)).squeeze(-1)
        
        return change_scores
        
    def detect_events(
        self, 
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        threshold: float = 0.7
    ) -> List[TemporalEvent]:
        """
        Detect synchronized events across video and text.
        
        Args:
            video_features: Video features [batch, frames, dim]
            text_features: Text features [batch, seq, dim]
            threshold: Event detection threshold
            
        Returns:
            List of detected events
        """
        # Process video frames to find event triggers
        video_event_features = self.event_detector(video_features.mean(dim=1))
        
        # Process text to find event triggers
        text_event_features = self.event_detector(text_features.mean(dim=1))
        
        # Compute similarity between video and text events
        sim = F.cosine_similarity(
            video_event_features.unsqueeze(1),
            text_event_features.unsqueeze(0),
            dim=2
        )
        
        # Find high-similarity pairs as potential events
        if sim.max().item() > threshold:
            # Create event with both modalities
            event = TemporalEvent(
                name=f"event_{len(self.events)}",
                modalities=["vision", "text"],
                features=(video_event_features + text_event_features) / 2,
                timestamp=time.time()
            )
            
            self.events.append(event)
            return [event]
            
        return []
    
    def process_video_document_context(
        self,
        video_frames: torch.Tensor,  # [batch, frames, dim]
        document_text: torch.Tensor,  # [batch, seq, dim]
        query: Optional[torch.Tensor] = None  # [batch, query_len, dim]
    ) -> Dict[str, Union[torch.Tensor, List[TemporalEvent]]]:
        """
        Process video and document simultaneously with cross-modal attention.
        
        Args:
            video_frames: Video frame features
            document_text: Document text features
            query: Optional query features
            
        Returns:
            Dictionary with processed features
        """
        batch_size = video_frames.shape[0]
        
        # Add positional embeddings
        video_pos = self.video_temporal_pos_emb[:, :video_frames.shape[1]]
        video_frames = video_frames + video_pos
        
        text_pos = self.text_pos_emb[:, :document_text.shape[1]]
        document_text = document_text + text_pos
        
        # Detect scene changes
        scene_changes = self.detect_scene_changes(video_frames)
        
        # Define modality masks (for selective cross-attention)
        video_mask = torch.zeros(
            batch_size, video_frames.shape[1] + document_text.shape[1], 2,
            dtype=torch.bool, device=video_frames.device
        )
        
        video_mask[:, :video_frames.shape[1], 0] = True  # First part is video
        video_mask[:, video_frames.shape[1]:, 1] = True  # Second part is text
        
        # Concatenate features
        combined_features = torch.cat([video_frames, document_text], dim=1)
        
        # Process with cross-modal attention
        attended_features = self.cross_attn(
            query=combined_features,
            key_value=combined_features,
            mod_mask=video_mask,
            context_modality="multimodal"
        )
        
        # Split back into separate modalities
        attended_video = attended_features[:, :video_frames.shape[1]]
        attended_text = attended_features[:, video_frames.shape[1]:]
        
        # Detect cross-modal events
        events = self.detect_events(attended_video, attended_text)
        
        # If query is provided, answer it using both modalities
        if query is not None:
            # First attend over video
            video_context = self.cross_attn(
                query=query,
                key_value=attended_video,
                context_modality="vision"
            )
            
            # Then attend over text
            text_context = self.cross_attn(
                query=query,
                key_value=attended_text,
                context_modality="text"
            )
            
            # Final attention to integrate both contexts
            query_updated = query + video_context + text_context
            
            # Store this context in persistent memory
            self.memory.add(
                content=query_updated.mean(dim=1),
                modality="multimodal",
                importance=1.0
            )
            
            return {
                "video_features": attended_video,
                "text_features": attended_text,
                "query_response": query_updated,
                "events": events,
                "scene_changes": scene_changes
            }
        
        return {
            "video_features": attended_video,
            "text_features": attended_text,
            "events": events,
            "scene_changes": scene_changes
        }
    
    def stream_video_with_document_context(
        self,
        video_stream: torch.Tensor,  # [batch, 1, dim]
        document_context: torch.Tensor,  # [batch, doc_len, dim]
    ) -> Dict[str, Union[torch.Tensor, List[TemporalEvent]]]:
        """
        Stream video while referring to document context.
        
        This enables real-time processing of video with cross-modal
        attention to document context.
        
        Args:
            video_stream: Current video frame features
            document_context: Document context features
            
        Returns:
            Dictionary with processed features including tensors and event lists
        """
        # Retrieve relevant document context based on video content
        memory_results = self.memory.retrieve(
            query=video_stream.mean(dim=1),
            top_k=5
        )
        
        # If we have relevant memory items, use them to enhance context
        if (memory_results):
            memory_contexts = torch.stack([item[1] for item in memory_results], dim=0)
            memory_context = memory_contexts.mean(dim=0, keepdim=True)
            
            # Enhance document context with memory
            document_context = document_context + memory_context
        
        # Process with standard method
        return self.process_video_document_context(
            video_frames=video_stream,
            document_text=document_context
        )

class CrossModalTemporalAlignment(nn.Module):
    """
    Revolutionary mechanism for aligning events and content across video and text modalities
    with variable temporal granularity.
    
    This enables precise synchronization between timestamps in video and positions in text,
    allowing the model to correctly reference content across modalities.
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 8,
        max_video_frames: int = 1000,
        max_text_tokens: int = 4096,
        alignment_resolution: int = 16  # Temporal granularity for alignment
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.alignment_resolution = alignment_resolution
        
        # Projections for alignment scoring
        self.video_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)
        
        # Alignment score predictor
        self.alignment_attn = CrossModalAttention3D(
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Temporal position embeddings
        self.video_pos_emb = nn.Parameter(torch.zeros(1, max_video_frames, d_model))
        self.text_pos_emb = nn.Parameter(torch.zeros(1, max_text_tokens, d_model))
        
        # Event detection and tracking
        self.event_detector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Confidence estimator for alignment quality
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Storage for detected events and alignments
        self.events = []
        self.alignment_history = []
        
        # Initialize embeddings
        nn.init.normal_(self.video_pos_emb, std=0.02)
        nn.init.normal_(self.text_pos_emb, std=0.02)
    
    def compute_alignment_matrix(
        self,
        video_features: torch.Tensor,  # [batch, v_frames, dim]
        text_features: torch.Tensor,  # [batch, t_tokens, dim]
        mask: Optional[torch.Tensor] = None  # Optional mask for invalid alignments
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alignment matrix between video frames and text tokens.
        
        Args:
            video_features: Video frame features
            text_features: Text token features
            mask: Optional mask for invalid alignments
            
        Returns:
            Tuple of (alignment_matrix, confidence_scores)
        """
        batch_size, v_frames, _ = video_features.shape
        _, t_tokens, _ = text_features.shape
        
        # Apply projections
        video_proj = self.video_proj(video_features)  # [batch, v_frames, dim]
        text_proj = self.text_proj(text_features)    # [batch, t_tokens, dim]
        
        # Add positional embeddings
        video_pos = self.video_pos_emb[:, :v_frames]
        text_pos = self.text_pos_emb[:, :t_tokens]
        
        video_proj = video_proj + video_pos
        text_proj = text_proj + text_pos
        
        # Compute alignment scores with attention
        # Scale video features to create query
        video_query = video_proj.view(batch_size * v_frames, 1, self.d_model)
        
        # Expand text features for each video frame
        text_key_value = text_proj.unsqueeze(1).expand(-1, v_frames, -1, -1)
        text_key_value = text_key_value.reshape(batch_size * v_frames, t_tokens, self.d_model)
        
        # Compute attention from video to text
        attended = self.alignment_attn(
            query=video_query,
            key_value=text_key_value
        )
        
        # Reshape to get alignment scores
        alignment_scores = torch.bmm(
            video_query, 
            text_key_value.transpose(1, 2)
        ) / math.sqrt(self.d_model)
        
        alignment_matrix = alignment_scores.view(batch_size, v_frames, t_tokens)
        
        # Apply mask if provided
        if mask is not None:
            alignment_matrix = alignment_matrix.masked_fill(~mask, float('-inf'))
        
        # Normalize to probabilities
        alignment_probs = F.softmax(alignment_matrix, dim=2)
        
        # Compute confidence for each alignment
        confidence_features = torch.cat([
            video_proj.unsqueeze(2).expand(-1, -1, t_tokens, -1),
            text_proj.unsqueeze(1).expand(-1, v_frames, -1, -1)
        ], dim=3)
        
        # Reshape for confidence estimation
        confidence_input = confidence_features.view(batch_size * v_frames * t_tokens, self.d_model * 2)
        confidence = self.confidence_estimator(confidence_input)
        confidence = confidence.view(batch_size, v_frames, t_tokens)
        
        return alignment_probs, confidence
    
    def detect_synchronized_events(
        self,
        video_features: torch.Tensor,  # [batch, v_frames, dim]
        text_features: torch.Tensor,  # [batch, t_tokens, dim]
        alignment_matrix: torch.Tensor,  # [batch, v_frames, t_tokens]
        confidence: torch.Tensor,  # [batch, v_frames, t_tokens]
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Detect events that are synchronized across video and text.
        
        Args:
            video_features: Video frame features
            text_features: Text token features
            alignment_matrix: Alignment probabilities between frames and tokens
            confidence: Confidence scores for alignments
            threshold: Detection threshold
            
        Returns:
            List of detected synchronized events
        """
        batch_size, v_frames, _ = video_features.shape
        _, t_tokens, _ = text_features.shape
        
        # Process video and text with event detector
        video_event_features = self.event_detector(video_features)
        text_event_features = self.event_detector(text_features)
        
        detected_events = []
        
        # Find high-confidence alignments
        high_conf = confidence > threshold
        
        for b in range(batch_size):
            # Get indices of high-confidence alignments
            v_indices, t_indices = torch.where(high_conf[b])
            
            # Group by temporal proximity
            if len(v_indices) > 0:
                # Sort by video frame index
                sorted_indices = torch.argsort(v_indices)
                v_sorted = v_indices[sorted_indices]
                t_sorted = t_indices[sorted_indices]
                
                # Group into events with temporal continuity
                current_event = {
                    'video_indices': [v_sorted[0].item()],
                    'text_indices': [t_sorted[0].item()],
                    'confidence': [confidence[b, v_sorted[0], t_sorted[0]].item()]
                }
                
                for i in range(1, len(v_sorted)):
                    # If next frame is within temporal window, add to current event
                    if v_sorted[i] - v_sorted[i-1] <= self.alignment_resolution:
                        current_event['video_indices'].append(v_sorted[i].item())
                        current_event['text_indices'].append(t_sorted[i].item())
                        current_event['confidence'].append(
                            confidence[b, v_sorted[i], t_sorted[i]].item()
                        )
                    else:
                        # Create new event
                        if len(current_event['video_indices']) >= 3:  # Minimum length threshold
                            # Extract features for this event
                            v_idxs = current_event['video_indices']
                            t_idxs = current_event['text_indices']
                            
                            event_v_feat = video_event_features[b, v_idxs].mean(dim=0)
                            event_t_feat = text_event_features[b, t_idxs].mean(dim=0)
                            
                            # Create event with metadata
                            event = {
                                'batch_idx': b,
                                'video_indices': current_event['video_indices'],
                                'text_indices': current_event['text_indices'],
                                'video_features': event_v_feat,
                                'text_features': event_t_feat,
                                'mean_confidence': sum(current_event['confidence']) / len(current_event['confidence']),
                                'timestamp': time.time()
                            }
                            
                            detected_events.append(event)
                        
                        # Start new event
                        current_event = {
                            'video_indices': [v_sorted[i].item()],
                            'text_indices': [t_sorted[i].item()],
                            'confidence': [confidence[b, v_sorted[i], t_sorted[i]].item()]
                        }
                
                # Don't forget the last event
                if len(current_event['video_indices']) >= 3:
                    v_idxs = current_event['video_indices']
                    t_idxs = current_event['text_indices']
                    
                    event_v_feat = video_event_features[b, v_idxs].mean(dim=0)
                    event_t_feat = text_event_features[b, t_idxs].mean(dim=0)
                    
                    event = {
                        'batch_idx': b,
                        'video_indices': current_event['video_indices'],
                        'text_indices': current_event['text_indices'],
                        'video_features': event_v_feat,
                        'text_features': event_t_feat,
                        'mean_confidence': sum(current_event['confidence']) / len(current_event['confidence']),
                        'timestamp': time.time()
                    }
                    
                    detected_events.append(event)
        
        # Store events for future reference
        self.events.extend(detected_events)
        
        return detected_events
    
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:  # Changed return type to Dict[str, Any]
        """
        Compute temporal alignment between video and text features.
        
        Args:
            video_features: Video frame features [batch, v_frames, dim]
            text_features: Text token features [batch, t_tokens, dim]
            mask: Optional mask for invalid alignments
            
        Returns:
            Dictionary with alignment results
        """
        # Compute alignment matrix and confidence
        alignment_matrix, confidence = self.compute_alignment_matrix(
            video_features, text_features, mask
        )
        
        # Detect synchronized events
        events = self.detect_synchronized_events(
            video_features, text_features, alignment_matrix, confidence
        )
        
        # Store alignment history
        self.alignment_history.append({
            'alignment': alignment_matrix.detach(),
            'confidence': confidence.detach(),
            'timestamp': time.time()
        })
        
        # Limit history length
        if len(self.alignment_history) > 100:
            self.alignment_history = self.alignment_history[-100:]
        
        return {
            'alignment_matrix': alignment_matrix,
            'confidence': confidence,
            'events': events,
            'num_events': len(events)
        }


class AttentionGuidanceController(nn.Module):
    """
    Revolutionary controller for dynamically guiding attention between video and text
    based on task relevance, uncertainty, and content saliency.
    
    This allows for intelligent focus shifting between modalities, enabling the model to
    concentrate on the most relevant modality for the current reasoning step.
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        num_modalities: int = 2,  # Default: video and text
        temperature: float = 1.0,
        uncertainty_threshold: float = 0.3
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.temperature = temperature
        self.uncertainty_threshold = uncertainty_threshold
        
        # Content saliency predictors
        self.saliency_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(num_modalities)
        ])
        
        # Task relevance estimator
        self.relevance_estimator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_modalities)
        )
        
        # Uncertainty tracker
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_modalities),
            nn.Sigmoid()
        )
        
        # Attention distribution tracker
        self.attention_history = deque(maxlen=100)
        
        # EMA parameters for smoothing
        self.register_buffer("modality_importance", torch.ones(1, num_modalities) / num_modalities)
        self.ema_decay = 0.9
    
    def compute_saliency(
        self, 
        features_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute content saliency for each modality.
        
        Args:
            features_list: List of feature tensors for each modality
            
        Returns:
            List of saliency maps for each modality
        """
        saliency_maps = []
        
        for i, features in enumerate(features_list):
            # Apply saliency predictor
            saliency = self.saliency_predictors[i](features)
            
            # Normalize
            saliency = F.softmax(saliency / self.temperature, dim=1)
            saliency_maps.append(saliency)
            
        return saliency_maps
    
    def estimate_task_relevance(
        self,
        query: torch.Tensor,  # [batch, q_len, dim]
        context: torch.Tensor  # [batch, c_len, dim]
    ) -> torch.Tensor:
        """
        Estimate task relevance of each modality for the current query.
        
        Args:
            query: Query representation
            context: Context representation
            
        Returns:
            Relevance scores for each modality [batch, num_modalities]
        """
        # Pool query and context
        query_pooled = query.mean(dim=1)  # [batch, dim]
        context_pooled = context.mean(dim=1)  # [batch, dim]
        
        # Concatenate
        combined = torch.cat([query_pooled, context_pooled], dim=1)
        
        # Estimate relevance
        relevance = self.relevance_estimator(combined)
        
        return F.softmax(relevance, dim=1)
    
    def estimate_uncertainty(
        self,
        features_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Estimate uncertainty in each modality.
        
        Args:
            features_list: List of feature tensors for each modality
            
        Returns:
            Uncertainty scores for each modality [batch, num_modalities]
        """
        # Pool features from each modality
        pooled_features = [features.mean(dim=1) for features in features_list]
        
        # Concatenate
        combined = torch.cat(pooled_features, dim=1)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(combined)
        
        return uncertainty
    
    def forward(
        self,
        query: torch.Tensor,  # [batch, q_len, dim]
        features_list: List[torch.Tensor],  # List of [batch, seq, dim] for each modality
        task_type: Optional[str] = None,  # Optional task type hint
        prev_attention_weights: Optional[torch.Tensor] = None  # Optional previous weights
    ) -> Dict[str, Any]:  # Changed return type to Dict[str, Any]
        """
        Compute guided attention distribution across modalities.
        
        Args:
            query: Query representation
            features_list: List of feature tensors for each modality
            task_type: Optional task type hint
            prev_attention_weights: Optional previous attention weights
            
        Returns:
            Dictionary with attention guidance results
        """
        batch_size = query.shape[0]
        
        # Ensure we have the correct number of modalities
        assert len(features_list) == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {len(features_list)}"
        
        # Compute content saliency
        saliency_maps = self.compute_saliency(features_list)
        
        # Estimate task relevance using first modality as context
        # (typically this would be the primary modality)
        modality_relevance = self.estimate_task_relevance(query, features_list[0])
        
        # Estimate uncertainty
        uncertainty = self.estimate_uncertainty(features_list)
        
        # Adjust for high uncertainty
        high_uncertainty = uncertainty > self.uncertainty_threshold
        
        # If high uncertainty in a modality, reduce its importance
        modality_weights = modality_relevance * (1 - uncertainty)
        
        # Re-normalize weights
        modality_weights = F.normalize(modality_weights, p=1, dim=1)
        
        # Apply EMA smoothing
        old_weights = self.modality_importance.expand(batch_size, -1)
        smoothed_weights = self.ema_decay * old_weights + (1 - self.ema_decay) * modality_weights
        
        # Update EMA
        with torch.no_grad():
            self.modality_importance = smoothed_weights.mean(dim=0, keepdim=True)
        
        # Save to history
        self.attention_history.append({
            'weights': modality_weights.detach().cpu(),
            'uncertainty': uncertainty.detach().cpu(),
            'timestamp': time.time()
        })
        
        return {
            'modality_weights': modality_weights,
            'saliency_maps': saliency_maps,
            'uncertainty': uncertainty,
            'smoothed_weights': smoothed_weights,
            'high_uncertainty': high_uncertainty
        }


class SynchronizedStreamProcessor(nn.Module):
    """
    Revolutionary processor for synchronized processing of video and text streams
    with dynamic cross-modal attention and temporal alignment.
    
    This enables true simultaneous processing of video and text, allowing models to
    watch videos while reading and answering questions about both modalities in real-time.
    """
    
    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 16,
        max_video_buffer: int = 64,
        max_text_buffer: int = 2048,
        memory_size: int = 1000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_video_buffer = max_video_buffer
        self.max_text_buffer = max_text_buffer
        
        # Core attention mechanism
        self.cross_attn = CrossModalAttention3D(
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Enhanced components
        self.temporal_alignment = CrossModalTemporalAlignment(
            d_model=d_model,
            num_heads=num_heads // 2,
            max_video_frames=max_video_buffer,
            max_text_tokens=max_text_buffer
        )
        
        self.attention_controller = AttentionGuidanceController(
            d_model=d_model,
            num_modalities=2  # Video and text
        )
        
        # Memory for cross-modal context
        self.memory = PersistentMemory(
            capacity=memory_size,
            dim=d_model
        )
        
        # Video stream buffer
        self.video_buffer = []
        self.video_buffer_features = None
        
        # Text stream buffer
        self.text_buffer = []
        self.text_buffer_features = None
        
        # Modality fusion
        self.video_to_text_fusion = nn.Linear(d_model, d_model)
        self.text_to_video_fusion = nn.Linear(d_model, d_model)
        
        # Temporal position encoding
        self.video_pos_encoding = nn.Parameter(
            self._generate_positional_encoding(max_video_buffer, d_model)
        )
        self.text_pos_encoding = nn.Parameter(
            self._generate_positional_encoding(max_text_buffer, d_model)
        )
        
        # Multimodal integration
        self.multimodal_integration = MultiStreamAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_streams=3,  # Video, text, and integrated
            stream_configs=[
                StreamConfig(AttentionStreamType.PRIMARY, priority=1.0, modalities=["vision"]),
                StreamConfig(AttentionStreamType.PRIMARY, priority=1.0, modalities=["text"]),
                StreamConfig(AttentionStreamType.INTEGRATION, priority=0.8, modalities=["vision", "text"])
            ]
        )
        
        # Output adapters
        self.output_projector = nn.Linear(d_model, d_model)
        
        # For tracking synchronization
        self.sync_points = []
        
    @staticmethod
    def _generate_positional_encoding(seq_len: int, dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encodings."""
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        
        pos_enc = torch.zeros(1, seq_len, dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_enc
    
    def add_video_frame(
        self,
        frame_features: torch.Tensor  # [batch, 1, dim] or [batch, dim]
    ) -> None:
        """
        Add a new video frame to the processing buffer.
        
        Args:
            frame_features: Features for new video frame
        """
        # Ensure correct shape
        if frame_features.dim() == 2:
            frame_features = frame_features.unsqueeze(1)
            
        # Add to buffer
        self.video_buffer.append(frame_features)
        
        # Limit buffer size
        if len(self.video_buffer) > self.max_video_buffer:
            self.video_buffer.pop(0)
            
        # Update buffer features
        self.video_buffer_features = torch.cat(self.video_buffer, dim=1)
    
    def add_text_chunk(
        self,
        text_features: torch.Tensor  # [batch, seq, dim]
    ) -> None:
        """
        Add a new text chunk to the processing buffer.
        
        Args:
            text_features: Features for new text chunk
        """
        # Add to buffer
        self.text_buffer.append(text_features)
        
        # Concatenate all chunks
        cat_text = torch.cat(self.text_buffer, dim=1)
        
        # Limit buffer size
        if cat_text.size(1) > self.max_text_buffer:
            excess = cat_text.size(1) - self.max_text_buffer
            
            # Remove oldest tokens
            self.text_buffer = []
            self.text_buffer.append(cat_text[:, excess:])
            
        # Update buffer features
        self.text_buffer_features = torch.cat(self.text_buffer, dim=1)
    
    def process_streams(
        self,
        query: Optional[torch.Tensor] = None  # [batch, q_len, dim]
    ) -> Dict[str, Any]:  # Changed return type to Dict[str, Any]
        """
        Process video and text streams with synchronized cross-modal attention.
        
        Args:
            query: Optional query for directed attention
            
        Returns:
            Dictionary with processing results
        """
        # Check if we have content to process
        if self.video_buffer_features is None or self.text_buffer_features is None:
            return {'status': 'insufficient_content'}
            
        batch_size = self.video_buffer_features.shape[0]
        video_seq_len = self.video_buffer_features.shape[1]
        text_seq_len = self.text_buffer_features.shape[1]
        
        # Add positional encodings
        video_features = self.video_buffer_features + self.video_pos_encoding[:, :video_seq_len]
        text_features = self.text_buffer_features + self.text_pos_encoding[:, :text_seq_len]
        
        # Compute temporal alignment between streams
        alignment_results = self.temporal_alignment(
            video_features=video_features,
            text_features=text_features
        )
        
        alignment_matrix = alignment_results['alignment_matrix']
        confidence = alignment_results['confidence']
        
        # Check if we detected synchronized events
        if (alignment_results['num_events'] > 0):
            self.sync_points.extend(alignment_results['events'])
            
        # Apply attention guidance
        guidance_results = self.attention_controller(
            query=query if query is not None else video_features.mean(dim=1, keepdim=True),
            features_list=[video_features, text_features]
        )
        
        modality_weights = guidance_results['modality_weights']
        
        # Cross-modal fusion with guided attention
        # Using alignment matrix for informed fusion
        v2t_attention = torch.matmul(video_features.transpose(1, 2), alignment_matrix)
        v2t_attention = v2t_attention.transpose(1, 2)  # [batch, text_seq, d_model]
        
        t2v_attention = torch.matmul(text_features.transpose(1, 2), alignment_matrix.transpose(1, 2))
        t2v_attention = t2v_attention.transpose(1, 2)  # [batch, video_seq, d_model]
        
        # Apply fusion
        video_enhanced = video_features + self.text_to_video_fusion(t2v_attention)
        text_enhanced = text_features + self.video_to_text_fusion(v2t_attention)
        
        # Weight each modality by its importance
        video_weighted = video_enhanced * modality_weights[:, 0:1, None]
        text_weighted = text_enhanced * modality_weights[:, 1:2, None]
        
        # Process multimodal streams
        multimodal_results = self.multimodal_integration(
            query=torch.cat([video_weighted, text_weighted], dim=1),
            key_value=torch.cat([video_weighted, text_weighted], dim=1)
        )
        
        # Store key events and contexts in memory
        high_confidence_points = (confidence > 0.8).any(dim=2)
        
        for b in range(batch_size):
            for v_idx in range(video_seq_len):
                if high_confidence_points[b, v_idx]:
                    # Store this high-confidence point in memory
                    self.memory.add(
                        content=torch.cat([
                            video_weighted[b, v_idx],
                            text_weighted[b, torch.argmax(alignment_matrix[b, v_idx])]
                        ]),
                        modality="multimodal",
                        importance=confidence[b, v_idx].max().item()
                    )
        
        # If query is provided, answer it using both modalities
        if query is not None:
            # Use cross-modal attention to process query
            query_video_context = self.cross_attn(
                query=query,
                key_value=video_weighted,
                context_modality="vision"
            )
            
            query_text_context = self.cross_attn(
                query=query,
                key_value=text_weighted,
                context_modality="text"
            )
            
            # Weight contexts by modality importance
            query_video_context = query_video_context * modality_weights[:, 0:1, None]
            query_text_context = query_text_context * modality_weights[:, 1:2, None]
            
            # Integrate contexts
            query_response = query + query_video_context + query_text_context
            query_response = self.output_projector(query_response)
            
            return {
                'video_features': video_weighted,
                'text_features': text_weighted,
                'alignment': alignment_matrix,
                'confidence': confidence,
                'modality_weights': modality_weights,
                'query_response': query_response,
                'events': alignment_results['events'],
                'integrated_features': multimodal_results
            }
        
        return {
            'video_features': video_weighted,
            'text_features': text_weighted,
            'alignment': alignment_matrix,
            'confidence': confidence,
            'modality_weights': modality_weights,
            'events': alignment_results['events'],
            'integrated_features': multimodal_results
        }
    
    def answer_query(
        self,
        query: torch.Tensor,  # [batch, q_len, dim]
        modality_preference: Optional[str] = None  # 'video', 'text', or None for balanced
    ) -> torch.Tensor:
        """
        Answer a query using synchronized video and text context.
        
        Args:
            query: Query tensor
            modality_preference: Optional preference for which modality to prioritize
            
        Returns:
            Response tensor
        """
        # Process streams with query
        results = self.process_streams(query=query)
        
        if 'query_response' in results:
            response = results['query_response']
            
            # Apply modality preference if specified
            if modality_preference == 'video':
                # Retrieve relevant video memories
                video_memories = self.memory.retrieve(
                    query=query.mean(dim=1),
                    modality="video",
                    top_k=3
                )
                
                if video_memories:
                    # Enhance with video context
                    video_context = torch.stack([m[1] for m in video_memories], dim=0).mean(0)
                    response = response + 0.3 * self.video_to_text_fusion(video_context)
                    
            elif modality_preference == 'text':
                # Retrieve relevant text memories
                text_memories = self.memory.retrieve(
                    query=query.mean(dim=1),
                    modality="text",
                    top_k=3
                )
                
                if text_memories:
                    # Enhance with text context
                    text_context = torch.stack([m[1] for m in text_memories], dim=0).mean(0)
                    response = response + 0.3 * self.text_to_video_fusion(text_context)
            
            return response
        
        # If we don't have a query response, return the original query
        return query


class EnhancedTemporalPersistentMemory(PersistentMemory):
    """
    Enhanced persistent memory with specialized temporal indexing and cross-modal
    reference management for video and document content.
    
    This enables more efficient storage and retrieval of synchronized content across
    modalities with temporal relationships preserved.
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        dim: int = 4096,
        temporal_window: int = 30,  # Window size in seconds
        cross_ref_limit: int = 5,   # Max cross-references per item
        **kwargs
    ):
        super().__init__(capacity=capacity, dim=dim, **kwargs)
        
        self.temporal_window = temporal_window
        self.cross_ref_limit = cross_ref_limit
        
        # Temporal index
        self.temporal_index: Dict[str, List[str]] = {}  # timestamp -> [item_ids]
        
        # Cross-references between modalities
        self.cross_references = defaultdict(set)  # item_id -> {ref_ids}
        
        # Event tracking
        self.events: Dict[str, Dict[str, Any]] = {}  # event_id -> event_data
    
    def add_with_timestamp(
        self,
        content: torch.Tensor,
        modality: str,
        timestamp: float,
        key: Optional[torch.Tensor] = None,
        importance: float = 1.0,
        ttl: Optional[float] = None,
        references: Optional[List[str]] = None,
        event_id: Optional[str] = None
    ) -> str:
        """
        Add item with explicit timestamp for temporal indexing.
        
        Args:
            content: Content tensor
            modality: Content modality
            timestamp: Timestamp in seconds
            key: Optional search key
            importance: Item importance
            ttl: Time-to-live
            references: Optional related item IDs
            event_id: Optional event identifier
            
        Returns:
            ID of stored item
        """
        # Add basic item
        item_id = super().add(
            content=content,
            modality=modality,
            key=key,
            importance=importance,
            ttl=ttl
        )
        
        # Round timestamp to nearest second for indexing
        ts_key = str(int(timestamp))
        
        # Add to temporal index
        if ts_key not in self.temporal_index:
            self.temporal_index[ts_key] = []
        self.temporal_index[ts_key].append(item_id)
        
        # Add cross-references if provided
        if references:
            for ref_id in references[:self.cross_ref_limit]:
                if ref_id in self.items:
                    self.cross_references[item_id].add(ref_id)
                    self.cross_references[ref_id].add(item_id)
        
        # Link to event if provided
        if event_id:
            if event_id not in self.events:
                self.events[event_id] = {
                    'items': [],
                    'modalities': set(),
                    'timestamp': timestamp
                }
            
            self.events[event_id]['items'].append(item_id)
            self.events[event_id]['modalities'].add(modality)
        
        return item_id
    
    def retrieve_temporal_window(
        self,
        center_time: float,
        window_size: Optional[float] = None,
        modalities: Optional[List[str]] = None
    ) -> List[Tuple[str, torch.Tensor, float, float]]:
        """
        Retrieve items within a temporal window.
        
        Args:
            center_time: Center time in seconds
            window_size: Window size in seconds (default: self.temporal_window)
            modalities: Optional filter by modalities
            
        Returns:
            List of (item_id, content, importance, timestamp) tuples
        """
        if window_size is None:
            window_size = self.temporal_window
        
        half_window = window_size / 2
        min_time = int(center_time - half_window)
        max_time = int(center_time + half_window)
        
        results = []
        
        # Collect all item_ids in the time window
        item_ids = []
        for ts in range(min_time, max_time + 1):
            ts_key = str(ts)
            if ts_key in self.temporal_index:
                item_ids.extend(self.temporal_index[ts_key])
        
        # Filter and collect items
        for item_id in item_ids:
            if item_id in self.items:
                item = self.items[item_id]
                
                # Apply modality filter
                if modalities and item.modality not in modalities:
                    continue
                
                # Update access stats
                item.update_access()
                
                # Add to results
                results.append((
                    item_id, 
                    item.content, 
                    item.importance,
                    item.creation_time
                ))
        
        return results
    
    def retrieve_related(
        self,
        item_id: str,
        depth: int = 1,
        max_results: int = 10
    ) -> List[Tuple[str, torch.Tensor, float, int]]:
        """
        Retrieve items related to the specified item through cross-references.
        
        Args:
            item_id: Source item ID
            depth: How far to follow references (1-3)
            max_results: Maximum number of results
            
        Returns:
            List of (item_id, content, importance, distance) tuples
        """
        if item_id not in self.items:
            return []
            
        visited = {item_id}
        to_visit = [(ref, 1) for ref in self.cross_references.get(item_id, set())]
        results = []
        
        while to_visit and len(results) < max_results:
            current_id, distance = to_visit.pop(0)
            
            if current_id in visited or current_id not in self.items:
                continue
                
            visited.add(current_id)
            
            # Add to results
            item = self.items[current_id]
            item.update_access()
            results.append((current_id, item.content, item.importance, distance))
            
            # Add neighbors if within depth
            if distance < depth:
                for ref in self.cross_references.get(current_id, set()):
                    if ref not in visited:
                        to_visit.append((ref, distance + 1))
        
        return results
    
    def retrieve_event(
        self,
        event_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve all items associated with an event.
        
        Args:
            event_id: Event identifier
            
        Returns:
            Event data dictionary with items
        """
        if event_id not in self.events:
            return {'found': False}
            
        event_data = self.events[event_id]
        items_data = []
        
        for item_id in event_data['items']:
            if item_id in self.items:
                item = self.items[item_id]
                item.update_access()
                items_data.append({
                    'id': item_id,
                    'content': item.content,
                    'modality': item.modality,
                    'importance': item.importance
                })
        
        return {
            'found': True,
            'timestamp': event_data['timestamp'],
            'modalities': list(event_data['modalities']),
            'items': items_data
        }
