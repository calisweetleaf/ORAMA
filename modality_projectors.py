"""
Modality Projectors for GPT-4o Architecture

This module implements the projectors that convert encoded modality outputs 
(from text, image, audio encoders) into a unified latent tensor space for the 
transformer backbone. It handles modality-specific linear projections, adds 
learned modality bias tokens, and normalizes sequence lengths with padding/masking.

The module is designed according to the GPT-4o architecture guidelines,
implementing the following key features:
- Modality-specific linear projections to align dimensions
- Learned modality bias tokens for each modality type
- Padding and masking for sequence normalization
- Temporal embedding injection
- Support for text, image, audio, and video modalities

Architecture Note: This projector is placed after modality-specific encoders and
before the cross-modal attention mechanisms in the GPT-4o pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List, Tuple, Any, Callable, Type
from dataclasses import dataclass, field
import enum
import logging
import numpy as np
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
import os
import json
from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)

class ModalityProjectorError(Exception):
    """Base exception class for all modality projector errors."""
    
    ERROR_CODE = "MP-BASE-ERROR"
    
    def __init__(self, message: str, code: Optional[str] = None):
        self.code = code or self.ERROR_CODE
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class DimensionMismatchError(ModalityProjectorError):
    """Raised when there's a dimension mismatch in the input tensor."""
    ERROR_CODE = "MP-DIM-ERROR"


class UnknownModalityError(ModalityProjectorError):
    """Raised when an unknown modality is encountered."""
    ERROR_CODE = "MP-MOD-ERROR"


class ConfigurationError(ModalityProjectorError):
    """Raised for configuration-related errors."""
    ERROR_CODE = "MP-CFG-ERROR"


class SequenceLengthError(ModalityProjectorError):
    """Raised when sequence length exceeds the maximum allowable length."""
    ERROR_CODE = "MP-SEQ-ERROR"


#------------------------------------------------------------------------------
# Modality Enumerations and Constants
#------------------------------------------------------------------------------

class ModalityType(str, enum.Enum):
    """Enumeration of supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    
    @classmethod
    def all_types(cls) -> List[str]:
        """Returns list of all supported modality types."""
        return [m.value for m in cls]


# Default dimensions for each modality
DEFAULT_DIMENSIONS = {
    ModalityType.TEXT: 4096,
    ModalityType.IMAGE: 4096,
    ModalityType.AUDIO: 4096,
    ModalityType.VIDEO: 4096
}

# Default bias initialization parameters
DEFAULT_BIAS_INIT = {
    "std": 0.02, 
    "mean": 0.0
}


#------------------------------------------------------------------------------
# Configuration Classes
#------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModalityConfig:
    """Configuration for a single modality's projection.
    
    Args:
        input_dim: Input dimension from the modality encoder
        output_dim: Target dimension for the modality projection
        use_bias: Whether to use bias in the projection layer
        dropout: Dropout probability
        layernorm: Whether to apply layer normalization
    
    Examples:
        >>> text_config = ModalityConfig(input_dim=4096, output_dim=4096)
        >>> image_config = ModalityConfig(input_dim=1024, output_dim=4096, dropout=0.2)
    """
    input_dim: int
    output_dim: int
    use_bias: bool = True
    dropout: float = 0.1
    layernorm: bool = True


@dataclass
class ProjectorConfig:
    """Configuration for the modality projectors module.
    
    Centralizes all configuration parameters for modality projections,
    supporting dynamic addition of new modalities and ensuring type safety.
    
    Args:
        d_model: Target dimension for all modalities
        modalities: Dictionary mapping modality names to their configurations
        bias_init_std: Standard deviation for bias initialization
        bias_init_mean: Mean for bias initialization
        max_seq_len: Maximum allowable sequence length per modality
        device: Device to use for computation
        dtype: Data type for computations
        
    Examples:
        >>> config = ProjectorConfig(
        ...     d_model=4096,
        ...     modalities={
        ...         "text": ModalityConfig(input_dim=4096, output_dim=4096),
        ...         "image": ModalityConfig(input_dim=1024, output_dim=4096)
        ...     }
        ... )
    """
    d_model: int = 4096
    modalities: Dict[str, ModalityConfig] = field(default_factory=dict)
    bias_init_std: float = DEFAULT_BIAS_INIT["std"]
    bias_init_mean: float = DEFAULT_BIAS_INIT["mean"]
    max_seq_len: int = 2048
    device: Optional[Union[str, torch.device]] = None
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        # Initialize with defaults if modalities is empty
        if not self.modalities:
            # Use the enum members directly instead of their string values
            for mod_type in ModalityType:  # Iterate over enum members
                self.modalities[mod_type] = ModalityConfig(
                    input_dim=DEFAULT_DIMENSIONS[mod_type],
                    output_dim=self.d_model
                )
        
        # Validate configurations
        for mod_name, config in self.modalities.items():
            if config.output_dim != self.d_model:
                logger.warning(
                    f"Output dimension mismatch for modality {mod_name}: "
                    f"{config.output_dim} != {self.d_model}. "
                    f"Setting output_dim to {self.d_model}."
                )
                # Create a new config with the corrected output_dim
                self.modalities[mod_name] = ModalityConfig(
                    input_dim=config.input_dim,
                    output_dim=self.d_model,
                    use_bias=config.use_bias,
                    dropout=config.dropout,
                    layernorm=config.layernorm
                )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProjectorConfig':
        """Create a ProjectorConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            ProjectorConfig instance
            
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        try:
            # Extract main parameters
            d_model = config_dict.get("d_model", 4096)
            modalities = {}
            
            # Extract modalities configurations
            mod_configs = config_dict.get("modalities", {})
            for mod_name, mod_config in mod_configs.items():
                if isinstance(mod_config, dict):
                    modalities[mod_name] = ModalityConfig(**mod_config)
                elif isinstance(mod_config, int):
                    # If only dimension is provided, use it as input_dim
                    modalities[mod_name] = ModalityConfig(
                        input_dim=mod_config,
                        output_dim=d_model
                    )
                else:
                    raise ConfigurationError(f"Invalid configuration for modality {mod_name}")
            
            # Create and return the config
            return cls(
                d_model=d_model,
                modalities=modalities,
                bias_init_std=config_dict.get("bias_init_std", DEFAULT_BIAS_INIT["std"]),
                bias_init_mean=config_dict.get("bias_init_mean", DEFAULT_BIAS_INIT["mean"]),
                max_seq_len=config_dict.get("max_seq_len", 2048),
                device=config_dict.get("device", None),
                dtype=getattr(torch, config_dict.get("dtype", "float32"))
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create ProjectorConfig: {str(e)}")
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ProjectorConfig':
        """Load a ProjectorConfig from a JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            ProjectorConfig instance
            
        Raises:
            ConfigurationError: If loading or parsing fails
        """
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {json_path}: {str(e)}")

@contextmanager
def timer(name: str, threshold_ms: float = 10.0, log_level: int = logging.DEBUG):
    """Context manager for timing code blocks.
    
    Args:
        name: Name of the operation being timed
        threshold_ms: Log only if execution time exceeds this threshold (ms)
        log_level: Logging level to use
        
    Yields:
        None
    """
    start = time.time()
    try:
        yield  # This makes it a proper generator
    finally:
        elapsed_ms = (time.time() - start) * 1000
        if elapsed_ms > threshold_ms:
            logger.log(log_level, f"Operation '{name}' took {elapsed_ms:.2f}ms")


class LayerNormWithBias(nn.Module):
    """Layer normalization with optional bias.
    
    Implements a more efficient layer normalization that can disable bias
    while maintaining the same interface.
    
    Args:
        hidden_size: Size of the hidden dimension
        eps: Epsilon for numerical stability
        bias: Whether to include a bias term
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Normalized tensor of same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            return normalized * self.weight + self.bias
        else:
            return normalized * self.weight


class SequencePadder:
    """Utility class for padding and unpadding sequences.
    
    Handles padding sequences to a uniform length and creating attention masks.
    """
    @staticmethod
    def pad_sequence(
        x: torch.Tensor, 
        pad_len: int, 
        pad_value: float = 0.0
    ) -> torch.Tensor:
        """Pad sequence to a specific length.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            pad_len: Target sequence length
            pad_value: Value to use for padding
            
        Returns:
            Padded tensor of shape (batch_size, pad_len, hidden_size)
        """
        assert x.dim() == 3, "Expected 3D tensor (batch_size, seq_len, hidden_size)"
        batch_size, seq_len, hidden_size = x.shape
        if seq_len >= pad_len:
            return x[:, :pad_len, :]
        
        padding = torch.full(
            (batch_size, pad_len - seq_len, hidden_size),
            pad_value,
            dtype=x.dtype,
            device=x.device
        )
        return torch.cat([x, padding], dim=1)
    
    @staticmethod
    def create_padding_mask(
        x: torch.Tensor, 
        seq_lens: Union[torch.Tensor, List[int]]
    ) -> torch.Tensor:
        """Create an attention mask for padded sequences.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            seq_lens: Original sequence lengths before padding
            
        Returns:
            Boolean mask of shape (batch_size, seq_len) where True indicates valid positions
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if isinstance(seq_lens, list):
            seq_lens = torch.tensor(seq_lens, device=device)
            
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) < seq_lens.unsqueeze(1)
        return mask


class ModalityEmbedding(nn.Module):
    """Learnable embeddings for each modality type.
    
    Creates distinct embeddings for different modality types.
    
    Args:
        d_model: Embedding dimension
        modalities: List of modality types
        init_std: Standard deviation for initialization
        init_mean: Mean for initialization
    """
    def __init__(
        self, 
        d_model: int, 
        modalities: List[str],
        init_std: float = 0.02,
        init_mean: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.ParameterDict()
        
        for mod in modalities:
            param = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(param, mean=init_mean, std=init_std)
            self.embeddings[mod] = param
    
    def forward(self, mod: str) -> torch.Tensor:
        """Get embeddings for a specific modality.
        
        Args:
            mod: Modality type name
            
        Returns:
            Embedding tensor of shape (1, 1, d_model)
            
        Raises:
            UnknownModalityError: If modality is not recognized
        """
        if mod not in self.embeddings:
            raise UnknownModalityError(f"Unknown modality: {mod}")
        return self.embeddings[mod]


class TemporalPositionEmbedding(nn.Module):
    """Temporal position embeddings for token sequences.
    
    Creates position embeddings that encode temporal information for 
    cross-modality alignment.
    
    Args:
        d_model: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        learnable: Whether positions are learnable parameters
    """
    def __init__(
        self, 
        d_model: int, 
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        learnable: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        if learnable:
            self.positions = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.normal_(self.positions, mean=0.0, std=0.02)
        else:
            # Use fixed sinusoidal embeddings
            position = torch.arange(max_seq_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pos_embedding = torch.zeros(1, max_seq_len, d_model)
            pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
            pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('positions', pos_embedding)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply temporal position embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            position_ids: Optional explicit position indices
            
        Returns:
            Tensor with position embeddings added
            
        Raises:
            SequenceLengthError: If sequence length exceeds maximum
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_seq_len:
            raise SequenceLengthError(
                f"Input sequence length {seq_len} exceeds maximum allowed {self.max_seq_len}"
            )
        
        if position_ids is None:
            positions = self.positions[:, :seq_len, :]
        else:
            positions = self.positions[position_ids]
            
        x = x + positions
        return self.dropout(x)
        token_masks: Optional[Dict[str, torch.Tensor]] = None
        
        
class ModalityProjector(nn.Module):
    """Projects encoded modality outputs into a unified latent tensor space.
    
    This module is a critical component in the GPT-4o architecture, responsible for
    projecting encoded outputs from different modalities (text, image, audio, video)
    into a unified latent tensor space that can be processed by the transformer backbone.
    
    Features:
    - Modality-specific linear projections
    - Learned modality bias tokens
    - Sequence normalization with padding and masking
    - Support for parallel processing of multiple modalities
    - Optional layer normalization and dropout
    - Context managers for mixed precision and timing
    
    Usage example:
        >>> config = ProjectorConfig(
        ...     d_model=4096,
        ...     modalities={
        ...         "text": ModalityConfig(input_dim=4096, output_dim=4096),
        ...         "image": ModalityConfig(input_dim=1024, output_dim=4096)
        ...     }
        ... )
        >>> projector = ModalityProjector(config)
        >>> encoded = {
        ...     "text": torch.randn(2, 512, 4096),
        ...     "image": torch.randn(2, 256, 1024)
        ... }
        >>> outputs, masks = projector(encoded)
        >>> outputs.shape
        torch.Size([2, 768, 4096])
        >>> masks.shape
        torch.Size([2, 768])
    
    Args:
        config: Configuration parameters for the projector
    """
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.dtype = config.dtype
        self.max_seq_len = config.max_seq_len
        
        # Get all modality types from config
        self.modalities = list(config.modalities.keys())
        
        # Initialize submodules
        self._build_projectors()
        self._build_normalization()
        self._build_modality_embeddings()
        
        # Initialize sequence padder
        self.padder = SequencePadder()
        
        # Set device if specified
        if config.device is not None:
            self.to(config.device)
    
    def _build_projectors(self) -> None:
        """Build projection layers for each modality."""
        self.proj = nn.ModuleDict()
        
        for mod, mod_config in self.config.modalities.items():
            # Validate dimensions
            if mod_config.input_dim <= 0:
                raise ConfigurationError(f"Invalid input dimension for {mod}: {mod_config.input_dim}")
            if self.d_model <= 0:
                raise ConfigurationError(f"Invalid model dimension: {self.d_model}")
            
            # Create linear projection
            projection = nn.Linear(
                mod_config.input_dim, 
                self.d_model, 
                bias=mod_config.use_bias
            )
            
            # Initialize projection weights
            nn.init.normal_(projection.weight, std=0.02)
            if mod_config.use_bias and projection.bias is not None:
                nn.init.zeros_(projection.bias)
            
            # Ensure correct dtype and device
            if hasattr(self, 'dtype') and self.dtype is not torch.float32:
                projection = projection.to(dtype=self.dtype)
                
            # Add to module dictionary
            self.proj[mod] = projection
            
            logger.debug(f"Created projection for {mod}: {mod_config.input_dim} → {self.d_model}")
    
    def _build_normalization(self) -> None:
        """Build normalization and dropout layers for each modality."""
        self.norm = nn.ModuleDict()
        self.dropout = nn.ModuleDict()
        
        for mod, mod_config in self.config.modalities.items():
            if mod_config.layernorm:
                self.norm[mod] = LayerNormWithBias(
                    self.d_model, 
                    eps=1e-5, 
                    bias=mod_config.use_bias
                )
            
            self.dropout[mod] = nn.Dropout(mod_config.dropout)
    
    def _build_modality_embeddings(self) -> None:
        """Build learnable modality embeddings."""
        # Modality-specific embeddings
        self.mod_embed = ModalityEmbedding(
            self.d_model,
            self.modalities,
            init_std=self.config.bias_init_std,
            init_mean=self.config.bias_init_mean
        )
        
        # Temporal position embeddings
        self.pos_embed = TemporalPositionEmbedding(
            self.d_model,
            max_seq_len=self.max_seq_len
        )
    
    def _validate_inputs(self, encoded: Dict[str, torch.Tensor]) -> None:
        """Validate input tensors.
        
        Args:
            encoded: Dictionary of modality encoded tensors
            
        Raises:
            ConfigurationError: If no supported modalities are present
            DimensionMismatchError: If dimension doesn't match configuration
            SequenceLengthError: If sequence length exceeds maximum
        """
        found_modalities = [m for m in encoded.keys() if m in self.modalities]
        
        if not found_modalities:
            raise ConfigurationError(
                f"No supported modalities found in input. Expected one of: {self.modalities}"
            )
        
        for mod in found_modalities:
            x = encoded[mod]
            # Check dimensions
            if x.dim() != 3:
                raise DimensionMismatchError(
                    f"Expected 3D tensor (batch_size, seq_len, hidden_size) for {mod}, "
                    f"got shape {x.shape}"
                )
            
            # Check input dimension
            expected_dim = self.config.modalities[mod].input_dim
            if x.size(2) != expected_dim:
                raise DimensionMismatchError(
                    f"Input dimension mismatch for {mod}: expected {expected_dim}, "
                    f"got {x.size(2)}"
                )
            
            # Check sequence length
            if x.size(1) > self.max_seq_len:
                raise SequenceLengthError(
                    f"Sequence length {x.size(1)} for {mod} exceeds maximum allowed "
                    f"{self.max_seq_len}"
                )
    
    def _project_modality(
        self, 
        x: torch.Tensor, 
        mod: str,
        apply_modality_embedding: bool = True
    ) -> torch.Tensor:
        """Project a single modality tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mod: Modality name
            apply_modality_embedding: Whether to apply modality embeddings
            
        Returns:
            Projected tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to target dimension
        with timer(f"project_{mod}", threshold_ms=5.0):
            x = self.proj[mod](x)
        
        # Apply modality embedding if requested
        if apply_modality_embedding:
            with timer(f"mod_embed_{mod}", threshold_ms=5.0):
                # Get and expand modality embedding to match sequence length
                mod_emb = self.mod_embed(mod)
                mod_emb = mod_emb.expand(batch_size, seq_len, self.d_model)
                x = x + mod_emb
        
        # Apply normalization if configured
        if mod in self.norm:
            with timer(f"norm_{mod}", threshold_ms=5.0):
                x = self.norm[mod](x)
        
        # Apply dropout
        with timer(f"dropout_{mod}", threshold_ms=5.0):
            x = self.dropout[mod](x)
        
        return x
    
    def forward(
        self, 
        encoded: Dict[str, torch.Tensor],
        position_ids: Optional[Dict[str, torch.Tensor]] = None,
        apply_positional_embedding: bool = True,
        apply_modality_embedding: bool = True,
        token_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project and fuse all modality sequences into a single tensor.
        
        Args:
            encoded: Dict mapping modality name to tensor of shape (batch, seq, dim)
            position_ids: Optional dict mapping modality to position indices
            apply_positional_embedding: Whether to apply positional embeddings
            apply_modality_embedding: Whether to apply modality embeddings
            token_masks: Optional dict mapping modality to boolean mask (batch, seq)
            
        Returns:
            Tuple of:
                - Projected tensor of shape (batch, total_seq, d_model)
                - Boolean mask of shape (batch, total_seq) (True=valid, False=pad)
                
        Raises:
            Various exceptions defined in exception hierarchy
        """
        # Input validation
        with timer("validate_inputs", threshold_ms=1.0):
            self._validate_inputs(encoded)
        
        # Get batch size from first modality
        batch_size = next(iter(encoded.values())).size(0)
        
        # Use mixed precision if available
        with autocast(enabled=torch.cuda.is_available()):
            # Process each modality
            projected_tensors = []
            sequence_lengths = []
            valid_modalities = []
            
            for mod in self.modalities:
                if mod in encoded:
                    valid_modalities.append(mod)
                    
                    # Get input tensor and its length
                    x = encoded[mod]
                    seq_len = x.size(1)
                    sequence_lengths.append(seq_len)
                    
                    # Apply projection
                    x_proj = self._project_modality(
                        x, 
                        mod, 
                        apply_modality_embedding=apply_modality_embedding
                    )
                    
                    # Create or get mask for this modality
                    if token_masks is not None and mod in token_masks:
                        mask = token_masks[mod]
                        if mask.size() != (batch_size, seq_len):
                            raise DimensionMismatchError(
                                f"Mask shape mismatch for {mod}: expected {(batch_size, seq_len)}, "
                                f"got {mask.size()}"
                            )
                    else:
                        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
                    
                    projected_tensors.append((x_proj, mask))
            
            if not projected_tensors:
                logger.warning("No valid modalities found in input.")
                # Return empty tensors with correct dimensions
                empty_tensor = torch.zeros(
                    batch_size, 0, self.d_model, 
                    dtype=self.dtype, 
                    device=next(self.parameters()).device
                )
                empty_mask = torch.zeros(
                    batch_size, 0, 
                    dtype=torch.bool, 
                    device=next(self.parameters()).device
                )
                return empty_tensor, empty_mask
                
            # Concatenate all projections
            with timer("concatenate_tensors", threshold_ms=5.0):
                all_tensors = []
                all_masks = []
                
                for tensor, mask in projected_tensors:
                    all_tensors.append(tensor)
                    all_masks.append(mask)
                
                # Concatenate along sequence dimension
                output_tensor = torch.cat(all_tensors, dim=1)
                output_mask = torch.cat(all_masks, dim=1)
            
            # Apply temporal position embeddings if requested
            if apply_positional_embedding:
                with timer("apply_pos_embed", threshold_ms=5.0):
                    # Use provided position IDs if available
                    pos_ids = None
                    if position_ids is not None:
                        # Concatenate position IDs from all modalities
                        all_pos_ids = []
                        for mod in valid_modalities:
                            if mod in position_ids:
                                all_pos_ids.append(position_ids[mod])
                        
                        if all_pos_ids:
                            pos_ids = torch.cat(all_pos_ids, dim=1)
                    
                    # Apply position embeddings
                    output_tensor = self.pos_embed(output_tensor, pos_ids)
            
            # Return projected tensor and mask
            return output_tensor, output_mask
    
    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> 'ModalityProjector':
        """Load a pretrained ModalityProjector from a directory.
        
        Args:
            path: Path to the directory containing the configuration and weights
            
        Returns:
            ModalityProjector instance
            
        Raises:
            ConfigurationError: If loading fails
        """
        path = Path(path)
        try:
            # Load configuration
            config_path = path / 'config.json'
            config = ProjectorConfig.from_json(config_path)
            
            # Create model
            model = cls(config)
            
            # Load weights
            weights_path = path / 'weights.pt'
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            return model
        except Exception as e:
            raise ConfigurationError(f"Failed to load model from {path}: {str(e)}")
    
    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the model to a directory.
        
        Args:
            path: Path to the directory where the model should be saved
            
        Raises:
            ConfigurationError: If saving fails
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save configuration
            config_dict = {
                'd_model': self.d_model,
                'modalities': {
                    name: {
                        'input_dim': config.input_dim,
                        'output_dim': config.output_dim,
                        'use_bias': config.use_bias,
                        'dropout': config.dropout,
                        'layernorm': config.layernorm
                    }
                    for name, config in self.config.modalities.items()
                },
                'bias_init_std': self.config.bias_init_std,
                'bias_init_mean': self.config.bias_init_mean,
                'max_seq_len': self.max_seq_len,
                'dtype': self.dtype.__str__().split('.')[-1]
            }
            
            with open(path / 'config.json', 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Save weights
            torch.save(self.state_dict(), path / 'weights.pt')
        except Exception as e:
            raise ConfigurationError(f"Failed to save model to {path}: {str(e)}")

class BatchProcessor:
    """Utility class for efficient batch processing of modality inputs.
    
    Provides optimized batch processing with automatic chunking for large inputs,
    memory management, and caching capabilities.
    
    Args:
        projector: ModalityProjector instance
        chunk_size: Maximum number of sequences per processing batch
        cache_embeddings: Whether to cache projection results
        max_cache_size: Maximum number of cached embeddings
    """
    def __init__(
        self,
        projector: ModalityProjector,
        chunk_size: int = 32,
        cache_embeddings: bool = True,
        max_cache_size: int = 1000,
    ):
        self.projector = projector
        self.chunk_size = chunk_size
        self.cache_embeddings = cache_embeddings
        self.max_cache_size = max_cache_size
        self.embedding_cache = {}
    
    def _get_cache_key(self, tensor: torch.Tensor, mod: str) -> str:
        """Generate a unique cache key for a tensor.
        
        Args:
            tensor: Input tensor
            mod: Modality name
            
        Returns:
            Cache key string
        """
        # Use the first and last values along with shape as a simple hash
        if tensor.numel() > 0:
            first_val = float(tensor[0, 0, 0].cpu().detach())
            last_val = float(tensor[-1, -1, -1].cpu().detach())
            return f"{mod}_{tensor.shape}_{first_val:.4f}_{last_val:.4f}"
        return f"{mod}_{tensor.shape}_empty"
    
    def _get_from_cache(self, tensor: torch.Tensor, mod: str) -> Optional[torch.Tensor]:
        """Retrieve cached embedding if available.
        
        Args:
            tensor: Input tensor
            mod: Modality name
            
        Returns:
            Cached projection if available, None otherwise
        """
        if not self.cache_embeddings:
            return None
        
        key = self._get_cache_key(tensor, mod)
        return self.embedding_cache.get(key, None)
    
    def _add_to_cache(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, mod: str) -> None:
        """Add projection result to cache.
        
        Args:
            input_tensor: Original input tensor
            output_tensor: Projected output tensor
            mod: Modality name
        """
        if not self.cache_embeddings:
            return
        
        # Basic cache management - remove oldest entry if at capacity
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remove least recently used item (first item in dict)
            self.embedding_cache.pop(next(iter(self.embedding_cache.keys())))
        
        key = self._get_cache_key(input_tensor, mod)
        self.embedding_cache[key] = output_tensor
    
    def process_batch(
        self, 
        encoded: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of inputs with automatic chunking and caching.
        
        Args:
            encoded: Dict mapping modality name to tensor
            **kwargs: Additional arguments to pass to the projector
            
        Returns:
            Tuple of projected tensor and attention mask
        """
        # Check if batch size exceeds chunk size
        batch_sizes = [tensor.size(0) for tensor in encoded.values()]
        if not batch_sizes:
            return torch.tensor([]), torch.tensor([])
        
        max_batch = max(batch_sizes)
        if max_batch <= self.chunk_size:
            # Process entire batch at once
            return self.projector(encoded, **kwargs)
        
        # Process in chunks
        outputs = []
        masks = []
        
        for i in range(0, max_batch, self.chunk_size):
            end = min(i + self.chunk_size, max_batch)
            chunk_encoded = {
                mod: tensor[i:end] if tensor.size(0) > i else tensor
                for mod, tensor in encoded.items()
            }
            
            # Process chunk
            chunk_output, chunk_mask = self.projector(chunk_encoded, **kwargs)
            outputs.append(chunk_output)
            masks.append(chunk_mask)
        
        # Concatenate results
        return torch.cat(outputs, dim=0), torch.cat(masks, dim=0)


#------------------------------------------------------------------------------
# Multimodal Fusion Strategies
#------------------------------------------------------------------------------

class FusionStrategy(nn.Module):
    """Base class for multimodal fusion strategies.
    
    Defines the interface for fusion strategies that combine modality projections.
    """
    def forward(
        self, 
        encoded_modalities: Dict[str, torch.Tensor],
        modality_projector: ModalityProjector,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fusion strategy to modality projections.
        
        Args:
            encoded_modalities: Dict mapping modality name to tensor
            modality_projector: ModalityProjector instance
            **kwargs: Additional arguments
            
        Returns:
            Tuple of fused tensor and attention mask
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward method")


class SequentialFusion(FusionStrategy):
    """Sequential fusion strategy.
    
    Concatenates modality projections in a fixed order.
    
    Args:
        modality_order: Order in which to concatenate modalities
    """
    def __init__(self, modality_order: Optional[List[str]] = None):
        super().__init__()
        self.modality_order = modality_order
    
    def forward(
        self, 
        encoded_modalities: Dict[str, torch.Tensor],
        modality_projector: ModalityProjector,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply sequential fusion to modality projections.
        
        Args:
            encoded_modalities: Dict mapping modality name to tensor
            modality_projector: ModalityProjector instance
            **kwargs: Additional arguments to pass to the projector
            
        Returns:
            Tuple of fused tensor and attention mask
        """
        # Determine modality order
        if self.modality_order is None:
            # Default order based on available modalities
            available = list(encoded_modalities.keys())
            # Prioritize text first if available
            if "text" in available:
                available.remove("text")
                modality_order = ["text"] + available
            else:
                modality_order = available
        else:
            # Filter by available modalities
            modality_order = [
                mod for mod in self.modality_order 
                if mod in encoded_modalities
            ]
        
        # Reorder encoded modalities
        ordered_modalities = {
            mod: encoded_modalities[mod] 
            for mod in modality_order 
            if mod in encoded_modalities
        }
        
        # Project and fuse using the projector
        return modality_projector(ordered_modalities, **kwargs)


class WeightedFusion(FusionStrategy):
    """Weighted fusion strategy.
    
    Applies learnable weights to each modality's projection before fusion.
    
    Args:
        modality_weights: Initial weights for each modality
        learnable: Whether weights are learnable parameters
    """
    def __init__(
        self, 
        modality_weights: Optional[Dict[str, float]] = None,
        learnable: bool = True
    ):
        super().__init__()
        self.modality_weights = modality_weights or {}
        self.learnable = learnable
        self.weights = nn.ParameterDict()
        
        # Initialize with identity weights if not specified
        for mod, weight in self.modality_weights.items():
            if learnable:
                self.weights[mod] = nn.Parameter(torch.tensor(weight))
            else:
                self.register_buffer(f"weight_{mod}", torch.tensor(weight))
    
    def _get_weight(self, mod: str) -> torch.Tensor:
        """Get weight for a modality.
        
        Args:
            mod: Modality name
            
        Returns:
            Weight tensor
        """
        if self.learnable:
            if mod not in self.weights:
                self.weights[mod] = nn.Parameter(torch.tensor(1.0))
            return self.weights[mod]
        else:
            buffer_name = f"weight_{mod}"
            if not hasattr(self, buffer_name):
                weight = self.modality_weights.get(mod, 1.0)
                self.register_buffer(buffer_name, torch.tensor(weight))
            return getattr(self, buffer_name)
    
    def forward(
        self, 
        encoded_modalities: Dict[str, torch.Tensor],
        modality_projector: ModalityProjector,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply weighted fusion to modality projections.
        
        Args:
            encoded_modalities: Dict mapping modality name to tensor
            modality_projector: ModalityProjector instance
            **kwargs: Additional arguments to pass to the projector
            
        Returns:
            Tuple of fused tensor and attention mask
        """
        # Process each modality separately
        projected = {}
        masks = {}
        
        for mod, tensor in encoded_modalities.items():
            # Create single-modality input
            single_input = {mod: tensor}
            
            # Project modality
            proj, mask = modality_projector(single_input, **kwargs)
            
            # Apply weight
            weight = self._get_weight(mod)
            proj = proj * weight
            
            projected[mod] = proj
            masks[mod] = mask
        
        # Concatenate all projections
        all_projections = []
        all_masks = []
        
        for mod in projected:
            all_projections.append(projected[mod])
            all_masks.append(masks[mod])
        
        if not all_projections:
            # Return empty tensors with correct dimensions
            batch_size = next(iter(encoded_modalities.values())).size(0)
            device = next(modality_projector.parameters()).device
            return (
                torch.zeros(batch_size, 0, modality_projector.d_model, device=device),
                torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            )
        
        # Concatenate along sequence dimension
        output_tensor = torch.cat(all_projections, dim=1)
        output_mask = torch.cat(all_masks, dim=1)
        
        return output_tensor, output_mask


class AttentionFusion(FusionStrategy):
    """Cross-attention fusion strategy.
    
    Uses cross-attention to dynamically weight and fuse modality projections.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(
        self, 
        d_model: int = 4096, 
        num_heads: int = 16, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Multi-head attention for fusion
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer norm
        self.norm = LayerNormWithBias(d_model)
    
    def forward(
        self, 
        encoded_modalities: Dict[str, torch.Tensor],
        modality_projector: ModalityProjector,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention-based fusion to modality projections.
        
        Args:
            encoded_modalities: Dict mapping modality name to tensor
            modality_projector: ModalityProjector instance
            **kwargs: Additional arguments to pass to the projector
            
        Returns:
            Tuple of fused tensor and attention mask
        """
        # First get regular projection
        base_projection, base_mask = modality_projector(encoded_modalities, **kwargs)
        
        if base_projection.size(1) == 0:
            # Empty projection, nothing to fuse
            return base_projection, base_mask
        
        # Create attention mask for padding
        attn_mask = ~base_mask.unsqueeze(1).expand(-1, base_mask.size(1), -1)
        
        # Apply cross-attention
        attn_output, _ = self.mha(
            query=base_projection,
            key=base_projection,
            value=base_projection,
            attn_mask=attn_mask
        )
        
        # Add residual and normalize
        output = self.norm(base_projection + self.out_proj(attn_output))
        
        return output, base_mask


#------------------------------------------------------------------------------
# Factory Methods
#------------------------------------------------------------------------------

def create_projector(
    config_path: Optional[Union[str, Path]] = None,
    d_model: int = 4096,
    modalities: Optional[Dict[str, Union[int, Dict[str, Any]]]] = None,
    fusion_strategy: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> Union[ModalityProjector, FusionStrategy]:
    """Create a ModalityProjector with optional fusion strategy.
    
    Factory function to simplify creating projector instances with various configurations.
    
    Args:
        config_path: Path to JSON configuration file
        d_model: Model dimension if not using config file
        modalities: Dict mapping modality names to input dimensions or configs
        fusion_strategy: Name of fusion strategy to use
        device: Device to place model on
        **kwargs: Additional arguments for ProjectorConfig
        
    Returns:
        ModalityProjector or FusionStrategy instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Create config
    if config_path is not None:
        config = ProjectorConfig.from_json(config_path)
    else:
        # Create modalities dict from input
        modality_configs = {}
        if modalities:
            for mod_name, mod_config in modalities.items():
                if isinstance(mod_config, int):
                    modality_configs[mod_name] = ModalityConfig(
                        input_dim=mod_config,
                        output_dim=d_model
                    )
                elif isinstance(mod_config, dict):
                    # Ensure output_dim is set to d_model
                    mod_config["output_dim"] = d_model
                    modality_configs[mod_name] = ModalityConfig(**mod_config)
        
        # Create config with other parameters
        config = ProjectorConfig(
            d_model=d_model,
            modalities=modality_configs,
            device=device,
            **kwargs
        )
    
    # Create projector
    projector = ModalityProjector(config)
    
    # Add fusion strategy if specified
    if fusion_strategy:
        if fusion_strategy.lower() == "sequential":
            return SequentialFusion()(projector)
        elif fusion_strategy.lower() == "weighted":
            return WeightedFusion()(projector)
        elif fusion_strategy.lower() == "attention":
            return AttentionFusion(d_model=d_model)(projector)
        else:
            raise ConfigurationError(f"Unknown fusion strategy: {fusion_strategy}")
    
    return projector


def load_pretrained(
    path: Union[str, Path], 
    device: Optional[Union[str, torch.device]] = None
) -> ModalityProjector:
    """Load a pretrained ModalityProjector.
    
    Simplified function to load a pretrained model.
    
    Args:
        path: Path to the directory containing the model
        device: Device to place model on
        
    Returns:
        Loaded ModalityProjector
    """
    model = ModalityProjector.from_pretrained(path)
    
    if device is not None:
        model = model.to(device)
    
    return model


#------------------------------------------------------------------------------
# Benchmarking Utilities
#------------------------------------------------------------------------------

class ProjectorBenchmark:
    """Benchmark utility for ModalityProjector.
    
    Measures performance metrics for the modality projector, including
    latency, throughput, and memory usage.
    
    Args:
        projector: ModalityProjector instance
        device: Device for benchmarking
    """
    def __init__(
        self,
        projector: ModalityProjector,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.projector = projector
        self.device = device or (
            next(projector.parameters()).device 
            if list(projector.parameters()) else torch.device("cpu")
        )
    
    def generate_synthetic_data(
        self,
        batch_size: int = 4,
        seq_lens: Optional[Dict[str, int]] = None,
        modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic data for benchmarking.
        
        Args:
            batch_size: Batch size
            seq_lens: Dict mapping modality name to sequence length
            modalities: List of modalities to include
            
        Returns:
            Dict mapping modality name to synthetic tensor
        """
        device = self.device
        modalities = modalities or self.projector.modalities
        seq_lens = seq_lens or {mod: 128 for mod in modalities}
        
        synthetic_data = {}
        
        for mod in modalities:
            if mod in self.projector.config.modalities:
                mod_config = self.projector.config.modalities[mod]
                synthetic_data[mod] = torch.randn(
                    batch_size,
                    seq_lens.get(mod, 128),
                    mod_config.input_dim,
                    device=device
                )
        
        return synthetic_data
    
    def measure_latency(
        self,
        batch_size: int = 4,
        seq_lens: Optional[Dict[str, int]] = None,
        modalities: Optional[List[str]] = None,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """Measure latency for various operations.
        
        Args:
            batch_size: Batch size
            seq_lens: Dict mapping modality name to sequence length
            modalities: List of modalities to include
            num_iterations: Number of iterations to average over
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dict with latency metrics in milliseconds
        """
        # Generate synthetic data
        data = self.generate_synthetic_data(
            batch_size=batch_size,
            seq_lens=seq_lens,
            modalities=modalities
        )
        
        # Warm up
        for _ in range(warmup_iterations):
            with torch.no_grad():
                self.projector(data)
        
        # Measure latency
        torch.cuda.synchronize(self.device)
        start = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                self.projector(data)
        
        torch.cuda.synchronize(self.device)
        elapsed = time.time() - start
        
        # Compute metrics
        total_latency = elapsed * 1000 / num_iterations  # ms
        samples_per_sec = num_iterations * batch_size / elapsed
        
        return {
            "latency_ms": total_latency,
            "samples_per_sec": samples_per_sec,
            "batch_size": batch_size,
            "num_iterations": num_iterations
        }
    
    def profile_memory(
        self,
        batch_size: int = 4,
        seq_lens: Optional[Dict[str, int]] = None,
        modalities: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Profile memory usage during projection.
        
        Args:
            batch_size: Batch size
            seq_lens: Dict mapping modality name to sequence length
            modalities: List of modalities to include
            
        Returns:
            Dict with memory usage metrics in MB
        """
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for memory profiling.")
        
        # Record starting memory
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start_mem = torch.cuda.memory_allocated(self.device) / (1024 * 1024)  # MB
        
        # Generate synthetic data
        data = self.generate_synthetic_data(
            batch_size=batch_size,
            seq_lens=seq_lens,
            modalities=modalities
        )
        
        # Run projection
        with torch.no_grad():
            self.projector(data)
        
        # Record peak memory
        torch.cuda.synchronize(self.device)
        peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)  # MB
        current_mem = torch.cuda.memory_allocated(self.device) / (1024 * 1024)  # MB
        
        return {
            "start_memory_mb": start_mem,
            "peak_memory_mb": peak_mem,
            "current_memory_mb": current_mem,
            "used_memory_mb": peak_mem - start_mem
        }
    
    def run_comprehensive_benchmark(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        modalities: Optional[List[str]] = None,
        seq_lens: Optional[Dict[str, int]] = None,
        num_iterations: int = 50,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite.
        
        Args:
            batch_sizes: List of batch sizes to benchmark
            modalities: List of modalities to include
            seq_lens: Dict mapping modality name to sequence length
            num_iterations: Number of iterations to average over
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dict with comprehensive benchmark results
        """
        results = {
            "device": str(self.device),
            "model_params": sum(p.numel() for p in self.projector.parameters()),
            "latency": {},
            "memory": {}
        }
        
        for batch_size in batch_sizes:
            # Measure latency
            latency = self.measure_latency(
                batch_size=batch_size,
                seq_lens=seq_lens,
                modalities=modalities,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations
            )
            
            results["latency"][f"batch_{batch_size}"] = latency
            
            # Profile memory
            if torch.cuda.is_available():
                memory = self.profile_memory(
                    batch_size=batch_size,
                    seq_lens=seq_lens,
                    modalities=modalities
                )
                
                results["memory"][f"batch_{batch_size}"] = memory
        
        return results

def run_simple_example(
    batch_size: int = 2,
    text_seq_len: int = 512,
    image_seq_len: int = 256,
    audio_seq_len: int = 128,
    d_model: int = 4096,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Run a simple example of modality projection.
    
    This function demonstrates basic usage of the ModalityProjector with
    synthetic data, useful for testing and verification.
    
    Args:
        batch_size: Number of sequences in batch
        text_seq_len: Length of text sequences
        image_seq_len: Length of image sequences
        audio_seq_len: Length of audio sequences
        d_model: Model dimension
        device: Device to run on
        
    Returns:
        Dict with results and timings
    """
    # Use CUDA if available and not specified
    if device is None and torch.cuda.is_available():
        device = "cuda"
    elif device is None:
        device = "cpu"
        
    # Create synthetic data for each modality
    encoded = {
        "text": torch.randn(batch_size, text_seq_len, d_model, device=device),
        "image": torch.randn(batch_size, image_seq_len, d_model, device=device),
        "audio": torch.randn(batch_size, audio_seq_len, d_model, device=device)
    }
    
    # Create configuration
    config = ProjectorConfig(
        d_model=d_model,
        modalities={
            "text": ModalityConfig(input_dim=d_model, output_dim=d_model),
            "image": ModalityConfig(input_dim=d_model, output_dim=d_model),
            "audio": ModalityConfig(input_dim=d_model, output_dim=d_model),
        },
        device=device
    )
    
    # Create projector
    projector = ModalityProjector(config)
    
    # Measure projection time
    start = time.time()
    
    # Forward pass
    output, mask = projector(encoded)
    
    elapsed = time.time() - start
    
    # Return results
    return {
        "output_shape": output.shape,
        "mask_shape": mask.shape,
        "elapsed_ms": elapsed * 1000,
        "device": device,
        "config": config
    }


class ExampleGenerator:
    """Generates example inputs and outputs for documentation and testing.
    
    This class provides utilities for generating synthetic modality data
    and running examples with the ModalityProjector.
    
    Args:
        d_model: Model dimension
        device: Device to use
    """
    def __init__(self, d_model: int = 4096, device: Optional[str] = None):
        self.d_model = d_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate_example_input(
        self,
        batch_size: int = 1,
        modalities: Optional[Dict[str, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate example inputs for multiple modalities.
        
        Args:
            batch_size: Batch size
            modalities: Dict mapping modality name to sequence length
            
        Returns:
            Dict mapping modality name to input tensor
        """
        modalities = modalities or {
            "text": 512,
            "image": 256,
            "audio": 128
        }
        
        input_dict = {}
        
        for mod_name, seq_len in modalities.items():
            input_dict[mod_name] = torch.randn(
                batch_size, seq_len, self.d_model, device=self.device
            )
        
        return input_dict
    
    def generate_example_masks(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate example attention masks for inputs.
        
        Args:
            inputs: Dict mapping modality name to input tensor
            
        Returns:
            Dict mapping modality name to attention mask
        """
        masks = {}
        
        for mod_name, tensor in inputs.items():
            batch_size, seq_len, _ = tensor.shape
            
            # Create mask with some random padding
            if seq_len > 1:
                # Randomly mask out 0-20% of tokens
                mask_prob = 0.2 * torch.rand(1).item()
                mask = torch.rand(batch_size, seq_len, device=self.device) > mask_prob
            else:
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
                
            masks[mod_name] = mask
        
        return masks
    
    def run_example_with_fusion(
        self,
        fusion_type: str = "sequential",
        batch_size: int = 2,
        modalities: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Run example with different fusion strategies.
        
        Args:
            fusion_type: Type of fusion strategy to use
            batch_size: Batch size
            modalities: Dict mapping modality name to sequence length
            
        Returns:
            Dict with results
            
        Raises:
            ValueError: If fusion_type is unknown
        """
        # Generate inputs
        inputs = self.generate_example_input(batch_size, modalities)
        masks = self.generate_example_masks(inputs)
        
        # Create projector with chosen fusion strategy
        if fusion_type == "sequential":
            fusion = SequentialFusion()
        elif fusion_type == "weighted":
            fusion = WeightedFusion()
        elif fusion_type == "attention":
            fusion = AttentionFusion(d_model=self.d_model)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Create projector
        projector = ModalityProjector(
            ProjectorConfig(
                d_model=self.d_model,
                modalities={
                    mod: ModalityConfig(input_dim=self.d_model, output_dim=self.d_model)
                    for mod in inputs.keys()
                },
                device=self.device
            )
        )
        
        # Apply fusion
        output, output_mask = fusion(inputs, projector, token_masks=masks)
        
        return {
            "fusion_type": fusion_type,
            "output_shape": output.shape,
            "output_mask_shape": output_mask.shape,
            "inputs": {mod: tensor.shape for mod, tensor in inputs.items()},
            "device": self.device
        }
