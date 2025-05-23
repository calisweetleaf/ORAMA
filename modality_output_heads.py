import logging
from typing import Dict, Any, Optional, Union, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MultimodalOutput(nn.Module):
    def __init__(self, d_model: int = 4096, num_classes: int = 10):
        """
        Initializes the multimodal output layer.

        Args:
            d_model: Dimensionality of the input features.
            num_classes: Number of output classes for classification tasks.
        """
        super(MultimodalOutput, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the multimodal output layer.

        Args:
            x: Input tensor of shape (batch_size, d_model).

        Returns:
            A dictionary containing the output logits and probabilities.
        """
        assert x.shape[1] == self.d_model, "Input tensor must have shape (batch_size, d_model)"
        x = self.dropout(x)
        logits = self.fc(x)
        probabilities = torch.softmax(logits, dim=-1)
        return {'logits': logits, 'probabilities': probabilities}

    def get_no_output_token(self) -> int:
        """
        Returns the token ID for 'no output'.

        Returns:
            An integer representing the 'no output' token ID.
        """
        return -1  # Placeholder for 'no output' token ID

    def stream_output(self, x: torch.Tensor, priority: Optional[bool] = False) -> Dict[str, Any]:
        """
        Stream output based on priority settings.

        Args:
            x: Input tensor of shape (batch_size, d_model).
            priority: If True, prioritize certain outputs.

        Returns:
            A dictionary containing streamed output information.
        """
        if priority:
            # Implement priority logic here
            pass
        return self.forward(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, cast
from enum import Enum, auto
from dataclasses import dataclass, field
import time
from contextlib import contextmanager

from .transformers_backbone import InvalidInputError

logger = logging.getLogger("gpt4o.output_heads")

class OutputModalityType(Enum):
    """Types of output modalities supported by GPT-4o."""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()  # Experimental support
    LATENT = auto()  # Internal latent space representations
    MULTIMODAL = auto()  # Combined output across modalities


class OutputHeadError(Exception):
    """Base error class for output head exceptions."""
    pass


class UnknownModalityError(OutputHeadError):
    """Error raised when trying to use an unsupported modality."""
    pass


class ModalityConfigurationError(OutputHeadError):
    """Error raised with invalid modality configuration."""
    pass


class UnavailableModalityError(OutputHeadError):
    """Error raised when a requested modality is not available."""
    pass


@dataclass
class OutputHeadConfig:
    """Configuration for an output modality head."""
    modality: OutputModalityType
    hidden_dim: int
    output_dim: int
    dropout: float = 0.1
    use_bias: bool = True
    norm_layer: Optional[str] = "layer_norm"  # "layer_norm", "rms_norm", or None
    activation: Optional[str] = None  # "gelu", "relu", "silu", or None
    tie_weights: bool = False  # Whether to tie with input embedding weights
    tied_embedding: Optional[nn.Module] = None
    distributional: bool = False  # Whether to output distribution parameters
    special_tokens: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate configuration
        if self.tie_weights and self.tied_embedding is None:
            raise ModalityConfigurationError("When tie_weights=True, tied_embedding must be provided")
        
        if self.distributional and not (self.modality == OutputModalityType.AUDIO or 
                                       self.modality == OutputModalityType.VIDEO):
            logger.warning(f"Distributional output typically used only for AUDIO/VIDEO, not {self.modality}")

    @property
    def token_bias(self):
        raise NotImplementedError

    @token_bias.setter
    def token_bias(self, value):
        raise NotImplementedError


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization as described in the GPT-4o architecture.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class BaseOutputHead(nn.Module):
    """Base class for all output modality heads."""
    
    def __init__(self, config: OutputHeadConfig):
        super().__init__()
        self.config = config
        self.modality = config.modality
        
        # Optional normalization layer
        self.norm = None
        if config.norm_layer == "layer_norm":
            self.norm = nn.LayerNorm(config.hidden_dim)
        elif config.norm_layer == "rms_norm":
            self.norm = RMSNorm(config.hidden_dim)
        
        # Optional activation
        self.activation = None
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "silu":
            self.activation = nn.SiLU()
        
        # Output projection
        if config.tie_weights and config.tied_embedding is not None:
            # Reuse input embedding matrix (transposed)
            self.out_proj = None
        else:
            # Create new output projection
            self.out_proj = nn.Linear(
                config.hidden_dim, 
                config.output_dim, 
                bias=config.use_bias
            )
            
        # Common dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Track generation metrics
        self.generation_count = 0
        self._generation_cache = {}
        self._perf_metrics = []

    def tie_weights(self):
        """Tie weights with input embeddings if configured."""
        if self.config.tie_weights and self.config.tied_embedding is not None:
            # Access weight attribute correctly and check for None
            if not hasattr(self.config.tied_embedding, 'weight') or (
                hasattr(self.config.tied_embedding, 'weight') and 
                getattr(self.config.tied_embedding, 'weight').shape[1] != self.config.hidden_dim):
                raise ModalityConfigurationError(
                    f"Tied embedding dimension {getattr(self.config.tied_embedding, 'weight').shape[1] if hasattr(self.config.tied_embedding, 'weight') else 'unknown'} "
                    f"doesn't match output dimension {self.config.hidden_dim}"
                )
            if self.out_proj is not None:
                del self.out_proj
            self.out_proj = None
            logger.info(f"Tied weights for {self.modality} output head")
    
    def init_generation_cache(self):
        self._generation_cache = {}
    
    def clear_generation_cache(self):
        self._generation_cache = {}
        torch.cuda.empty_cache()

    @contextmanager
    def profile_generation(self, batch_size: int = 1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Provide current stream to record method
        stream = torch.cuda.current_stream()
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            tokens_per_second = (batch_size * 1000) / elapsed_ms
            logger.info(f"Generation performance: {elapsed_ms:.2f}ms, {tokens_per_second:.2f} tokens/sec")
            self._perf_metrics.append({
                'elapsed_ms': elapsed_ms,
                'tokens_per_second': tokens_per_second,
                'batch_size': batch_size,
                'timestamp': time.time()
            })

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Basic forward pass for output head.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            
        Returns:
            logits: Output logits [batch, seq_len, output_dim]
        """
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
            
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
            
        hidden_states = self.dropout(hidden_states)
        
        # Use tied weights or dedicated output projection
        if self.out_proj is None and self.config.tied_embedding is not None:
            # Weight tying: ensure we're using a tensor for the weight
            if hasattr(self.config.tied_embedding, 'weight'):
                embedding_weight = self.config.tied_embedding.weight
                if isinstance(embedding_weight, torch.Tensor):
                    logits = F.linear(hidden_states, embedding_weight)
                else:
                    raise TypeError("Expected tied_embedding.weight to be a tensor")
            else:
                raise AttributeError("tied_embedding has no attribute 'weight'")
        else:
            # Standard output projection - check for None
            if self.out_proj is not None:
                logits = self.out_proj(hidden_states)
            else:
                raise ValueError("out_proj is None and tied_embedding configuration is invalid")
            
        return logits

    def safely_generate(self, hidden_states: torch.Tensor, fallback_strategy: str = "greedy", **kwargs) -> Dict[str, Any]:
        try:
            with torch.cuda.amp.autocast(enabled=True):
                generation_method = getattr(self, f"generate_{self.modality.name.lower()}_token", None)
                if generation_method is None:
                    raise NotImplementedError(f"Generation not implemented for {self.modality.name}")
                return generation_method(hidden_states, **kwargs)
        except (RuntimeError, ValueError, NotImplementedError) as e:
            logger.warning(f"Generation failed in {self.modality.name} head: {str(e)}")
            logger.warning(f"Using fallback strategy: {fallback_strategy}")
            logits = self.forward(hidden_states[:, -1:]).squeeze(1)
            if fallback_strategy == "greedy":
                token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            elif fallback_strategy == "random":
                probs = F.softmax(logits, dim=-1)
                token_ids = torch.multinomial(probs, num_samples=1)
            elif fallback_strategy == "special_token":
                special_token_id = self.config.special_tokens.get("eos_token_id", 0)
                token_ids = torch.full((hidden_states.shape[0], 1), special_token_id, device=hidden_states.device)
            else:
                raise ValueError(f"Unknown fallback strategy: {fallback_strategy}")
            # Modified return type to match Dict[str, Any] instead of Dict[str, Tensor]
            return {
                "token_ids": token_ids,
                "is_fallback": True,
                "error": str(e)
            }

    def to_numpy(self, tensor_outputs: Dict[str, Any]) -> Dict[str, Any]:
        numpy_outputs = {}
        for k, v in tensor_outputs.items():
            if isinstance(v, torch.Tensor):
                numpy_outputs[k] = v.detach().cpu().numpy()
            elif isinstance(v, list):
                # Add explicit check before iterating
                if all(isinstance(item, torch.Tensor) for item in v):
                    numpy_outputs[k] = [item.detach().cpu().numpy() for item in v]
                else:
                    numpy_outputs[k] = v
            else:
                numpy_outputs[k] = v
        return numpy_outputs

    def quantize(self, quantization_scheme: str = "dynamic_int8") -> None:
        if (quantization_scheme == "dynamic_int8"):
            if self.out_proj is not None:
                self.out_proj = torch.quantization.quantize_dynamic(
                    self.out_proj, {nn.Linear}, dtype=torch.qint8
                )
        elif quantization_scheme == "static_int8":
            self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self, inplace=True)
            torch.quantization.convert(self, inplace=True)
        elif quantization_scheme == "fp16":
            self.half()
        else:
            raise ValueError(f"Unsupported quantization scheme: {quantization_scheme}")
        logger.info(f"Quantized {self.modality} head using {quantization_scheme}")

    def add_prompt_tuning(self, num_virtual_tokens: int = 20) -> nn.Parameter:
        self.num_virtual_tokens = num_virtual_tokens
        self.prompt_embedding = nn.Parameter(
            torch.zeros(1, num_virtual_tokens, self.config.hidden_dim)
        )
        nn.init.normal_(self.prompt_embedding, std=0.02)
        self._original_forward = self.forward
        def forward_with_prompt(hidden_states: torch.Tensor, **kwargs):
            batch_size = hidden_states.shape[0]
            prompt = self.prompt_embedding.expand(batch_size, -1, -1)
            augmented_hidden_states = torch.cat([prompt, hidden_states], dim=1)
            return self._original_forward(augmented_hidden_states, **kwargs)
        self.forward = forward_with_prompt
        return self.prompt_embedding


class TextOutputHead(BaseOutputHead):
    """
    Text output modality head for GPT-4o.
    Projects hidden states to vocabulary logits for next token prediction.
    """
    
    def __init__(self, config: OutputHeadConfig):
        assert config.modality == OutputModalityType.TEXT, "TextOutputHead requires TEXT modality config"
        super().__init__(config)
        
        # Special tokens in vocabulary (for masking/bias if needed)
        self.special_tokens = config.special_tokens
        
        # Optional bias for token preferences
        self.token_bias = None
        if hasattr(config, 'token_bias') and config.token_bias is not None:
            self.token_bias = nn.Parameter(torch.zeros(config.output_dim))
            
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for text output head.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]
            labels: Optional target labels [batch, seq_len]
            
        Returns:
            Dictionary with logits and optional loss
        """
        logits = super().forward(hidden_states)
        
        # Apply token bias if configured
        if self.token_bias is not None:
            logits = logits + self.token_bias
            
        outputs = {"logits": logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Enable fp16 compatibility
            shift_logits = shift_logits.to(torch.float32)
            
            # Calculate loss
            outputs["loss"] = loss_fct(shift_logits, shift_labels)
            
        return outputs
    
    def generate_token(
        self, 
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        no_output_token_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate next token from logits with sampling options.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            temperature: Sampling temperature
            top_k: Top-k filtering value
            top_p: Top-p (nucleus) filtering value
            do_sample: Whether to sample or take argmax
            no_output_token_id: Token ID representing "no output" (optional)
            
        Returns:
            Dictionary with next token IDs and probabilities
        """
        # Get logits for the last position only
        logits = super().forward(hidden_states[:, -1:, :]).squeeze(1)
        
        # Track sampling
        self.generation_count += 1
        
        # Optional no_output token handling
        if no_output_token_id is not None:
            # Ensure "no output" is a valid option but not overly favored
            no_output_mask = torch.zeros_like(logits)
            no_output_mask[:, no_output_token_id] = 0
            logits = logits + no_output_mask
        
        # Apply temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
            
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            # Zero out all values below the top-k values
            indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
            probs = probs.masked_fill(indices_to_remove, 0.0)
            # Re-normalize
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0.0)
            # Re-normalize
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            
        # Sample or argmax
        if do_sample:
            # Multinomial sampling based on probabilities
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy selection
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True)
            
        # Get probability of selected token
        token_probs = torch.gather(probs, -1, next_tokens)
        
        return {
            "token_ids": next_tokens,
            "token_probs": token_probs,
            "distribution": probs
        }


class ImageOutputHead(BaseOutputHead):
    """
    Image output modality head for GPT-4o.
    Projects hidden states to image token predictions using VQ-VAE style.
    """
    
    def __init__(
        self, 
        config: OutputHeadConfig, 
        image_size: Tuple[int, int] = (16, 16),
        codebook_size: int = 65536,
        no_output_token_id: Optional[int] = None
    ):
        assert config.modality == OutputModalityType.IMAGE, "ImageOutputHead requires IMAGE modality config"
        super().__init__(config)
        
        self.image_size = image_size  # Size of image in tokens (height, width)
        self.codebook_size = codebook_size
        self.no_output_token_id = no_output_token_id
        
        # Image sequence position embeddings
        max_image_tokens = image_size[0] * image_size[1]
        self.register_buffer(
            "position_ids", 
            torch.arange(max_image_tokens).expand((1, -1)),
            persistent=False
        )
        self.position_embeddings = nn.Embedding(max_image_tokens, config.hidden_dim)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for image output head.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            labels: Optional target image token labels [batch, img_tokens]
            image_mask: Optional mask for image tokens [batch, img_tokens]
            
        Returns:
            Dictionary with logits and optional loss
        """
        logits = super().forward(hidden_states)
        
        outputs = {"logits": logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # Apply image masking if provided
            if image_mask is not None:
                active_logits = logits.view(-1, self.config.output_dim)
                active_labels = torch.where(
                    image_mask.view(-1), 
                    labels.view(-1),
                    torch.tensor(-100, device=labels.device)
                )
                outputs["loss"] = loss_fct(active_logits, active_labels)
            else:
                outputs["loss"] = loss_fct(
                    logits.view(-1, self.config.output_dim), 
                    labels.view(-1)
                )
                
        return outputs
    
    def generate_image_token(
        self,
        hidden_states: torch.Tensor,
        generated_tokens: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate next image token.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            generated_tokens: Previously generated tokens in the image
            temperature: Sampling temperature
            top_k: Top-k filtering value
            top_p: Top-p (nucleus) filtering value
            do_sample: Whether to sample or take argmax
            
        Returns:
            Dictionary with next token IDs and probabilities
        """
        batch_size = hidden_states.shape[0]
        device, pos_idx, pos_emb = self.get_position_embedding(hidden_states, generated_tokens, batch_size)
        
        # Combine with hidden state
        last_hidden = hidden_states[:, -1:, :]
        image_hidden = last_hidden + pos_emb
        
        # Get logits
        logits = super().forward(image_hidden).squeeze(1)
        
        # Track sampling
        self.generation_count += 1
        
        # Optional no_output token handling
        if self.no_output_token_id is not None:
            # Option to not generate image tokens
            no_output_mask = torch.zeros_like(logits)
            no_output_mask[:, self.no_output_token_id] = 0  # Keep "no output" token probability unchanged
            logits = logits + no_output_mask
        
        # Apply temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
            
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            # Zero out all values below the top-k values
            indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
            probs = probs.masked_fill(indices_to_remove, 0.0)
            # Re-normalize
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0.0)
            # Re-normalize
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            
        # Sample or argmax
        if do_sample:
            # Multinomial sampling based on probabilities
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy selection
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True)
            
        # Get probability of selected token
        token_probs = torch.gather(probs, -1, next_tokens)
        
        # Convert position index to tensor to satisfy return type
        position_tensor = torch.tensor([pos_idx], device=device)
        
        return {
            "token_ids": next_tokens,
            "token_probs": token_probs,
            "position": position_tensor,
            "distribution": probs
        }

    def get_position_embedding(self, hidden_states, generated_tokens, batch_size):
        device = hidden_states.device
        
        # Add position embedding for the next token
        pos_idx = 0 if generated_tokens is None else len(generated_tokens)
        
        # Get the position_ids tensor directly - fix for "__getitem__" method not defined error
        position_ids_tensor = self.position_ids  # This is a tensor, not a module
        
        # Ensure position_ids_tensor is a tensor
        if not torch.is_tensor(position_ids_tensor):
            position_ids_tensor = torch.tensor(position_ids_tensor, device=device)
        
        # Use .size() instead of .shape if needed, but .shape is fine for tensors
        if pos_idx >= position_ids_tensor.shape[1]:
            raise ValueError(f"Cannot generate position {pos_idx}, max is {position_ids_tensor.shape[1]-1}")
        
        # Use the tensor indexing operations on the position_ids_tensor
        position_idx = position_ids_tensor[:, pos_idx].expand(batch_size)
        pos_emb = self.position_embeddings(position_idx)
        
        return device, pos_idx, pos_emb


class AudioOutputHead(BaseOutputHead):
    """
    Audio output modality head for GPT-4o.
    Projects hidden states to audio token predictions using a multi-codebook approach.
    """
    
    def __init__(
        self, 
        config: OutputHeadConfig,
        num_codebooks: int = 4,
        codebook_size: int = 65536,
        no_output_token_id: Optional[int] = None
    ):
        assert config.modality == OutputModalityType.AUDIO, "AudioOutputHead requires AUDIO modality config"
        super().__init__(config)
        
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.no_output_token_id = no_output_token_id
        
        # Each codebook gets its own output projector
        self.codebook_projectors = nn.ModuleList([
            nn.Linear(config.hidden_dim, codebook_size)
            for _ in range(num_codebooks)
        ])
        
        # For distributional output (e.g., hierarchical RVQ or VQ-VAE)
        self.is_distributional = config.distributional
        if self.is_distributional:
            # Predict both mean and scale parameters for each codebook
            self.distribution_projector = nn.Linear(
                config.hidden_dim, 
                2 * config.hidden_dim
            )
            
    def forward(
        self, 
        hidden_states: torch.Tensor,
        labels: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass for audio output head.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            labels: Optional target audio token labels per codebook
                   [List of [batch, seq_len] tensors]
            
        Returns:
            Dictionary with logits for each codebook and optional loss
        """
        # Apply normalization and dropout
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Project to each codebook
        codebook_logits = []
        for projector in self.codebook_projectors:
            codebook_logits.append(projector(hidden_states))
            
        outputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {"logits": codebook_logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            if len(labels) != self.num_codebooks:
                raise ValueError(
                    f"Expected labels for {self.num_codebooks} codebooks, got {len(labels)}"
                )
            
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            
            # Calculate loss for each codebook
            for i, (logits, label) in enumerate(zip(codebook_logits, labels)):
                # Reshape logits and labels
                logits_flat = logits.view(-1, self.codebook_size)
                labels_flat = label.view(-1)
                
                # Calculate loss for this codebook
                losses.append(loss_fct(logits_flat, labels_flat))
                
            outputs["loss"] = torch.stack(losses).mean()
            outputs["codebook_losses"] = torch.stack(losses)
            
        return outputs
    
    def generate_audio_token(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate next audio tokens for all codebooks.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            temperature: Sampling temperature
            top_k: Top-k filtering value
            top_p: Top-p (nucleus) filtering value
            do_sample: Whether to sample or take argmax
            
        Returns:
            Dictionary with next token IDs and probabilities for each codebook
        """
        # Apply normalization and dropout
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Get hidden states for the last position only
        last_hidden = hidden_states[:, -1:, :].squeeze(1)
        
        # Track sampling
        self.generation_count += 1
        
        # Get logits for each codebook
        token_ids = []
        token_probs = []
        distributions = []
        
        for i, projector in enumerate(self.codebook_projectors):
            # Get logits for this codebook
            logits = projector(last_hidden)
            
            # Optional no_output token handling
            if self.no_output_token_id is not None:
                # Option to not generate audio tokens
                no_output_mask = torch.zeros_like(logits)
                no_output_mask[:, self.no_output_token_id] = 0  # Keep "no output" token probability unchanged
                logits = logits + no_output_mask
            
            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature
                
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                # Zero out all values below the top-k values
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs = probs.masked_fill(indices_to_remove, 0.0)
                # Re-normalize
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                probs = probs.masked_fill(indices_to_remove, 0.0)
                # Re-normalize
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                
            # Sample or argmax
            if do_sample:
                # Multinomial sampling based on probabilities
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy selection
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
            # Get probability of selected token
            token_prob = torch.gather(probs, -1, next_token)
            
            # Store results for this codebook
            token_ids.append(next_token)
            token_probs.append(token_prob)
            distributions.append(probs)
            
        return self.new_method(token_ids, token_probs, distributions)

    def new_method(self, token_ids, token_probs, distributions):
        return {
            "token_ids": token_ids,
            "token_probs": token_probs,
            "distributions": distributions
        }


class ModalityOutputHeads(nn.Module):
    """
    Container for multiple output modality heads in GPT-4o.
    """
    
    def __init__(self, configs: Dict[OutputModalityType, OutputHeadConfig]):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.configs = configs
        
        # Create output heads for each configured modality
        for modality_type, config in configs.items():
            if modality_type == OutputModalityType.TEXT:
                self.heads[modality_type.name] = TextOutputHead(config)
            elif modality_type == OutputModalityType.IMAGE:
                self.heads[modality_type.name] = ImageOutputHead(config)
            elif modality_type == OutputModalityType.AUDIO:
                self.heads[modality_type.name] = AudioOutputHead(config)
            else:
                logger.warning(f"Output modality {modality_type} not fully implemented")
                self.heads[modality_type.name] = BaseOutputHead(config)
                
        # Default active modality
        self.active_modality = OutputModalityType.TEXT
        
        logger.info(f"Initialized {len(self.heads)} output modality heads: {list(self.heads.keys())}")
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        modality: Optional[Union[OutputModalityType, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through the selected output head.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            modality: Which modality head to use (defaults to active_modality)
            **kwargs: Additional arguments to pass to the specific head
            
        Returns:
            Output from the selected modality head
        """
        # Determine which modality to use
        if modality is None:
            modality = self.active_modality
        elif isinstance(modality, str):
            try:
                modality = OutputModalityType[modality]
            except KeyError:
                raise UnknownModalityError(f"Unknown modality: {modality}")
                
        # Ensure the modality is available
        modality_name = modality.name
        if modality_name not in self.heads:
            raise UnavailableModalityError(
                f"Modality {modality_name} is not available. "
                f"Available modalities: {list(self.heads.keys())}"
            )
            
        # Forward through the selected head
        return self.heads[modality_name](hidden_states, **kwargs)
    
    def set_active_modality(self, modality: Union[OutputModalityType, str]) -> None:
        """
        Set the active modality for default output.
        
        Args:
            modality: Modality to activate
        """
        if isinstance(modality, str):
            try:
                modality = OutputModalityType[modality]
            except KeyError:
                raise UnknownModalityError(f"Unknown modality: {modality}")
                
        if modality.name not in self.heads:
            raise UnavailableModalityError(f"Modality {modality.name} not available")
            
        self.active_modality = modality
        logger.info(f"Set active modality to {modality.name}")
        
    def generate(
        self,
        hidden_states: torch.Tensor,
        modality: Optional[Union[OutputModalityType, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate outputs from the selected modality head.
        
        Args:
            hidden_states: Transformer hidden states [batch, seq_len, hidden_dim]
            modality: Which modality head to use (defaults to active_modality)
            **kwargs: Generation parameters (temperature, top_k, top_p, etc.)
            
        Returns:
            Generated outputs for the selected modality
        """
        # Determine which modality to use
        if modality is None:
            modality = self.active_modality
        elif isinstance(modality, str):
            try:
                modality = OutputModalityType[modality]
            except KeyError:
                raise UnknownModalityError(f"Unknown modality: {modality}")
                
        # Ensure the modality is available
        modality_name = modality.name
        if modality_name not in self.heads:
            raise UnavailableModalityError(
                f"Modality {modality_name} is not available. "
                f"Available modalities: {list(self.heads.keys())}"
            )
            
        # Call the appropriate generation method based on modality
        head = self.heads[modality_name]
        if modality == OutputModalityType.TEXT:
            generate_method = getattr(head, 'generate_token')
            return generate_method(hidden_states, **kwargs)
        elif modality == OutputModalityType.IMAGE:
            generate_method = getattr(head, 'generate_image_token')
            return generate_method(hidden_states, **kwargs)
        elif modality == OutputModalityType.AUDIO:
            generate_method = getattr(head, 'generate_audio_token')
            return generate_method(hidden_states, **kwargs)
        else:
            raise NotImplementedError(
                f"Generation not implemented for modality {modality_name}"
            )
            
    def get_head(self, modality: Union[OutputModalityType, str]) -> BaseOutputHead:
        """
        Get the output head for a specific modality.
        
        Args:
            modality: Which modality head to retrieve
            
        Returns:
            The requested output head module
        """
        if isinstance(modality, str):
            modality_name = modality.upper()
        else:
            modality_name = modality.name
            
        if modality_name not in self.heads:
            raise UnavailableModalityError(f"Modality {modality_name} not available")
        
        # Cast the module to BaseOutputHead to satisfy type checking
        head = self.heads[modality_name]
        assert isinstance(head, BaseOutputHead), f"Head for modality {modality_name} is not a BaseOutputHead"
        return cast(BaseOutputHead, head)

    def detect_best_modality(self, hidden_states: torch.Tensor) -> OutputModalityType:
        confidence_scores = {}
        for modality_name, head in self.heads.items():
            logits = head.forward(hidden_states[:, -1:])
            probs = F.softmax(logits, dim=-1).squeeze(1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            confidence = 1.0 / (entropy + 1e-10)
            confidence_scores[modality_name] = confidence.item()
        best_modality_name = max(confidence_scores, key=lambda k: confidence_scores[k])
        return OutputModalityType[best_modality_name]

    def to_onnx_format(self, modality: Union[OutputModalityType, str]) -> Dict[str, Any]:
        head = self.get_head(modality)
        onnx_inputs = {
            "input_name": "hidden_states",
            "input_dims": [-1, -1, head.config.hidden_dim]
        }
        onnx_outputs = {
            "output_name": "logits",
            "output_dims": [-1, -1, head.config.output_dim]
        }
        import dataclasses
        return {
            "head_config": dataclasses.asdict(head.config),
            "onnx_inputs": onnx_inputs,
            "onnx_outputs": onnx_outputs
        }
