"""
Modality Encoders for GPT-4o Architecture

This module implements the encoders for processing different modality inputs (text, image, audio, video)
for the GPT-4o architecture. Each encoder transforms raw input data into a unified embedding space,
with specific designs optimized for each modality's unique characteristics.

The encoders serve as the initial processing stage in the GPT-4o pipeline, converting
raw inputs into embeddings that can be processed by the transformer backbone.

Key Features:
- Specialized encoders for text, image, audio, and video modalities
- Consistent output embedding dimensions across all modalities
- Support for fine-tuning with LoRA adapters
- Robust error handling and graceful degradation
- Memory-efficient processing with streaming capabilities
"""

import os
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Tuple, Any, Callable, Type, NamedTuple, Protocol
from enum import Enum
import threading
import time
from contextlib import contextmanager, nullcontext
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
try:
    from transformers.models.auto.modeling_auto import AutoModel
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class DummyLoRALayer:
    pass

try:
    from loralib import LoRALayer
    LORA_AVAILABLE = True
except ImportError:
    LoRALayer = DummyLoRALayer  
    LORA_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(os.environ.get("GPT4O_LOG_LEVEL", "INFO"))

class ModalityEncoderError(Exception):
    """Base exception for all modality encoder errors."""
    def __init__(self, message: str, modality: Optional[str] = None):
        self.modality = modality
        self.message = message
        super_message = f"[{modality or 'unknown'}] {message}" if modality else message
        super().__init__(super_message)


class ModelInitializationError(ModalityEncoderError):
    """Raised when encoder model initialization fails."""
    def __init__(self, message: str, modality: str, model_name: Optional[str] = None):
        self.model_name = model_name
        super_message = f"Failed to initialize {model_name or 'model'}: {message}"
        super().__init__(super_message, modality)


class InputValidationError(ModalityEncoderError):
    """Raised when input validation fails."""
    def __init__(self, message: str, modality: str, expected: Optional[str] = None, received: Optional[str] = None):
        self.expected = expected
        self.received = received
        details = f" Expected: {expected}, Received: {received}" if expected and received else ""
        super().__init__(f"Input validation failed: {message}{details}", modality)


class ProcessingError(ModalityEncoderError):
    """Raised when processing inputs fails."""
    def __init__(self, message: str, modality: str, stage: Optional[str] = None):
        self.stage = stage
        stage_prefix = f"[{stage}] " if stage else ""
        super().__init__(f"{stage_prefix}Processing error: {message}", modality)


class ResourceExhaustedError(ModalityEncoderError):
    """Raised when resources (memory, compute) are exhausted."""
    def __init__(self, message: str, modality: str, resource_type: Optional[str] = None):
        self.resource_type = resource_type
        resource_prefix = f"[{resource_type}] " if resource_type else ""
        super().__init__(f"{resource_prefix}Resource exhausted: {message}", modality)

@dataclass
class EncoderConfig:
    """Base configuration for modality encoders."""
    d_model: int = 4096  # Output embedding dimension
    batch_first: bool = True  # Whether batch is the first dimension (B, S, D)
    use_mixed_precision: bool = True  # Whether to use mixed precision (FP16/BF16)
    checkpoint_path: Optional[str] = None  # Path to pretrained weights
    dropout: float = 0.1  # Dropout rate


@dataclass
class TextEncoderConfig(EncoderConfig):
    """Configuration for text encoder."""
    model_name: str = "bert-base-uncased"  # Base model architecture
    max_seq_len: int = 512  # Maximum sequence length
    lora_rank: int = 8  # LoRA rank for adaptation
    vocab_size: int = 32000  # Vocabulary size
    hidden_dim: int = 768  # Hidden dimension for embedding (commonly 768 for BERT models)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    """Configuration for image encoder."""
    patch_size: int = 16  # Size of image patches
    image_size: int = 768  # Input image size
    hidden_dim: int = 1024  # Hidden dimension
    num_layers: int = 12  # Number of transformer layers
    num_heads: int = 16  # Number of attention heads
    codebook_size: int = 65536 # Size of the codebook for VQ-VAE


@dataclass
class AudioEncoderConfig(EncoderConfig):
    """Configuration for audio encoder."""
    sample_rate: int = 24000  # Audio sample rate
    n_fft: int = 400  # FFT size for spectrogram
    hop_length: int = 160  # Hop length for spectrogram
    num_mel_bins: int = 80  # Number of mel bins
    hidden_dim: int = 768  # Hidden dimension
    num_layers: int = 12  # Number of layers
    codebook_size: int = 65536  # Size of audio codebook
    num_codebooks: int = 4  # Number of codebooks for RVQ
    frame_length: int = 6000  # Frame length (250ms at 24kHz)


@dataclass
class VideoEncoderConfig(EncoderConfig):
    """Configuration for video encoder."""
    frame_rate: int = 4  # Frames per second to process
    max_frames: int = 16  # Maximum number of frames to process at once
    temporal_stride: int = 2  # Stride for temporal convolution


#------------------------------------------------------------------------------
# Module performance and resource monitoring
#------------------------------------------------------------------------------

class ResourceMonitor:
    """Monitor resource usage during encoding operations."""
    
    def __init__(self):
        """Initialize the resource monitor."""
        self.memory_peaks = {}
        self.operation_times = {}
        self.lock = threading.Lock()
    
    def reset_stats(self):
        """Reset all tracking statistics."""
        with self.lock:
            self.memory_peaks = {}
            self.operation_times = {}
    
    def track_operation(self, name: str, duration_ms: float):
        """Track operation execution time."""
        with self.lock:
            if name not in self.operation_times:
                self.operation_times[name] = []
            self.operation_times[name].append(duration_ms)
            # Keep only the last 100 measurements
            self.operation_times[name] = self.operation_times[name][-100:]
    
    def track_memory_usage(self, name: str, memory_usage_bytes: int):
        """Track peak memory usage."""
        with self.lock:
            current_peak = self.memory_peaks.get(name, 0)
            self.memory_peaks[name] = max(current_peak, memory_usage_bytes)
    
    def get_average_times(self) -> Dict[str, float]:
        """Get average operation times."""
        with self.lock:
            return {
                name: sum(times) / len(times) if times else 0
                for name, times in self.operation_times.items()
            }
    
    def get_memory_peaks(self) -> Dict[str, int]:
        """Get peak memory usage for each operation."""
        with self.lock:
            return self.memory_peaks.copy()


#------------------------------------------------------------------------------
# Base Encoder Class
#------------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """Base class for all modality encoders."""
    
    def __init__(self, config: EncoderConfig):
        """Initialize the encoder."""
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.resource_monitor = ResourceMonitor()
        self._is_initialized = False
        self._start_time = 0
    
    @contextmanager
    def monitor_execution(self, operation_name: str):
        """Context manager for monitoring operation execution."""
        try:
            start_time = time.time()
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.resource_monitor.track_operation(operation_name, duration_ms)
            if duration_ms > 100:  # Log slow operations (>100ms)
                logger.warning(f"{operation_name} took {duration_ms:.2f}ms")
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """Preprocess inputs before encoding."""
        raise NotImplementedError("Subclasses must implement preprocess()")
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed inputs into embeddings."""
        raise NotImplementedError("Subclasses must implement encode()")
    
    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass through the encoder."""
        with self.monitor_execution(f"{self.__class__.__name__}.forward"):
            preprocessed = self.preprocess(inputs)
            encoded = self.encode(preprocessed)
            return encoded
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        return {
            "avg_times": self.resource_monitor.get_average_times(),
            "memory_peaks": self.resource_monitor.get_memory_peaks(),
            "params": sum(p.numel() for p in self.parameters()),
        }
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load checkpoint weights."""
        checkpoint_path = path or self.config.checkpoint_path
        if not checkpoint_path:
            logger.warning(f"No checkpoint path provided for {self.__class__.__name__}")
            return
        
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            missing, unexpected = self.load_state_dict(
                state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict,
                strict=False
            )
            if missing:
                logger.warning(f"Missing keys in checkpoint: {missing[:5]}" + 
                               (f"... and {len(missing)-5} more" if len(missing) > 5 else ""))
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected[:5]}" + 
                               (f"... and {len(unexpected)-5} more" if len(unexpected) > 5 else ""))
            self._is_initialized = True
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to load checkpoint: {e}", 
                modality=self.__class__.__name__.lower().replace("encoder", ""),
                model_name=checkpoint_path
            )


#------------------------------------------------------------------------------
# Text Encoder
#------------------------------------------------------------------------------

class TextEncoder(ModalityEncoder):
    """Text encoder based on SentencePiece with BPE tokenization."""
    
    def __init__(self, config: TextEncoderConfig):
        """Initialize the text encoder."""
        super().__init__(config)
        self.config = config
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is required for TextEncoder")
            
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
            # Initialize embedding layer
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
            
            # Initialize projection to d_model
            if config.hidden_dim != config.d_model:
                self.projection = nn.Linear(config.hidden_dim, config.d_model)
            else:
                self.projection = nn.Identity()
                
            self._is_initialized = True
        except Exception as e:
            raise ModelInitializationError(
                str(e), modality="text", model_name=config.model_name
            )

    def preprocess(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: Input text or list of texts
            
        Returns:
            Dictionary of tokenized inputs
        """
        with self.monitor_execution("TextEncoder.preprocess"):
            if isinstance(texts, str):
                texts = [texts]
                
            try:
                # Tokenize inputs
                tokenized = self.tokenizer(
                    texts,
                    padding="max_length" if hasattr(self.config, "max_seq_len") else "longest",
                    max_length=self.config.max_seq_len if hasattr(self.config, "max_seq_len") else None,
                    truncation=True,
                    return_tensors="pt"
                )
                
                return tokenized
            except Exception as e:
                raise ProcessingError(str(e), modality="text", stage="tokenization")

    def encode(self, tokenized: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode tokenized text into embeddings.
        
        Args:
            tokenized: Dictionary of tokenized inputs
            
        Returns:
            Text embeddings (batch_size, seq_len, d_model)
        """
        with self.monitor_execution("TextEncoder.encode"):
            device = next(self.parameters()).device
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            
            # Use autocast for mixed precision if configured
            context = autocast() if self.config.use_mixed_precision else nullcontext()
            
            with context:
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized.get("attention_mask")
                
                # Get embeddings
                try:
                    embeddings = self.embedding(input_ids)
                    
                    # Apply projection to match d_model
                    embeddings = self.projection(embeddings)
                    
                    # Apply attention mask
                    if attention_mask is not None:
                        embeddings = embeddings * attention_mask.unsqueeze(-1)
                        
                    return embeddings
                except Exception as e:
                    raise ProcessingError(str(e), modality="text", stage="embedding")


#------------------------------------------------------------------------------
# Image Encoder
#------------------------------------------------------------------------------

class VQVAEEncoder(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) Encoder for image tokenization.
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_embeddings: int = 65536, embedding_dim: int = 32):
        super().__init__()
        # Encoder network to transform 768x768 image into 128x128 latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        )
        # Codebook for vector quantization
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE encoder.
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            Tuple of (indices, quantized vectors)
        """
        # Encode the input image
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, z.shape[-1])
        
        # Compute distances to codebook vectors
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.codebook.weight.t())
        
        # Find nearest codebook vectors
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.codebook(min_encoding_indices).view(z.shape)
        
        return min_encoding_indices.view(z.shape[0], z.shape[1], z.shape[2]), z_q


class ImageEncoder(ModalityEncoder):
    """
    Image encoder based on Vector-Quantized Variational Autoencoder (VQ-VAE).
    
    This encoder transforms input images into discrete token indices
    that can be processed by the transformer backbone.
    """
    
    def __init__(self, config: ImageEncoderConfig):
        """Initialize the image encoder."""
        super().__init__(config)
        self.config = config
        
        try:
            # Initialize VQ-VAE encoder
            self.vqvae = VQVAEEncoder(
                in_channels=3,
                hidden_dim=config.hidden_dim,
                num_embeddings=config.codebook_size,
                embedding_dim=config.d_model // 16  # Scale embedding dim to reduce memory usage
            )
            
            # Embedding layer to convert indices to embeddings
            self.embedding = nn.Embedding(config.codebook_size, config.d_model)
            
            # Projection layer to match d_model
            self.projection = nn.Linear(config.d_model, config.d_model)
            
            self._is_initialized = True
        except Exception as e:
            raise ModelInitializationError(
                str(e), modality="image"
            )

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images.
        
        Args:
            images: Input images (batch_size, channels, height, width)
            
        Returns:
            Preprocessed images
        """
        with self.monitor_execution("ImageEncoder.preprocess"):
            try:
                # Ensure correct shape and normalization
                if images.dim() == 3:
                    images = images.unsqueeze(0)  # Add batch dimension
                    
                # Resize to expected image size
                if images.shape[-1] != self.config.image_size or images.shape[-2] != self.config.image_size:
                    images = F.interpolate(
                        images,
                        size=(self.config.image_size, self.config.image_size),
                        mode="bilinear",
                        align_corners=False
                    )
                
                # Normalize to [-1, 1]
                if images.max() > 1.0:
                    images = images / 255.0
                    
                if images.max() <= 1.0 and images.min() >= 0.0:
                    images = images * 2.0 - 1.0
                    
                return images
            except Exception as e:
                raise ProcessingError(str(e), modality="image", stage="preprocessing")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images into embeddings.
        
        Args:
            images: Preprocessed images (batch_size, channels, height, width)
            
        Returns:
            Image embeddings (batch_size, sequence_length, d_model)
        """
        with self.monitor_execution("ImageEncoder.encode"):
            device = next(self.parameters()).device
            images = images.to(device)
            
            # Use autocast for mixed precision if configured
            context = autocast() if self.config.use_mixed_precision else nullcontext()
            
            with context:
                try:
                    # Get VQ-VAE tokens
                    indices, _ = self.vqvae(images)
                    batch_size, height, width = indices.shape
                    
                    # Flatten indices for embedding
                    indices_flat = indices.view(batch_size, -1)
                    
                    # Convert token indices to embeddings
                    embeddings = self.embedding(indices_flat)
                    
                    # Apply projection
                    embeddings = self.projection(embeddings)
                    
                    return embeddings
                except Exception as e:
                    raise ProcessingError(str(e), modality="image", stage="encoding")

    def get_token_indices(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get token indices for images without computing embeddings.
        Useful for visualization and debugging.
        
        Args:
            images: Input images (batch_size, channels, height, width)
            
        Returns:
            Token indices (batch_size, height, width)
        """
        with torch.no_grad():
            images = self.preprocess(images)
            device = next(self.parameters()).device
            images = images.to(device)
            indices, _ = self.vqvae(images)
            return indices


#------------------------------------------------------------------------------
# Audio Encoder
#------------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    Encoder block for the audio tokenizer, featuring residual connections and downsampling.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=stride*2, stride=stride, padding=stride//2)
        self.elu = nn.ELU()
        self.norm = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder block.
        
        Args:
            x: Input tensor (batch_size, channels, time)
            
        Returns:
            Output tensor with downsampling applied
        """
        residual = self.residual(x)
        shortcut = self.downsample(x)
        return self.elu(self.norm(residual + shortcut))


class AudioTokenizer(nn.Module):
    """
    Audio tokenizer using a series of convolutional layers, LSTM,
    and Residual Vector Quantization (RVQ) as described in the EnCodec model.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_codebooks: int, codebook_size: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        
        # Encoder blocks with different strides
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(hidden_dim * 2**min(i, 3), hidden_dim * 2**min(i+1, 3), stride)
            for i, stride in enumerate([2, 4, 5, 8])
        ])
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(hidden_dim * 8, hidden_dim * 8, num_layers=2, batch_first=True)
        
        # Final convolution
        self.final_conv = nn.Conv1d(hidden_dim * 8, hidden_dim * 8, kernel_size=7, padding=3)
        
        # RVQ codebooks
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, hidden_dim * 8) for _ in range(num_codebooks)])
        
        # Projection layers for each codebook
        self.projections = nn.ModuleList([nn.Linear(hidden_dim * 8, codebook_size) for _ in range(num_codebooks)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the audio tokenizer.
        
        Args:
            x: Input audio waveform (batch_size, 1, time)
            
        Returns:
            Tuple of (indices, quantized vectors)
        """
        # Initial convolution and activation
        x = F.elu(self.initial_conv(x))

        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # LSTM processing
        x_t = x.transpose(1, 2)  # (batch_size, time, channels)
        x_t, _ = self.lstm(x_t)
        x = x_t.transpose(1, 2)  # (batch_size, channels, time)

        # Final convolution
        x = self.final_conv(x)
        
        # Transpose for RVQ
        x_t = x.transpose(1, 2).contiguous()  # (batch_size, time, channels)
        
        # Residual Vector Quantization
        indices = []
        quantized = []
        residual = x_t
        
        for i in range(self.num_codebooks):
            # Project to logits
            logits = self.projections[i](residual)
            
            # Get indices of nearest codebook vectors
            idx = torch.argmax(logits, dim=-1)
            
            # Get quantized vectors
            quant = self.codebooks[i](idx)
            
            # Add to results
            indices.append(idx)
            quantized.append(quant)
            
            # Update residual
            residual = residual - quant
        
        # Stack results
        indices = torch.stack(indices, dim=-1)  # (batch_size, time, num_codebooks)
        quantized = torch.stack(quantized, dim=-1).sum(dim=-1)  # (batch_size, time, channels)
        
        return indices, quantized


class AudioEncoder(ModalityEncoder):
    """
    Audio encoder based on EnCodec model with Residual Vector Quantization.
    
    This encoder transforms raw audio waveforms into discrete token indices
    that can be processed by the transformer backbone.
    """
    
    def __init__(self, config: AudioEncoderConfig):
        """Initialize the audio encoder."""
        super().__init__(config)
        self.config = config
        
        try:
            # Initialize audio tokenizer
            self.audio_tokenizer = AudioTokenizer(
                input_dim=1,  # Mono audio
                hidden_dim=config.hidden_dim // 8,  # Scale down hidden dim
                num_codebooks=config.num_codebooks,
                codebook_size=config.codebook_size
            )
            
            # Embedding layer to convert indices to embeddings
            self.embedding = nn.Linear(config.hidden_dim, config.d_model)
            
            self._is_initialized = True
        except Exception as e:
            raise ModelInitializationError(
                str(e), modality="audio"
            )
    
    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio waveforms.
        
        Args:
            audio: Audio waveforms (batch_size, time) or (batch_size, channels, time)
            
        Returns:
            Preprocessed audio
        """
        with self.monitor_execution("AudioEncoder.preprocess"):
            try:
                # Ensure correct shape
                if audio.dim() == 2:
                    audio = audio.unsqueeze(1)  # Add channel dimension
                elif audio.dim() == 3 and audio.shape[1] > 1:
                    # Convert multi-channel to mono by averaging
                    audio = audio.mean(dim=1, keepdim=True)
                
                # Ensure correct sampling rate (assume input is at config.sample_rate)
                
                # Normalize to [-1, 1]
                if audio.abs().max() > 1.0:
                    audio = audio / audio.abs().max()
                
                # Split into frames of 250 ms
                batch_size, channels, time = audio.shape
                frame_length = self.config.frame_length  # 250 ms at 24 kHz
                
                # Pad if needed
                if time % frame_length != 0:
                    padding = frame_length - (time % frame_length)
                    audio = F.pad(audio, (0, padding))
                    time += padding
                
                # Reshape to (batch_size * num_frames, channels, frame_length)
                num_frames = time // frame_length
                audio = audio.view(batch_size, channels, num_frames, frame_length)
                audio = audio.transpose(1, 2).contiguous()
                audio = audio.view(batch_size * num_frames, channels, frame_length)
                
                return audio
            except Exception as e:
                raise ProcessingError(str(e), modality="audio", stage="preprocessing")
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio into embeddings.
        
        Args:
            audio: Preprocessed audio (batch_size, channels, time)
            
        Returns:
            Audio embeddings (batch_size, sequence_length, d_model)
        """
        with self.monitor_execution("AudioEncoder.encode"):
            device = next(self.parameters()).device
            audio = audio.to(device)
            
            # Use autocast for mixed precision if configured
            context = autocast() if self.config.use_mixed_precision else nullcontext()
            
            with context:
                try:
                    # Get tokenizer output
                    indices, quantized = self.audio_tokenizer(audio)
                    
                    # Apply embedding layer
                    embeddings = self.embedding(quantized)
                    
                    # Reshape back to batch structure
                    original_batch_size = audio.shape[0]
                    embeddings = embeddings.view(original_batch_size, -1, self.config.d_model)
                    
                    return embeddings
                except Exception as e:
                    raise ProcessingError(str(e), modality="audio", stage="encoding")
    
    def get_token_indices(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Get token indices for audio without computing embeddings.
        
        Args:
            audio: Input audio (batch_size, channels, time)
            
        Returns:
            Token indices (batch_size, time, num_codebooks)
        """
        with torch.no_grad():
            audio = self.preprocess(audio)
            device = next(self.parameters()).device
            audio = audio.to(device)
            indices, _ = self.audio_tokenizer(audio)
            return indices


#------------------------------------------------------------------------------
# Video Encoder
#------------------------------------------------------------------------------

class VideoEncoder(ModalityEncoder):
    """
    Video encoder based on a hybrid approach combining frame-wise image encoding
    with temporal modeling via a Transformer.
    
    This encoder processes video as a sequence of images with added temporal context.
    """
    
    def __init__(self, config: VideoEncoderConfig):
        """Initialize the video encoder."""
        super().__init__(config)
        self.config = config
        
        try:
            # Initialize image encoder for frame-wise encoding
            image_config = ImageEncoderConfig(
                d_model=config.d_model,
                batch_first=config.batch_first,
                use_mixed_precision=config.use_mixed_precision,
                dropout=config.dropout,
                # Use smaller image size for video frames to reduce compute
                image_size=384
            )
            self.image_encoder = ImageEncoder(image_config)
            
            # Temporal modeling
            self.temporal_conv = nn.Conv1d(
                in_channels=config.d_model,
                out_channels=config.d_model,
                kernel_size=3,
                stride=config.temporal_stride,
                padding=1
            )
            
            # Frame position embedding
            self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_frames, config.d_model))
            
            # Temporal transformer for modeling frame relationships
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=8,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True
            )
            self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            self._is_initialized = True
        except Exception as e:
            raise ModelInitializationError(
                str(e), modality="video"
            )
    
    def preprocess(self, videos: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Preprocess video frames.
        
        Args:
            videos: Video frames (batch_size, frames, channels, height, width)
            
        Returns:
            Tuple of (flattened frames, original frame counts)
        """
        with self.monitor_execution("VideoEncoder.preprocess"):
            try:
                # Ensure correct shape
                if videos.dim() == 4:
                    videos = videos.unsqueeze(0)  # Add batch dimension
                    
                batch_size, num_frames, channels, height, width = videos.shape
                
                # Limit to max frames
                if num_frames > self.config.max_frames:
                    # Subsample frames
                    stride = num_frames // self.config.max_frames
                    indices = torch.arange(0, num_frames, stride)[:self.config.max_frames]
                    videos = videos[:, indices]
                    num_frames = videos.shape[1]
                
                # Store original frame counts
                frame_counts = [num_frames] * batch_size
                
                # Flatten batch and frames
                videos_flat = videos.view(batch_size * num_frames, channels, height, width)
                
                return videos_flat, frame_counts
            except Exception as e:
                raise ProcessingError(str(e), modality="video", stage="preprocessing")
    
    def encode(self, videos_with_counts: Tuple[torch.Tensor, List[int]]) -> torch.Tensor:
        """
        Encode video frames into embeddings with temporal context.
        
        Args:
            videos_with_counts: Tuple of (flattened frames, original frame counts)
            
        Returns:
            Video embeddings (batch_size, sequence_length, d_model)
        """
        with self.monitor_execution("VideoEncoder.encode"):
            videos_flat, frame_counts = videos_with_counts
            device = next(self.parameters()).device
            videos_flat = videos_flat.to(device)
            
            # Use autocast for mixed precision if configured
            context = autocast() if self.config.use_mixed_precision else nullcontext()
            
            with context:
                try:
                    # Encode each frame using image encoder
                    frame_embeddings = self.image_encoder(videos_flat)
                    
                    # Reshape back to (batch_size, frames, tokens_per_frame, d_model)
                    batch_size = len(frame_counts)
                    tokens_per_frame = frame_embeddings.shape[1]
                    d_model = frame_embeddings.shape[2]
                    
                    # Calculate total frames
                    max_frames = max(frame_counts)
                    
                    # Reshape frame embeddings
                    frame_embeddings = frame_embeddings.view(batch_size, max_frames, tokens_per_frame, d_model)
                    
                    # Average over tokens to get one embedding per frame
                    frame_embeddings = frame_embeddings.mean(dim=2)  # (batch_size, frames, d_model)
                    
                    # Add positional embeddings
                    positional_embedding = self.pos_embedding[:, :max_frames, :]
                    frame_embeddings = frame_embeddings + positional_embedding
                    
                    # Apply temporal transformer
                    # Create attention mask to handle variable frame counts
                    attention_mask = torch.zeros(batch_size, max_frames, device=device)
                    for i, count in enumerate(frame_counts):
                        attention_mask[i, count:] = float('-inf')
                    
                    # Apply temporal transformer
                    video_embeddings = self.temporal_transformer(frame_embeddings)
                    
                    return video_embeddings
                except Exception as e:
                    raise ProcessingError(str(e), modality="video", stage="encoding")


#------------------------------------------------------------------------------
# Unified embeddings and projection
#------------------------------------------------------------------------------

class ModalityProjection(nn.Module):
    """
    Projects embeddings from different modalities to a common embedding space.
    Adds modality-specific position encodings and source flags.
    """
    
    def __init__(self, d_model: int, max_positions: int = 5000, max_time_steps: int = 1000):
        """
        Initialize the modality projection.
        
        Args:
            d_model: Model dimension
            max_positions: Maximum number of positions
            max_time_steps: Maximum number of time steps
        """
        super().__init__()
        self.d_model = d_model
        
        # Modality-specific position encoding (represents position within a specific modality)
        self.pos_encoding = nn.Embedding(max_positions, d_model)
        
        # Time-based encoding (function of absolute time)
        self.time_encoding = nn.Embedding(max_time_steps, d_model)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        times: torch.Tensor,
        source_flags: torch.Tensor
    ) -> torch.Tensor:
        """
        Project embeddings and add position/time encodings and source flags.
        
        Args:
            embeddings: Input embeddings (batch_size, seq_len, d_model)
            positions: Position indices (batch_size, seq_len)
            times: Time step indices (batch_size, seq_len)
            source_flags: Source flags (0 = input, 1 = output) (batch_size, seq_len)
            
        Returns:
            Projected embeddings (batch_size, seq_len, d_model + d_model + 1)
        """
        # Get position and time encodings
        pos_encodings = self.pos_encoding(positions)
        time_encodings = self.time_encoding(times)
        
        # Concatenate embeddings with position and time encodings
        # Concatenation preserves their individual information rather than adding
        embeddings = torch.cat([embeddings, pos_encodings, time_encodings], dim=-1)
        
        # Add source flag
        source_flags = source_flags.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        embeddings = torch.cat([embeddings, source_flags], dim=-1)
        
        return embeddings


#------------------------------------------------------------------------------
# Factory function for creating modality encoders
#------------------------------------------------------------------------------

def create_encoder(modality: str, config: Optional[EncoderConfig] = None) -> ModalityEncoder:
    """
    Factory function for creating modality encoders.
    
    Args:
        modality: Modality name ('text', 'image', 'audio', 'video')
        config: Encoder configuration (optional, default configs used if not provided)
        
    Returns:
        Initialized modality encoder
    """
    if modality == "text":
        # Type check and use provided config if it's the right type, otherwise create a new one
        if isinstance(config, TextEncoderConfig):
            return TextEncoder(config)
        return TextEncoder(TextEncoderConfig())
    elif modality == "image":
        if isinstance(config, ImageEncoderConfig):
            return ImageEncoder(config)
        return ImageEncoder(ImageEncoderConfig())
    elif modality == "audio":
        if isinstance(config, AudioEncoderConfig):
            return AudioEncoder(config)
        return AudioEncoder(AudioEncoderConfig())
    elif modality == "video":
        if isinstance(config, VideoEncoderConfig):
            return VideoEncoder(config)
        return VideoEncoder(VideoEncoderConfig())
    else:
        raise ValueError(f"Unknown modality: {modality}")