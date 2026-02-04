"""
Core Device & Memory Management Module
Handles device selection, precision settings, and OOM-safe execution strategies.
"""

import torch
import gc
import logging
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager

from backend.core.logging import get_logger, log_gpu_memory
from backend.core.exceptions import OutOfMemoryError

logger = get_logger("core.device")

class DeviceManager:
    """Manages global device configuration and memory strategies."""
    
    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        self._device = self._select_device()
        self._dtype = self._select_dtype()
        logger.info(f"Initialized DeviceManager: Device={self._device}, Dtype={self._dtype}")

    def _select_device(self) -> torch.device:
        """Select best available device."""
        if self.force_cpu:
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            # Basic benchmark or check
            props = torch.cuda.get_device_properties(0)
            logger.info(f"Found GPU: {props.name} ({props.total_memory / 1024**3:.2f} GB)")
            return torch.device("cuda")
        
        return torch.device("cpu")

    def _select_dtype(self) -> torch.dtype:
        """Select appropriate precision."""
        if self._device.type == "cuda":
            # Prefer fp16 for SD1.5/SDXL on consumer GPUs
            return torch.float16
        return torch.float32

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._dtype

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self._device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=self._dtype):
                yield
        else:
            yield

    def empty_cache(self):
        """Aggressively clear memory."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

# Global instance
device_manager = DeviceManager()

def get_device() -> torch.device:
    return device_manager.device

def get_dtype() -> torch.dtype:
    return device_manager.dtype

def oom_safe(logger: logging.Logger = logger):
    """
    Decorator for OOM-safe function execution with fallback strategies.
    Strategies tried in order:
    1. Standard execution
    2. Empty cache & retry
    3. Half batch size (if 'batch_size' in kwargs)
    4. Sliced attention (if applicable/configurable)
    5. CPU fallback (extreme case, optional)
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            try:
                # Attempt 1: Standard run
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"OOM detected in {func.__name__}. Attempting recovery...")
                log_gpu_memory(logger, "OOM State")
                
                # Strategy 1: Clear Cache & Retry
                device_manager.empty_cache()
                try:
                    logger.info("Retrying after cache clear...")
                    return func(*args, **kwargs)
                except torch.cuda.OutOfMemoryError:
                     pass # Continue to next strategy

                # Strategy 2: Reduce logical batch/chunk size if present
                if 'chunk_size' in kwargs and kwargs['chunk_size'] > 1:
                    original_chunk = kwargs['chunk_size']
                    new_chunk = max(1, original_chunk // 2)
                    logger.warning(f"Retrying with chunk_size {original_chunk} -> {new_chunk}")
                    kwargs['chunk_size'] = new_chunk
                    device_manager.empty_cache()
                    try:
                        return func(*args, **kwargs)
                    except torch.cuda.OutOfMemoryError:
                        pass # Continue

                # Strategy 3: Enable attention slicing (if object has method)
                # Assumes the first arg is 'self' and has enable_attention_slicing
                if args and hasattr(args[0], 'enable_attention_slicing'):
                     logger.warning("Enabling attention slicing...")
                     args[0].enable_attention_slicing()
                     device_manager.empty_cache()
                     try:
                         return func(*args, **kwargs)
                     except torch.cuda.OutOfMemoryError:
                         pass

                # Strategy 4: Error-out or strict CPU fallback
                # For now, we raise a custom error to be handled by the pipeline orchestrator
                raise OutOfMemoryError(
                    device=str(device_manager.device),
                    required="unknown",
                    available="exhausted"
                ) from e
        return wrapper
    return decorator
