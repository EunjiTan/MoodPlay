"""
Core Logging Module
Provides structured logging with timing decorators and GPU memory tracking.
"""

import sys
import logging
import time
import functools
import psutil
import torch
from typing import Optional, Callable, Any
from pathlib import Path

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)

def log_gpu_memory(logger: logging.Logger, context: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[GPU Memory] {context}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

def get_memory_usage() -> dict:
    """Get current system and GPU memory usage stats."""
    stats = {
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": psutil.virtual_memory().used / 1024**3,
        "gpu_allocated_gb": 0.0,
        "gpu_reserved_gb": 0.0
    }
    
    if torch.cuda.is_available():
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        
    return stats

def time_execution(logger: Optional[logging.Logger] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            if logger:
                logger.debug(f"Starting {func.__name__}...")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if logger:
                    logger.info(f"{func.__name__} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                if logger:
                    logger.error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator

class ProgressLogger:
    """Simple progress logger for long running tasks."""
    def __init__(self, logger: logging.Logger, total_steps: int, desc: str = "Processing"):
        self.logger = logger
        self.total = total_steps
        self.params = desc
        self.start_time = time.time()
        
    def update(self, step: int, info: str = ""):
        if step % max(1, self.total // 10) == 0:  # Log every 10%
            elapsed = time.time() - self.start_time
            if step > 0:
                fps = step / elapsed
                remaining = (self.total - step) / fps
                self.logger.info(
                    f"{self.params}: {step}/{self.total} ({step/self.total*100:.0f}%) "
                    f"- {fps:.2f}it/s - ETA: {remaining:.0f}s - {info}"
                )

# Example Usage
if __name__ == "__main__":
    logger = get_logger("test")
    log_gpu_memory(logger, "Before ops")
