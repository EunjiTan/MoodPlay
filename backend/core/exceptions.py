"""
Core Exceptions Module
Defines custom exception hierarchy for the video colorization pipeline.
"""

class PipelineError(Exception):
    """Base class for all pipeline exceptions."""
    pass

class ResourceError(PipelineError):
    """Base class for resource-related errors (Memory, GPU, Files)."""
    pass

class OutOfMemoryError(ResourceError):
    """
    Raised when GPU or System memory is exhausted.
    Used to trigger fallback strategies (e.g. tiling, CPU offload).
    """
    def __init__(self, device: str, required: str = "unknown", available: str = "unknown"):
        self.device = device
        self.message = f"OOM on {device}. Required: {required}, Available: {available}"
        super().__init__(self.message)

class ModelLoadError(PipelineError):
    """Raised when a model fails to load (missing file, corruption)."""
    pass

class VideoReadError(ResourceError):
    """Raised when video decoding fails."""
    pass

class VideoWriteError(ResourceError):
    """Raised when video encoding fails."""
    pass

class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""
    pass

class DependencyError(PipelineError):
    """Raised when an external dependency (ffmpeg, etc) is missing."""
    pass
