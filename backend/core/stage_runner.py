"""
Stage Runner Base Module
Base class for isolated stage execution. Enforces setup/teardown lifecycles.
"""

import gc
import torch
import abc
from typing import Any, Dict
from backend.core.logging import get_logger, log_gpu_memory
from backend.core.data_manager import DataManager
from backend.core.device import device_manager

class StageRunner(abc.ABC):
    """
    Abstract base class for a pipeline stage.
    """
    
    def __init__(self, data_manager: DataManager, config: Dict = None):
        self.dm = data_manager
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def run(self) -> Any:
        """Main execution logic for the stage."""
        pass

    def release_resources(self):
        """Force cleanup of models/tensors."""
        # Derived classes should set their models to None before calling super
        device_manager.empty_cache()
        log_gpu_memory(self.logger, "Post-Stage Cleanup")

    def execute(self, *args, **kwargs):
        """Wrapper to ensure cleanup happens."""
        self.logger.info(">>> Starting Stage")
        log_gpu_memory(self.logger, "Start")
        
        try:
            return self.run(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Stage failed: {e}")
            raise
        finally:
            self.logger.info("<<< Finishing Stage")
            self.release_resources()
