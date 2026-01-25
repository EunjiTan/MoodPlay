"""
Processing module for video segmentation and temporal consistency.
"""

from backend.processing.tracking import SAM2Tracker, SegmentationPrompt

# Lazy imports to avoid issues when ML libraries are not available
def get_optical_flow_estimator():
    from backend.processing.optical_flow import OpticalFlowEstimator
    return OpticalFlowEstimator

def get_temporal_blender():
    from backend.processing.optical_flow import TemporalBlender
    return TemporalBlender

def get_consistency_enforcer():
    from backend.processing.temporal import TemporalConsistencyEnforcer
    return TemporalConsistencyEnforcer

__all__ = [
    'SAM2Tracker',
    'SegmentationPrompt',
    'get_optical_flow_estimator',
    'get_temporal_blender',
    'get_consistency_enforcer',
]
