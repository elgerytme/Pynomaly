"""Shared interfaces for cross-domain communication.

This module provides domain interfaces that allow packages to communicate
without direct dependencies. It implements the Interface Segregation Principle
and enables dependency inversion.
"""

from .mlops import *
from .enterprise import *
from .data import *

__all__ = [
    # MLOps interfaces
    "ExperimentTrackingInterface",
    "ModelRegistryInterface", 
    "ModelDeploymentInterface",
    
    # Enterprise interfaces  
    "AuthServiceInterface",
    "MultiTenantInterface",
    "OperationsInterface",
    
    # Data interfaces
    "DataProcessingInterface",
    "QualityAssessmentInterface",
    "ProfilingEngineInterface"
]