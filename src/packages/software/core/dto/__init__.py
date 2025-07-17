"""
Generic Software Data Transfer Objects (DTOs)

This module provides generic DTOs for software applications.
Contains only domain-agnostic data structures.
"""

from .configuration_dto import ConfigurationDTO
from .export_options import ExportOptions
from .monitoring_dto import MonitoringDTO
from .mfa_dto import MfaDTO
from .cost_optimization_dto import CostOptimizationDTO

__all__ = [
    "ConfigurationDTO",
    "ExportOptions", 
    "MonitoringDTO",
    "MfaDTO",
    "CostOptimizationDTO"
]