"""
Neuro-Symbolic AI Package

Combines neural networks with symbolic reasoning for interpretable AI systems.
"""

from .domain.entities.neuro_symbolic_model import NeuroSymbolicModel
from .domain.value_objects.reasoning_result import ReasoningResult
from .application.services.neuro_symbolic_service import NeuroSymbolicService

__version__ = "0.1.0"
__all__ = [
    "NeuroSymbolicModel",
    "ReasoningResult",
    "NeuroSymbolicService",
]