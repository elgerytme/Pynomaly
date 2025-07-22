"""Infrastructure adapters for neuro-symbolic AI."""

from .neural_adapter import (
    NeuralAdapter,
    NeuralBackbone,
    TransformerBackbone,
    CNNBackbone,
    AutoencoderBackbone
)

from .symbolic_adapter import (
    SymbolicAdapter,
    SymbolicReasoner,
    PropositionalReasoner,
    FirstOrderReasoner,
    SMTSolver,
    RDFReasoner,
    LogicalRule,
    InferenceResult,
    LogicType
)

__all__ = [
    # Neural adapters
    "NeuralAdapter",
    "NeuralBackbone", 
    "TransformerBackbone",
    "CNNBackbone",
    "AutoencoderBackbone",
    
    # Symbolic adapters
    "SymbolicAdapter",
    "SymbolicReasoner",
    "PropositionalReasoner", 
    "FirstOrderReasoner",
    "SMTSolver",
    "RDFReasoner",
    "LogicalRule",
    "InferenceResult",
    "LogicType"
]