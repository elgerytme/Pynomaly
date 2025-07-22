"""Value objects for neuro-symbolic AI."""

from .reasoning_result import ReasoningResult
from .causal_explanation import (
    CausalExplanation,
    CausalFactor,
    CausalLink,
    CausalChain,
    CausalAnalysisResult,
    CausalRelationType,
    TemporalRelation
)
from .counterfactual_result import (
    CounterfactualResult,
    CounterfactualScenario,
    FeatureChange,
    CounterfactualAnalysisResult,
    CounterfactualType,
    ChangeDirection
)

__all__ = [
    # Basic reasoning
    "ReasoningResult",
    
    # Causal reasoning
    "CausalExplanation",
    "CausalFactor",
    "CausalLink", 
    "CausalChain",
    "CausalAnalysisResult",
    "CausalRelationType",
    "TemporalRelation",
    
    # Counterfactual reasoning
    "CounterfactualResult",
    "CounterfactualScenario",
    "FeatureChange",
    "CounterfactualAnalysisResult",
    "CounterfactualType",
    "ChangeDirection"
]