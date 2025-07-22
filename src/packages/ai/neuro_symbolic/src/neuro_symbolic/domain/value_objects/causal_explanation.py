"""Causal explanation value objects for neuro-symbolic reasoning."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class CausalRelationType(Enum):
    """Types of causal relationships."""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    NECESSARY_CAUSE = "necessary_cause"
    SUFFICIENT_CAUSE = "sufficient_cause"
    CONTRIBUTING_CAUSE = "contributing_cause"
    INHIBITING_CAUSE = "inhibiting_cause"
    SPURIOUS_CORRELATION = "spurious_correlation"


class TemporalRelation(Enum):
    """Temporal relationships in causal chains."""
    BEFORE = "before"
    AFTER = "after"
    SIMULTANEOUS = "simultaneous"
    DELAYED = "delayed"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class CausalFactor:
    """Represents a factor in a causal explanation."""
    id: str
    name: str
    value: Any
    confidence: float
    evidence: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")
    
    def __str__(self) -> str:
        return f"{self.name}={self.value} (confidence: {self.confidence:.3f})"


@dataclass(frozen=True)
class CausalLink:
    """Represents a causal relationship between two factors."""
    cause: CausalFactor
    effect: CausalFactor
    relation_type: CausalRelationType
    strength: float
    temporal_relation: TemporalRelation
    evidence: List[str] = field(default_factory=list)
    confounders: List[CausalFactor] = field(default_factory=list)
    mediators: List[CausalFactor] = field(default_factory=list)
    
    def __post_init__(self):
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError("Causal strength must be between 0 and 1")
    
    def __str__(self) -> str:
        arrow = "→" if self.relation_type != CausalRelationType.SPURIOUS_CORRELATION else "↔"
        return f"{self.cause.name} {arrow} {self.effect.name} (strength: {self.strength:.3f})"


@dataclass(frozen=True) 
class CausalChain:
    """Represents a sequence of causal links."""
    links: List[CausalLink]
    total_strength: float
    confidence: float
    
    def __post_init__(self):
        if not self.links:
            raise ValueError("Causal chain must contain at least one link")
        
        if not (0.0 <= self.total_strength <= 1.0):
            raise ValueError("Total strength must be between 0 and 1")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def root_cause(self) -> CausalFactor:
        """Get the root cause of the chain."""
        return self.links[0].cause
    
    @property
    def final_effect(self) -> CausalFactor:
        """Get the final effect of the chain."""
        return self.links[-1].effect
    
    @property 
    def length(self) -> int:
        """Get the length of the causal chain."""
        return len(self.links)
    
    def get_intermediate_factors(self) -> List[CausalFactor]:
        """Get all intermediate factors in the chain."""
        factors = []
        for link in self.links[:-1]:
            factors.append(link.effect)
        return factors
    
    def __str__(self) -> str:
        chain_str = " → ".join([link.cause.name for link in self.links] + [self.links[-1].effect.name])
        return f"Chain: {chain_str} (strength: {self.total_strength:.3f}, confidence: {self.confidence:.3f})"


@dataclass(frozen=True)
class CausalExplanation:
    """
    Immutable value object representing a causal explanation for a prediction or outcome.
    Provides detailed causal reasoning about why something occurred.
    """
    
    id: str
    target_outcome: CausalFactor
    primary_causes: List[CausalFactor]
    causal_chains: List[CausalChain]
    alternative_explanations: List["CausalExplanation"] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    methodology: str = "neural_symbolic_reasoning"
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")
        
        if not self.primary_causes:
            raise ValueError("Must have at least one primary cause")
        
        if not self.causal_chains:
            raise ValueError("Must have at least one causal chain")
    
    @classmethod
    def create(
        cls,
        target_outcome: CausalFactor,
        primary_causes: List[CausalFactor],
        causal_chains: List[CausalChain],
        confidence: float = 1.0,
        methodology: str = "neural_symbolic_reasoning",
        assumptions: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "CausalExplanation":
        """Create a new causal explanation."""
        return cls(
            id=str(uuid.uuid4()),
            target_outcome=target_outcome,
            primary_causes=primary_causes,
            causal_chains=causal_chains,
            confidence=confidence,
            methodology=methodology,
            assumptions=assumptions or [],
            limitations=limitations or [],
            metadata=metadata
        )
    
    def get_all_factors(self) -> Set[CausalFactor]:
        """Get all unique factors mentioned in the explanation."""
        factors = {self.target_outcome}
        factors.update(self.primary_causes)
        
        for chain in self.causal_chains:
            for link in chain.links:
                factors.add(link.cause)
                factors.add(link.effect)
                factors.update(link.confounders)
                factors.update(link.mediators)
        
        return factors
    
    def get_strongest_chain(self) -> CausalChain:
        """Get the causal chain with the highest total strength."""
        return max(self.causal_chains, key=lambda c: c.total_strength)
    
    def get_weakest_chain(self) -> CausalChain:
        """Get the causal chain with the lowest total strength."""
        return min(self.causal_chains, key=lambda c: c.total_strength)
    
    def get_direct_causes(self) -> List[CausalFactor]:
        """Get factors that directly cause the target outcome."""
        direct_causes = []
        for chain in self.causal_chains:
            for link in chain.links:
                if (link.effect == self.target_outcome and 
                    link.relation_type == CausalRelationType.DIRECT_CAUSE):
                    direct_causes.append(link.cause)
        return list(set(direct_causes))
    
    def get_root_causes(self) -> List[CausalFactor]:
        """Get the root causes from all causal chains."""
        return list(set(chain.root_cause for chain in self.causal_chains))
    
    def has_spurious_correlations(self) -> bool:
        """Check if the explanation contains spurious correlations."""
        for chain in self.causal_chains:
            for link in chain.links:
                if link.relation_type == CausalRelationType.SPURIOUS_CORRELATION:
                    return True
        return False
    
    def get_confounders(self) -> Set[CausalFactor]:
        """Get all confounding factors across chains."""
        confounders = set()
        for chain in self.causal_chains:
            for link in chain.links:
                confounders.update(link.confounders)
        return confounders
    
    def get_mediators(self) -> Set[CausalFactor]:
        """Get all mediating factors across chains."""
        mediators = set()
        for chain in self.causal_chains:
            for link in chain.links:
                mediators.update(link.mediators)
        return mediators
    
    def get_explanation_summary(self) -> str:
        """Get a human-readable summary of the causal explanation."""
        primary_causes_str = ", ".join([f.name for f in self.primary_causes])
        
        summary_parts = [
            f"Target: {self.target_outcome.name}",
            f"Primary causes: {primary_causes_str}",
            f"Causal chains: {len(self.causal_chains)}",
            f"Strongest chain: {self.get_strongest_chain()}",
            f"Overall confidence: {self.confidence:.3f}"
        ]
        
        if self.has_spurious_correlations():
            summary_parts.append("⚠️ Contains spurious correlations")
        
        if self.get_confounders():
            confounders = ", ".join([f.name for f in self.get_confounders()])
            summary_parts.append(f"Confounders: {confounders}")
        
        return " | ".join(summary_parts)
    
    def get_detailed_explanation(self) -> Dict[str, Any]:
        """Get a detailed breakdown of the causal explanation."""
        return {
            'id': self.id,
            'target_outcome': {
                'name': self.target_outcome.name,
                'value': self.target_outcome.value,
                'confidence': self.target_outcome.confidence
            },
            'primary_causes': [
                {
                    'name': cause.name,
                    'value': cause.value,
                    'confidence': cause.confidence,
                    'evidence': cause.evidence
                }
                for cause in self.primary_causes
            ],
            'causal_chains': [
                {
                    'length': chain.length,
                    'total_strength': chain.total_strength,
                    'confidence': chain.confidence,
                    'root_cause': chain.root_cause.name,
                    'final_effect': chain.final_effect.name,
                    'links': [
                        {
                            'cause': link.cause.name,
                            'effect': link.effect.name,
                            'relation_type': link.relation_type.value,
                            'strength': link.strength,
                            'temporal_relation': link.temporal_relation.value
                        }
                        for link in chain.links
                    ]
                }
                for chain in self.causal_chains
            ],
            'statistics': {
                'total_factors': len(self.get_all_factors()),
                'direct_causes': len(self.get_direct_causes()),
                'root_causes': len(self.get_root_causes()),
                'confounders': len(self.get_confounders()),
                'mediators': len(self.get_mediators()),
                'has_spurious_correlations': self.has_spurious_correlations(),
                'average_chain_strength': sum(c.total_strength for c in self.causal_chains) / len(self.causal_chains),
                'strongest_chain_strength': self.get_strongest_chain().total_strength,
                'weakest_chain_strength': self.get_weakest_chain().total_strength
            },
            'methodology': self.methodology,
            'confidence': self.confidence,
            'assumptions': self.assumptions,
            'limitations': self.limitations,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def compare_with(self, other: "CausalExplanation") -> Dict[str, Any]:
        """Compare this explanation with another."""
        if not isinstance(other, CausalExplanation):
            raise TypeError("Can only compare with another CausalExplanation")
        
        my_factors = self.get_all_factors()
        other_factors = other.get_all_factors()
        
        common_factors = my_factors.intersection(other_factors)
        my_unique = my_factors.difference(other_factors)
        other_unique = other_factors.difference(my_factors)
        
        my_causes = set(self.primary_causes)
        other_causes = set(other.primary_causes)
        
        return {
            'agreement_score': len(common_factors) / len(my_factors.union(other_factors)),
            'common_factors': [f.name for f in common_factors],
            'unique_to_self': [f.name for f in my_unique],
            'unique_to_other': [f.name for f in other_unique],
            'primary_cause_overlap': len(my_causes.intersection(other_causes)) / len(my_causes.union(other_causes)),
            'confidence_difference': abs(self.confidence - other.confidence),
            'chain_count_difference': abs(len(self.causal_chains) - len(other.causal_chains)),
            'methodology_match': self.methodology == other.methodology
        }
    
    def validate_consistency(self) -> List[str]:
        """Validate the internal consistency of the causal explanation."""
        issues = []
        
        # Check if primary causes appear in chains
        chain_causes = set()
        for chain in self.causal_chains:
            chain_causes.add(chain.root_cause)
        
        missing_primary_causes = set(self.primary_causes) - chain_causes
        if missing_primary_causes:
            issues.append(f"Primary causes not found in chains: {[c.name for c in missing_primary_causes]}")
        
        # Check for circular causation
        for chain in self.causal_chains:
            factors_in_chain = []
            for link in chain.links:
                factors_in_chain.extend([link.cause, link.effect])
            
            if len(set(factors_in_chain)) < len(factors_in_chain):
                issues.append(f"Circular causation detected in chain: {chain}")
        
        # Check if target outcome appears as intended
        target_found = False
        for chain in self.causal_chains:
            if chain.final_effect == self.target_outcome:
                target_found = True
                break
        
        if not target_found:
            issues.append("Target outcome not found as final effect in any chain")
        
        # Check temporal consistency
        for chain in self.causal_chains:
            for i, link in enumerate(chain.links[:-1]):
                next_link = chain.links[i + 1]
                if (link.temporal_relation == TemporalRelation.AFTER and 
                    next_link.temporal_relation == TemporalRelation.BEFORE):
                    issues.append(f"Temporal inconsistency in chain: {chain}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.get_detailed_explanation()
    
    def __str__(self) -> str:
        return self.get_explanation_summary()


@dataclass(frozen=True)
class CausalAnalysisResult:
    """Result of causal analysis containing multiple explanations and analysis."""
    
    explanations: List[CausalExplanation]
    best_explanation: CausalExplanation
    analysis_metadata: Dict[str, Any]
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.explanations:
            raise ValueError("Must have at least one explanation")
        
        if self.best_explanation not in self.explanations:
            raise ValueError("Best explanation must be in the explanations list")
        
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0 and 1")
    
    @classmethod
    def create(
        cls,
        explanations: List[CausalExplanation],
        analysis_metadata: Optional[Dict[str, Any]] = None
    ) -> "CausalAnalysisResult":
        """Create causal analysis result with automatic best explanation selection."""
        if not explanations:
            raise ValueError("Must provide at least one explanation")
        
        # Select best explanation based on confidence and chain strength
        best_explanation = max(
            explanations,
            key=lambda e: (e.confidence * e.get_strongest_chain().total_strength)
        )
        
        # Calculate overall confidence
        confidence_score = sum(e.confidence for e in explanations) / len(explanations)
        
        return cls(
            explanations=explanations,
            best_explanation=best_explanation,
            analysis_metadata=analysis_metadata or {},
            confidence_score=confidence_score
        )
    
    def get_consensus_factors(self) -> List[CausalFactor]:
        """Get factors that appear in multiple explanations."""
        factor_counts = {}
        
        for explanation in self.explanations:
            for factor in explanation.get_all_factors():
                factor_counts[factor.name] = factor_counts.get(factor.name, 0) + 1
        
        # Return factors that appear in more than half of explanations
        threshold = len(self.explanations) / 2
        consensus_names = [name for name, count in factor_counts.items() if count > threshold]
        
        consensus_factors = []
        for explanation in self.explanations:
            for factor in explanation.get_all_factors():
                if factor.name in consensus_names and factor not in consensus_factors:
                    consensus_factors.append(factor)
        
        return consensus_factors
    
    def get_analysis_summary(self) -> str:
        """Get a summary of the causal analysis."""
        return (
            f"Causal Analysis: {len(self.explanations)} explanations, "
            f"confidence: {self.confidence_score:.3f}, "
            f"best: {self.best_explanation.get_explanation_summary()}"
        )