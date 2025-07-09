"""Causal anomaly detection domain models for identifying causal relationships and root causes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.value_objects import PerformanceMetrics

# Type alias for backward compatibility
ModelMetrics = PerformanceMetrics

class CausalMethod(Enum):
    """Causal inference methods supported."""
    
    # Structure learning methods
    PC_ALGORITHM = "pc_algorithm"
    GES_ALGORITHM = "ges_algorithm"
    LINGAM = "lingam"
    NOTEARS = "notears"
    
    # Intervention-based methods
    RANDOMIZED_INTERVENTION = "randomized_intervention"
    OBSERVATIONAL_INTERVENTION = "observational_intervention"
    
    # Time series causal methods
    GRANGER_CAUSALITY = "granger_causality"
    TRANSFER_ENTROPY = "transfer_entropy"
    CCM = "convergent_cross_mapping"
    
    # Counterfactual methods
    COUNTERFACTUAL_INFERENCE = "counterfactual_inference"
    PROPENSITY_SCORE = "propensity_score"
    
    # Deep learning causal methods
    CAUSAL_CNN = "causal_cnn"
    CAUSAL_RNN = "causal_rnn"
    NEURAL_CAUSAL_MODEL = "neural_causal_model"


class CausalGraphType(Enum):
    """Types of causal graphs."""
    
    DAG = "directed_acyclic_graph"  # Standard causal DAG
    PDAG = "partially_directed_graph"  # Partially directed graph
    CPDAG = "completed_pdag"  # Completed PDAG
    MAG = "maximal_ancestral_graph"  # With latent confounders
    PAG = "partial_ancestral_graph"  # Partial ancestral graph


class CausalRelationType(Enum):
    """Types of causal relationships."""
    
    DIRECT_CAUSE = "direct_cause"  # X -> Y
    INDIRECT_CAUSE = "indirect_cause"  # X -> Z -> Y
    COMMON_CAUSE = "common_cause"  # Z -> X, Z -> Y
    COMMON_EFFECT = "common_effect"  # X -> Z, Y -> Z
    CONFOUNDER = "confounder"  # Z -> X, Z -> Y, X -> Y
    MEDIATOR = "mediator"  # X -> Z -> Y
    MODERATOR = "moderator"  # Z modifies X -> Y
    COLLIDER = "collider"  # X -> Z, Y -> Z
    SPURIOUS = "spurious"  # No causal relationship


class AnomalyType(Enum):
    """Types of anomalies in causal context."""
    
    STRUCTURAL_BREAK = "structural_break"  # Change in causal structure
    INTERVENTION_EFFECT = "intervention_effect"  # Effect of intervention
    CONFOUNDING_CHANGE = "confounding_change"  # Change in confounding
    MECHANISM_FAILURE = "mechanism_failure"  # Causal mechanism failure
    FEEDBACK_LOOP = "feedback_loop"  # Unexpected feedback
    CAUSAL_REVERSAL = "causal_reversal"  # Direction reversal
    SPURIOUS_CORRELATION = "spurious_correlation"  # False correlation
    TREATMENT_EFFECT = "treatment_effect"  # Treatment effect anomaly


@dataclass
class CausalEdge:
    """Represents a causal edge in a causal graph."""
    
    source: str  # Source variable
    target: str  # Target variable
    edge_type: CausalRelationType
    strength: float  # Causal strength [0, 1]
    confidence: float  # Confidence in edge [0, 1]
    lag: int = 0  # Time lag (for time series)
    mechanism: Optional[str] = None  # Causal mechanism description
    
    def __post_init__(self):
        if not 0 <= self.strength <= 1:
            raise ValueError("Causal strength must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.lag < 0:
            raise ValueError("Lag must be non-negative")


@dataclass
class CausalGraph:
    """Represents a causal graph structure."""
    
    graph_id: UUID
    graph_type: CausalGraphType
    variables: List[str]  # Variable names
    edges: List[CausalEdge]  # Causal edges
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        # Validate that all edge variables are in the variable list
        all_vars = set(self.variables)
        for edge in self.edges:
            if edge.source not in all_vars:
                raise ValueError(f"Edge source '{edge.source}' not in variables")
            if edge.target not in all_vars:
                raise ValueError(f"Edge target '{edge.target}' not in variables")
    
    def get_parents(self, variable: str) -> List[str]:
        """Get parent variables (direct causes) of a variable."""
        return [edge.source for edge in self.edges if edge.target == variable]
    
    def get_children(self, variable: str) -> List[str]:
        """Get child variables (direct effects) of a variable."""
        return [edge.target for edge in self.edges if edge.source == variable]
    
    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all ancestral variables (causes) of a variable."""
        ancestors = set()
        to_process = [variable]
        
        while to_process:
            current = to_process.pop()
            parents = self.get_parents(current)
            for parent in parents:
                if parent not in ancestors:
                    ancestors.add(parent)
                    to_process.append(parent)
        
        return ancestors
    
    def get_descendants(self, variable: str) -> Set[str]:
        """Get all descendant variables (effects) of a variable."""
        descendants = set()
        to_process = [variable]
        
        while to_process:
            current = to_process.pop()
            children = self.get_children(current)
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    to_process.append(child)
        
        return descendants
    
    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic (DAG)."""
        # Simplified cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for child in self.get_children(node):
                if child not in visited:
                    if has_cycle(child):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for var in self.variables:
            if var not in visited:
                if has_cycle(var):
                    return False
        
        return True
    
    def get_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        """Get edge between two variables if it exists."""
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge
        return None
    
    def add_edge(self, edge: CausalEdge) -> None:
        """Add a causal edge to the graph."""
        if edge.source not in self.variables:
            raise ValueError(f"Source variable '{edge.source}' not in graph")
        if edge.target not in self.variables:
            raise ValueError(f"Target variable '{edge.target}' not in graph")
        
        # Remove existing edge if present
        self.edges = [e for e in self.edges if not (e.source == edge.source and e.target == edge.target)]
        self.edges.append(edge)
    
    def remove_edge(self, source: str, target: str) -> bool:
        """Remove edge between two variables."""
        original_length = len(self.edges)
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]
        return len(self.edges) < original_length


@dataclass
class InterventionSpec:
    """Specification for a causal intervention."""
    
    intervention_id: UUID
    target_variable: str  # Variable being intervened on
    intervention_type: str  # Type of intervention (do, set, shift, etc.)
    intervention_value: Any  # Value or function for intervention
    duration: Optional[int] = None  # Duration in time steps
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        valid_types = {"do", "set", "shift", "noise", "remove", "block"}
        if self.intervention_type not in valid_types:
            raise ValueError(f"Intervention type must be one of {valid_types}")


@dataclass
class CausalAnomalyEvent:
    """Represents a detected causal anomaly."""
    
    event_id: UUID
    anomaly_type: AnomalyType
    affected_variables: List[str]
    causal_score: float  # Strength of causal evidence [0, 1]
    timestamp: datetime
    
    # Causal explanations
    root_causes: List[str] = field(default_factory=list)
    causal_path: List[str] = field(default_factory=list)
    confounders: List[str] = field(default_factory=list)
    
    # Evidence and metrics
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    
    # Context
    pre_intervention_graph: Optional[UUID] = None
    post_intervention_graph: Optional[UUID] = None
    intervention_spec: Optional[UUID] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.causal_score <= 1:
            raise ValueError("Causal score must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.p_value is not None and not 0 <= self.p_value <= 1:
            raise ValueError("P-value must be between 0 and 1")


@dataclass
class CausalAnalysisConfig:
    """Configuration for causal analysis."""
    
    method: CausalMethod
    graph_type: CausalGraphType = CausalGraphType.DAG
    
    # Structure learning parameters
    alpha: float = 0.05  # Significance level
    max_lag: int = 5  # Maximum time lag
    min_causal_effect: float = 0.1  # Minimum effect size
    
    # Intervention parameters
    intervention_strength: float = 1.0
    num_bootstrap_samples: int = 1000
    
    # Algorithm-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Anomaly detection parameters
    anomaly_threshold: float = 0.8
    confidence_threshold: float = 0.9
    
    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        if self.max_lag < 0:
            raise ValueError("Max lag must be non-negative")
        if not 0 <= self.min_causal_effect <= 1:
            raise ValueError("Min causal effect must be between 0 and 1")


@dataclass
class CausalDetector:
    """Causal anomaly detector for identifying causal anomalies."""
    
    detector_id: UUID
    name: str
    config: CausalAnalysisConfig
    
    # Model state
    baseline_graph: Optional[CausalGraph] = None
    learned_mechanisms: Dict[str, Any] = field(default_factory=dict)
    intervention_history: List[InterventionSpec] = field(default_factory=list)
    
    # Training state
    is_trained: bool = False
    training_samples: int = 0
    validation_metrics: Optional[ModelMetrics] = None
    
    # Performance tracking
    detected_anomalies: List[CausalAnomalyEvent] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Detector name cannot be empty")
    
    def add_intervention(self, intervention: InterventionSpec) -> None:
        """Add an intervention to the history."""
        self.intervention_history.append(intervention)
        self.last_updated = datetime.utcnow()
    
    def get_recent_interventions(self, n: int = 10) -> List[InterventionSpec]:
        """Get the n most recent interventions."""
        return self.intervention_history[-n:]
    
    def detect_causal_anomaly(
        self, 
        data: np.ndarray, 
        variable_names: List[str],
        timestamp: Optional[datetime] = None
    ) -> Optional[CausalAnomalyEvent]:
        """Detect causal anomalies in the given data."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection")
        
        timestamp = timestamp or datetime.utcnow()
        
        # Placeholder for actual causal anomaly detection
        # This would implement the specific causal inference algorithm
        
        # For now, return a simple anomaly based on data variance
        anomaly_score = np.std(data) if len(data) > 0 else 0.0
        normalized_score = min(1.0, anomaly_score / 10.0)  # Normalize to [0, 1]
        
        if normalized_score > self.config.anomaly_threshold:
            event = CausalAnomalyEvent(
                event_id=uuid4(),
                anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                affected_variables=variable_names,
                causal_score=normalized_score,
                timestamp=timestamp,
                confidence=min(1.0, normalized_score * 1.2),
                evidence={"data_variance": float(anomaly_score)},
            )
            
            self.detected_anomalies.append(event)
            return event
        
        return None
    
    def explain_anomaly(self, anomaly: CausalAnomalyEvent) -> Dict[str, Any]:
        """Provide causal explanation for an anomaly."""
        explanation = {
            "anomaly_id": str(anomaly.event_id),
            "type": anomaly.anomaly_type.value,
            "causal_explanation": {
                "root_causes": anomaly.root_causes,
                "causal_path": anomaly.causal_path,
                "confounders": anomaly.confounders,
            },
            "evidence": anomaly.evidence,
            "confidence": anomaly.confidence,
            "recommendations": self._generate_recommendations(anomaly),
        }
        
        return explanation
    
    def _generate_recommendations(self, anomaly: CausalAnomalyEvent) -> List[str]:
        """Generate recommendations based on anomaly type."""
        recommendations = []
        
        if anomaly.anomaly_type == AnomalyType.STRUCTURAL_BREAK:
            recommendations.extend([
                "Investigate recent changes in system configuration",
                "Check for external interventions or environmental changes",
                "Re-estimate causal model with recent data",
            ])
        elif anomaly.anomaly_type == AnomalyType.INTERVENTION_EFFECT:
            recommendations.extend([
                "Verify intervention was applied correctly",
                "Check for unintended side effects",
                "Monitor intervention duration and decay",
            ])
        elif anomaly.anomaly_type == AnomalyType.CONFOUNDING_CHANGE:
            recommendations.extend([
                "Identify new confounding variables",
                "Adjust for discovered confounders",
                "Re-run causal analysis with expanded variable set",
            ])
        
        return recommendations
    
    def simulate_intervention(
        self, 
        intervention: InterventionSpec,
        data: np.ndarray,
        steps: int = 100
    ) -> Dict[str, Any]:
        """Simulate the effect of a causal intervention."""
        if not self.baseline_graph:
            raise ValueError("Baseline causal graph required for simulation")
        
        # Placeholder for intervention simulation
        # This would use the learned causal model to predict intervention effects
        
        simulated_effect = {
            "intervention": {
                "variable": intervention.target_variable,
                "type": intervention.intervention_type,
                "value": intervention.intervention_value,
            },
            "predicted_effects": {},
            "confidence_intervals": {},
            "time_horizon": steps,
        }
        
        # Simulate effects on all variables
        if self.baseline_graph:
            for var in self.baseline_graph.variables:
                if var != intervention.target_variable:
                    # Simple simulation: effect proportional to causal path strength
                    effect_magnitude = np.random.normal(0, 0.1)
                    simulated_effect["predicted_effects"][var] = effect_magnitude
                    simulated_effect["confidence_intervals"][var] = [
                        effect_magnitude - 0.2, effect_magnitude + 0.2
                    ]
        
        return simulated_effect
    
    def get_detector_summary(self) -> Dict[str, Any]:
        """Get comprehensive detector summary."""
        return {
            "detector_id": str(self.detector_id),
            "name": self.name,
            "method": self.config.method.value,
            "graph_type": self.config.graph_type.value,
            "is_trained": self.is_trained,
            "training_samples": self.training_samples,
            "baseline_graph": {
                "variables": len(self.baseline_graph.variables) if self.baseline_graph else 0,
                "edges": len(self.baseline_graph.edges) if self.baseline_graph else 0,
                "is_acyclic": self.baseline_graph.is_acyclic() if self.baseline_graph else None,
            },
            "anomaly_statistics": {
                "total_detected": len(self.detected_anomalies),
                "by_type": self._get_anomaly_type_counts(),
                "avg_causal_score": self._get_avg_causal_score(),
            },
            "intervention_history": len(self.intervention_history),
            "validation_metrics": {
                "accuracy": self.validation_metrics.accuracy,
                "precision": self.validation_metrics.precision,
                "recall": self.validation_metrics.recall,
                "f1_score": self.validation_metrics.f1_score,
            } if self.validation_metrics else None,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }
    
    def _get_anomaly_type_counts(self) -> Dict[str, int]:
        """Get counts of anomalies by type."""
        type_counts = {}
        for anomaly in self.detected_anomalies:
            type_name = anomaly.anomaly_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts
    
    def _get_avg_causal_score(self) -> float:
        """Get average causal score of detected anomalies."""
        if not self.detected_anomalies:
            return 0.0
        
        total_score = sum(anomaly.causal_score for anomaly in self.detected_anomalies)
        return total_score / len(self.detected_anomalies)


@dataclass
class CausalDataset:
    """Dataset for causal analysis with metadata."""
    
    dataset_id: UUID
    name: str
    data: np.ndarray  # Shape: (n_samples, n_variables)
    variable_names: List[str]
    timestamps: Optional[List[datetime]] = None
    
    # Causal metadata
    known_edges: List[CausalEdge] = field(default_factory=list)
    interventions: List[InterventionSpec] = field(default_factory=list)
    confounders: List[str] = field(default_factory=list)
    
    # Dataset properties
    is_time_series: bool = False
    is_experimental: bool = False  # Contains interventions
    is_observational: bool = True   # Purely observational
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.data.shape[1] != len(self.variable_names):
            raise ValueError("Number of variables must match data dimensions")
        if self.timestamps and len(self.timestamps) != self.data.shape[0]:
            raise ValueError("Number of timestamps must match data samples")
        
        # Set flags based on interventions
        if self.interventions:
            self.is_experimental = True
            self.is_observational = False
    
    def get_variable_data(self, variable_name: str) -> np.ndarray:
        """Get data for a specific variable."""
        if variable_name not in self.variable_names:
            raise ValueError(f"Variable '{variable_name}' not found")
        
        var_index = self.variable_names.index(variable_name)
        return self.data[:, var_index]
    
    def get_subset(self, start_idx: int, end_idx: int) -> 'CausalDataset':
        """Get a subset of the dataset."""
        subset_data = self.data[start_idx:end_idx]
        subset_timestamps = None
        if self.timestamps:
            subset_timestamps = self.timestamps[start_idx:end_idx]
        
        return CausalDataset(
            dataset_id=uuid4(),
            name=f"{self.name}_subset_{start_idx}_{end_idx}",
            data=subset_data,
            variable_names=self.variable_names.copy(),
            timestamps=subset_timestamps,
            known_edges=self.known_edges.copy(),
            interventions=[],  # Don't copy interventions to subset
            confounders=self.confounders.copy(),
            is_time_series=self.is_time_series,
            is_experimental=False,  # Subset is observational
            is_observational=True,
            metadata=self.metadata.copy(),
        )
    
    def add_intervention_period(
        self, 
        intervention: InterventionSpec,
        start_idx: int,
        end_idx: int
    ) -> None:
        """Mark a period as having an intervention."""
        intervention_with_period = InterventionSpec(
            intervention_id=intervention.intervention_id,
            target_variable=intervention.target_variable,
            intervention_type=intervention.intervention_type,
            intervention_value=intervention.intervention_value,
            duration=end_idx - start_idx,
            metadata={
                **intervention.metadata,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )
        
        self.interventions.append(intervention_with_period)
        
        # Update flags
        self.is_experimental = True
        self.is_observational = len(self.interventions) == 0


# Aliases for backward compatibility
CausalInferenceMethod = CausalMethod
InterventionSpecification = InterventionSpec