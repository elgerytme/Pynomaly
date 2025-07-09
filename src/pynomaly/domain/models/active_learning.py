"""Active learning domain models for intelligent sample selection and iterative model improvement."""

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


class QueryStrategy(Enum):
    """Active learning query strategies for sample selection."""
    
    # Uncertainty-based strategies
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    LEAST_CONFIDENT = "least_confident"
    MARGIN_SAMPLING = "margin_sampling"
    ENTROPY_SAMPLING = "entropy_sampling"
    
    # Diversity-based strategies
    DIVERSITY_SAMPLING = "diversity_sampling"
    CLUSTER_BASED = "cluster_based"
    REPRESENTATIVE_SAMPLING = "representative_sampling"
    
    # Hybrid strategies
    UNCERTAINTY_DIVERSITY = "uncertainty_diversity"
    EXPECTED_MODEL_CHANGE = "expected_model_change"
    EXPECTED_ERROR_REDUCTION = "expected_error_reduction"
    
    # Committee-based strategies
    QUERY_BY_COMMITTEE = "query_by_committee"
    VOTE_ENTROPY = "vote_entropy"
    KL_DIVERGENCE = "kl_divergence"
    
    # Information-theoretic strategies
    MUTUAL_INFORMATION = "mutual_information"
    FISHER_INFORMATION = "fisher_information"
    EXPECTED_GRADIENT_LENGTH = "expected_gradient_length"
    
    # Density-based strategies
    DENSITY_WEIGHTED = "density_weighted"
    OUTLIER_DETECTION = "outlier_detection"
    
    # Specialized for anomaly detection
    ANOMALY_UNCERTAINTY = "anomaly_uncertainty"
    BOUNDARY_SAMPLING = "boundary_sampling"
    RARE_CLASS_SAMPLING = "rare_class_sampling"


class LabelType(Enum):
    """Types of labels in active learning."""
    
    BINARY = "binary"  # Normal/Anomaly
    MULTI_CLASS = "multi_class"  # Multiple anomaly types
    CONFIDENCE = "confidence"  # Confidence scores
    RANKING = "ranking"  # Relative ranking
    SEVERITY = "severity"  # Anomaly severity


class SampleStatus(Enum):
    """Status of samples in active learning pipeline."""
    
    UNLABELED = "unlabeled"
    QUERIED = "queried"  # Selected for labeling
    LABELED = "labeled"
    VALIDATED = "validated"  # Expert validated
    REJECTED = "rejected"  # Poor quality label
    UNCERTAIN = "uncertain"  # Uncertain label


@dataclass
class LabeledSample:
    """Container for a labeled data sample."""
    
    sample_id: str
    data: np.ndarray
    label: Any  # Could be binary, multi-class, or continuous
    label_type: LabelType
    confidence: float = 1.0  # Labeler confidence
    
    # Metadata
    labeler_id: Optional[str] = None
    labeling_time: Optional[datetime] = None
    labeling_cost: float = 1.0  # Cost of obtaining this label
    annotation_notes: Optional[str] = None
    
    # Quality metrics
    label_quality: Optional[float] = None  # Estimated quality [0, 1]
    inter_annotator_agreement: Optional[float] = None
    
    # Active learning metadata
    selection_method: Optional[QueryStrategy] = None
    selection_score: Optional[float] = None
    selection_iteration: int = 0
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.labeling_cost < 0:
            raise ValueError("Labeling cost must be non-negative")
        if self.label_quality is not None and not 0 <= self.label_quality <= 1:
            raise ValueError("Label quality must be between 0 and 1")


@dataclass
class QueryResult:
    """Result of an active learning query."""
    
    query_id: UUID
    strategy: QueryStrategy
    selected_samples: List[str]  # Sample IDs
    selection_scores: Dict[str, float]  # Sample ID -> selection score
    query_time: datetime
    
    # Selection criteria
    selection_criteria: Dict[str, Any] = field(default_factory=dict)
    diversity_score: Optional[float] = None
    uncertainty_score: Optional[float] = None
    
    # Expected improvement
    expected_performance_gain: Optional[float] = None
    expected_cost: float = 0.0
    
    # Query metadata
    model_version: Optional[str] = None
    dataset_size: int = 0
    labeled_ratio: float = 0.0
    
    def __post_init__(self):
        if self.expected_cost < 0:
            raise ValueError("Expected cost must be non-negative")
        if self.labeled_ratio < 0 or self.labeled_ratio > 1:
            raise ValueError("Labeled ratio must be between 0 and 1")


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning pipeline."""
    
    query_strategy: QueryStrategy
    batch_size: int = 10  # Number of samples to query per iteration
    max_iterations: int = 100
    max_budget: float = 1000.0  # Maximum labeling budget
    
    # Stopping criteria
    performance_threshold: float = 0.95  # Stop when this performance is reached
    improvement_threshold: float = 0.01  # Stop when improvement < this
    min_samples_per_iteration: int = 1
    max_samples_per_iteration: int = 100
    
    # Query strategy parameters
    uncertainty_threshold: float = 0.5
    diversity_weight: float = 0.5  # For hybrid strategies
    committee_size: int = 5  # For committee-based strategies
    
    # Sampling parameters
    exploration_rate: float = 0.1  # Exploration vs exploitation
    temperature: float = 1.0  # For softmax sampling
    
    # Quality control
    enable_quality_control: bool = True
    min_label_quality: float = 0.7
    require_validation: bool = False
    
    # Budget management
    adaptive_budget: bool = True
    cost_sensitivity: float = 1.0
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.max_budget <= 0:
            raise ValueError("Max budget must be positive")
        if not 0 <= self.performance_threshold <= 1:
            raise ValueError("Performance threshold must be between 0 and 1")


@dataclass
class LearningIteration:
    """Record of a single active learning iteration."""
    
    iteration_id: UUID
    iteration_number: int
    query_result: QueryResult
    new_labels: List[LabeledSample]
    
    # Model performance before/after
    pre_iteration_performance: Optional[ModelMetrics] = None
    post_iteration_performance: Optional[ModelMetrics] = None
    
    # Iteration statistics
    labeling_cost: float = 0.0
    labeling_time: float = 0.0  # Time in seconds
    model_training_time: float = 0.0
    
    # Learning progress
    performance_improvement: float = 0.0
    cumulative_budget_used: float = 0.0
    total_labeled_samples: int = 0
    
    # Iteration metadata
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.iteration_number < 0:
            raise ValueError("Iteration number must be non-negative")
        if self.labeling_cost < 0:
            raise ValueError("Labeling cost must be non-negative")
        if self.cumulative_budget_used < 0:
            raise ValueError("Cumulative budget must be non-negative")
    
    def get_duration(self) -> Optional[float]:
        """Get iteration duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_cost_efficiency(self) -> float:
        """Get performance improvement per cost unit."""
        if self.labeling_cost > 0:
            return self.performance_improvement / self.labeling_cost
        return 0.0


@dataclass
class ActiveLearningSession:
    """Complete active learning session tracking."""
    
    session_id: UUID
    name: str
    config: ActiveLearningConfig
    
    # Session state
    current_iteration: int = 0
    total_budget_used: float = 0.0
    is_active: bool = True
    
    # Data tracking
    initial_labeled_samples: int = 0
    total_samples: int = 0
    labeled_samples: List[LabeledSample] = field(default_factory=list)
    unlabeled_pool: Set[str] = field(default_factory=set)  # Sample IDs
    
    # Iteration history
    iterations: List[LearningIteration] = field(default_factory=list)
    
    # Performance tracking
    initial_performance: Optional[ModelMetrics] = None
    current_performance: Optional[ModelMetrics] = None
    best_performance: Optional[ModelMetrics] = None
    
    # Session metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.current_iteration < 0:
            raise ValueError("Current iteration must be non-negative")
        if self.total_budget_used < 0:
            raise ValueError("Total budget used must be non-negative")
        if self.initial_labeled_samples < 0:
            raise ValueError("Initial labeled samples must be non-negative")
    
    def add_iteration(self, iteration: LearningIteration) -> None:
        """Add a learning iteration to the session."""
        iteration.iteration_number = self.current_iteration
        iteration.cumulative_budget_used = self.total_budget_used + iteration.labeling_cost
        
        self.iterations.append(iteration)
        self.current_iteration += 1
        self.total_budget_used += iteration.labeling_cost
        
        # Update performance tracking
        if iteration.post_iteration_performance:
            self.current_performance = iteration.post_iteration_performance
            
            if (self.best_performance is None or 
                iteration.post_iteration_performance.f1_score > self.best_performance.f1_score):
                self.best_performance = iteration.post_iteration_performance
    
    def get_labeled_count(self) -> int:
        """Get current number of labeled samples."""
        return len(self.labeled_samples)
    
    def get_labeling_ratio(self) -> float:
        """Get ratio of labeled to total samples."""
        if self.total_samples > 0:
            return len(self.labeled_samples) / self.total_samples
        return 0.0
    
    def get_budget_utilization(self) -> float:
        """Get ratio of budget used to total budget."""
        return self.total_budget_used / self.config.max_budget
    
    def should_stop(self) -> Tuple[bool, str]:
        """Check if learning should stop based on criteria."""
        
        # Budget exhausted
        if self.total_budget_used >= self.config.max_budget:
            return True, "Budget exhausted"
        
        # Max iterations reached
        if self.current_iteration >= self.config.max_iterations:
            return True, "Max iterations reached"
        
        # Performance threshold reached
        if (self.current_performance and 
            self.current_performance.f1_score >= self.config.performance_threshold):
            return True, "Performance threshold reached"
        
        # No improvement in recent iterations
        if len(self.iterations) >= 3:
            recent_improvements = [iter.performance_improvement for iter in self.iterations[-3:]]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            
            if avg_improvement < self.config.improvement_threshold:
                return True, "Insufficient improvement"
        
        # All samples labeled
        if len(self.unlabeled_pool) == 0:
            return True, "No unlabeled samples remaining"
        
        return False, ""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        return {
            "session_id": str(self.session_id),
            "name": self.name,
            "strategy": self.config.query_strategy.value,
            "status": "active" if self.is_active else "completed",
            "iterations_completed": self.current_iteration,
            "budget_utilization": self.get_budget_utilization(),
            "labeling_ratio": self.get_labeling_ratio(),
            "performance_improvement": {
                "initial": {
                    "f1_score": self.initial_performance.f1_score if self.initial_performance else 0,
                    "accuracy": self.initial_performance.accuracy if self.initial_performance else 0,
                } if self.initial_performance else None,
                "current": {
                    "f1_score": self.current_performance.f1_score if self.current_performance else 0,
                    "accuracy": self.current_performance.accuracy if self.current_performance else 0,
                } if self.current_performance else None,
                "best": {
                    "f1_score": self.best_performance.f1_score if self.best_performance else 0,
                    "accuracy": self.best_performance.accuracy if self.best_performance else 0,
                } if self.best_performance else None,
            },
            "efficiency_metrics": {
                "total_cost": self.total_budget_used,
                "average_cost_per_iteration": (
                    self.total_budget_used / self.current_iteration if self.current_iteration > 0 else 0
                ),
                "samples_per_iteration": (
                    len(self.labeled_samples) / self.current_iteration if self.current_iteration > 0 else 0
                ),
            },
            "timing": {
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "ended_at": self.ended_at.isoformat() if self.ended_at else None,
                "duration_hours": (
                    (self.ended_at - self.started_at).total_seconds() / 3600 
                    if self.started_at and self.ended_at else None
                ),
            },
        }


@dataclass
class SamplePool:
    """Pool of samples for active learning."""
    
    pool_id: UUID
    name: str
    samples: Dict[str, np.ndarray]  # Sample ID -> data
    
    # Sample metadata
    sample_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sample_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Pool statistics
    feature_statistics: Dict[str, Any] = field(default_factory=dict)
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Pool organization
    clusters: Dict[str, List[str]] = field(default_factory=dict)  # Cluster ID -> sample IDs
    outliers: Set[str] = field(default_factory=set)
    representatives: Set[str] = field(default_factory=set)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.samples:
            raise ValueError("Sample pool cannot be empty")
        
        # Validate that all samples have consistent dimensions
        sample_shapes = [data.shape for data in self.samples.values()]
        if len(set(sample_shapes)) > 1:
            raise ValueError("All samples must have the same shape")
    
    def add_sample(self, sample_id: str, data: np.ndarray, features: Optional[Dict[str, Any]] = None) -> None:
        """Add a sample to the pool."""
        if sample_id in self.samples:
            raise ValueError(f"Sample {sample_id} already exists in pool")
        
        # Validate shape consistency
        if self.samples:
            expected_shape = next(iter(self.samples.values())).shape
            if data.shape != expected_shape:
                raise ValueError(f"Sample shape {data.shape} doesn't match expected {expected_shape}")
        
        self.samples[sample_id] = data
        if features:
            self.sample_features[sample_id] = features
        
        self.last_updated = datetime.utcnow()
    
    def remove_sample(self, sample_id: str) -> bool:
        """Remove a sample from the pool."""
        if sample_id not in self.samples:
            return False
        
        del self.samples[sample_id]
        
        # Clean up related data
        if sample_id in self.sample_features:
            del self.sample_features[sample_id]
        if sample_id in self.sample_embeddings:
            del self.sample_embeddings[sample_id]
        
        # Remove from sets
        self.outliers.discard(sample_id)
        self.representatives.discard(sample_id)
        
        # Remove from clusters
        for cluster_samples in self.clusters.values():
            if sample_id in cluster_samples:
                cluster_samples.remove(sample_id)
        
        self.last_updated = datetime.utcnow()
        return True
    
    def get_sample_data(self, sample_id: str) -> Optional[np.ndarray]:
        """Get data for a specific sample."""
        return self.samples.get(sample_id)
    
    def get_random_samples(self, n: int) -> List[str]:
        """Get n random sample IDs."""
        sample_ids = list(self.samples.keys())
        n = min(n, len(sample_ids))
        return np.random.choice(sample_ids, size=n, replace=False).tolist()
    
    def get_pool_size(self) -> int:
        """Get number of samples in pool."""
        return len(self.samples)
    
    def get_data_matrix(self, sample_ids: Optional[List[str]] = None) -> np.ndarray:
        """Get data matrix for specified samples (or all samples)."""
        if sample_ids is None:
            sample_ids = list(self.samples.keys())
        
        return np.array([self.samples[sid] for sid in sample_ids if sid in self.samples])
    
    def compute_diversity_metrics(self) -> Dict[str, float]:
        """Compute diversity metrics for the pool."""
        if len(self.samples) < 2:
            return {"diversity": 0.0}
        
        data_matrix = self.get_data_matrix()
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(data_matrix)
        
        metrics = {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "diversity_index": float(np.mean(distances) / np.std(distances)) if np.std(distances) > 0 else 0.0,
        }
        
        self.diversity_metrics = metrics
        return metrics