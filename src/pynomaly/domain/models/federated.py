"""Federated learning domain models for privacy-preserving anomaly detection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.models.base import DomainModel
from pynomaly.domain.value_objects import ModelMetrics


class FederationStrategy(Enum):
    """Federated learning aggregation strategies."""
    
    FEDERATED_AVERAGING = "federated_averaging"
    FEDERATED_SGD = "federated_sgd"
    BYZANTINE_ROBUST = "byzantine_robust"
    DIFFERENTIAL_PRIVATE = "differential_private"
    SECURE_AGGREGATION = "secure_aggregation"


class ParticipantRole(Enum):
    """Roles in federated learning."""
    
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    VALIDATOR = "validator"
    OBSERVER = "observer"


class AggregationMethod(Enum):
    """Model parameter aggregation methods."""
    
    WEIGHTED_AVERAGE = "weighted_average"
    SIMPLE_AVERAGE = "simple_average"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    BYZANTINE_RESILIENT = "byzantine_resilient"


class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms."""
    
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_MULTIPARTY = "secure_multiparty"
    FEDERATED_DISTILLATION = "federated_distillation"
    NOISE_INJECTION = "noise_injection"


@dataclass
class PrivacyBudget:
    """Privacy budget for differential privacy."""
    
    epsilon: float
    delta: float
    spent_epsilon: float = 0.0
    max_queries: int = 1000
    query_count: int = 0
    
    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not 0 <= self.delta <= 1:
            raise ValueError("Delta must be between 0 and 1")
    
    def can_answer_query(self, query_epsilon: float) -> bool:
        """Check if query can be answered within privacy budget."""
        return (
            self.spent_epsilon + query_epsilon <= self.epsilon and
            self.query_count < self.max_queries
        )
    
    def consume_budget(self, query_epsilon: float) -> None:
        """Consume privacy budget for a query."""
        if not self.can_answer_query(query_epsilon):
            raise ValueError("Insufficient privacy budget")
        
        self.spent_epsilon += query_epsilon
        self.query_count += 1


@dataclass
class FederatedParticipant(DomainModel):
    """Participant in federated learning network."""
    
    participant_id: UUID
    name: str
    role: ParticipantRole
    public_key: str
    trust_score: float = 1.0
    data_size: int = 0
    computation_capacity: float = 1.0
    network_latency_ms: float = 0.0
    privacy_budget: Optional[PrivacyBudget] = None
    is_active: bool = True
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    contribution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not 0 <= self.trust_score <= 1:
            raise ValueError("Trust score must be between 0 and 1")
        if self.data_size < 0:
            raise ValueError("Data size cannot be negative")
        if self.computation_capacity <= 0:
            raise ValueError("Computation capacity must be positive")


@dataclass
class ModelUpdate:
    """Model parameter update from a participant."""
    
    update_id: UUID
    participant_id: UUID
    round_number: int
    parameters: Dict[str, np.ndarray]
    gradients: Optional[Dict[str, np.ndarray]] = None
    metrics: Optional[ModelMetrics] = None
    data_size: int = 0
    computation_time_seconds: float = 0.0
    privacy_spent: float = 0.0
    checksum: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        content = f"{self.participant_id}{self.round_number}{self.data_size}"
        for name, array in self.parameters.items():
            content += f"{name}{array.tobytes().hex()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify update integrity."""
        return self.checksum == self._compute_checksum()


@dataclass
class FederatedRound:
    """Single round of federated training."""
    
    round_id: UUID
    round_number: int
    federation_id: UUID
    global_model_version: str
    target_participants: Set[UUID]
    received_updates: Dict[UUID, ModelUpdate] = field(default_factory=dict)
    aggregated_parameters: Optional[Dict[str, np.ndarray]] = None
    aggregation_metrics: Optional[Dict[str, float]] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    is_completed: bool = False
    
    @property
    def participation_rate(self) -> float:
        """Calculate participation rate for this round."""
        if not self.target_participants:
            return 0.0
        return len(self.received_updates) / len(self.target_participants)
    
    @property
    def is_valid(self) -> bool:
        """Check if round has sufficient participation."""
        return self.participation_rate >= 0.5  # Require 50% participation
    
    def add_update(self, update: ModelUpdate) -> None:
        """Add participant update to round."""
        if update.participant_id not in self.target_participants:
            raise ValueError("Update from non-target participant")
        
        if not update.verify_integrity():
            raise ValueError("Update failed integrity check")
        
        self.received_updates[update.participant_id] = update
    
    def complete_round(self, aggregated_params: Dict[str, np.ndarray]) -> None:
        """Mark round as completed with aggregated parameters."""
        self.aggregated_parameters = aggregated_params
        self.completed_at = datetime.utcnow()
        self.is_completed = True


@dataclass
class FederatedDetector(DomainModel):
    """Federated anomaly detector coordinating distributed training."""
    
    federation_id: UUID
    name: str
    base_detector: Detector
    strategy: FederationStrategy
    aggregation_method: AggregationMethod
    privacy_mechanism: PrivacyMechanism
    participants: Dict[UUID, FederatedParticipant] = field(default_factory=dict)
    current_round: Optional[FederatedRound] = None
    training_rounds: List[FederatedRound] = field(default_factory=list)
    global_model_version: str = "v1.0.0"
    min_participants: int = 3
    max_participants: int = 100
    round_timeout_minutes: int = 60
    convergence_threshold: float = 1e-4
    max_rounds: int = 100
    byzantine_tolerance: float = 0.1
    differential_privacy_budget: Optional[PrivacyBudget] = None
    coordinator_public_key: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_training: Optional[datetime] = None
    
    def __post_init__(self):
        if self.min_participants < 2:
            raise ValueError("Minimum participants must be at least 2")
        if self.max_participants < self.min_participants:
            raise ValueError("Maximum participants must be >= minimum")
        if not 0 <= self.byzantine_tolerance <= 0.5:
            raise ValueError("Byzantine tolerance must be between 0 and 0.5")
    
    @property
    def is_active(self) -> bool:
        """Check if federation is actively training."""
        return self.current_round is not None and not self.current_round.is_completed
    
    @property
    def active_participants(self) -> List[FederatedParticipant]:
        """Get list of active participants."""
        return [p for p in self.participants.values() if p.is_active]
    
    @property
    def can_start_training(self) -> bool:
        """Check if federation can start training."""
        return (
            len(self.active_participants) >= self.min_participants and
            not self.is_active
        )
    
    def add_participant(self, participant: FederatedParticipant) -> None:
        """Add participant to federation."""
        if len(self.participants) >= self.max_participants:
            raise ValueError("Federation at maximum capacity")
        
        if participant.participant_id in self.participants:
            raise ValueError("Participant already in federation")
        
        self.participants[participant.participant_id] = participant
    
    def remove_participant(self, participant_id: UUID) -> None:
        """Remove participant from federation."""
        if participant_id not in self.participants:
            raise ValueError("Participant not in federation")
        
        # Mark as inactive rather than removing to preserve history
        self.participants[participant_id].is_active = False
    
    def start_training_round(self) -> FederatedRound:
        """Start new training round."""
        if not self.can_start_training:
            raise ValueError("Cannot start training round")
        
        round_number = len(self.training_rounds) + 1
        target_participants = {p.participant_id for p in self.active_participants}
        
        self.current_round = FederatedRound(
            round_id=uuid4(),
            round_number=round_number,
            federation_id=self.federation_id,
            global_model_version=self.global_model_version,
            target_participants=target_participants,
            deadline=datetime.utcnow().replace(
                minute=datetime.utcnow().minute + self.round_timeout_minutes
            ),
        )
        
        return self.current_round
    
    def complete_current_round(
        self, 
        aggregated_params: Dict[str, np.ndarray]
    ) -> None:
        """Complete current training round."""
        if not self.current_round:
            raise ValueError("No active training round")
        
        self.current_round.complete_round(aggregated_params)
        self.training_rounds.append(self.current_round)
        self.current_round = None
        self.last_training = datetime.utcnow()
        
        # Update global model version
        version_parts = self.global_model_version.split('.')
        patch_version = int(version_parts[2]) + 1
        self.global_model_version = f"{version_parts[0]}.{version_parts[1]}.{patch_version}"
    
    def get_participant_weights(self) -> Dict[UUID, float]:
        """Calculate participant weights for aggregation."""
        active_participants = self.active_participants
        
        if self.aggregation_method == AggregationMethod.SIMPLE_AVERAGE:
            return {p.participant_id: 1.0 for p in active_participants}
        
        elif self.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            total_data = sum(p.data_size for p in active_participants)
            if total_data == 0:
                return {p.participant_id: 1.0 for p in active_participants}
            
            return {
                p.participant_id: p.data_size / total_data 
                for p in active_participants
            }
        
        else:
            # Default to simple average
            return {p.participant_id: 1.0 for p in active_participants}
    
    def calculate_convergence_metric(self) -> Optional[float]:
        """Calculate convergence metric based on recent rounds."""
        if len(self.training_rounds) < 2:
            return None
        
        # Simple convergence based on aggregation metrics
        recent_rounds = self.training_rounds[-5:]  # Last 5 rounds
        
        if not all(r.aggregation_metrics for r in recent_rounds):
            return None
        
        losses = [r.aggregation_metrics.get('loss', 0) for r in recent_rounds]
        if len(losses) < 2:
            return None
        
        # Calculate relative change in loss
        recent_loss = losses[-1]
        previous_loss = losses[-2]
        
        if previous_loss == 0:
            return float('inf')
        
        return abs(recent_loss - previous_loss) / previous_loss