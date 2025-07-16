"""Federated learning infrastructure for privacy-preserving distributed training."""

from .aggregation import (
    AggregationStrategy,
    FederatedAggregationService,
    FederatedAveragingAggregation,
    GeometricMedianAggregation,
    KrumAggregation,
    MultiKrumAggregation,
    TrimmedMeanAggregation,
)
from .coordinator import FederatedCoordinator
from .participant import FederatedParticipantClient

__all__ = [
    "FederatedCoordinator",
    "FederatedParticipantClient",
    "FederatedAggregationService",
    "AggregationStrategy",
    "FederatedAveragingAggregation",
    "TrimmedMeanAggregation",
    "KrumAggregation",
    "MultiKrumAggregation",
    "GeometricMedianAggregation",
]
