"""Model version entity for model persistence framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pynomaly.domain.value_objects.model_storage_info import ModelStorageInfo
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics
from pynomaly.domain.value_objects.semantic_version import SemanticVersion


class ModelStatus(Enum):
    """Status of a model version."""

    DRAFT = "draft"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Represents a specific version of a trained model.

    This entity captures a complete snapshot of a model at a specific point
    in time, including its performance, storage information, and metadata.

    Attributes:
        id: Unique identifier for this model version
        model_id: Identifier of the parent model
        version: Semantic version number
        detector_id: ID of the detector this version was created from
        created_at: When this version was created
        created_by: User who created this version
        tags: Semantic tags for organization and discovery
        performance_metrics: Performance measurements for this version
        storage_info: Information about where and how the model is stored
        metadata: Additional metadata and annotations
        status: Current status of this model version
        parent_version_id: ID of the version this was derived from
        description: Human-readable description of this version
    """

    model_id: UUID
    version: SemanticVersion
    detector_id: UUID
    created_by: str
    performance_metrics: PerformanceMetrics
    storage_info: ModelStorageInfo
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: ModelStatus = ModelStatus.DRAFT
    parent_version_id: UUID | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate model version after initialization."""
        if not isinstance(self.version, SemanticVersion):
            raise TypeError(
                f"Version must be SemanticVersion instance, got {type(self.version)}"
            )

        if not isinstance(self.performance_metrics, PerformanceMetrics):
            raise TypeError(
                f"Performance metrics must be PerformanceMetrics instance, "
                f"got {type(self.performance_metrics)}"
            )

        if not isinstance(self.storage_info, ModelStorageInfo):
            raise TypeError(
                f"Storage info must be ModelStorageInfo instance, "
                f"got {type(self.storage_info)}"
            )

        if not self.created_by:
            raise ValueError("Created by cannot be empty")

    @property
    def version_string(self) -> str:
        """Get version as string."""
        return self.version.version_string

    @property
    def is_deployed(self) -> bool:
        """Check if this version is currently deployed."""
        return self.status == ModelStatus.DEPLOYED

    @property
    def is_deprecated(self) -> bool:
        """Check if this version is deprecated."""
        return self.status == ModelStatus.DEPRECATED

    @property
    def is_archived(self) -> bool:
        """Check if this version is archived."""
        return self.status == ModelStatus.ARCHIVED

    def add_tag(self, tag: str) -> None:
        """Add a tag to this version."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this version."""
        if tag in self.tags:
            self.tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if this version has a specific tag."""
        return tag in self.tags

    def update_status(self, new_status: ModelStatus) -> None:
        """Update the status of this version."""
        self.status = new_status
        self.metadata["status_updated_at"] = datetime.utcnow().isoformat()

    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata for this version."""
        self.metadata[key] = value
        self.metadata["last_updated"] = datetime.utcnow().isoformat()

    def get_performance_summary(self) -> dict[str, float]:
        """Get a summary of performance metrics."""
        return {
            "accuracy": self.performance_metrics.accuracy,
            "precision": self.performance_metrics.precision,
            "recall": self.performance_metrics.recall,
            "f1_score": self.performance_metrics.f1_score,
            "training_time": self.performance_metrics.training_time,
            "inference_time": self.performance_metrics.inference_time,
        }

    def compare_performance(self, other: ModelVersion) -> dict[str, float]:
        """Compare performance metrics with another version."""
        if not isinstance(other, ModelVersion):
            raise TypeError("Can only compare with another ModelVersion")

        current = self.get_performance_summary()
        other_metrics = other.get_performance_summary()

        return {
            metric: current[metric] - other_metrics[metric]
            for metric in current.keys()
            if metric in other_metrics
        }

    def get_info(self) -> dict[str, Any]:
        """Get comprehensive information about this model version."""
        return {
            "id": str(self.id),
            "model_id": str(self.model_id),
            "version": self.version_string,
            "detector_id": str(self.detector_id),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "status": self.status.value,
            "tags": self.tags.copy(),
            "description": self.description,
            "parent_version_id": (
                str(self.parent_version_id) if self.parent_version_id else None
            ),
            "performance_metrics": self.performance_metrics.to_dict(),
            "storage_info": self.storage_info.to_dict(),
            "metadata": self.metadata.copy(),
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"ModelVersion(v{self.version_string}, "
            f"status={self.status.value}, "
            f"accuracy={self.performance_metrics.accuracy:.3f})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"ModelVersion(id={self.id}, model_id={self.model_id}, "
            f"version={self.version_string}, status={self.status.value})"
        )
