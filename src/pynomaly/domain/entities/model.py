"""Model entity for representing ML models in production."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

# Import for alias
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics

# Create alias for backward compatibility
ModelMetrics = PerformanceMetrics


class ModelType(Enum):
    """Type of anomaly detection model."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    TIME_SERIES = "time_series"


class ModelStage(Enum):
    """Stage of model in the ML lifecycle."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class Model:
    """Represents an ML model in the system.

    A Model is the conceptual representation of a machine learning solution
    for anomaly detection. It can have multiple versions (ModelVersion entities)
    and tracks the model's lifecycle from development to production.

    Attributes:
        id: Unique identifier for the model
        name: Human-readable name for the model
        description: Detailed description of the model's purpose
        model_type: Type of anomaly detection approach
        algorithm_family: The family of algorithms (e.g., 'isolation_forest', 'neural_network')
        created_at: When the model was first created
        created_by: User who created the model
        team: Team or organization responsible for the model
        tags: Semantic tags for organization and discovery
        stage: Current stage in the ML lifecycle
        current_version_id: ID of the currently active version
        latest_version_id: ID of the most recently created version
        metadata: Additional metadata and configuration
        use_cases: Specific use cases this model addresses
        data_requirements: Requirements for input data
    """

    name: str
    description: str
    model_type: ModelType
    algorithm_family: str
    created_by: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    team: str = ""
    tags: list[str] = field(default_factory=list)
    stage: ModelStage = ModelStage.DEVELOPMENT
    current_version_id: UUID | None = None
    latest_version_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    use_cases: list[str] = field(default_factory=list)
    data_requirements: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model after initialization."""
        if not self.name:
            raise ValueError("Model name cannot be empty")

        if not self.description:
            raise ValueError("Model description cannot be empty")

        if not isinstance(self.model_type, ModelType):
            raise TypeError(
                f"Model type must be ModelType, got {type(self.model_type)}"
            )

        if not isinstance(self.stage, ModelStage):
            raise TypeError(f"Model stage must be ModelStage, got {type(self.stage)}")

        if not self.created_by:
            raise ValueError("Created by cannot be empty")

        if not self.algorithm_family:
            raise ValueError("Algorithm family cannot be empty")

    @property
    def is_in_production(self) -> bool:
        """Check if model is currently in production."""
        return self.stage == ModelStage.PRODUCTION

    @property
    def is_archived(self) -> bool:
        """Check if model is archived."""
        return self.stage == ModelStage.ARCHIVED

    @property
    def has_current_version(self) -> bool:
        """Check if model has a current deployed version."""
        return self.current_version_id is not None

    @property
    def version_count(self) -> int:
        """Get number of versions (would need repository to implement)."""
        return self.metadata.get("version_count", 0)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        if tag in self.tags:
            self.tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if model has a specific tag."""
        return tag in self.tags

    def update_stage(self, new_stage: ModelStage) -> None:
        """Update the model's lifecycle stage."""
        old_stage = self.stage
        self.stage = new_stage

        # Record stage transition
        transitions = self.metadata.get("stage_transitions", [])
        transitions.append(
            {
                "from": old_stage.value,
                "to": new_stage.value,
                "timestamp": datetime.utcnow().isoformat(),
                "updated_by": self.metadata.get("last_updated_by", "system"),
            }
        )
        self.metadata["stage_transitions"] = transitions
        self.metadata["last_stage_update"] = datetime.utcnow().isoformat()

    def promote_to_production(self, version_id: UUID, updated_by: str) -> None:
        """Promote a specific version to production."""
        self.current_version_id = version_id
        self.update_stage(ModelStage.PRODUCTION)
        self.metadata["promoted_at"] = datetime.utcnow().isoformat()
        self.metadata["promoted_by"] = updated_by
        self.metadata["production_version_id"] = str(version_id)

    def add_use_case(self, use_case: str) -> None:
        """Add a use case for this model."""
        if use_case and use_case not in self.use_cases:
            self.use_cases.append(use_case)

    def remove_use_case(self, use_case: str) -> None:
        """Remove a use case from this model."""
        if use_case in self.use_cases:
            self.use_cases.remove(use_case)

    def update_data_requirements(self, requirements: dict[str, Any]) -> None:
        """Update data requirements for the model."""
        self.data_requirements.update(requirements)
        self.metadata["data_requirements_updated"] = datetime.utcnow().isoformat()

    def set_current_version(self, version_id: UUID) -> None:
        """Set the current active version."""
        self.current_version_id = version_id
        self.metadata["current_version_updated"] = datetime.utcnow().isoformat()

    def set_latest_version(self, version_id: UUID) -> None:
        """Set the latest created version."""
        self.latest_version_id = version_id
        self.metadata["latest_version_updated"] = datetime.utcnow().isoformat()

        # Increment version count
        self.metadata["version_count"] = self.metadata.get("version_count", 0) + 1

    def update_metadata(
        self, key: str, value: Any, updated_by: str | None = None
    ) -> None:
        """Update model metadata."""
        self.metadata[key] = value
        self.metadata["last_updated"] = datetime.utcnow().isoformat()
        if updated_by:
            self.metadata["last_updated_by"] = updated_by

    def get_info(self) -> dict[str, Any]:
        """Get comprehensive information about the model."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type.value,
            "algorithm_family": self.algorithm_family,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "team": self.team,
            "stage": self.stage.value,
            "is_in_production": self.is_in_production,
            "has_current_version": self.has_current_version,
            "current_version_id": (
                str(self.current_version_id) if self.current_version_id else None
            ),
            "latest_version_id": (
                str(self.latest_version_id) if self.latest_version_id else None
            ),
            "version_count": self.version_count,
            "tags": self.tags.copy(),
            "use_cases": self.use_cases.copy(),
            "data_requirements": self.data_requirements.copy(),
            "metadata": self.metadata.copy(),
        }

    def get_stage_history(self) -> list[dict[str, Any]]:
        """Get the history of stage transitions."""
        return self.metadata.get("stage_transitions", [])

    def can_deploy(self) -> tuple[bool, list[str]]:
        """Check if model can be deployed to production."""
        issues = []

        if not self.has_current_version:
            issues.append("No current version set")

        if not self.use_cases:
            issues.append("No use cases defined")

        if not self.data_requirements:
            issues.append("No data requirements specified")

        if self.stage == ModelStage.ARCHIVED:
            issues.append("Cannot deploy archived model")

        # Check for required metadata
        required_metadata = [
            "business_impact",
            "data_validation",
            "performance_baseline",
        ]
        for req in required_metadata:
            if req not in self.metadata:
                issues.append(f"Missing required metadata: {req}")

        return len(issues) == 0, issues

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Model('{self.name}', {self.model_type.value}, "
            f"stage={self.stage.value}, versions={self.version_count})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Model(id={self.id}, name='{self.name}', "
            f"type={self.model_type.value}, stage={self.stage.value})"
        )
