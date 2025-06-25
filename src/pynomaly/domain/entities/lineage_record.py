"""Model lineage tracking domain entities."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LineageRelationType(str, Enum):
    """Model lineage relationship types."""

    DERIVED_FROM = "derived_from"
    FINE_TUNED_FROM = "fine_tuned_from"
    ENSEMBLE_OF = "ensemble_of"
    DISTILLED_FROM = "distilled_from"
    RETRAINED_FROM = "retrained_from"
    MERGED_FROM = "merged_from"
    PRUNED_FROM = "pruned_from"
    QUANTIZED_FROM = "quantized_from"
    FEDERATED_FROM = "federated_from"
    CUSTOM = "custom"


class TransformationType(str, Enum):
    """Model transformation types."""

    TRAINING = "training"
    FINE_TUNING = "fine_tuning"
    TRANSFER_LEARNING = "transfer_learning"
    ENSEMBLE = "ensemble"
    DISTILLATION = "distillation"
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_AUGMENTATION = "data_augmentation"
    FEDERATED_LEARNING = "federated_learning"
    MODEL_MERGING = "model_merging"
    ARCHITECTURE_MODIFICATION = "architecture_modification"
    CUSTOM = "custom"


class LineageArtifact(BaseModel):
    """Artifact information in model lineage."""

    id: UUID = Field(..., description="Artifact identifier")
    type: str = Field(..., description="Artifact type (model, dataset, configuration, etc.)")
    name: str = Field(..., description="Artifact name")
    version: str | None = Field(None, description="Artifact version")
    checksum: str | None = Field(None, description="Artifact checksum for integrity")
    location: str | None = Field(None, description="Artifact storage location")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Artifact metadata")


class LineageTransformation(BaseModel):
    """Transformation information in model lineage."""

    type: TransformationType = Field(..., description="Transformation type")
    description: str | None = Field(None, description="Transformation description")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Transformation parameters")
    algorithm: str | None = Field(None, description="Algorithm used")
    tool: str | None = Field(None, description="Tool or framework used")
    version: str | None = Field(None, description="Tool/framework version")
    execution_time: float | None = Field(None, description="Execution time in seconds")
    resource_usage: dict[str, Any] = Field(
        default_factory=dict, description="Resource usage during transformation"
    )


class LineageRecord(BaseModel):
    """Model lineage record tracking relationships between models."""

    id: UUID = Field(default_factory=uuid4, description="Lineage record identifier")
    
    # Relationship information
    child_model_id: UUID = Field(..., description="Child (derived) model identifier")
    parent_model_ids: list[UUID] = Field(..., description="Parent model identifiers")
    relation_type: LineageRelationType = Field(..., description="Type of lineage relationship")
    
    # Transformation details
    transformation: LineageTransformation = Field(..., description="Transformation applied")
    
    # Input artifacts
    input_artifacts: list[LineageArtifact] = Field(
        default_factory=list, description="Input artifacts (datasets, configurations, etc.)"
    )
    
    # Output artifacts
    output_artifacts: list[LineageArtifact] = Field(
        default_factory=list, description="Output artifacts (models, metrics, etc.)"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    created_by: str = Field(..., description="User who created the record")
    experiment_id: UUID | None = Field(None, description="Associated experiment identifier")
    run_id: str | None = Field(None, description="Associated run identifier")
    tags: list[str] = Field(default_factory=list, description="Lineage tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Provenance information
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="Detailed provenance information"
    )

    def add_input_artifact(self, artifact: LineageArtifact) -> None:
        """Add input artifact to lineage record."""
        self.input_artifacts.append(artifact)

    def add_output_artifact(self, artifact: LineageArtifact) -> None:
        """Add output artifact to lineage record."""
        self.output_artifacts.append(artifact)

    def get_all_model_ids(self) -> set[UUID]:
        """Get all model IDs involved in this lineage record."""
        model_ids = set(self.parent_model_ids)
        model_ids.add(self.child_model_id)
        return model_ids

    def is_direct_descendant(self, parent_id: UUID, child_id: UUID) -> bool:
        """Check if child is a direct descendant of parent."""
        return (
            child_id == self.child_model_id and
            parent_id in self.parent_model_ids
        )

    class Config:
        """Pydantic model configuration."""
        
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class LineageNode(BaseModel):
    """Node in a lineage graph."""

    model_id: UUID = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    created_at: datetime = Field(..., description="Model creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Node metadata")


class LineageEdge(BaseModel):
    """Edge in a lineage graph."""

    parent_id: UUID = Field(..., description="Parent model identifier")
    child_id: UUID = Field(..., description="Child model identifier")
    relation_type: LineageRelationType = Field(..., description="Relationship type")
    transformation: LineageTransformation = Field(..., description="Transformation applied")
    created_at: datetime = Field(..., description="Relationship creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Edge metadata")


class LineageGraph(BaseModel):
    """Complete lineage graph for a model or set of models."""

    root_model_id: UUID = Field(..., description="Root model identifier")
    nodes: dict[UUID, LineageNode] = Field(..., description="Graph nodes by model ID")
    edges: list[LineageEdge] = Field(..., description="Graph edges")
    depth: int = Field(..., description="Maximum depth of the graph")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Graph creation timestamp")

    def get_parents(self, model_id: UUID) -> list[UUID]:
        """Get direct parents of a model."""
        parents = []
        for edge in self.edges:
            if edge.child_id == model_id:
                parents.append(edge.parent_id)
        return parents

    def get_children(self, model_id: UUID) -> list[UUID]:
        """Get direct children of a model."""
        children = []
        for edge in self.edges:
            if edge.parent_id == model_id:
                children.append(edge.child_id)
        return children

    def get_ancestors(self, model_id: UUID) -> set[UUID]:
        """Get all ancestors of a model."""
        ancestors = set()
        to_visit = [model_id]
        visited = set()

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)

            parents = self.get_parents(current)
            ancestors.update(parents)
            to_visit.extend(parents)

        return ancestors

    def get_descendants(self, model_id: UUID) -> set[UUID]:
        """Get all descendants of a model."""
        descendants = set()
        to_visit = [model_id]
        visited = set()

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)

            children = self.get_children(current)
            descendants.update(children)
            to_visit.extend(children)

        return descendants

    def find_path(self, from_model_id: UUID, to_model_id: UUID) -> list[UUID] | None:
        """Find path between two models in the lineage graph."""
        if from_model_id == to_model_id:
            return [from_model_id]

        queue = [(from_model_id, [from_model_id])]
        visited = set()

        while queue:
            current_id, path = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id == to_model_id:
                return path

            # Check both directions (parents and children)
            neighbors = self.get_parents(current_id) + self.get_children(current_id)
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_model_generations(self) -> dict[int, list[UUID]]:
        """Get models grouped by generation (distance from root)."""
        generations = {0: [self.root_model_id]}
        current_generation = 0
        visited = {self.root_model_id}

        while True:
            next_generation = []
            for model_id in generations[current_generation]:
                children = self.get_children(model_id)
                for child_id in children:
                    if child_id not in visited:
                        next_generation.append(child_id)
                        visited.add(child_id)

            if not next_generation:
                break

            current_generation += 1
            generations[current_generation] = next_generation

        return generations


class LineageQuery(BaseModel):
    """Query for lineage information."""

    model_id: UUID = Field(..., description="Target model identifier")
    include_ancestors: bool = Field(True, description="Include ancestor models")
    include_descendants: bool = Field(True, description="Include descendant models")
    max_depth: int = Field(10, description="Maximum depth to traverse")
    relation_types: list[LineageRelationType] | None = Field(
        None, description="Filter by relation types"
    )
    transformation_types: list[TransformationType] | None = Field(
        None, description="Filter by transformation types"
    )
    created_after: datetime | None = Field(None, description="Filter by creation date")
    created_before: datetime | None = Field(None, description="Filter by creation date")
    created_by: str | None = Field(None, description="Filter by creator")
    tags: list[str] | None = Field(None, description="Filter by tags")


class LineageStatistics(BaseModel):
    """Statistics about model lineage."""

    total_models: int = Field(..., description="Total number of models")
    total_relationships: int = Field(..., description="Total number of relationships")
    max_depth: int = Field(..., description="Maximum lineage depth")
    avg_branching_factor: float = Field(..., description="Average branching factor")
    relation_type_counts: dict[str, int] = Field(..., description="Count by relation type")
    transformation_type_counts: dict[str, int] = Field(..., description="Count by transformation type")
    orphaned_models: int = Field(..., description="Number of models without lineage")
    most_derived_model: UUID | None = Field(None, description="Model with most derivations")
    oldest_lineage: datetime | None = Field(None, description="Oldest lineage record")
    newest_lineage: datetime | None = Field(None, description="Newest lineage record")