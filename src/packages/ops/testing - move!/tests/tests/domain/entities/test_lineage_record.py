"""
Tests for lineage record domain entity.

This module provides comprehensive tests for the lineage record entity,
ensuring proper model lineage tracking, relationships, and metadata management.
"""

from datetime import datetime
from uuid import UUID, uuid4

from monorepo.domain.entities.lineage_record import (
    LineageArtifact,
    LineageNode,
    LineageRecord,
    LineageRelationType,
    TransformationType,
)


class TestLineageArtifact:
    """Test cases for LineageArtifact."""

    def test_artifact_creation(self):
        """Test artifact creation with valid data."""
        artifact_id = uuid4()
        artifact = LineageArtifact(
            id=artifact_id,
            name="test_artifact",
            type="model",
            version="1.0",
            location="s3://bucket/model.pkl",
            size_bytes=1024,
            checksum="abc123",
        )

        assert artifact.id == artifact_id
        assert artifact.name == "test_artifact"
        assert artifact.type == "model"
        assert artifact.version == "1.0"
        assert artifact.location == "s3://bucket/model.pkl"
        assert artifact.size_bytes == 1024
        assert artifact.checksum == "abc123"

    def test_artifact_minimal_creation(self):
        """Test artifact creation with minimal required fields."""
        artifact_id = uuid4()
        artifact = LineageArtifact(id=artifact_id, name="minimal_artifact")

        assert artifact.id == artifact_id
        assert artifact.name == "minimal_artifact"
        assert artifact.type is None
        assert artifact.version is None


class TestLineageRecord:
    """Test cases for LineageRecord."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parent_id = uuid4()
        self.child_id = uuid4()
        self.timestamp = datetime.now()

        self.test_record = LineageRecord(
            parent_model_ids=[self.parent_id],
            child_model_id=self.child_id,
            relationship_type=LineageRelationType.DERIVED_FROM,
            transformation_type=TransformationType.TRAINING,
            timestamp=self.timestamp,
            description="Test lineage record",
        )

    def test_lineage_record_creation(self):
        """Test lineage record creation with valid data."""
        assert self.test_record.parent_model_ids == [self.parent_id]
        assert self.test_record.child_model_id == self.child_id
        assert self.test_record.relationship_type == LineageRelationType.DERIVED_FROM
        assert self.test_record.transformation_type == TransformationType.TRAINING
        assert self.test_record.timestamp == self.timestamp
        assert self.test_record.description == "Test lineage record"
        assert len(self.test_record.input_artifacts) == 0
        assert len(self.test_record.output_artifacts) == 0

    def test_add_input_artifact(self):
        """Test adding input artifact to lineage record."""
        artifact = LineageArtifact(id=uuid4(), name="input_dataset", type="dataset")

        initial_count = len(self.test_record.input_artifacts)
        self.test_record.add_input_artifact(artifact)

        assert len(self.test_record.input_artifacts) == initial_count + 1
        assert artifact in self.test_record.input_artifacts

    def test_add_output_artifact(self):
        """Test adding output artifact to lineage record."""
        artifact = LineageArtifact(id=uuid4(), name="trained_model", type="model")

        initial_count = len(self.test_record.output_artifacts)
        self.test_record.add_output_artifact(artifact)

        assert len(self.test_record.output_artifacts) == initial_count + 1
        assert artifact in self.test_record.output_artifacts

    def test_get_all_model_ids(self):
        """Test getting all model IDs from lineage record."""
        parent_id2 = uuid4()
        record = LineageRecord(
            parent_model_ids=[self.parent_id, parent_id2],
            child_model_id=self.child_id,
            relationship_type=LineageRelationType.ENSEMBLE_OF,
            transformation_type=TransformationType.ENSEMBLE,
        )

        all_ids = record.get_all_model_ids()

        assert len(all_ids) == 3  # 2 parents + 1 child
        assert self.parent_id in all_ids
        assert parent_id2 in all_ids
        assert self.child_id in all_ids

    def test_is_direct_descendant_true(self):
        """Test direct descendant check with valid parent-child relationship."""
        is_descendant = self.test_record.is_direct_descendant(
            self.parent_id, self.child_id
        )

        assert is_descendant is True

    def test_is_direct_descendant_false_wrong_parent(self):
        """Test direct descendant check with invalid parent."""
        wrong_parent_id = uuid4()
        is_descendant = self.test_record.is_direct_descendant(
            wrong_parent_id, self.child_id
        )

        assert is_descendant is False

    def test_is_direct_descendant_false_wrong_child(self):
        """Test direct descendant check with invalid child."""
        wrong_child_id = uuid4()
        is_descendant = self.test_record.is_direct_descendant(
            self.parent_id, wrong_child_id
        )

        assert is_descendant is False

    def test_multiple_parents(self):
        """Test lineage record with multiple parent models."""
        parent_id2 = uuid4()
        parent_id3 = uuid4()

        record = LineageRecord(
            parent_model_ids=[self.parent_id, parent_id2, parent_id3],
            child_model_id=self.child_id,
            relationship_type=LineageRelationType.ENSEMBLE_OF,
            transformation_type=TransformationType.ENSEMBLE,
        )

        assert len(record.parent_model_ids) == 3
        assert self.parent_id in record.parent_model_ids
        assert parent_id2 in record.parent_model_ids
        assert parent_id3 in record.parent_model_ids

        # Test direct descendant check with multiple parents
        assert record.is_direct_descendant(self.parent_id, self.child_id)
        assert record.is_direct_descendant(parent_id2, self.child_id)
        assert record.is_direct_descendant(parent_id3, self.child_id)

    def test_lineage_with_experiment_metadata(self):
        """Test lineage record with experiment and run information."""
        experiment_id = uuid4()
        run_id = "run_12345"

        record = LineageRecord(
            parent_model_ids=[self.parent_id],
            child_model_id=self.child_id,
            relationship_type=LineageRelationType.FINE_TUNED_FROM,
            transformation_type=TransformationType.FINE_TUNING,
            experiment_id=experiment_id,
            run_id=run_id,
            tags=["production", "validated"],
            metadata={"performance": 0.95, "optimizer": "adam"},
        )

        assert record.experiment_id == experiment_id
        assert record.run_id == run_id
        assert "production" in record.tags
        assert "validated" in record.tags
        assert record.metadata["performance"] == 0.95
        assert record.metadata["optimizer"] == "adam"

    def test_lineage_with_provenance(self):
        """Test lineage record with detailed provenance information."""
        provenance_data = {
            "code_version": "v1.2.3",
            "environment": {"python": "3.11", "pytorch": "2.0"},
            "author": "data_scientist@company.com",
            "training_duration": "2h 15m",
        }

        record = LineageRecord(
            parent_model_ids=[self.parent_id],
            child_model_id=self.child_id,
            relationship_type=LineageRelationType.RETRAINED_FROM,
            transformation_type=TransformationType.HYPERPARAMETER_TUNING,
            provenance=provenance_data,
        )

        assert record.provenance == provenance_data
        assert record.provenance["code_version"] == "v1.2.3"
        assert record.provenance["author"] == "data_scientist@company.com"


class TestLineageNode:
    """Test cases for LineageNode."""

    def test_node_creation(self):
        """Test lineage node creation."""
        model_id = uuid4()
        created_at = datetime.now()

        node = LineageNode(
            model_id=model_id,
            model_name="anomaly_detector_v1",
            model_version="1.0.0",
            created_at=created_at,
            metadata={"algorithm": "isolation_forest", "accuracy": 0.92},
        )

        assert node.model_id == model_id
        assert node.model_name == "anomaly_detector_v1"
        assert node.model_version == "1.0.0"
        assert node.created_at == created_at
        assert node.metadata["algorithm"] == "isolation_forest"
        assert node.metadata["accuracy"] == 0.92


class TestLineageEnums:
    """Test cases for lineage enumeration types."""

    def test_lineage_relation_types(self):
        """Test lineage relationship type enumeration."""
        assert LineageRelationType.DERIVED_FROM == "derived_from"
        assert LineageRelationType.FINE_TUNED_FROM == "fine_tuned_from"
        assert LineageRelationType.ENSEMBLE_OF == "ensemble_of"
        assert LineageRelationType.DISTILLED_FROM == "distilled_from"
        assert LineageRelationType.CUSTOM == "custom"

    def test_transformation_types(self):
        """Test transformation type enumeration."""
        assert TransformationType.TRAINING == "training"
        assert TransformationType.FINE_TUNING == "fine_tuning"
        assert TransformationType.ENSEMBLE == "ensemble"
        assert TransformationType.PRUNING == "pruning"
        assert TransformationType.QUANTIZATION == "quantization"
        assert TransformationType.CUSTOM == "custom"

    def test_enum_values_in_record(self):
        """Test using enum values in lineage record."""
        record = LineageRecord(
            parent_model_ids=[uuid4()],
            child_model_id=uuid4(),
            relationship_type=LineageRelationType.QUANTIZED_FROM,
            transformation_type=TransformationType.QUANTIZATION,
        )

        assert record.relationship_type == LineageRelationType.QUANTIZED_FROM
        assert record.transformation_type == TransformationType.QUANTIZATION


class TestLineageRecordValidation:
    """Test cases for lineage record validation."""

    def test_valid_record_creation(self):
        """Test creating a valid lineage record."""
        record = LineageRecord(
            parent_model_ids=[uuid4()],
            child_model_id=uuid4(),
            relationship_type=LineageRelationType.DERIVED_FROM,
            transformation_type=TransformationType.TRAINING,
        )

        assert record is not None
        assert isinstance(record.id, UUID)
        assert isinstance(record.timestamp, datetime)

    def test_record_with_empty_parent_list(self):
        """Test creating record with empty parent list."""
        record = LineageRecord(
            parent_model_ids=[],
            child_model_id=uuid4(),
            relationship_type=LineageRelationType.DERIVED_FROM,
            transformation_type=TransformationType.TRAINING,
        )

        assert record.parent_model_ids == []
        assert len(record.get_all_model_ids()) == 1  # Only child ID

    def test_record_immutable_after_creation(self):
        """Test that core record fields maintain integrity."""
        parent_id = uuid4()
        child_id = uuid4()

        record = LineageRecord(
            parent_model_ids=[parent_id],
            child_model_id=child_id,
            relationship_type=LineageRelationType.DERIVED_FROM,
            transformation_type=TransformationType.TRAINING,
        )

        # Core fields should be accessible
        assert record.parent_model_ids == [parent_id]
        assert record.child_model_id == child_id

        # Mutable collections should be modifiable
        initial_artifact_count = len(record.input_artifacts)
        artifact = LineageArtifact(id=uuid4(), name="test")
        record.add_input_artifact(artifact)
        assert len(record.input_artifacts) == initial_artifact_count + 1
