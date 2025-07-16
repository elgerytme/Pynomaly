#!/usr/bin/env python3
"""
Comprehensive tests for ModelVersion domain entity.
Tests all model version functionality including status management, performance tracking,
metadata handling, and business logic validation.
"""

from datetime import datetime
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from monorepo.domain.entities.model_version import ModelStatus, ModelVersion
from monorepo.domain.value_objects.model_storage_info import (
    ModelStorageInfo,
    SerializationFormat,
    StorageBackend,
)
from monorepo.domain.value_objects.performance_metrics import PerformanceMetrics
from monorepo.domain.value_objects.semantic_version import SemanticVersion


class TestModelStatus:
    """Test cases for ModelStatus enum."""

    def test_all_model_statuses(self):
        """Test all model status values are defined correctly."""
        assert ModelStatus.DRAFT.value == "draft"
        assert ModelStatus.VALIDATED.value == "validated"
        assert ModelStatus.DEPLOYED.value == "deployed"
        assert ModelStatus.DEPRECATED.value == "deprecated"
        assert ModelStatus.ARCHIVED.value == "archived"

    def test_status_workflow(self):
        """Test status represents a typical workflow."""
        workflow_statuses = [
            ModelStatus.DRAFT,
            ModelStatus.VALIDATED,
            ModelStatus.DEPLOYED,
            ModelStatus.DEPRECATED,
            ModelStatus.ARCHIVED
        ]
        assert len(workflow_statuses) == 5
        assert all(isinstance(status.value, str) for status in workflow_statuses)


class TestModelVersion:
    """Test cases for ModelVersion entity."""

    def create_valid_model_version(self, **kwargs):
        """Create a valid model version for testing."""
        defaults = {
            "model_id": uuid4(),
            "version": SemanticVersion(1, 0, 0),
            "detector_id": uuid4(),
            "created_by": "test_user",
            "performance_metrics": PerformanceMetrics(
                accuracy=0.95,
                precision=0.92,
                recall=0.88,
                f1_score=0.90,
                training_time=120.5,
                inference_time=0.05,
                model_size=1024000
            ),
            "storage_info": ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/models/test_model",
                format=SerializationFormat.PICKLE,
                size_bytes=1024000,
                checksum="a" * 64  # Valid SHA-256 checksum format
            )
        }
        defaults.update(kwargs)
        return ModelVersion(**defaults)

    def test_valid_creation(self):
        """Test creating a valid model version."""
        model_id = uuid4()
        detector_id = uuid4()
        version = SemanticVersion(1, 2, 3)
        performance = PerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            training_time=300.0,
            inference_time=0.1,
            model_size=2048000
        )
        storage_info = ModelStorageInfo(
            storage_backend=StorageBackend.LOCAL_FILESYSTEM,
            storage_path="/models/test_model_v1.2.3",
            format=SerializationFormat.JOBLIB,
            size_bytes=2048000,
            checksum="b" * 64  # Valid SHA-256 checksum format
        )

        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            detector_id=detector_id,
            created_by="data_scientist",
            performance_metrics=performance,
            storage_info=storage_info,
            description="Test model version"
        )

        assert model_version.model_id == model_id
        assert model_version.version == version
        assert model_version.detector_id == detector_id
        assert model_version.created_by == "data_scientist"
        assert model_version.performance_metrics == performance
        assert model_version.storage_info == storage_info
        assert model_version.description == "Test model version"
        assert isinstance(model_version.id, UUID)
        assert isinstance(model_version.created_at, datetime)
        assert model_version.status == ModelStatus.DRAFT
        assert model_version.tags == []
        assert model_version.metadata == {}
        assert model_version.parent_version_id is None

    def test_auto_generated_fields(self):
        """Test auto-generated fields are set correctly."""
        model_version = self.create_valid_model_version()

        assert isinstance(model_version.id, UUID)
        assert isinstance(model_version.created_at, datetime)
        assert model_version.status == ModelStatus.DRAFT
        assert model_version.tags == []
        assert model_version.metadata == {}
        assert model_version.parent_version_id is None
        assert model_version.description == ""

    def test_with_optional_fields(self):
        """Test creation with optional fields."""
        model_version = self.create_valid_model_version(
            tags=["production", "validated"],
            metadata={"experiment_id": "exp_123", "notes": "Best performing model"},
            status=ModelStatus.VALIDATED,
            parent_version_id=uuid4(),
            description="Production-ready model"
        )

        assert model_version.tags == ["production", "validated"]
        assert model_version.metadata == {"experiment_id": "exp_123", "notes": "Best performing model"}
        assert model_version.status == ModelStatus.VALIDATED
        assert isinstance(model_version.parent_version_id, UUID)
        assert model_version.description == "Production-ready model"

    def test_version_type_validation(self):
        """Test version type validation."""
        with pytest.raises(TypeError, match="Version must be SemanticVersion instance"):
            ModelVersion(
                model_id=uuid4(),
                version="1.0.0",  # String instead of SemanticVersion
                detector_id=uuid4(),
                created_by="test_user",
                performance_metrics=PerformanceMetrics(
                    accuracy=0.9, precision=0.8, recall=0.85, f1_score=0.82,
                    training_time=100.0, inference_time=0.1, model_size=1000000
                ),
                storage_info=ModelStorageInfo(
                    storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                    storage_path="/test",
                    format=SerializationFormat.PICKLE,
                    size_bytes=1000,
                    checksum="c" * 64
                )
            )

    def test_performance_metrics_type_validation(self):
        """Test performance metrics type validation."""
        with pytest.raises(TypeError, match="Performance metrics must be PerformanceMetrics instance"):
            ModelVersion(
                model_id=uuid4(),
                version=SemanticVersion(1, 0, 0),
                detector_id=uuid4(),
                created_by="test_user",
                performance_metrics={"accuracy": 0.9},  # Dict instead of PerformanceMetrics
                storage_info=ModelStorageInfo(
                    storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                    storage_path="/test",
                    format=SerializationFormat.PICKLE,
                    size_bytes=1000,
                    checksum="c" * 64
                )
            )

    def test_storage_info_type_validation(self):
        """Test storage info type validation."""
        with pytest.raises(TypeError, match="Storage info must be ModelStorageInfo instance"):
            ModelVersion(
                model_id=uuid4(),
                version=SemanticVersion(1, 0, 0),
                detector_id=uuid4(),
                created_by="test_user",
                performance_metrics=PerformanceMetrics(
                    accuracy=0.9, precision=0.8, recall=0.85, f1_score=0.82,
                    training_time=100.0, inference_time=0.1, model_size=1000000
                ),
                storage_info={"path": "/test"}  # Dict instead of ModelStorageInfo
            )

    def test_created_by_validation(self):
        """Test created_by validation."""
        with pytest.raises(ValueError, match="Created by cannot be empty"):
            ModelVersion(
                model_id=uuid4(),
                version=SemanticVersion(1, 0, 0),
                detector_id=uuid4(),
                created_by="",  # Empty string
                performance_metrics=PerformanceMetrics(
                    accuracy=0.9, precision=0.8, recall=0.85, f1_score=0.82,
                    training_time=100.0, inference_time=0.1, model_size=1000000
                ),
                storage_info=ModelStorageInfo(
                    storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                    storage_path="/test",
                    format=SerializationFormat.PICKLE,
                    size_bytes=1000,
                    checksum="c" * 64
                )
            )

    def test_version_string_property(self):
        """Test version string property."""
        model_version = self.create_valid_model_version(
            version=SemanticVersion(2, 1, 3, "alpha", 1)
        )

        assert model_version.version_string == "2.1.3-alpha.1"

    def test_is_deployed_property(self):
        """Test is_deployed property."""
        model_version = self.create_valid_model_version()

        # Initially not deployed
        assert model_version.is_deployed is False

        # After setting to deployed
        model_version.status = ModelStatus.DEPLOYED
        assert model_version.is_deployed is True

        # After setting to other status
        model_version.status = ModelStatus.DEPRECATED
        assert model_version.is_deployed is False

    def test_is_deprecated_property(self):
        """Test is_deprecated property."""
        model_version = self.create_valid_model_version()

        # Initially not deprecated
        assert model_version.is_deprecated is False

        # After setting to deprecated
        model_version.status = ModelStatus.DEPRECATED
        assert model_version.is_deprecated is True

        # After setting to other status
        model_version.status = ModelStatus.VALIDATED
        assert model_version.is_deprecated is False

    def test_is_archived_property(self):
        """Test is_archived property."""
        model_version = self.create_valid_model_version()

        # Initially not archived
        assert model_version.is_archived is False

        # After setting to archived
        model_version.status = ModelStatus.ARCHIVED
        assert model_version.is_archived is True

        # After setting to other status
        model_version.status = ModelStatus.DRAFT
        assert model_version.is_archived is False

    def test_add_tag(self):
        """Test adding tags."""
        model_version = self.create_valid_model_version()

        # Add first tag
        model_version.add_tag("production")
        assert "production" in model_version.tags
        assert len(model_version.tags) == 1

        # Add second tag
        model_version.add_tag("validated")
        assert "validated" in model_version.tags
        assert len(model_version.tags) == 2

        # Add duplicate tag - should not be added
        model_version.add_tag("production")
        assert model_version.tags.count("production") == 1
        assert len(model_version.tags) == 2

    def test_remove_tag(self):
        """Test removing tags."""
        model_version = self.create_valid_model_version(
            tags=["production", "validated", "experimental"]
        )

        # Remove existing tag
        model_version.remove_tag("validated")
        assert "validated" not in model_version.tags
        assert len(model_version.tags) == 2

        # Remove non-existing tag - should not raise error
        model_version.remove_tag("nonexistent")
        assert len(model_version.tags) == 2

        # Remove another existing tag
        model_version.remove_tag("production")
        assert "production" not in model_version.tags
        assert len(model_version.tags) == 1
        assert model_version.tags == ["experimental"]

    def test_has_tag(self):
        """Test checking for tag existence."""
        model_version = self.create_valid_model_version(
            tags=["production", "validated"]
        )

        assert model_version.has_tag("production") is True
        assert model_version.has_tag("validated") is True
        assert model_version.has_tag("experimental") is False
        assert model_version.has_tag("nonexistent") is False

    def test_update_status(self):
        """Test updating model status."""
        model_version = self.create_valid_model_version()

        # Initially draft
        assert model_version.status == ModelStatus.DRAFT

        with patch('monorepo.domain.entities.model_version.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            # Update to validated
            model_version.update_status(ModelStatus.VALIDATED)
            assert model_version.status == ModelStatus.VALIDATED
            assert model_version.metadata["status_updated_at"] == mock_now.isoformat()

    def test_update_metadata(self):
        """Test updating metadata."""
        model_version = self.create_valid_model_version()

        with patch('monorepo.domain.entities.model_version.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            # Update metadata
            model_version.update_metadata("experiment_id", "exp_123")
            assert model_version.metadata["experiment_id"] == "exp_123"
            assert model_version.metadata["last_updated"] == mock_now.isoformat()

            # Update another metadata field
            model_version.update_metadata("notes", "Updated model")
            assert model_version.metadata["notes"] == "Updated model"
            assert model_version.metadata["experiment_id"] == "exp_123"  # Previous value preserved

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        performance = PerformanceMetrics(
            accuracy=0.92,
            precision=0.89,
            recall=0.94,
            f1_score=0.91,
            training_time=250.5,
            inference_time=0.08,
            model_size=1500000
        )

        model_version = self.create_valid_model_version(
            performance_metrics=performance
        )

        summary = model_version.get_performance_summary()

        expected_summary = {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91,
            "training_time": 250.5,
            "inference_time": 0.08
        }

        assert summary == expected_summary

    def test_compare_performance(self):
        """Test comparing performance with another model version."""
        # Create first model version
        performance1 = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            training_time=200.0,
            inference_time=0.05,
            model_size=1200000
        )
        model_version1 = self.create_valid_model_version(
            performance_metrics=performance1
        )

        # Create second model version
        performance2 = PerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            training_time=180.0,
            inference_time=0.08,
            model_size=1100000
        )
        model_version2 = self.create_valid_model_version(
            performance_metrics=performance2
        )

        # Compare performance
        comparison = model_version1.compare_performance(model_version2)

        expected_comparison = {
            "accuracy": 0.05,     # 0.90 - 0.85
            "precision": 0.06,    # 0.88 - 0.82
            "recall": 0.04,       # 0.92 - 0.88
            "f1_score": 0.05,     # 0.90 - 0.85
            "training_time": 20.0, # 200.0 - 180.0
            "inference_time": -0.03 # 0.05 - 0.08
        }

        for metric, expected_diff in expected_comparison.items():
            assert abs(comparison[metric] - expected_diff) < 0.001

    def test_compare_performance_type_validation(self):
        """Test compare_performance type validation."""
        model_version = self.create_valid_model_version()

        with pytest.raises(TypeError, match="Can only compare with another ModelVersion"):
            model_version.compare_performance({"accuracy": 0.8})

    def test_get_info(self):
        """Test getting comprehensive model version info."""
        model_id = uuid4()
        detector_id = uuid4()
        parent_version_id = uuid4()
        version = SemanticVersion(1, 2, 3)

        model_version = self.create_valid_model_version(
            model_id=model_id,
            version=version,
            detector_id=detector_id,
            created_by="test_user",
            tags=["production", "validated"],
            metadata={"experiment_id": "exp_123"},
            status=ModelStatus.DEPLOYED,
            parent_version_id=parent_version_id,
            description="Production model"
        )

        info = model_version.get_info()

        assert info["id"] == str(model_version.id)
        assert info["model_id"] == str(model_id)
        assert info["version"] == "1.2.3"
        assert info["detector_id"] == str(detector_id)
        assert info["created_at"] == model_version.created_at.isoformat()
        assert info["created_by"] == "test_user"
        assert info["status"] == "deployed"
        assert info["tags"] == ["production", "validated"]
        assert info["description"] == "Production model"
        assert info["parent_version_id"] == str(parent_version_id)
        assert isinstance(info["performance_metrics"], dict)
        assert isinstance(info["storage_info"], dict)
        assert info["metadata"] == {"experiment_id": "exp_123"}

    def test_get_info_without_parent_version(self):
        """Test getting info without parent version."""
        model_version = self.create_valid_model_version()

        info = model_version.get_info()

        assert info["parent_version_id"] is None

    def test_str_representation(self):
        """Test string representation."""
        model_version = self.create_valid_model_version(
            version=SemanticVersion(2, 1, 0),
            status=ModelStatus.VALIDATED
        )

        str_repr = str(model_version)

        assert "ModelVersion(v2.1.0" in str_repr
        assert "status=validated" in str_repr
        assert "accuracy=" in str_repr
        assert str(model_version.performance_metrics.accuracy) in str_repr

    def test_repr_representation(self):
        """Test developer representation."""
        model_version = self.create_valid_model_version(
            version=SemanticVersion(1, 0, 0),
            status=ModelStatus.DRAFT
        )

        repr_str = repr(model_version)

        assert "ModelVersion(" in repr_str
        assert f"id={model_version.id}" in repr_str
        assert f"model_id={model_version.model_id}" in repr_str
        assert "version=1.0.0" in repr_str
        assert "status=draft" in repr_str

    def test_tag_operations_workflow(self):
        """Test complete tag operations workflow."""
        model_version = self.create_valid_model_version()

        # Start with no tags
        assert len(model_version.tags) == 0

        # Add tags for different stages
        model_version.add_tag("experimental")
        assert model_version.has_tag("experimental") is True

        model_version.add_tag("validated")
        assert model_version.has_tag("validated") is True
        assert len(model_version.tags) == 2

        model_version.add_tag("production")
        assert model_version.has_tag("production") is True
        assert len(model_version.tags) == 3

        # Remove experimental tag when moving to production
        model_version.remove_tag("experimental")
        assert model_version.has_tag("experimental") is False
        assert model_version.has_tag("validated") is True
        assert model_version.has_tag("production") is True
        assert len(model_version.tags) == 2

    def test_status_lifecycle_workflow(self):
        """Test complete status lifecycle workflow."""
        model_version = self.create_valid_model_version()

        # Start as draft
        assert model_version.status == ModelStatus.DRAFT
        assert model_version.is_deployed is False
        assert model_version.is_deprecated is False
        assert model_version.is_archived is False

        # Move to validated
        model_version.update_status(ModelStatus.VALIDATED)
        assert model_version.status == ModelStatus.VALIDATED
        assert model_version.is_deployed is False

        # Deploy to production
        model_version.update_status(ModelStatus.DEPLOYED)
        assert model_version.status == ModelStatus.DEPLOYED
        assert model_version.is_deployed is True
        assert model_version.is_deprecated is False

        # Deprecate when new version is deployed
        model_version.update_status(ModelStatus.DEPRECATED)
        assert model_version.status == ModelStatus.DEPRECATED
        assert model_version.is_deployed is False
        assert model_version.is_deprecated is True
        assert model_version.is_archived is False

        # Archive for long-term storage
        model_version.update_status(ModelStatus.ARCHIVED)
        assert model_version.status == ModelStatus.ARCHIVED
        assert model_version.is_deployed is False
        assert model_version.is_deprecated is False
        assert model_version.is_archived is True

    def test_metadata_operations_workflow(self):
        """Test complete metadata operations workflow."""
        model_version = self.create_valid_model_version()

        # Start with empty metadata
        assert len(model_version.metadata) == 0

        # Add experiment tracking info
        model_version.update_metadata("experiment_id", "exp_001")
        model_version.update_metadata("hyperparameters", {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100
        })

        assert model_version.metadata["experiment_id"] == "exp_001"
        assert model_version.metadata["hyperparameters"]["learning_rate"] == 0.01
        assert "last_updated" in model_version.metadata

        # Add deployment info
        model_version.update_metadata("deployment_info", {
            "deployed_at": "2024-01-01T10:00:00Z",
            "deployed_by": "deployment_service",
            "environment": "production"
        })

        assert model_version.metadata["deployment_info"]["environment"] == "production"
        assert model_version.metadata["experiment_id"] == "exp_001"  # Previous data preserved

        # Add performance tracking
        model_version.update_metadata("performance_history", [
            {"timestamp": "2024-01-01T11:00:00Z", "accuracy": 0.95},
            {"timestamp": "2024-01-01T12:00:00Z", "accuracy": 0.94}
        ])

        assert len(model_version.metadata["performance_history"]) == 2
        assert model_version.metadata["performance_history"][0]["accuracy"] == 0.95


class TestModelVersionIntegration:
    """Test cases for integration scenarios."""

    def create_valid_model_version(self, **kwargs):
        """Create a valid model version for testing."""
        defaults = {
            "model_id": uuid4(),
            "version": SemanticVersion(1, 0, 0),
            "detector_id": uuid4(),
            "created_by": "test_user",
            "performance_metrics": PerformanceMetrics(
                accuracy=0.95,
                precision=0.92,
                recall=0.88,
                f1_score=0.90,
                training_time=120.5,
                inference_time=0.05,
                model_size=1024000
            ),
            "storage_info": ModelStorageInfo(
                storage_backend=StorageBackend.LOCAL_FILESYSTEM,
                storage_path="/models/test_model",
                format=SerializationFormat.PICKLE,
                size_bytes=1024000,
                checksum="a" * 64  # Valid SHA-256 checksum format
            )
        }
        defaults.update(kwargs)
        return ModelVersion(**defaults)

    def test_model_version_lineage(self):
        """Test model version lineage tracking."""
        # Create parent version
        parent_version = self.create_valid_model_version(
            version=SemanticVersion(1, 0, 0),
            status=ModelStatus.DEPRECATED
        )

        # Create child version derived from parent
        child_version = self.create_valid_model_version(
            version=SemanticVersion(1, 1, 0),
            parent_version_id=parent_version.id,
            status=ModelStatus.DEPLOYED
        )

        # Verify lineage
        assert child_version.parent_version_id == parent_version.id
        assert parent_version.is_deprecated is True
        assert child_version.is_deployed is True

        # Compare performance between versions
        comparison = child_version.compare_performance(parent_version)
        assert isinstance(comparison, dict)
        assert "accuracy" in comparison
        assert "precision" in comparison
        assert "recall" in comparison
        assert "f1_score" in comparison

    def test_model_deployment_workflow(self):
        """Test complete model deployment workflow."""
        # Create model version
        model_version = self.create_valid_model_version(
            version=SemanticVersion(2, 0, 0),
            created_by="data_scientist",
            description="New model with improved accuracy"
        )

        # Initial state
        assert model_version.status == ModelStatus.DRAFT
        assert len(model_version.tags) == 0

        # Add experimental tag and validate
        model_version.add_tag("experimental")
        model_version.update_status(ModelStatus.VALIDATED)
        model_version.update_metadata("validation_results", {
            "cross_validation_score": 0.92,
            "test_accuracy": 0.94,
            "validated_by": "validation_service"
        })

        assert model_version.status == ModelStatus.VALIDATED
        assert model_version.has_tag("experimental") is True
        assert model_version.metadata["validation_results"]["cross_validation_score"] == 0.92

        # Move to production
        model_version.remove_tag("experimental")
        model_version.add_tag("production")
        model_version.update_status(ModelStatus.DEPLOYED)
        model_version.update_metadata("deployment_config", {
            "replicas": 3,
            "cpu_limit": "2000m",
            "memory_limit": "4Gi"
        })

        assert model_version.status == ModelStatus.DEPLOYED
        assert model_version.is_deployed is True
        assert model_version.has_tag("production") is True
        assert model_version.has_tag("experimental") is False
        assert model_version.metadata["deployment_config"]["replicas"] == 3

        # Get deployment info
        info = model_version.get_info()
        assert info["status"] == "deployed"
        assert "production" in info["tags"]
        assert "deployment_config" in info["metadata"]
        assert "validation_results" in info["metadata"]

    def test_performance_comparison_scenario(self):
        """Test performance comparison scenario."""
        # Create baseline model
        baseline_performance = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1_score=0.85,
            training_time=300.0,
            inference_time=0.1,
            model_size=1800000
        )

        baseline_model = self.create_valid_model_version(
            version=SemanticVersion(1, 0, 0),
            performance_metrics=baseline_performance,
            status=ModelStatus.DEPRECATED
        )
        baseline_model.add_tag("baseline")

        # Create improved model
        improved_performance = PerformanceMetrics(
            accuracy=0.92,
            precision=0.88,
            recall=0.94,
            f1_score=0.91,
            training_time=250.0,
            inference_time=0.08,
            model_size=1600000
        )

        improved_model = self.create_valid_model_version(
            version=SemanticVersion(2, 0, 0),
            performance_metrics=improved_performance,
            parent_version_id=baseline_model.id,
            status=ModelStatus.DEPLOYED
        )
        improved_model.add_tag("production")

        # Compare performance
        comparison = improved_model.compare_performance(baseline_model)

        # Verify improvements
        assert comparison["accuracy"] > 0  # Improved accuracy
        assert comparison["precision"] > 0  # Improved precision
        assert comparison["recall"] > 0  # Improved recall
        assert comparison["f1_score"] > 0  # Improved F1 score
        assert comparison["training_time"] < 0  # Faster training
        assert comparison["inference_time"] < 0  # Faster inference

        # Check specific improvements
        assert abs(comparison["accuracy"] - 0.07) < 0.01  # 0.92 - 0.85
        assert abs(comparison["f1_score"] - 0.06) < 0.01  # 0.91 - 0.85
        assert abs(comparison["training_time"] - (-50.0)) < 0.01  # 250.0 - 300.0
        assert abs(comparison["inference_time"] - (-0.02)) < 0.01  # 0.08 - 0.1

    def test_model_archival_workflow(self):
        """Test model archival workflow."""
        # Create old production model
        old_model = self.create_valid_model_version(
            version=SemanticVersion(1, 0, 0),
            status=ModelStatus.DEPLOYED
        )
        old_model.add_tag("production")
        old_model.update_metadata("deployed_at", "2023-01-01T00:00:00Z")

        # Create new model to replace it
        new_model = self.create_valid_model_version(
            version=SemanticVersion(2, 0, 0),
            parent_version_id=old_model.id,
            status=ModelStatus.VALIDATED
        )
        new_model.add_tag("candidate")

        # Deploy new model
        new_model.update_status(ModelStatus.DEPLOYED)
        new_model.remove_tag("candidate")
        new_model.add_tag("production")

        # Deprecate old model
        old_model.update_status(ModelStatus.DEPRECATED)
        old_model.remove_tag("production")
        old_model.add_tag("deprecated")

        # Archive old model after some time
        old_model.update_status(ModelStatus.ARCHIVED)
        old_model.remove_tag("deprecated")
        old_model.add_tag("archived")
        old_model.update_metadata("archived_at", "2024-01-01T00:00:00Z")
        old_model.update_metadata("archive_reason", "Replaced by newer version")

        # Verify final states
        assert old_model.is_archived is True
        assert old_model.is_deployed is False
        assert old_model.is_deprecated is False
        assert old_model.has_tag("archived") is True
        assert old_model.has_tag("production") is False
        assert old_model.metadata["archive_reason"] == "Replaced by newer version"

        assert new_model.is_deployed is True
        assert new_model.has_tag("production") is True
        assert new_model.parent_version_id == old_model.id

    def test_comprehensive_model_info(self):
        """Test comprehensive model information retrieval."""
        # Create a fully configured model version
        model_version = self.create_valid_model_version(
            version=SemanticVersion(3, 1, 4, "rc", 2),
            created_by="senior_data_scientist",
            description="Advanced model with feature engineering improvements",
            tags=["production", "validated", "high_performance"],
            status=ModelStatus.DEPLOYED,
            metadata={
                "experiment_id": "exp_456",
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "epochs": 200,
                    "regularization": 0.01
                },
                "feature_engineering": {
                    "pca_components": 50,
                    "scaling_method": "standard",
                    "feature_selection": "mutual_info"
                },
                "validation_metrics": {
                    "cross_val_score": 0.93,
                    "test_score": 0.94,
                    "validation_date": "2024-01-15"
                }
            }
        )

        # Get comprehensive info
        info = model_version.get_info()

        # Verify all information is present
        assert info["version"] == "3.1.4-rc.2"
        assert info["created_by"] == "senior_data_scientist"
        assert info["description"] == "Advanced model with feature engineering improvements"
        assert info["status"] == "deployed"
        assert len(info["tags"]) == 3
        assert "production" in info["tags"]
        assert "validated" in info["tags"]
        assert "high_performance" in info["tags"]

        # Verify metadata structure
        assert info["metadata"]["experiment_id"] == "exp_456"
        assert info["metadata"]["hyperparameters"]["learning_rate"] == 0.001
        assert info["metadata"]["feature_engineering"]["pca_components"] == 50
        assert info["metadata"]["validation_metrics"]["cross_val_score"] == 0.93

        # Verify performance metrics are included
        assert "performance_metrics" in info
        assert isinstance(info["performance_metrics"], dict)
        assert "accuracy" in info["performance_metrics"]

        # Verify storage info is included
        assert "storage_info" in info
        assert isinstance(info["storage_info"], dict)
        assert "path" in info["storage_info"]
        assert "format" in info["storage_info"]

        # Test string representations
        str_repr = str(model_version)
        assert "3.1.4-rc.2" in str_repr
        assert "deployed" in str_repr

        repr_str = repr(model_version)
        assert "3.1.4-rc.2" in repr_str
        assert "deployed" in repr_str
        assert str(model_version.id) in repr_str

    def test_model_version_equality_and_comparison(self):
        """Test model version equality and comparison scenarios."""
        # Create two identical model versions
        model_version1 = self.create_valid_model_version(
            version=SemanticVersion(1, 0, 0)
        )
        model_version2 = self.create_valid_model_version(
            version=SemanticVersion(1, 0, 0)
        )

        # They should have different IDs
        assert model_version1.id != model_version2.id

        # But same version strings
        assert model_version1.version_string == model_version2.version_string

        # Performance comparison should return zero differences
        comparison = model_version1.compare_performance(model_version2)
        for metric, diff in comparison.items():
            assert abs(diff) < 0.001  # Should be approximately zero

    def test_error_handling_scenarios(self):
        """Test various error handling scenarios."""
        model_version = self.create_valid_model_version()

        # Test invalid performance comparison
        with pytest.raises(TypeError):
            model_version.compare_performance("not_a_model_version")

        # Test that operations still work after errors
        model_version.add_tag("test")
        assert model_version.has_tag("test") is True

        model_version.update_metadata("test_key", "test_value")
        assert model_version.metadata["test_key"] == "test_value"

        # Test that status updates work properly
        model_version.update_status(ModelStatus.VALIDATED)
        assert model_version.status == ModelStatus.VALIDATED
