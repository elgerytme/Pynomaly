"""Comprehensive tests for Model domain entity."""

from datetime import datetime
from uuid import uuid4

import pytest

from monorepo.domain.entities.model import Model, ModelStage, ModelType


class TestModelInitialization:
    """Test model initialization and validation."""

    def test_model_initialization_with_required_fields(self):
        """Test model initialization with required fields only."""
        model = Model(
            name="Fraud Detection Model",
            description="Detects fraudulent transactions",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="data_scientist@company.com",
        )

        assert model.name == "Fraud Detection Model"
        assert model.description == "Detects fraudulent transactions"
        assert model.model_type == ModelType.SUPERVISED
        assert model.algorithm_family == "isolation_forest"
        assert model.created_by == "data_scientist@company.com"
        assert isinstance(model.id, type(uuid4()))
        assert isinstance(model.created_at, datetime)
        assert model.team == ""
        assert model.tags == []
        assert model.stage == ModelStage.DEVELOPMENT
        assert model.current_version_id is None
        assert model.latest_version_id is None
        assert model.metadata == {}
        assert model.use_cases == []
        assert model.data_requirements == {}

    def test_model_initialization_with_all_fields(self):
        """Test model initialization with all fields."""
        model_id = uuid4()
        current_version_id = uuid4()
        latest_version_id = uuid4()
        created_at = datetime(2023, 1, 1, 12, 0, 0)

        model = Model(
            id=model_id,
            name="Advanced Anomaly Detector",
            description="Advanced ML model for anomaly detection",
            model_type=ModelType.ENSEMBLE,
            algorithm_family="deep_learning",
            created_by="ml_engineer@company.com",
            created_at=created_at,
            team="Data Science Team",
            tags=["production", "critical"],
            stage=ModelStage.PRODUCTION,
            current_version_id=current_version_id,
            latest_version_id=latest_version_id,
            metadata={"version_count": 5, "last_updated": "2023-01-01"},
            use_cases=["fraud_detection", "system_monitoring"],
            data_requirements={"features": ["amount", "location"], "format": "json"},
        )

        assert model.id == model_id
        assert model.name == "Advanced Anomaly Detector"
        assert model.description == "Advanced ML model for anomaly detection"
        assert model.model_type == ModelType.ENSEMBLE
        assert model.algorithm_family == "deep_learning"
        assert model.created_by == "ml_engineer@company.com"
        assert model.created_at == created_at
        assert model.team == "Data Science Team"
        assert model.tags == ["production", "critical"]
        assert model.stage == ModelStage.PRODUCTION
        assert model.current_version_id == current_version_id
        assert model.latest_version_id == latest_version_id
        assert model.metadata == {"version_count": 5, "last_updated": "2023-01-01"}
        assert model.use_cases == ["fraud_detection", "system_monitoring"]
        assert model.data_requirements == {
            "features": ["amount", "location"],
            "format": "json",
        }

    def test_model_validation_empty_name(self):
        """Test model validation with empty name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            Model(
                name="",
                description="Test description",
                model_type=ModelType.SUPERVISED,
                algorithm_family="isolation_forest",
                created_by="test@company.com",
            )

    def test_model_validation_empty_description(self):
        """Test model validation with empty description."""
        with pytest.raises(ValueError, match="Model description cannot be empty"):
            Model(
                name="Test Model",
                description="",
                model_type=ModelType.SUPERVISED,
                algorithm_family="isolation_forest",
                created_by="test@company.com",
            )

    def test_model_validation_invalid_model_type(self):
        """Test model validation with invalid model type."""
        with pytest.raises(TypeError, match="Model type must be ModelType"):
            Model(
                name="Test Model",
                description="Test description",
                model_type="invalid_type",  # Should be ModelType enum
                algorithm_family="isolation_forest",
                created_by="test@company.com",
            )

    def test_model_validation_invalid_model_stage(self):
        """Test model validation with invalid model stage."""
        with pytest.raises(TypeError, match="Model stage must be ModelStage"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=ModelType.SUPERVISED,
                algorithm_family="isolation_forest",
                created_by="test@company.com",
                stage="invalid_stage",  # Should be ModelStage enum
            )

    def test_model_validation_empty_created_by(self):
        """Test model validation with empty created_by."""
        with pytest.raises(ValueError, match="Created by cannot be empty"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=ModelType.SUPERVISED,
                algorithm_family="isolation_forest",
                created_by="",
            )

    def test_model_validation_empty_algorithm_family(self):
        """Test model validation with empty algorithm family."""
        with pytest.raises(ValueError, match="Algorithm family cannot be empty"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=ModelType.SUPERVISED,
                algorithm_family="",
                created_by="test@company.com",
            )


class TestModelProperties:
    """Test model properties."""

    def test_is_in_production_true(self):
        """Test is_in_production property when true."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.PRODUCTION,
        )

        assert model.is_in_production is True

    def test_is_in_production_false(self):
        """Test is_in_production property when false."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.DEVELOPMENT,
        )

        assert model.is_in_production is False

    def test_is_archived_true(self):
        """Test is_archived property when true."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.ARCHIVED,
        )

        assert model.is_archived is True

    def test_is_archived_false(self):
        """Test is_archived property when false."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.PRODUCTION,
        )

        assert model.is_archived is False

    def test_has_current_version_true(self):
        """Test has_current_version property when true."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            current_version_id=uuid4(),
        )

        assert model.has_current_version is True

    def test_has_current_version_false(self):
        """Test has_current_version property when false."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        assert model.has_current_version is False

    def test_version_count_property(self):
        """Test version_count property."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            metadata={"version_count": 3},
        )

        assert model.version_count == 3

    def test_version_count_property_default(self):
        """Test version_count property with default value."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        assert model.version_count == 0


class TestModelTagManagement:
    """Test model tag management methods."""

    def test_add_tag(self):
        """Test adding tags to model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        model.add_tag("production")
        model.add_tag("critical")

        assert "production" in model.tags
        assert "critical" in model.tags
        assert len(model.tags) == 2

    def test_add_duplicate_tag(self):
        """Test adding duplicate tags."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            tags=["production"],
        )

        model.add_tag("production")  # Duplicate

        assert model.tags == ["production"]  # No duplicate added

    def test_add_empty_tag(self):
        """Test adding empty tag."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        model.add_tag("")  # Empty tag
        model.add_tag(None)  # None tag

        assert model.tags == []

    def test_remove_tag(self):
        """Test removing tags from model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            tags=["production", "critical", "experimental"],
        )

        model.remove_tag("critical")

        assert "critical" not in model.tags
        assert "production" in model.tags
        assert "experimental" in model.tags
        assert len(model.tags) == 2

    def test_remove_nonexistent_tag(self):
        """Test removing non-existent tag."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            tags=["production"],
        )

        model.remove_tag("nonexistent")

        assert model.tags == ["production"]

    def test_has_tag(self):
        """Test checking if model has a specific tag."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            tags=["production", "critical"],
        )

        assert model.has_tag("production") is True
        assert model.has_tag("critical") is True
        assert model.has_tag("experimental") is False


class TestModelStageManagement:
    """Test model stage management."""

    def test_update_stage(self):
        """Test updating model stage."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.DEVELOPMENT,
        )

        model.update_stage(ModelStage.STAGING)

        assert model.stage == ModelStage.STAGING
        assert "stage_transitions" in model.metadata
        assert len(model.metadata["stage_transitions"]) == 1

        transition = model.metadata["stage_transitions"][0]
        assert transition["from"] == "development"
        assert transition["to"] == "staging"
        assert "timestamp" in transition

    def test_update_stage_multiple_transitions(self):
        """Test multiple stage transitions."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.DEVELOPMENT,
        )

        model.update_stage(ModelStage.STAGING)
        model.update_stage(ModelStage.PRODUCTION)

        assert model.stage == ModelStage.PRODUCTION
        assert len(model.metadata["stage_transitions"]) == 2

        # Check first transition
        transition1 = model.metadata["stage_transitions"][0]
        assert transition1["from"] == "development"
        assert transition1["to"] == "staging"

        # Check second transition
        transition2 = model.metadata["stage_transitions"][1]
        assert transition2["from"] == "staging"
        assert transition2["to"] == "production"

    def test_promote_to_production(self):
        """Test promoting model to production."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.STAGING,
        )

        version_id = uuid4()
        model.promote_to_production(version_id, "ml_engineer@company.com")

        assert model.stage == ModelStage.PRODUCTION
        assert model.current_version_id == version_id
        assert "promoted_at" in model.metadata
        assert model.metadata["promoted_by"] == "ml_engineer@company.com"
        assert model.metadata["production_version_id"] == str(version_id)

    def test_get_stage_history(self):
        """Test getting stage history."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        # No transitions yet
        history = model.get_stage_history()
        assert history == []

        # Add some transitions
        model.update_stage(ModelStage.STAGING)
        model.update_stage(ModelStage.PRODUCTION)

        history = model.get_stage_history()
        assert len(history) == 2
        assert history[0]["from"] == "development"
        assert history[0]["to"] == "staging"
        assert history[1]["from"] == "staging"
        assert history[1]["to"] == "production"


class TestModelVersionManagement:
    """Test model version management."""

    def test_set_current_version(self):
        """Test setting current version."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        version_id = uuid4()
        model.set_current_version(version_id)

        assert model.current_version_id == version_id
        assert "current_version_updated" in model.metadata

    def test_set_latest_version(self):
        """Test setting latest version."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        version_id = uuid4()
        model.set_latest_version(version_id)

        assert model.latest_version_id == version_id
        assert "latest_version_updated" in model.metadata
        assert model.metadata["version_count"] == 1

    def test_set_latest_version_increments_count(self):
        """Test that setting latest version increments version count."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            metadata={"version_count": 2},
        )

        version_id = uuid4()
        model.set_latest_version(version_id)

        assert model.metadata["version_count"] == 3


class TestModelUseCaseManagement:
    """Test model use case management."""

    def test_add_use_case(self):
        """Test adding use cases to model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        model.add_use_case("fraud_detection")
        model.add_use_case("system_monitoring")

        assert "fraud_detection" in model.use_cases
        assert "system_monitoring" in model.use_cases
        assert len(model.use_cases) == 2

    def test_add_duplicate_use_case(self):
        """Test adding duplicate use cases."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            use_cases=["fraud_detection"],
        )

        model.add_use_case("fraud_detection")  # Duplicate

        assert model.use_cases == ["fraud_detection"]  # No duplicate added

    def test_add_empty_use_case(self):
        """Test adding empty use case."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        model.add_use_case("")  # Empty use case
        model.add_use_case(None)  # None use case

        assert model.use_cases == []

    def test_remove_use_case(self):
        """Test removing use cases from model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            use_cases=["fraud_detection", "system_monitoring", "quality_control"],
        )

        model.remove_use_case("system_monitoring")

        assert "system_monitoring" not in model.use_cases
        assert "fraud_detection" in model.use_cases
        assert "quality_control" in model.use_cases
        assert len(model.use_cases) == 2

    def test_remove_nonexistent_use_case(self):
        """Test removing non-existent use case."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            use_cases=["fraud_detection"],
        )

        model.remove_use_case("nonexistent")

        assert model.use_cases == ["fraud_detection"]


class TestModelDataRequirements:
    """Test model data requirements management."""

    def test_update_data_requirements(self):
        """Test updating data requirements."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        requirements = {
            "features": ["amount", "location", "timestamp"],
            "format": "json",
            "frequency": "real-time",
        }

        model.update_data_requirements(requirements)

        assert model.data_requirements == requirements
        assert "data_requirements_updated" in model.metadata

    def test_update_data_requirements_partial(self):
        """Test partial update of data requirements."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            data_requirements={"features": ["amount"], "format": "json"},
        )

        model.update_data_requirements({"frequency": "real-time"})

        assert model.data_requirements["features"] == ["amount"]
        assert model.data_requirements["format"] == "json"
        assert model.data_requirements["frequency"] == "real-time"


class TestModelMetadata:
    """Test model metadata management."""

    def test_update_metadata(self):
        """Test updating metadata."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        model.update_metadata("business_impact", "high", "manager@company.com")

        assert model.metadata["business_impact"] == "high"
        assert "last_updated" in model.metadata
        assert model.metadata["last_updated_by"] == "manager@company.com"

    def test_update_metadata_without_user(self):
        """Test updating metadata without specifying user."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        model.update_metadata("performance_baseline", 0.95)

        assert model.metadata["performance_baseline"] == 0.95
        assert "last_updated" in model.metadata
        assert "last_updated_by" not in model.metadata


class TestModelInfo:
    """Test model information methods."""

    def test_get_info(self):
        """Test getting comprehensive model information."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.ENSEMBLE,
            algorithm_family="random_forest",
            created_by="test@company.com",
            team="Data Science",
            tags=["production", "critical"],
            stage=ModelStage.PRODUCTION,
            current_version_id=uuid4(),
            latest_version_id=uuid4(),
            metadata={"version_count": 3, "business_impact": "high"},
            use_cases=["fraud_detection"],
            data_requirements={"features": ["amount"], "format": "json"},
        )

        info = model.get_info()

        assert info["name"] == "Test Model"
        assert info["description"] == "Test description"
        assert info["model_type"] == "ensemble"
        assert info["algorithm_family"] == "random_forest"
        assert info["created_by"] == "test@company.com"
        assert info["team"] == "Data Science"
        assert info["stage"] == "production"
        assert info["is_in_production"] is True
        assert info["has_current_version"] is True
        assert info["version_count"] == 3
        assert info["tags"] == ["production", "critical"]
        assert info["use_cases"] == ["fraud_detection"]
        assert info["data_requirements"] == {"features": ["amount"], "format": "json"}
        assert info["metadata"] == {"version_count": 3, "business_impact": "high"}
        assert "id" in info
        assert "created_at" in info
        assert "current_version_id" in info
        assert "latest_version_id" in info

    def test_get_info_minimal(self):
        """Test getting info with minimal model."""
        model = Model(
            name="Minimal Model",
            description="Minimal description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
        )

        info = model.get_info()

        assert info["name"] == "Minimal Model"
        assert info["current_version_id"] is None
        assert info["latest_version_id"] is None
        assert info["has_current_version"] is False
        assert info["is_in_production"] is False
        assert info["version_count"] == 0
        assert info["tags"] == []
        assert info["use_cases"] == []
        assert info["data_requirements"] == {}
        assert info["metadata"] == {}


class TestModelValidation:
    """Test model validation and deployment readiness."""

    def test_can_deploy_success(self):
        """Test successful deployment validation."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            current_version_id=uuid4(),
            use_cases=["fraud_detection"],
            data_requirements={"features": ["amount"], "format": "json"},
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": 0.95,
            },
        )

        can_deploy, issues = model.can_deploy()

        assert can_deploy is True
        assert issues == []

    def test_can_deploy_no_current_version(self):
        """Test deployment validation fails without current version."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            use_cases=["fraud_detection"],
            data_requirements={"features": ["amount"], "format": "json"},
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": 0.95,
            },
        )

        can_deploy, issues = model.can_deploy()

        assert can_deploy is False
        assert "No current version set" in issues

    def test_can_deploy_no_use_cases(self):
        """Test deployment validation fails without use cases."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            current_version_id=uuid4(),
            data_requirements={"features": ["amount"], "format": "json"},
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": 0.95,
            },
        )

        can_deploy, issues = model.can_deploy()

        assert can_deploy is False
        assert "No use cases defined" in issues

    def test_can_deploy_no_data_requirements(self):
        """Test deployment validation fails without data requirements."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            current_version_id=uuid4(),
            use_cases=["fraud_detection"],
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": 0.95,
            },
        )

        can_deploy, issues = model.can_deploy()

        assert can_deploy is False
        assert "No data requirements specified" in issues

    def test_can_deploy_archived_model(self):
        """Test deployment validation fails for archived model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.ARCHIVED,
            current_version_id=uuid4(),
            use_cases=["fraud_detection"],
            data_requirements={"features": ["amount"], "format": "json"},
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": 0.95,
            },
        )

        can_deploy, issues = model.can_deploy()

        assert can_deploy is False
        assert "Cannot deploy archived model" in issues

    def test_can_deploy_missing_metadata(self):
        """Test deployment validation fails with missing required metadata."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            current_version_id=uuid4(),
            use_cases=["fraud_detection"],
            data_requirements={"features": ["amount"], "format": "json"},
            metadata={
                "business_impact": "high"
            },  # Missing data_validation and performance_baseline
        )

        can_deploy, issues = model.can_deploy()

        assert can_deploy is False
        assert "Missing required metadata: data_validation" in issues
        assert "Missing required metadata: performance_baseline" in issues

    def test_can_deploy_multiple_issues(self):
        """Test deployment validation with multiple issues."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.ARCHIVED,  # Issue 1
            # Missing current_version_id, use_cases, data_requirements, metadata
        )

        can_deploy, issues = model.can_deploy()

        assert can_deploy is False
        assert len(issues) >= 4  # Multiple issues should be reported


class TestModelRepresentations:
    """Test model string representations."""

    def test_str_representation(self):
        """Test human-readable string representation."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.ENSEMBLE,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.PRODUCTION,
            metadata={"version_count": 5},
        )

        str_repr = str(model)
        assert "Test Model" in str_repr
        assert "ensemble" in str_repr
        assert "stage=production" in str_repr
        assert "versions=5" in str_repr

    def test_repr_representation(self):
        """Test developer representation."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="isolation_forest",
            created_by="test@company.com",
            stage=ModelStage.DEVELOPMENT,
        )

        repr_str = repr(model)
        assert "Model(" in repr_str
        assert "name='Test Model'" in repr_str
        assert "type=supervised" in repr_str
        assert "stage=development" in repr_str
