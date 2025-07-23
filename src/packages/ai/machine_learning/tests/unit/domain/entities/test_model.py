"""Comprehensive unit tests for Model entity."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from machine_learning.domain.entities.model import Model, ModelType, ModelStage


class TestModel:
    """Test cases for Model entity."""

    def test_model_creation_with_required_fields(self):
        """Test successful model creation with all required fields."""
        model = Model(
            name="Test Model",
            description="A test model for unit testing",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        assert model.name == "Test Model"
        assert model.description == "A test model for unit testing"
        assert model.model_type == ModelType.SUPERVISED
        assert model.algorithm_family == "random_forest"
        assert model.created_by == "test_user"
        assert isinstance(model.id, UUID)
        assert isinstance(model.created_at, datetime)
        assert model.stage == ModelStage.DEVELOPMENT
        assert model.current_version_id is None
        assert model.latest_version_id is None
        assert model.team == ""
        assert model.tags == []
        assert model.metadata == {}
        assert model.use_cases == []
        assert model.data_requirements == {}

    def test_model_creation_with_all_fields(self):
        """Test model creation with all optional fields provided."""
        model_id = uuid4()
        created_at = datetime.utcnow()
        version_id = uuid4()
        
        model = Model(
            name="Full Model",
            description="A complete test model",
            model_type=ModelType.DEEP_LEARNING,
            algorithm_family="neural_network",
            created_by="test_user",
            id=model_id,
            created_at=created_at,
            team="ML Team",
            tags=["production", "nlp"],
            stage=ModelStage.STAGING,
            current_version_id=version_id,
            latest_version_id=version_id,
            metadata={"key": "value"},
            use_cases=["text_classification"],
            data_requirements={"format": "text"}
        )
        
        assert model.id == model_id
        assert model.created_at == created_at
        assert model.team == "ML Team"
        assert model.tags == ["production", "nlp"]
        assert model.stage == ModelStage.STAGING
        assert model.current_version_id == version_id
        assert model.latest_version_id == version_id
        assert model.metadata == {"key": "value"}
        assert model.use_cases == ["text_classification"]
        assert model.data_requirements == {"format": "text"}

    def test_model_validation_empty_name(self):
        """Test model validation fails with empty name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            Model(
                name="",
                description="Test description",
                model_type=ModelType.SUPERVISED,
                algorithm_family="random_forest",
                created_by="test_user"
            )

    def test_model_validation_empty_description(self):
        """Test model validation fails with empty description."""
        with pytest.raises(ValueError, match="Model description cannot be empty"):
            Model(
                name="Test Model",
                description="",
                model_type=ModelType.SUPERVISED,
                algorithm_family="random_forest",
                created_by="test_user"
            )

    def test_model_validation_invalid_model_type(self):
        """Test model validation fails with invalid model type."""
        with pytest.raises(TypeError, match="Model type must be ModelType"):
            Model(
                name="Test Model",
                description="Test description",
                model_type="invalid_type",  # type: ignore
                algorithm_family="random_forest",
                created_by="test_user"
            )

    def test_model_validation_invalid_stage(self):
        """Test model validation fails with invalid stage."""
        with pytest.raises(TypeError, match="Model stage must be ModelStage"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=ModelType.SUPERVISED,
                algorithm_family="random_forest",
                created_by="test_user",
                stage="invalid_stage"  # type: ignore
            )

    def test_model_validation_empty_created_by(self):
        """Test model validation fails with empty created_by."""
        with pytest.raises(ValueError, match="Created by cannot be empty"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=ModelType.SUPERVISED,
                algorithm_family="random_forest",
                created_by=""
            )

    def test_model_validation_empty_algorithm_family(self):
        """Test model validation fails with empty algorithm_family."""
        with pytest.raises(ValueError, match="Algorithm family cannot be empty"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=ModelType.SUPERVISED,
                algorithm_family="",
                created_by="test_user"
            )

    def test_is_in_production_property(self):
        """Test is_in_production property."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        assert not model.is_in_production
        
        model.stage = ModelStage.PRODUCTION
        assert model.is_in_production

    def test_is_archived_property(self):
        """Test is_archived property."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        assert not model.is_archived
        
        model.stage = ModelStage.ARCHIVED
        assert model.is_archived

    def test_has_current_version_property(self):
        """Test has_current_version property."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        assert not model.has_current_version
        
        model.current_version_id = uuid4()
        assert model.has_current_version

    def test_version_count_property(self):
        """Test version_count property."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        assert model.version_count == 0
        
        model.metadata["version_count"] = 5
        assert model.version_count == 5

    def test_add_tag(self):
        """Test adding tags to model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        model.add_tag("production")
        assert "production" in model.tags
        
        # Adding same tag should not duplicate
        model.add_tag("production")
        assert model.tags.count("production") == 1
        
        # Adding empty tag should not add anything
        model.add_tag("")
        assert "" not in model.tags

    def test_remove_tag(self):
        """Test removing tags from model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            tags=["production", "stable"]
        )
        
        model.remove_tag("production")
        assert "production" not in model.tags
        assert "stable" in model.tags
        
        # Removing non-existent tag should not raise error
        model.remove_tag("non_existent")

    def test_has_tag(self):
        """Test checking if model has specific tag."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            tags=["production", "stable"]
        )
        
        assert model.has_tag("production")
        assert model.has_tag("stable")
        assert not model.has_tag("experimental")

    def test_update_stage(self):
        """Test updating model stage with transition tracking."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        old_stage = model.stage
        model.update_stage(ModelStage.STAGING)
        
        assert model.stage == ModelStage.STAGING
        
        # Check transition tracking
        transitions = model.metadata.get("stage_transitions", [])
        assert len(transitions) == 1
        assert transitions[0]["from"] == old_stage.value
        assert transitions[0]["to"] == ModelStage.STAGING.value
        assert "timestamp" in transitions[0]
        assert transitions[0]["updated_by"] == "system"
        assert "last_stage_update" in model.metadata

    def test_promote_to_production(self):
        """Test promoting model to production."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        version_id = uuid4()
        updated_by = "deployment_user"
        
        model.promote_to_production(version_id, updated_by)
        
        assert model.current_version_id == version_id
        assert model.stage == ModelStage.PRODUCTION
        assert model.metadata["promoted_by"] == updated_by
        assert model.metadata["production_version_id"] == str(version_id)
        assert "promoted_at" in model.metadata

    def test_add_use_case(self):
        """Test adding use cases to model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        model.add_use_case("fraud_detection")
        assert "fraud_detection" in model.use_cases
        
        # Adding same use case should not duplicate
        model.add_use_case("fraud_detection")
        assert model.use_cases.count("fraud_detection") == 1
        
        # Adding empty use case should not add anything
        model.add_use_case("")
        assert "" not in model.use_cases

    def test_remove_use_case(self):
        """Test removing use cases from model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            use_cases=["fraud_detection", "risk_assessment"]
        )
        
        model.remove_use_case("fraud_detection")
        assert "fraud_detection" not in model.use_cases
        assert "risk_assessment" in model.use_cases
        
        # Removing non-existent use case should not raise error
        model.remove_use_case("non_existent")

    def test_update_data_requirements(self):
        """Test updating data requirements."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        requirements = {
            "format": "csv",
            "columns": ["feature1", "feature2"],
            "min_rows": 1000
        }
        
        model.update_data_requirements(requirements)
        
        assert model.data_requirements == requirements
        assert "data_requirements_updated" in model.metadata

    def test_set_current_version(self):
        """Test setting current version."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        version_id = uuid4()
        model.set_current_version(version_id)
        
        assert model.current_version_id == version_id
        assert "current_version_updated" in model.metadata

    def test_set_latest_version(self):
        """Test setting latest version with version count increment."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        version_id = uuid4()
        model.set_latest_version(version_id)
        
        assert model.latest_version_id == version_id
        assert model.metadata["version_count"] == 1
        assert "latest_version_updated" in model.metadata
        
        # Adding another version should increment count
        version_id2 = uuid4()
        model.set_latest_version(version_id2)
        assert model.metadata["version_count"] == 2

    def test_update_metadata(self):
        """Test updating model metadata."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        model.update_metadata("test_key", "test_value", "test_user")
        
        assert model.metadata["test_key"] == "test_value"
        assert model.metadata["last_updated_by"] == "test_user"
        assert "last_updated" in model.metadata
        
        # Test without updated_by
        model.update_metadata("another_key", "another_value")
        assert model.metadata["another_key"] == "another_value"

    def test_get_info(self):
        """Test getting comprehensive model information."""
        model_id = uuid4()
        created_at = datetime.utcnow()
        current_version_id = uuid4()
        latest_version_id = uuid4()
        
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            id=model_id,
            created_at=created_at,
            team="ML Team",
            tags=["production"],
            stage=ModelStage.PRODUCTION,
            current_version_id=current_version_id,
            latest_version_id=latest_version_id,
            metadata={"version_count": 3},
            use_cases=["fraud_detection"],
            data_requirements={"format": "csv"}
        )
        
        info = model.get_info()
        
        assert info["id"] == str(model_id)
        assert info["name"] == "Test Model"
        assert info["description"] == "Test description"
        assert info["model_type"] == "supervised"
        assert info["algorithm_family"] == "random_forest"
        assert info["created_at"] == created_at.isoformat()
        assert info["created_by"] == "test_user"
        assert info["team"] == "ML Team"
        assert info["stage"] == "production"
        assert info["is_in_production"] is True
        assert info["has_current_version"] is True
        assert info["current_version_id"] == str(current_version_id)
        assert info["latest_version_id"] == str(latest_version_id)
        assert info["version_count"] == 3
        assert info["tags"] == ["production"]
        assert info["use_cases"] == ["fraud_detection"]
        assert info["data_requirements"] == {"format": "csv"}
        assert info["metadata"] == {"version_count": 3}

    def test_get_stage_history(self):
        """Test getting stage transition history."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user"
        )
        
        # Initially no history
        assert model.get_stage_history() == []
        
        # Add some transitions
        model.update_stage(ModelStage.STAGING)
        model.update_stage(ModelStage.PRODUCTION)
        
        history = model.get_stage_history()
        assert len(history) == 2
        assert history[0]["from"] == "development"
        assert history[0]["to"] == "staging"
        assert history[1]["from"] == "staging"
        assert history[1]["to"] == "production"

    def test_can_deploy_success(self):
        """Test successful deployment validation."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            current_version_id=uuid4(),
            use_cases=["fraud_detection"],
            data_requirements={"format": "csv"},
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": "0.85"
            }
        )
        
        can_deploy, issues = model.can_deploy()
        assert can_deploy is True
        assert issues == []

    def test_can_deploy_missing_version(self):
        """Test deployment validation fails without current version."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            use_cases=["fraud_detection"],
            data_requirements={"format": "csv"},
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": "0.85"
            }
        )
        
        can_deploy, issues = model.can_deploy()
        assert can_deploy is False
        assert "No current version set" in issues

    def test_can_deploy_missing_use_cases(self):
        """Test deployment validation fails without use cases."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            current_version_id=uuid4(),
            data_requirements={"format": "csv"},
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": "0.85"
            }
        )
        
        can_deploy, issues = model.can_deploy()
        assert can_deploy is False
        assert "No use cases defined" in issues

    def test_can_deploy_missing_data_requirements(self):
        """Test deployment validation fails without data requirements."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            current_version_id=uuid4(),
            use_cases=["fraud_detection"],
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": "0.85"
            }
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
            algorithm_family="random_forest",
            created_by="test_user",
            stage=ModelStage.ARCHIVED,
            current_version_id=uuid4(),
            use_cases=["fraud_detection"],
            data_requirements={"format": "csv"},
            metadata={
                "business_impact": "high",
                "data_validation": "passed",
                "performance_baseline": "0.85"
            }
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
            algorithm_family="random_forest",
            created_by="test_user",
            current_version_id=uuid4(),
            use_cases=["fraud_detection"],
            data_requirements={"format": "csv"}
        )
        
        can_deploy, issues = model.can_deploy()
        assert can_deploy is False
        assert "Missing required metadata: business_impact" in issues
        assert "Missing required metadata: data_validation" in issues
        assert "Missing required metadata: performance_baseline" in issues

    def test_str_representation(self):
        """Test string representation of model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            stage=ModelStage.PRODUCTION,
            metadata={"version_count": 3}
        )
        
        str_repr = str(model)
        assert "Test Model" in str_repr
        assert "supervised" in str_repr
        assert "production" in str_repr
        assert "versions=3" in str_repr

    def test_repr_representation(self):
        """Test developer representation of model."""
        model = Model(
            name="Test Model",
            description="Test description",
            model_type=ModelType.SUPERVISED,
            algorithm_family="random_forest",
            created_by="test_user",
            stage=ModelStage.PRODUCTION
        )
        
        repr_str = repr(model)
        assert f"id={model.id}" in repr_str
        assert "name='Test Model'" in repr_str
        assert "type=supervised" in repr_str
        assert "stage=production" in repr_str


class TestModelType:
    """Test cases for ModelType enum."""

    def test_model_type_values(self):
        """Test all ModelType enum values."""
        assert ModelType.SUPERVISED.value == "supervised"
        assert ModelType.UNSUPERVISED.value == "unsupervised"
        assert ModelType.SEMI_SUPERVISED.value == "semi_supervised"
        assert ModelType.ENSEMBLE.value == "ensemble"
        assert ModelType.DEEP_LEARNING.value == "deep_learning"
        assert ModelType.TIME_SERIES.value == "time_series"
        assert ModelType.CLASSIFICATION.value == "classification"
        assert ModelType.REGRESSION.value == "regression"
        assert ModelType.CLUSTERING.value == "clustering"


class TestModelStage:
    """Test cases for ModelStage enum."""

    def test_model_stage_values(self):
        """Test all ModelStage enum values."""
        assert ModelStage.DEVELOPMENT.value == "development"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"
        assert ModelStage.ARCHIVED.value == "archived"