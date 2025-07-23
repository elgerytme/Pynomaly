"""Comprehensive unit tests for MLOps Model domain entity."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import patch

from mlops.domain.entities.model import Model, ModelStage
from mlops.domain.value_objects.model_value_objects import ModelType


class MockModelType:
    """Mock ModelType enum for testing."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    
    def __init__(self, value):
        self.value = value


@pytest.fixture
def sample_model_data():
    """Sample model data for testing."""
    return {
        "name": "Test Model",
        "description": "A test model for unit testing",
        "model_type": MockModelType(MockModelType.CLASSIFICATION),
        "algorithm_family": "random_forest",
        "created_by": "test_user"
    }


@pytest.fixture
def sample_model(sample_model_data):
    """Sample model instance for testing."""
    return Model(**sample_model_data)


class TestModelStage:
    """Test cases for ModelStage enum."""

    def test_model_stage_values(self):
        """Test all ModelStage enum values."""
        assert ModelStage.DEVELOPMENT.value == "development"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"
        assert ModelStage.ARCHIVED.value == "archived"


class TestModel:
    """Test cases for Model domain entity."""

    def test_initialization_required_fields(self, sample_model_data):
        """Test model initialization with required fields only."""
        model = Model(**sample_model_data)
        
        assert model.name == "Test Model"
        assert model.description == "A test model for unit testing"
        assert model.model_type.value == MockModelType.CLASSIFICATION
        assert model.algorithm_family == "random_forest"
        assert model.created_by == "test_user"
        assert isinstance(model.id, UUID)
        assert isinstance(model.created_at, datetime)
        assert model.team == ""
        assert model.tags == []
        assert model.stage == ModelStage.DEVELOPMENT
        assert model.current_version_id is None
        assert model.latest_version_id is None
        assert model.metadata == {}
        assert model.use_cases == []
        assert model.data_requirements == {}

    def test_initialization_with_all_fields(self):
        """Test model initialization with all fields provided."""
        model_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 0, 0)
        current_version_id = uuid4()
        latest_version_id = uuid4()
        tenant_id = uuid4()
        experiment_id = uuid4()
        approved_at = datetime(2024, 1, 20, 14, 30, 0)
        
        model = Model(
            id=model_id,
            name="Production Model",
            description="A production-ready model",
            model_type=MockModelType(MockModelType.REGRESSION),
            algorithm_family="gradient_boosting",
            created_by="ml_engineer",
            created_at=created_at,
            team="ML Team",
            tags=["production", "validated"],
            stage=ModelStage.PRODUCTION,
            current_version_id=current_version_id,
            latest_version_id=latest_version_id,
            metadata={"business_impact": "high"},
            use_cases=["fraud_detection", "risk_assessment"],
            data_requirements={"min_samples": 1000},
            tenant_id=tenant_id,
            external_model_id="mlflow_123",
            framework="tensorflow",
            framework_version="2.13.0",
            model_uri="s3://models/prod_model",
            deployment_ids=[uuid4(), uuid4()],
            experiment_id=experiment_id,
            approval_status="approved",
            approved_by="data_scientist",
            approved_at=approved_at
        )
        
        assert model.id == model_id
        assert model.name == "Production Model"
        assert model.description == "A production-ready model"
        assert model.model_type.value == MockModelType.REGRESSION
        assert model.algorithm_family == "gradient_boosting"
        assert model.created_by == "ml_engineer"
        assert model.created_at == created_at
        assert model.team == "ML Team"
        assert model.tags == ["production", "validated"]
        assert model.stage == ModelStage.PRODUCTION
        assert model.current_version_id == current_version_id
        assert model.latest_version_id == latest_version_id
        assert model.metadata == {"business_impact": "high"}
        assert model.use_cases == ["fraud_detection", "risk_assessment"]
        assert model.data_requirements == {"min_samples": 1000}
        assert model.tenant_id == tenant_id
        assert model.external_model_id == "mlflow_123"
        assert model.framework == "tensorflow"
        assert model.framework_version == "2.13.0"
        assert model.model_uri == "s3://models/prod_model"
        assert len(model.deployment_ids) == 2
        assert model.experiment_id == experiment_id
        assert model.approval_status == "approved"
        assert model.approved_by == "data_scientist"
        assert model.approved_at == approved_at

    def test_post_init_validation_empty_name(self):
        """Test validation fails for empty name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            Model(
                name="",
                description="Test description",
                model_type=MockModelType(MockModelType.CLASSIFICATION),
                algorithm_family="random_forest",
                created_by="test_user"
            )

    def test_post_init_validation_empty_description(self):
        """Test validation fails for empty description."""
        with pytest.raises(ValueError, match="Model description cannot be empty"):
            Model(
                name="Test Model",
                description="",
                model_type=MockModelType(MockModelType.CLASSIFICATION),
                algorithm_family="random_forest",
                created_by="test_user"
            )

    def test_post_init_validation_invalid_model_type(self):
        """Test validation fails for invalid model type."""
        with pytest.raises(TypeError, match="Model type must be ModelType"):
            Model(
                name="Test Model",
                description="Test description",
                model_type="invalid_type",
                algorithm_family="random_forest",
                created_by="test_user"
            )

    def test_post_init_validation_invalid_stage(self):
        """Test validation fails for invalid stage."""
        with pytest.raises(TypeError, match="Model stage must be ModelStage"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=MockModelType(MockModelType.CLASSIFICATION),
                algorithm_family="random_forest",
                created_by="test_user",
                stage="invalid_stage"
            )

    def test_post_init_validation_empty_created_by(self):
        """Test validation fails for empty created_by."""
        with pytest.raises(ValueError, match="Created by cannot be empty"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=MockModelType(MockModelType.CLASSIFICATION),
                algorithm_family="random_forest",
                created_by=""
            )

    def test_post_init_validation_empty_algorithm_family(self):
        """Test validation fails for empty algorithm family."""
        with pytest.raises(ValueError, match="Algorithm family cannot be empty"):
            Model(
                name="Test Model",
                description="Test description",
                model_type=MockModelType(MockModelType.CLASSIFICATION),
                algorithm_family="",
                created_by="test_user"
            )

    def test_is_in_production_property(self, sample_model):
        """Test is_in_production property."""
        assert sample_model.is_in_production is False
        
        sample_model.stage = ModelStage.PRODUCTION
        assert sample_model.is_in_production is True
        
        sample_model.stage = ModelStage.STAGING
        assert sample_model.is_in_production is False

    def test_is_archived_property(self, sample_model):
        """Test is_archived property."""
        assert sample_model.is_archived is False
        
        sample_model.stage = ModelStage.ARCHIVED
        assert sample_model.is_archived is True
        
        sample_model.stage = ModelStage.PRODUCTION
        assert sample_model.is_archived is False

    def test_has_current_version_property(self, sample_model):
        """Test has_current_version property."""
        assert sample_model.has_current_version is False
        
        sample_model.current_version_id = uuid4()
        assert sample_model.has_current_version is True

    def test_version_count_property(self, sample_model):
        """Test version_count property."""
        assert sample_model.version_count == 0
        
        sample_model.metadata["version_count"] = 5
        assert sample_model.version_count == 5

    def test_add_tag(self, sample_model):
        """Test adding tags to model."""
        sample_model.add_tag("production")
        assert "production" in sample_model.tags
        
        sample_model.add_tag("validated")
        assert "validated" in sample_model.tags
        assert len(sample_model.tags) == 2
        
        # Test adding duplicate tag
        sample_model.add_tag("production")
        assert sample_model.tags.count("production") == 1

    def test_add_tag_empty_string(self, sample_model):
        """Test adding empty tag does nothing."""
        sample_model.add_tag("")
        assert len(sample_model.tags) == 0
        
        sample_model.add_tag(None)
        assert len(sample_model.tags) == 0

    def test_remove_tag(self, sample_model):
        """Test removing tags from model."""
        sample_model.tags = ["production", "validated", "tested"]
        
        sample_model.remove_tag("validated")
        assert "validated" not in sample_model.tags
        assert len(sample_model.tags) == 2
        
        # Test removing non-existent tag
        sample_model.remove_tag("non_existent")
        assert len(sample_model.tags) == 2

    def test_has_tag(self, sample_model):
        """Test checking if model has specific tag."""
        sample_model.tags = ["production", "validated"]
        
        assert sample_model.has_tag("production") is True
        assert sample_model.has_tag("validated") is True
        assert sample_model.has_tag("non_existent") is False

    def test_update_stage(self, sample_model):
        """Test updating model stage with transition tracking."""
        with patch('mlops.domain.entities.model.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_model.metadata["last_updated_by"] = "test_user"
            sample_model.update_stage(ModelStage.STAGING)
            
            assert sample_model.stage == ModelStage.STAGING
            assert "stage_transitions" in sample_model.metadata
            
            transitions = sample_model.metadata["stage_transitions"]
            assert len(transitions) == 1
            assert transitions[0]["from"] == "development"
            assert transitions[0]["to"] == "staging"
            assert transitions[0]["updated_by"] == "test_user"
            assert "last_stage_update" in sample_model.metadata

    def test_promote_to_production(self, sample_model):
        """Test promoting model to production."""
        version_id = uuid4()
        
        with patch('mlops.domain.entities.model.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 14, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_model.promote_to_production(version_id, "ml_engineer")
            
            assert sample_model.current_version_id == version_id
            assert sample_model.stage == ModelStage.PRODUCTION
            assert sample_model.metadata["promoted_by"] == "ml_engineer"
            assert sample_model.metadata["production_version_id"] == str(version_id)
            assert "promoted_at" in sample_model.metadata

    def test_add_use_case(self, sample_model):
        """Test adding use cases to model."""
        sample_model.add_use_case("fraud_detection")
        assert "fraud_detection" in sample_model.use_cases
        
        sample_model.add_use_case("risk_assessment")
        assert "risk_assessment" in sample_model.use_cases
        assert len(sample_model.use_cases) == 2
        
        # Test adding duplicate use case
        sample_model.add_use_case("fraud_detection")
        assert sample_model.use_cases.count("fraud_detection") == 1

    def test_add_use_case_empty_string(self, sample_model):
        """Test adding empty use case does nothing."""
        sample_model.add_use_case("")
        assert len(sample_model.use_cases) == 0

    def test_remove_use_case(self, sample_model):
        """Test removing use cases from model."""
        sample_model.use_cases = ["fraud_detection", "risk_assessment", "compliance"]
        
        sample_model.remove_use_case("risk_assessment")
        assert "risk_assessment" not in sample_model.use_cases
        assert len(sample_model.use_cases) == 2
        
        # Test removing non-existent use case
        sample_model.remove_use_case("non_existent")
        assert len(sample_model.use_cases) == 2

    def test_update_data_requirements(self, sample_model):
        """Test updating data requirements."""
        with patch('mlops.domain.entities.model.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 15, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            requirements = {
                "min_samples": 1000,
                "required_features": ["feature1", "feature2"],
                "data_quality_threshold": 0.95
            }
            
            sample_model.update_data_requirements(requirements)
            
            assert sample_model.data_requirements == requirements
            assert "data_requirements_updated" in sample_model.metadata

    def test_set_current_version(self, sample_model):
        """Test setting current version."""
        version_id = uuid4()
        
        with patch('mlops.domain.entities.model.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 16, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_model.set_current_version(version_id)
            
            assert sample_model.current_version_id == version_id
            assert "current_version_updated" in sample_model.metadata

    def test_set_latest_version(self, sample_model):
        """Test setting latest version with count increment."""
        version_id = uuid4()
        
        with patch('mlops.domain.entities.model.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 17, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_model.set_latest_version(version_id)
            
            assert sample_model.latest_version_id == version_id
            assert sample_model.metadata["version_count"] == 1
            assert "latest_version_updated" in sample_model.metadata
            
            # Test incrementing version count
            another_version_id = uuid4()
            sample_model.set_latest_version(another_version_id)
            assert sample_model.metadata["version_count"] == 2

    def test_update_metadata(self, sample_model):
        """Test updating model metadata."""
        with patch('mlops.domain.entities.model.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 18, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            sample_model.update_metadata("business_impact", "high", "data_scientist")
            
            assert sample_model.metadata["business_impact"] == "high"
            assert sample_model.metadata["last_updated_by"] == "data_scientist"
            assert "last_updated" in sample_model.metadata

    def test_update_metadata_without_user(self, sample_model):
        """Test updating metadata without specifying user."""
        sample_model.update_metadata("test_key", "test_value")
        
        assert sample_model.metadata["test_key"] == "test_value"
        assert "last_updated" in sample_model.metadata
        assert "last_updated_by" not in sample_model.metadata

    def test_get_info(self, sample_model):
        """Test getting comprehensive model information."""
        sample_model.tags = ["production", "validated"]
        sample_model.use_cases = ["fraud_detection"]
        sample_model.data_requirements = {"min_samples": 1000}
        sample_model.current_version_id = uuid4()
        sample_model.latest_version_id = uuid4()
        sample_model.metadata = {"business_impact": "high", "version_count": 3}
        
        info = sample_model.get_info()
        
        assert info["id"] == str(sample_model.id)
        assert info["name"] == sample_model.name
        assert info["description"] == sample_model.description
        assert info["model_type"] == sample_model.model_type.value
        assert info["algorithm_family"] == sample_model.algorithm_family
        assert info["created_at"] == sample_model.created_at.isoformat()
        assert info["created_by"] == sample_model.created_by
        assert info["team"] == sample_model.team
        assert info["stage"] == sample_model.stage.value
        assert info["is_in_production"] == sample_model.is_in_production
        assert info["has_current_version"] == sample_model.has_current_version
        assert info["current_version_id"] == str(sample_model.current_version_id)
        assert info["latest_version_id"] == str(sample_model.latest_version_id)
        assert info["version_count"] == 3
        assert info["tags"] == sample_model.tags
        assert info["use_cases"] == sample_model.use_cases
        assert info["data_requirements"] == sample_model.data_requirements
        assert info["metadata"] == sample_model.metadata

    def test_get_info_with_none_version_ids(self, sample_model):
        """Test get_info when version IDs are None."""
        info = sample_model.get_info()
        
        assert info["current_version_id"] is None
        assert info["latest_version_id"] is None

    def test_get_stage_history(self, sample_model):
        """Test getting stage transition history."""
        # Initially empty
        history = sample_model.get_stage_history()
        assert history == []
        
        # Add some transitions
        sample_model.metadata["stage_transitions"] = [
            {"from": "development", "to": "staging", "timestamp": "2024-01-15T10:00:00"},
            {"from": "staging", "to": "production", "timestamp": "2024-01-20T14:00:00"}
        ]
        
        history = sample_model.get_stage_history()
        assert len(history) == 2
        assert history[0]["from"] == "development"
        assert history[0]["to"] == "staging"

    def test_can_deploy_success(self, sample_model):
        """Test can_deploy returns True when all requirements are met."""
        sample_model.current_version_id = uuid4()
        sample_model.use_cases = ["fraud_detection"]
        sample_model.data_requirements = {"min_samples": 1000}
        sample_model.metadata = {
            "business_impact": "high",
            "data_validation": "passed",
            "performance_baseline": "established"
        }
        
        can_deploy, issues = sample_model.can_deploy()
        
        assert can_deploy is True
        assert issues == []

    def test_can_deploy_missing_current_version(self, sample_model):
        """Test can_deploy fails when no current version is set."""
        sample_model.use_cases = ["fraud_detection"]
        sample_model.data_requirements = {"min_samples": 1000}
        sample_model.metadata = {
            "business_impact": "high",
            "data_validation": "passed",
            "performance_baseline": "established"
        }
        
        can_deploy, issues = sample_model.can_deploy()
        
        assert can_deploy is False
        assert "No current version set" in issues

    def test_can_deploy_missing_use_cases(self, sample_model):
        """Test can_deploy fails when no use cases are defined."""
        sample_model.current_version_id = uuid4()
        sample_model.data_requirements = {"min_samples": 1000}
        sample_model.metadata = {
            "business_impact": "high",
            "data_validation": "passed",
            "performance_baseline": "established"
        }
        
        can_deploy, issues = sample_model.can_deploy()
        
        assert can_deploy is False
        assert "No use cases defined" in issues

    def test_can_deploy_missing_data_requirements(self, sample_model):
        """Test can_deploy fails when no data requirements are specified."""
        sample_model.current_version_id = uuid4()
        sample_model.use_cases = ["fraud_detection"]
        sample_model.metadata = {
            "business_impact": "high",
            "data_validation": "passed",
            "performance_baseline": "established"
        }
        
        can_deploy, issues = sample_model.can_deploy()
        
        assert can_deploy is False
        assert "No data requirements specified" in issues

    def test_can_deploy_archived_model(self, sample_model):
        """Test can_deploy fails for archived model."""
        sample_model.stage = ModelStage.ARCHIVED
        sample_model.current_version_id = uuid4()
        sample_model.use_cases = ["fraud_detection"]
        sample_model.data_requirements = {"min_samples": 1000}
        sample_model.metadata = {
            "business_impact": "high",
            "data_validation": "passed",
            "performance_baseline": "established"
        }
        
        can_deploy, issues = sample_model.can_deploy()
        
        assert can_deploy is False
        assert "Cannot deploy archived model" in issues

    def test_can_deploy_missing_required_metadata(self, sample_model):
        """Test can_deploy fails when required metadata is missing."""
        sample_model.current_version_id = uuid4()
        sample_model.use_cases = ["fraud_detection"]
        sample_model.data_requirements = {"min_samples": 1000}
        sample_model.metadata = {
            "business_impact": "high",
            # Missing data_validation and performance_baseline
        }
        
        can_deploy, issues = sample_model.can_deploy()
        
        assert can_deploy is False
        assert "Missing required metadata: data_validation" in issues
        assert "Missing required metadata: performance_baseline" in issues

    def test_can_deploy_multiple_issues(self, sample_model):
        """Test can_deploy with multiple issues."""
        # Archived model with no current version, use cases, or data requirements
        sample_model.stage = ModelStage.ARCHIVED
        
        can_deploy, issues = sample_model.can_deploy()
        
        assert can_deploy is False
        assert len(issues) > 1
        assert "Cannot deploy archived model" in issues
        assert "No current version set" in issues
        assert "No use cases defined" in issues

    def test_str_representation(self, sample_model):
        """Test string representation of model."""
        sample_model.metadata["version_count"] = 3
        
        str_repr = str(sample_model)
        
        assert "Test Model" in str_repr
        assert sample_model.model_type.value in str_repr
        assert "development" in str_repr
        assert "versions=3" in str_repr

    def test_repr_representation(self, sample_model):
        """Test developer representation of model."""
        repr_str = repr(sample_model)
        
        assert f"id={sample_model.id}" in repr_str
        assert "name='Test Model'" in repr_str
        assert f"type={sample_model.model_type.value}" in repr_str
        assert "stage=development" in repr_str

    def test_enterprise_features_defaults(self, sample_model):
        """Test enterprise features have correct defaults."""
        assert sample_model.tenant_id is None
        assert sample_model.external_model_id is None
        assert sample_model.framework == "scikit-learn"
        assert sample_model.framework_version is None
        assert sample_model.model_uri is None
        assert sample_model.deployment_ids == []
        assert sample_model.experiment_id is None
        assert sample_model.approval_status is None
        assert sample_model.approved_by is None
        assert sample_model.approved_at is None

    def test_deployment_ids_management(self, sample_model):
        """Test managing deployment IDs."""
        deployment_id1 = uuid4()
        deployment_id2 = uuid4()
        
        sample_model.deployment_ids = [deployment_id1, deployment_id2]
        
        assert len(sample_model.deployment_ids) == 2
        assert deployment_id1 in sample_model.deployment_ids
        assert deployment_id2 in sample_model.deployment_ids

    def test_approval_workflow(self, sample_model):
        """Test approval workflow fields."""
        approved_at = datetime(2024, 1, 25, 16, 0, 0)
        
        sample_model.approval_status = "approved"
        sample_model.approved_by = "senior_data_scientist"
        sample_model.approved_at = approved_at
        
        assert sample_model.approval_status == "approved"
        assert sample_model.approved_by == "senior_data_scientist"
        assert sample_model.approved_at == approved_at