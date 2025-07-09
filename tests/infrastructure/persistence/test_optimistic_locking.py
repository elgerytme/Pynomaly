"""
Unit tests for optimistic locking with VersionedMixin.
Tests simulate concurrent sessions editing the same row and assert StaleDataError is raised.
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from sqlalchemy import create_engine
from sqlalchemy.orm.exc import StaleDataError
from sqlalchemy.orm import sessionmaker

from pynomaly.domain.common import VersionedMixin
from pynomaly.infrastructure.persistence.database_repositories import (
    Base,
    DatasetModel,
    DetectionResultModel,
    DetectorModel,
    MetricModel,
    RoleModel,
    TenantModel,
    UserModel,
    UUIDType,
)


class TestOptimisticLocking:
    """Test optimistic locking with VersionedMixin."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary SQLite database for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield f"sqlite:///{path}"
        try:
            os.unlink(path)
        except OSError:
            pass

    @pytest.fixture
    def engine(self, temp_db_path):
        """Create SQLAlchemy engine."""
        engine = create_engine(temp_db_path, echo=False)
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def session_factory(self, engine):
        """Create session factory."""
        return sessionmaker(bind=engine)

    def test_versioned_mixin_inheritance(self):
        """Test that models inherit from VersionedMixin correctly."""
        # Check that all models inherit from VersionedMixin
        assert issubclass(DatasetModel, VersionedMixin)
        assert issubclass(DetectorModel, VersionedMixin)
        assert issubclass(DetectionResultModel, VersionedMixin)
        assert issubclass(UserModel, VersionedMixin)
        assert issubclass(TenantModel, VersionedMixin)
        assert issubclass(RoleModel, VersionedMixin)
        assert issubclass(MetricModel, VersionedMixin)

    def test_versioned_mixin_columns(self, session_factory):
        """Test that VersionedMixin adds version column."""
        session = session_factory()
        
        # Create a new dataset
        dataset = DatasetModel(
            id=uuid.uuid4(),
            name="Test Dataset",
            description="Test description",
            features=[],
            entity_metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(dataset)
        session.commit()
        
        # Check that version column exists and starts at 1
        assert hasattr(dataset, 'version')
        assert dataset.version == 1
        
        session.close()

    def test_dataset_model_optimistic_locking(self, session_factory):
        """Test optimistic locking with DatasetModel."""
        # Create initial dataset
        dataset_id = uuid.uuid4()
        session1 = session_factory()
        
        dataset = DatasetModel(
            id=dataset_id,
            name="Test Dataset",
            description="Initial description",
            features=[],
            entity_metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session1.add(dataset)
        session1.commit()
        
        # Simulate two concurrent sessions
        session2 = session_factory()
        
        # Both sessions load the same dataset
        dataset1 = session1.query(DatasetModel).filter_by(id=dataset_id).first()
        dataset2 = session2.query(DatasetModel).filter_by(id=dataset_id).first()
        
        # Both have the same version
        assert dataset1.version == dataset2.version == 1
        
        # Session 1 updates the dataset
        dataset1.description = "Updated by session 1"
        session1.commit()
        
        # Session 2 tries to update the same dataset
        dataset2.description = "Updated by session 2"
        
        # This should raise StaleDataError due to optimistic locking
        with pytest.raises(StaleDataError):
            session2.commit()
        
        # Clean up
        session1.close()
        session2.close()

    def test_detector_model_optimistic_locking(self, session_factory):
        """Test optimistic locking with DetectorModel."""
        # Create initial detector
        detector_id = uuid.uuid4()
        session1 = session_factory()
        
        detector = DetectorModel(
            id=detector_id,
            algorithm="IsolationForest",
            parameters={"n_estimators": 100},
            is_fitted=False,
            entity_metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session1.add(detector)
        session1.commit()
        
        # Simulate two concurrent sessions
        session2 = session_factory()
        
        # Both sessions load the same detector
        detector1 = session1.query(DetectorModel).filter_by(id=detector_id).first()
        detector2 = session2.query(DetectorModel).filter_by(id=detector_id).first()
        
        # Both have the same version
        assert detector1.version == detector2.version == 1
        
        # Session 1 updates the detector
        detector1.is_fitted = True
        session1.commit()
        
        # Session 2 tries to update the same detector
        detector2.parameters = {"n_estimators": 200}
        
        # This should raise StaleDataError due to optimistic locking
        with pytest.raises(StaleDataError):
            session2.commit()
        
        # Clean up
        session1.close()
        session2.close()

    def test_detection_result_model_optimistic_locking(self, session_factory):
        """Test optimistic locking with DetectionResultModel."""
        # Create initial detection result
        result_id = uuid.uuid4()
        session1 = session_factory()
        
        result = DetectionResultModel(
            id=result_id,
            detector_id=uuid.uuid4(),
            dataset_id=uuid.uuid4(),
            scores=[{"value": 0.5, "confidence": 0.8}],
            labels=[0, 1, 0],
            entity_metadata={},
            created_at=datetime.utcnow()
        )
        
        session1.add(result)
        session1.commit()
        
        # Simulate two concurrent sessions
        session2 = session_factory()
        
        # Both sessions load the same result
        result1 = session1.query(DetectionResultModel).filter_by(id=result_id).first()
        result2 = session2.query(DetectionResultModel).filter_by(id=result_id).first()
        
        # Both have the same version
        assert result1.version == result2.version == 1
        
        # Session 1 updates the result
        result1.scores = [{"value": 0.7, "confidence": 0.9}]
        session1.commit()
        
        # Session 2 tries to update the same result
        result2.labels = [1, 1, 0]
        
        # This should raise StaleDataError due to optimistic locking
        with pytest.raises(StaleDataError):
            session2.commit()
        
        # Clean up
        session1.close()
        session2.close()

    def test_user_model_optimistic_locking(self, session_factory):
        """Test optimistic locking with UserModel."""
        # Create initial user
        user_id = uuid.uuid4()
        session1 = session_factory()
        
        user = UserModel(
            id=user_id,
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            status="active",
            password_hash="hashed_password",
            settings={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session1.add(user)
        session1.commit()
        
        # Simulate two concurrent sessions
        session2 = session_factory()
        
        # Both sessions load the same user
        user1 = session1.query(UserModel).filter_by(id=user_id).first()
        user2 = session2.query(UserModel).filter_by(id=user_id).first()
        
        # Both have the same version
        assert user1.version == user2.version == 1
        
        # Session 1 updates the user
        user1.first_name = "Updated"
        session1.commit()
        
        # Session 2 tries to update the same user
        user2.last_name = "Updated"
        
        # This should raise StaleDataError due to optimistic locking
        with pytest.raises(StaleDataError):
            session2.commit()
        
        # Clean up
        session1.close()
        session2.close()

    def test_tenant_model_optimistic_locking(self, session_factory):
        """Test optimistic locking with TenantModel."""
        # Create initial tenant
        tenant_id = uuid.uuid4()
        session1 = session_factory()
        
        tenant = TenantModel(
            id=tenant_id,
            name="Test Tenant",
            domain="test.example.com",
            plan="basic",
            status="active",
            limits={},
            usage={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session1.add(tenant)
        session1.commit()
        
        # Simulate two concurrent sessions
        session2 = session_factory()
        
        # Both sessions load the same tenant
        tenant1 = session1.query(TenantModel).filter_by(id=tenant_id).first()
        tenant2 = session2.query(TenantModel).filter_by(id=tenant_id).first()
        
        # Both have the same version
        assert tenant1.version == tenant2.version == 1
        
        # Session 1 updates the tenant
        tenant1.plan = "premium"
        session1.commit()
        
        # Session 2 tries to update the same tenant
        tenant2.status = "inactive"
        
        # This should raise StaleDataError due to optimistic locking
        with pytest.raises(StaleDataError):
            session2.commit()
        
        # Clean up
        session1.close()
        session2.close()

    def test_role_model_optimistic_locking(self, session_factory):
        """Test optimistic locking with RoleModel."""
        # Create initial role
        role_id = uuid.uuid4()
        session1 = session_factory()
        
        role = RoleModel(
            id=role_id,
            name="Test Role",
            description="Test role description",
            permissions=[],
            is_system_role=False,
            created_at=datetime.utcnow()
        )
        
        session1.add(role)
        session1.commit()
        
        # Simulate two concurrent sessions
        session2 = session_factory()
        
        # Both sessions load the same role
        role1 = session1.query(RoleModel).filter_by(id=role_id).first()
        role2 = session2.query(RoleModel).filter_by(id=role_id).first()
        
        # Both have the same version
        assert role1.version == role2.version == 1
        
        # Session 1 updates the role
        role1.description = "Updated description"
        session1.commit()
        
        # Session 2 tries to update the same role
        role2.permissions = ["read", "write"]
        
        # This should raise StaleDataError due to optimistic locking
        with pytest.raises(StaleDataError):
            session2.commit()
        
        # Clean up
        session1.close()
        session2.close()

    def test_metric_model_optimistic_locking(self, session_factory):
        """Test optimistic locking with MetricModel."""
        # Create initial metric
        metric_id = uuid.uuid4()
        session1 = session_factory()
        
        metric = MetricModel(
            id=metric_id,
            name="test_metric",
            value=100.0,
            unit="count",
            tags={},
            timestamp=datetime.utcnow(),
            entity_type="detector",
            entity_id=uuid.uuid4(),
            meta_data={}
        )
        
        session1.add(metric)
        session1.commit()
        
        # Simulate two concurrent sessions
        session2 = session_factory()
        
        # Both sessions load the same metric
        metric1 = session1.query(MetricModel).filter_by(id=metric_id).first()
        metric2 = session2.query(MetricModel).filter_by(id=metric_id).first()
        
        # Both have the same version
        assert metric1.version == metric2.version == 1
        
        # Session 1 updates the metric
        metric1.value = 200.0
        session1.commit()
        
        # Session 2 tries to update the same metric
        metric2.unit = "percentage"
        
        # This should raise StaleDataError due to optimistic locking
        with pytest.raises(StaleDataError):
            session2.commit()
        
        # Clean up
        session1.close()
        session2.close()

    def test_version_increment_on_update(self, session_factory):
        """Test that version increments on each update."""
        # Create initial dataset
        dataset_id = uuid.uuid4()
        session = session_factory()
        
        dataset = DatasetModel(
            id=dataset_id,
            name="Test Dataset",
            description="Initial description",
            features=[],
            entity_metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(dataset)
        session.commit()
        
        # Check initial version
        assert dataset.version == 1
        
        # Update and check version increment
        dataset.description = "Updated description"
        session.commit()
        assert dataset.version == 2
        
        # Update again and check version increment
        dataset.description = "Updated again"
        session.commit()
        assert dataset.version == 3
        
        session.close()

    def test_optimistic_locking_recovery(self, session_factory):
        """Test recovery from optimistic locking conflict."""
        # Create initial dataset
        dataset_id = uuid.uuid4()
        session1 = session_factory()
        
        dataset = DatasetModel(
            id=dataset_id,
            name="Test Dataset",
            description="Initial description",
            features=[],
            entity_metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session1.add(dataset)
        session1.commit()
        
        # Simulate two concurrent sessions
        session2 = session_factory()
        
        # Both sessions load the same dataset
        dataset1 = session1.query(DatasetModel).filter_by(id=dataset_id).first()
        dataset2 = session2.query(DatasetModel).filter_by(id=dataset_id).first()
        
        # Session 1 successfully updates
        dataset1.description = "Updated by session 1"
        session1.commit()
        
        # Session 2 fails to update due to optimistic locking
        dataset2.description = "Updated by session 2"
        
        with pytest.raises(StaleDataError):
            session2.commit()
        
        # Session 2 can recover by rolling back and reloading
        session2.rollback()
        dataset2_fresh = session2.query(DatasetModel).filter_by(id=dataset_id).first()
        
        # Fresh instance should have updated version
        assert dataset2_fresh.version == 2
        assert dataset2_fresh.description == "Updated by session 1"
        
        # Now session 2 can successfully update
        dataset2_fresh.description = "Updated by session 2 after refresh"
        session2.commit()
        
        # Version should be incremented
        assert dataset2_fresh.version == 3
        
        # Clean up
        session1.close()
        session2.close()

    def test_versioned_mixin_configuration(self):
        """Test that VersionedMixin is configured correctly."""
        # Check that version column is configured properly
        assert hasattr(DatasetModel, 'version')
        
        # Check that __mapper_args__ contains version_id_col
        mapper_args = DatasetModel.__mapper_args__
        assert 'version_id_col' in mapper_args
        
