"""Comprehensive tests for database persistence infrastructure - Phase 2 Coverage Enhancement."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import text

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import RepositoryError
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)
from pynomaly.infrastructure.persistence import (
    DatasetModel,
    DetectionResultModel,
    DetectorModel,
)
from pynomaly.infrastructure.persistence.database import (
    DatabaseConnection,
    init_database,
)
from pynomaly.infrastructure.persistence.database_repositories import (
    DatabaseDatasetRepository,
    DatabaseDetectionResultRepository,
    DatabaseDetectorRepository,
)


class TestDatabaseConnection:
    """Comprehensive tests for database connection management."""

    @pytest.fixture
    def test_database_url(self):
        """Create in-memory SQLite database for testing."""
        return "sqlite:///:memory:"

    @pytest.fixture
    def db_connection(self, test_database_url):
        """Create database connection for testing."""
        return DatabaseConnection(test_database_url)

    def test_database_connection_initialization(self, db_connection, test_database_url):
        """Test database connection initialization."""
        assert db_connection.database_url == test_database_url
        assert db_connection.engine is not None
        assert db_connection.SessionLocal is not None

    def test_database_connection_context_manager(self, db_connection):
        """Test database connection context manager."""
        with db_connection.get_session() as session:
            assert session is not None
            # Session should be active
            assert session.is_active

    def test_database_connection_session_cleanup(self, db_connection):
        """Test proper session cleanup."""
        session = None
        with db_connection.get_session() as sess:
            session = sess
            assert session.is_active

        # Session should be closed after context exit
        assert not session.is_active

    def test_create_tables(self, db_connection):
        """Test database table creation."""
        # Create tables
        db_connection.create_tables()

        # Verify tables exist
        with db_connection.get_session() as session:
            # Check if we can query the tables
            result = session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            table_names = [row[0] for row in result.fetchall()]

            expected_tables = ["detectors", "datasets", "detection_results"]
            for table in expected_tables:
                assert table in table_names

    def test_drop_tables(self, db_connection):
        """Test database table dropping."""
        # Create tables first
        db_connection.create_tables()

        # Drop tables
        db_connection.drop_tables()

        # Verify tables are gone
        with db_connection.get_session() as session:
            result = session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            table_names = [row[0] for row in result.fetchall()]

            unexpected_tables = ["detectors", "datasets", "detection_results"]
            for table in unexpected_tables:
                assert table not in table_names

    def test_init_database_function(self, test_database_url):
        """Test global database initialization function."""
        db_connection = init_database(test_database_url)

        assert isinstance(db_connection, DatabaseConnection)
        assert db_connection.database_url == test_database_url


class TestDatabaseDetectorRepository:
    """Comprehensive tests for database detector repository."""

    @pytest.fixture
    def db_connection(self):
        """Create test database connection."""
        connection = DatabaseConnection("sqlite:///:memory:")
        connection.create_tables()
        return connection

    @pytest.fixture
    def detector_repository(self, db_connection):
        """Create detector repository for testing."""
        return DatabaseDetectorRepository(db_connection)

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector for testing."""
        return Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
            hyperparameters={"n_estimators": 100, "random_state": 42},
            metadata={"version": "1.0", "created_by": "test"},
        )

    @pytest.mark.asyncio
    async def test_save_detector_success(self, detector_repository, sample_detector):
        """Test successful detector saving."""
        await detector_repository.save(sample_detector)

        # Verify detector was saved
        retrieved = await detector_repository.find_by_id(sample_detector.id)
        assert retrieved is not None
        assert retrieved.name == sample_detector.name
        assert retrieved.algorithm == sample_detector.algorithm
        assert (
            retrieved.contamination_rate.value
            == sample_detector.contamination_rate.value
        )
        assert retrieved.hyperparameters == sample_detector.hyperparameters

    @pytest.mark.asyncio
    async def test_save_detector_update_existing(
        self, detector_repository, sample_detector
    ):
        """Test updating existing detector."""
        # Save initial detector
        await detector_repository.save(sample_detector)

        # Update detector
        sample_detector.name = "updated_detector"
        sample_detector.hyperparameters["n_estimators"] = 200

        # Save updated detector
        await detector_repository.save(sample_detector)

        # Verify update
        retrieved = await detector_repository.find_by_id(sample_detector.id)
        assert retrieved.name == "updated_detector"
        assert retrieved.hyperparameters["n_estimators"] == 200

    @pytest.mark.asyncio
    async def test_find_by_id_success(self, detector_repository, sample_detector):
        """Test finding detector by ID."""
        await detector_repository.save(sample_detector)

        found = await detector_repository.find_by_id(sample_detector.id)
        assert found is not None
        assert found.id == sample_detector.id

    @pytest.mark.asyncio
    async def test_find_by_id_not_found(self, detector_repository):
        """Test finding non-existent detector by ID."""
        non_existent_id = uuid4()
        found = await detector_repository.find_by_id(non_existent_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_find_by_name_success(self, detector_repository, sample_detector):
        """Test finding detector by name."""
        await detector_repository.save(sample_detector)

        found = await detector_repository.find_by_name(sample_detector.name)
        assert found is not None
        assert found.name == sample_detector.name

    @pytest.mark.asyncio
    async def test_find_by_name_not_found(self, detector_repository):
        """Test finding non-existent detector by name."""
        found = await detector_repository.find_by_name("non_existent_detector")
        assert found is None

    @pytest.mark.asyncio
    async def test_find_by_algorithm(self, detector_repository):
        """Test finding detectors by algorithm."""
        # Create detectors with different algorithms
        detector1 = Detector(
            name="det1",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
        )
        detector2 = Detector(
            name="det2", algorithm="lof", contamination=ContaminationRate(0.1)
        )
        detector3 = Detector(
            name="det3",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
        )

        await detector_repository.save(detector1)
        await detector_repository.save(detector2)
        await detector_repository.save(detector3)

        # Find by algorithm
        isolation_detectors = await detector_repository.find_by_algorithm(
            "isolation_forest"
        )
        lof_detectors = await detector_repository.find_by_algorithm("lof")

        assert len(isolation_detectors) == 2
        assert len(lof_detectors) == 1
        assert all(d.algorithm == "isolation_forest" for d in isolation_detectors)
        assert lof_detectors[0].algorithm == "lof"

    @pytest.mark.asyncio
    async def test_list_all_detectors(self, detector_repository):
        """Test listing all detectors."""
        # Create multiple detectors
        detectors = []
        for i in range(5):
            detector = Detector(
                name=f"detector_{i}",
                algorithm="test_algorithm",
                contamination=ContaminationRate(0.1),
            )
            detectors.append(detector)
            await detector_repository.save(detector)

        # List all
        all_detectors = await detector_repository.list_all()
        assert len(all_detectors) == 5

        # Verify all detectors are present
        detector_names = {d.name for d in all_detectors}
        expected_names = {f"detector_{i}" for i in range(5)}
        assert detector_names == expected_names

    @pytest.mark.asyncio
    async def test_delete_detector_success(self, detector_repository, sample_detector):
        """Test successful detector deletion."""
        await detector_repository.save(sample_detector)

        # Verify detector exists
        found = await detector_repository.find_by_id(sample_detector.id)
        assert found is not None

        # Delete detector
        result = await detector_repository.delete(sample_detector.id)
        assert result is True

        # Verify detector is gone
        found = await detector_repository.find_by_id(sample_detector.id)
        assert found is None

    @pytest.mark.asyncio
    async def test_delete_detector_not_found(self, detector_repository):
        """Test deleting non-existent detector."""
        non_existent_id = uuid4()
        result = await detector_repository.delete(non_existent_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_detector(self, detector_repository, sample_detector):
        """Test checking detector existence."""
        # Should not exist initially
        exists = await detector_repository.exists(sample_detector.id)
        assert exists is False

        # Save detector
        await detector_repository.save(sample_detector)

        # Should exist now
        exists = await detector_repository.exists(sample_detector.id)
        assert exists is True

    @pytest.mark.asyncio
    async def test_detector_json_serialization(self, detector_repository):
        """Test JSON serialization of complex detector fields."""
        complex_detector = Detector(
            name="complex_detector",
            algorithm="test",
            contamination=ContaminationRate(0.1),
            hyperparameters={
                "nested": {"param": "value"},
                "list": [1, 2, 3],
                "float": 3.14159,
            },
            metadata={
                "complex_meta": {"nested": True},
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        await detector_repository.save(complex_detector)
        retrieved = await detector_repository.find_by_id(complex_detector.id)

        assert retrieved.hyperparameters["nested"]["param"] == "value"
        assert retrieved.hyperparameters["list"] == [1, 2, 3]
        assert retrieved.metadata["complex_meta"]["nested"] is True

    @pytest.mark.asyncio
    async def test_repository_error_handling(self, detector_repository):
        """Test repository error handling."""
        # Mock database error
        with patch.object(
            detector_repository.db_connection, "get_session"
        ) as mock_session:
            mock_session.side_effect = Exception("Database connection failed")

            detector = Detector(
                name="test", algorithm="test", contamination=ContaminationRate(0.1)
            )

            with pytest.raises(RepositoryError):
                await detector_repository.save(detector)


class TestDatabaseDatasetRepository:
    """Comprehensive tests for database dataset repository."""

    @pytest.fixture
    def db_connection(self):
        """Create test database connection."""
        connection = DatabaseConnection("sqlite:///:memory:")
        connection.create_tables()
        return connection

    @pytest.fixture
    def dataset_repository(self, db_connection):
        """Create dataset repository for testing."""
        return DatabaseDatasetRepository(db_connection)

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        features = np.random.RandomState(42).normal(0, 1, (100, 5))
        targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])

        return Dataset(
            name="test_dataset",
            features=features,
            targets=targets,
            feature_names=[
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
                "feature_5",
            ],
            metadata={"source": "test", "version": "1.0"},
        )

    @pytest.mark.asyncio
    async def test_save_dataset_success(self, dataset_repository, sample_dataset):
        """Test successful dataset saving."""
        await dataset_repository.save(sample_dataset)

        # Verify dataset was saved
        retrieved = await dataset_repository.find_by_id(sample_dataset.id)
        assert retrieved is not None
        assert retrieved.name == sample_dataset.name
        assert retrieved.n_samples == sample_dataset.n_samples
        assert retrieved.n_features == sample_dataset.n_features
        np.testing.assert_array_equal(retrieved.features, sample_dataset.features)
        np.testing.assert_array_equal(retrieved.targets, sample_dataset.targets)

    @pytest.mark.asyncio
    async def test_save_dataset_without_targets(self, dataset_repository):
        """Test saving dataset without target labels."""
        features = np.random.random((50, 3))
        dataset = Dataset(name="unlabeled_dataset", features=features)

        await dataset_repository.save(dataset)

        retrieved = await dataset_repository.find_by_id(dataset.id)
        assert retrieved is not None
        assert retrieved.targets is None
        assert retrieved.target_column is None

    @pytest.mark.asyncio
    async def test_find_by_name_success(self, dataset_repository, sample_dataset):
        """Test finding dataset by name."""
        await dataset_repository.save(sample_dataset)

        found = await dataset_repository.find_by_name(sample_dataset.name)
        assert found is not None
        assert found.name == sample_dataset.name

    @pytest.mark.asyncio
    async def test_list_all_datasets(self, dataset_repository):
        """Test listing all datasets."""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            features = np.random.random((10, 2))
            dataset = Dataset(name=f"dataset_{i}", features=features)
            datasets.append(dataset)
            await dataset_repository.save(dataset)

        all_datasets = await dataset_repository.list_all()
        assert len(all_datasets) == 3

        dataset_names = {d.name for d in all_datasets}
        expected_names = {f"dataset_{i}" for i in range(3)}
        assert dataset_names == expected_names

    @pytest.mark.asyncio
    async def test_delete_dataset_success(self, dataset_repository, sample_dataset):
        """Test successful dataset deletion."""
        await dataset_repository.save(sample_dataset)

        result = await dataset_repository.delete(sample_dataset.id)
        assert result is True

        found = await dataset_repository.find_by_id(sample_dataset.id)
        assert found is None

    @pytest.mark.asyncio
    async def test_dataset_large_data_handling(self, dataset_repository):
        """Test handling large datasets."""
        # Create larger dataset
        features = np.random.random((1000, 20))
        targets = np.random.choice([0, 1], 1000)
        large_dataset = Dataset(
            name="large_dataset", features=features, targets=targets
        )

        await dataset_repository.save(large_dataset)
        retrieved = await dataset_repository.find_by_id(large_dataset.id)

        assert retrieved is not None
        assert retrieved.n_samples == 1000
        assert retrieved.n_features == 20
        np.testing.assert_array_equal(retrieved.features, features)
        np.testing.assert_array_equal(retrieved.targets, targets)


class TestDatabaseDetectionResultRepository:
    """Comprehensive tests for database detection result repository."""

    @pytest.fixture
    def db_connection(self):
        """Create test database connection."""
        connection = DatabaseConnection("sqlite:///:memory:")
        connection.create_tables()
        return connection

    @pytest.fixture
    def result_repository(self, db_connection):
        """Create detection result repository for testing."""
        return DatabaseDetectionResultRepository(db_connection)

    @pytest.fixture
    def sample_detection_result(self):
        """Create sample detection result for testing."""
        detector = Detector(
            name="test_detector", algorithm="test", contamination=ContaminationRate(0.1)
        )
        features = np.random.random((100, 5))
        dataset = Dataset(name="test_dataset", features=features)

        scores = [AnomalyScore(v) for v in np.random.random(100)]
        labels = np.random.choice([0, 1], 100)

        anomalies = []
        anomaly_indices = np.where(labels == 1)[0][:5]  # First 5 anomalies
        for idx in anomaly_indices:
            anomaly = Anomaly(
                score=scores[idx],
                index=int(idx),
                features=features[idx],
                confidence=ConfidenceInterval(0.8, 0.95, 0.9),
            )
            anomalies.append(anomaly)

        return DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.7,
            metadata={"processing_time": 1.23, "algorithm_version": "1.0"},
        )

    @pytest.mark.asyncio
    async def test_save_detection_result_success(
        self, result_repository, sample_detection_result
    ):
        """Test successful detection result saving."""
        await result_repository.save(sample_detection_result)

        # Verify result was saved
        retrieved = await result_repository.find_by_id(sample_detection_result.id)
        assert retrieved is not None
        assert retrieved.detector_id == sample_detection_result.detector_id
        assert retrieved.dataset_id == sample_detection_result.dataset_id
        assert retrieved.threshold == sample_detection_result.threshold
        assert len(retrieved.anomalies) == len(sample_detection_result.anomalies)

    @pytest.mark.asyncio
    async def test_find_by_detector(self, result_repository):
        """Test finding results by detector ID."""
        detector_id = uuid4()

        # Create multiple results for same detector
        for i in range(3):
            detector = Detector(
                name=f"detector_{i}",
                algorithm="test",
                contamination=ContaminationRate(0.1),
            )
            detector.id = detector_id  # Use same detector ID

            features = np.random.random((10, 2))
            dataset = Dataset(name=f"dataset_{i}", features=features)

            result = DetectionResult(
                detector=detector,
                dataset=dataset,
                anomalies=[],
                scores=[AnomalyScore(0.5)] * 10,
                labels=np.array([0] * 10),
                threshold=0.5,
            )
            await result_repository.save(result)

        # Find by detector
        results = await result_repository.find_by_detector(detector_id)
        assert len(results) == 3
        assert all(r.detector_id == detector_id for r in results)

    @pytest.mark.asyncio
    async def test_find_by_dataset(self, result_repository):
        """Test finding results by dataset ID."""
        dataset_id = uuid4()

        # Create multiple results for same dataset
        for i in range(2):
            detector = Detector(
                name=f"detector_{i}",
                algorithm="test",
                contamination=ContaminationRate(0.1),
            )
            features = np.random.random((10, 2))
            dataset = Dataset(name=f"dataset_{i}", features=features)
            dataset.id = dataset_id  # Use same dataset ID

            result = DetectionResult(
                detector=detector,
                dataset=dataset,
                anomalies=[],
                scores=[AnomalyScore(0.5)] * 10,
                labels=np.array([0] * 10),
                threshold=0.5,
            )
            await result_repository.save(result)

        # Find by dataset
        results = await result_repository.find_by_dataset(dataset_id)
        assert len(results) == 2
        assert all(r.dataset_id == dataset_id for r in results)

    @pytest.mark.asyncio
    async def test_find_recent_results(self, result_repository):
        """Test finding recent detection results."""
        # Create results with different timestamps
        for i in range(5):
            detector = Detector(
                name=f"detector_{i}",
                algorithm="test",
                contamination=ContaminationRate(0.1),
            )
            features = np.random.random((10, 2))
            dataset = Dataset(name=f"dataset_{i}", features=features)

            result = DetectionResult(
                detector=detector,
                dataset=dataset,
                anomalies=[],
                scores=[AnomalyScore(0.5)] * 10,
                labels=np.array([0] * 10),
                threshold=0.5,
            )
            await result_repository.save(result)

            # Simulate time passing
            await asyncio.sleep(0.01)

        # Find recent results
        recent = await result_repository.find_recent(limit=3)
        assert len(recent) == 3

        # Should be ordered by timestamp (newest first)
        for i in range(len(recent) - 1):
            assert recent[i].timestamp >= recent[i + 1].timestamp

    @pytest.mark.asyncio
    async def test_detection_result_complex_serialization(self, result_repository):
        """Test serialization of complex detection result data."""
        detector = Detector(
            name="test", algorithm="test", contamination=ContaminationRate(0.1)
        )
        features = np.random.random((50, 3))
        dataset = Dataset(name="test", features=features)

        # Create complex anomalies with various data types
        scores = [AnomalyScore(v) for v in np.random.beta(2, 8, 50)]
        labels = np.random.choice([0, 1], 50)

        anomalies = []
        for i in range(5):
            anomaly = Anomaly(
                score=scores[i],
                index=i,
                features=features[i],
                confidence=ConfidenceInterval(0.8, 0.95, 0.9),
                explanation={"feature_importance": {"f1": 0.3, "f2": 0.7}},
            )
            anomalies.append(anomaly)

        result = DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.8,
            metadata={
                "complex_data": {"nested": {"value": 123}},
                "array_data": [1, 2, 3, 4, 5],
            },
        )

        await result_repository.save(result)
        retrieved = await result_repository.find_by_id(result.id)

        assert retrieved is not None
        assert len(retrieved.anomalies) == 5
        assert retrieved.metadata["complex_data"]["nested"]["value"] == 123
        assert retrieved.metadata["array_data"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_delete_detection_result(
        self, result_repository, sample_detection_result
    ):
        """Test deletion of detection results."""
        await result_repository.save(sample_detection_result)

        # Verify result exists
        found = await result_repository.find_by_id(sample_detection_result.id)
        assert found is not None

        # Delete result
        result = await result_repository.delete(sample_detection_result.id)
        assert result is True

        # Verify result is gone
        found = await result_repository.find_by_id(sample_detection_result.id)
        assert found is None


class TestDatabaseModels:
    """Test database model conversions and validation."""

    def test_detector_model_from_entity(self):
        """Test converting detector entity to database model."""
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
            hyperparameters={"n_estimators": 100},
            metadata={"version": "1.0"},
        )

        model = DetectorModel.from_entity(detector)

        assert model.id == str(detector.id)
        assert model.name == detector.name
        assert model.algorithm == detector.algorithm
        assert model.contamination_rate == detector.contamination_rate.value
        assert model.hyperparameters == detector.hyperparameters
        assert model.metadata == detector.metadata

    def test_detector_model_to_entity(self):
        """Test converting database model to detector entity."""
        model = DetectorModel(
            id=str(uuid4()),
            name="test_detector",
            algorithm="isolation_forest",
            contamination_rate=0.1,
            hyperparameters={"n_estimators": 100},
            metadata={"version": "1.0"},
            is_fitted=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        entity = model.to_entity()

        assert str(entity.id) == model.id
        assert entity.name == model.name
        assert entity.algorithm == model.algorithm
        assert entity.contamination_rate.value == model.contamination_rate
        assert entity.hyperparameters == model.hyperparameters

    def test_dataset_model_from_entity(self):
        """Test converting dataset entity to database model."""
        features = np.random.random((100, 5))
        targets = np.random.choice([0, 1], 100)

        dataset = Dataset(
            name="test_dataset",
            features=features,
            targets=targets,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )

        model = DatasetModel.from_entity(dataset)

        assert model.id == str(dataset.id)
        assert model.name == dataset.name
        assert model.n_samples == dataset.n_samples
        assert model.n_features == dataset.n_features
        np.testing.assert_array_equal(
            np.frombuffer(model.features_data, dtype=np.float64).reshape(100, 5),
            features,
        )

    def test_detection_result_model_serialization(self):
        """Test detection result model data serialization."""
        detector = Detector(
            name="test", algorithm="test", contamination=ContaminationRate(0.1)
        )
        features = np.random.random((10, 2))
        dataset = Dataset(name="test", features=features)

        scores = [AnomalyScore(v) for v in np.random.random(10)]
        labels = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])

        result = DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=[],
            scores=scores,
            labels=labels,
            threshold=0.7,
        )

        model = DetectionResultModel.from_entity(result)

        assert model.detector_id == str(detector.id)
        assert model.dataset_id == str(dataset.id)
        assert model.threshold == 0.7
        assert model.n_anomalies == result.n_anomalies

        # Test round-trip conversion
        converted_result = model.to_entity()
        assert converted_result.threshold == result.threshold
        assert len(converted_result.scores) == len(result.scores)
