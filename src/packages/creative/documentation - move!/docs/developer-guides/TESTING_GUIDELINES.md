# Pynomaly Testing Guidelines

This comprehensive guide covers testing standards, strategies, and best practices for the Pynomaly project. Our goal is to maintain high-quality, reliable code through comprehensive testing.

## 📋 Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Pyramid](#test-pyramid)
- [Test Organization](#test-organization)
- [Writing Unit Tests](#writing-unit-tests)
- [Integration Testing](#integration-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Test Data Management](#test-data-management)
- [Performance Testing](#performance-testing)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)

## 🎯 Testing Philosophy

### Core Principles

1. **Test-Driven Development (TDD)**: Write tests first, then implement
2. **Fail Fast**: Tests should fail quickly and provide clear error messages
3. **Independent Tests**: Tests should not depend on each other
4. **Deterministic**: Tests should produce consistent results
5. **Comprehensive**: Test happy paths, edge cases, and error conditions

### Quality Metrics

- **Line Coverage**: Target 85%+ overall, 95%+ for domain layer
- **Function Coverage**: Target 90%+ overall
- **Branch Coverage**: Target 80%+ for critical paths
- **Test Execution Time**: Unit tests <1s each, integration tests <10s each

## 🏗️ Test Pyramid

Our testing strategy follows the test pyramid pattern:

```
        /\
       /  \
      / E2E \     10% - End-to-End (Slow, High Value)
     /      \
    /  INT   \    20% - Integration (Medium Speed)
   /          \
  /    UNIT    \  70% - Unit Tests (Fast, Low Level)
 /______________\
```

### Unit Tests (70%)
- **Purpose**: Test individual components in isolation
- **Speed**: Very fast (<1 second per test)
- **Scope**: Single functions, methods, or classes
- **Dependencies**: Mocked or stubbed

### Integration Tests (20%)
- **Purpose**: Test component interactions
- **Speed**: Medium (1-10 seconds per test)
- **Scope**: Multiple components working together
- **Dependencies**: Real dependencies where practical

### End-to-End Tests (10%)
- **Purpose**: Test complete user workflows
- **Speed**: Slow (10-60 seconds per test)
- **Scope**: Full application stack
- **Dependencies**: Real external systems

## 📁 Test Organization

### Directory Structure

```
tests/
├── unit/                       # Unit tests (fast, isolated)
│   ├── domain/                # Domain layer tests
│   │   ├── entities/          # Entity tests
│   │   ├── value_objects/     # Value object tests
│   │   └── services/          # Domain service tests
│   ├── application/           # Application layer tests
│   │   ├── use_cases/         # Use case tests
│   │   ├── services/          # Application service tests
│   │   └── dto/               # DTO tests
│   ├── infrastructure/        # Infrastructure tests
│   │   ├── adapters/          # Adapter tests (with mocks)
│   │   └── repositories/      # Repository tests
│   └── presentation/          # Presentation layer tests
│       ├── api/               # API controller tests
│       ├── cli/               # CLI command tests
│       └── web/               # Web interface tests
├── integration/               # Integration tests
│   ├── api/                   # API endpoint tests
│   ├── database/              # Database integration
│   ├── ml_algorithms/         # ML library integration
│   └── external_services/     # Third-party service integration
├── e2e/                       # End-to-end tests
│   ├── cli/                   # CLI workflow tests
│   ├── api/                   # API workflow tests
│   └── web/                   # Web interface tests
├── performance/               # Performance and benchmarks
│   ├── benchmarks/            # Performance benchmarks
│   └── load_tests/            # Load testing
├── fixtures/                  # Shared test data
│   ├── datasets/              # Sample datasets
│   ├── models/                # Pre-trained models
│   └── configs/               # Test configurations
└── helpers/                   # Test utilities
    ├── factories.py           # Test data factories
    ├── assertions.py          # Custom assertions
    └── mocks.py               # Mock objects
```

### Naming Conventions

```python
# Test files: test_*.py
test_anomaly_detector.py
test_detection_service.py
test_sklearn_adapter.py

# Test classes: Test + ClassName
class TestAnomalyDetector:
class TestDetectionService:
class TestSklearnAdapter:

# Test methods: test_method_with_scenario
def test_detector_fit_with_valid_data_succeeds():
def test_detector_predict_without_fit_raises_error():
def test_detection_service_returns_sorted_anomalies():
```

## 🧪 Writing Unit Tests

### Basic Test Structure

```python
# tests/unit/domain/entities/test_anomaly.py
import pytest
from datetime import datetime
from uuid import uuid4

from pynomaly.domain.entities import Anomaly
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.domain.exceptions import ValidationError


class TestAnomaly:
    """Test suite for Anomaly entity."""
    
    def test_anomaly_creation_with_valid_data_succeeds(self):
        """Test that anomaly can be created with valid data."""
        # Arrange
        anomaly_id = uuid4()
        score = AnomalyScore(0.85)
        detected_at = datetime.utcnow()
        data_point = {"feature1": 1.0, "feature2": 2.0}
        
        # Act
        anomaly = Anomaly(
            id=anomaly_id,
            score=score,
            detected_at=detected_at,
            data_point=data_point
        )
        
        # Assert
        assert anomaly.id == anomaly_id
        assert anomaly.score == score
        assert anomaly.detected_at == detected_at
        assert anomaly.data_point == data_point
    
    def test_anomaly_with_invalid_score_raises_validation_error(self):
        """Test that creating anomaly with invalid score raises ValidationError."""
        # Arrange
        invalid_score = AnomalyScore(1.5)  # Invalid: > 1.0
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Score must be between 0 and 1"):
            Anomaly(
                id=uuid4(),
                score=invalid_score,
                detected_at=datetime.utcnow(),
                data_point={}
            )
    
    @pytest.mark.parametrize("score_value,expected_severity", [
        (0.1, "low"),
        (0.5, "medium"),
        (0.8, "high"),
        (0.95, "critical"),
    ])
    def test_anomaly_severity_classification(self, score_value, expected_severity):
        """Test that anomaly severity is classified correctly."""
        # Arrange
        score = AnomalyScore(score_value)
        anomaly = self._create_anomaly(score=score)
        
        # Act
        severity = anomaly.get_severity()
        
        # Assert
        assert severity == expected_severity
    
    def _create_anomaly(self, **kwargs) -> Anomaly:
        """Helper method to create anomaly with defaults."""
        defaults = {
            "id": uuid4(),
            "score": AnomalyScore(0.5),
            "detected_at": datetime.utcnow(),
            "data_point": {"feature1": 1.0}
        }
        defaults.update(kwargs)
        return Anomaly(**defaults)
```

### Testing Domain Services

```python
# tests/unit/domain/services/test_ensemble_service.py
import pytest
import numpy as np
from unittest.mock import Mock

from pynomaly.domain.services import EnsembleService
from pynomaly.domain.protocols import DetectorProtocol
from pynomaly.domain.value_objects import AnomalyScore


class TestEnsembleService:
    """Test suite for EnsembleService."""
    
    def test_ensemble_with_single_detector_returns_original_scores(self):
        """Test that ensemble with single detector returns original scores."""
        # Arrange
        detector = Mock(spec=DetectorProtocol)
        scores = np.array([0.1, 0.5, 0.9])
        detector.score_samples.return_value = scores
        
        service = EnsembleService([detector])
        data = np.random.randn(3, 2)
        
        # Act
        result = service.score_samples(data)
        
        # Assert
        np.testing.assert_array_equal(result, scores)
        detector.score_samples.assert_called_once_with(data)
    
    def test_ensemble_with_multiple_detectors_averages_scores(self):
        """Test that ensemble averages scores from multiple detectors."""
        # Arrange
        detector1 = Mock(spec=DetectorProtocol)
        detector2 = Mock(spec=DetectorProtocol)
        
        detector1.score_samples.return_value = np.array([0.2, 0.4, 0.8])
        detector2.score_samples.return_value = np.array([0.4, 0.6, 0.6])
        
        service = EnsembleService([detector1, detector2])
        data = np.random.randn(3, 2)
        
        # Act
        result = service.score_samples(data)
        
        # Assert
        expected = np.array([0.3, 0.5, 0.7])  # Average of the two
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_ensemble_with_empty_detectors_raises_error(self):
        """Test that creating ensemble with no detectors raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="At least one detector required"):
            EnsembleService([])
```

### Testing with Fixtures

```python
# tests/fixtures/datasets.py
import pytest
import numpy as np
import pandas as pd
from pynomaly.domain.entities import Dataset


@pytest.fixture
def small_dataset():
    """Create a small dataset for testing."""
    data = np.random.randn(10, 3)
    return Dataset.from_array(data, name="small_test_dataset")


@pytest.fixture
def large_dataset():
    """Create a large dataset for performance testing."""
    data = np.random.randn(10000, 5)
    return Dataset.from_array(data, name="large_test_dataset")


@pytest.fixture
def anomalous_dataset():
    """Create a dataset with known anomalies."""
    # Normal data
    normal_data = np.random.normal(0, 1, (90, 2))
    
    # Outliers
    outliers = np.random.normal(5, 1, (10, 2))
    
    # Combine
    data = np.vstack([normal_data, outliers])
    return Dataset.from_array(data, name="anomalous_dataset")


@pytest.fixture
def pandas_dataset():
    """Create a pandas-based dataset."""
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
    })
    return Dataset.from_dataframe(df, name="pandas_dataset")
```

### Using Fixtures in Tests

```python
# tests/unit/application/use_cases/test_detect_anomalies.py
import pytest
from unittest.mock import Mock

from pynomaly.application.use_cases import DetectAnomaliesUseCase
from pynomaly.application.dto import DetectionRequest, DetectionResponse


class TestDetectAnomaliesUseCase:
    """Test suite for DetectAnomaliesUseCase."""
    
    def test_detect_anomalies_with_valid_data_returns_response(
        self, 
        small_dataset,
        mock_detector_service
    ):
        """Test successful anomaly detection."""
        # Arrange
        request = DetectionRequest(
            dataset_id=small_dataset.id,
            detector_id="test_detector",
            threshold=0.5
        )
        
        use_case = DetectAnomaliesUseCase(mock_detector_service)
        
        # Act
        response = use_case.execute(request)
        
        # Assert
        assert isinstance(response, DetectionResponse)
        assert response.success is True
        assert len(response.anomalies) >= 0
        mock_detector_service.detect.assert_called_once()
    
    @pytest.fixture
    def mock_detector_service(self):
        """Mock detector service for testing."""
        service = Mock()
        service.detect.return_value = []
        return service
```

## 🔗 Integration Testing

### Database Integration Tests

```python
# tests/integration/repositories/test_detector_repository.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pynomaly.infrastructure.persistence import DetectorRepository
from pynomaly.domain.entities import Detector


class TestDetectorRepository:
    """Integration tests for DetectorRepository."""
    
    @pytest.fixture(scope="class")
    def database_session(self):
        """Create test database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        yield session
        
        session.close()
    
    def test_save_and_retrieve_detector(self, database_session):
        """Test that detector can be saved and retrieved."""
        # Arrange
        repository = DetectorRepository(database_session)
        detector = self._create_test_detector()
        
        # Act
        repository.save(detector)
        retrieved = repository.find_by_id(detector.id)
        
        # Assert
        assert retrieved is not None
        assert retrieved.id == detector.id
        assert retrieved.name == detector.name
        assert retrieved.algorithm_name == detector.algorithm_name
    
    def test_find_by_name_returns_correct_detector(self, database_session):
        """Test finding detector by name."""
        # Arrange
        repository = DetectorRepository(database_session)
        detector = self._create_test_detector(name="unique_detector_name")
        repository.save(detector)
        
        # Act
        found = repository.find_by_name("unique_detector_name")
        
        # Assert
        assert found is not None
        assert found.name == "unique_detector_name"
    
    def _create_test_detector(self, **kwargs):
        """Helper to create test detector."""
        defaults = {
            "name": "test_detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"contamination": 0.1}
        }
        defaults.update(kwargs)
        return Detector(**defaults)
```

### API Integration Tests

```python
# tests/integration/api/test_detection_endpoints.py
import pytest
import httpx
from fastapi.testclient import TestClient

from pynomaly.presentation.api.app import create_app


class TestDetectionEndpoints:
    """Integration tests for detection API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_create_detector_endpoint(self, client):
        """Test detector creation endpoint."""
        # Arrange
        detector_data = {
            "name": "test_detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"contamination": 0.1}
        }
        
        # Act
        response = client.post("/api/v1/detectors", json=detector_data)
        
        # Assert
        assert response.status_code == 201
        
        created_detector = response.json()
        assert created_detector["name"] == "test_detector"
        assert created_detector["algorithm_name"] == "IsolationForest"
        assert "id" in created_detector
    
    def test_detect_anomalies_endpoint_with_valid_data(self, client):
        """Test anomaly detection endpoint."""
        # Arrange
        # First create a detector
        detector_response = client.post("/api/v1/detectors", json={
            "name": "test_detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"contamination": 0.1}
        })
        detector_id = detector_response.json()["id"]
        
        # Upload dataset
        dataset_response = client.post("/api/v1/datasets", json={
            "name": "test_dataset",
            "data": [[1, 2], [2, 3], [10, 15]]  # Last point is outlier
        })
        dataset_id = dataset_response.json()["id"]
        
        # Act
        detection_response = client.post("/api/v1/detect", json={
            "detector_id": detector_id,
            "dataset_id": dataset_id
        })
        
        # Assert
        assert detection_response.status_code == 200
        
        result = detection_response.json()
        assert "anomalies" in result
        assert "scores" in result
        assert len(result["anomalies"]) >= 0
    
    def test_detect_with_nonexistent_detector_returns_404(self, client):
        """Test detection with invalid detector ID."""
        # Act
        response = client.post("/api/v1/detect", json={
            "detector_id": "nonexistent",
            "dataset_id": "also_nonexistent"
        })
        
        # Assert
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
```

## 🌐 End-to-End Testing

### CLI Workflow Tests

```python
# tests/e2e/cli/test_detection_workflow.py
import pytest
import subprocess
import tempfile
import json
from pathlib import Path


class TestCLIDetectionWorkflow:
    """End-to-end tests for CLI detection workflow."""
    
    def test_complete_detection_workflow_via_cli(self, sample_csv_file):
        """Test complete detection workflow through CLI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Create detector
            result = subprocess.run([
                "python", "-m", "pynomaly.cli", 
                "detector", "create",
                "--name", "test_detector",
                "--algorithm", "IsolationForest",
                "--contamination", "0.1"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            detector_output = json.loads(result.stdout)
            detector_id = detector_output["id"]
            
            # Step 2: Load dataset
            result = subprocess.run([
                "python", "-m", "pynomaly.cli",
                "dataset", "load",
                "--file", str(sample_csv_file),
                "--name", "test_dataset"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            dataset_output = json.loads(result.stdout)
            dataset_id = dataset_output["id"]
            
            # Step 3: Run detection
            output_file = temp_path / "results.json"
            result = subprocess.run([
                "python", "-m", "pynomaly.cli",
                "detect",
                "--detector", detector_id,
                "--dataset", dataset_id,
                "--output", str(output_file)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            
            # Step 4: Verify results
            assert output_file.exists()
            with open(output_file) as f:
                results = json.load(f)
            
            assert "anomalies" in results
            assert "statistics" in results
            assert isinstance(results["anomalies"], list)
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,feature3\n")
            # Normal data
            for i in range(100):
                f.write(f"{i},{i*2},{i*3}\n")
            # Outliers
            f.write("1000,2000,3000\n")
            f.write("-1000,-2000,-3000\n")
            
            temp_file = Path(f.name)
        
        yield temp_file
        
        # Cleanup
        temp_file.unlink()
```

### Web Interface Tests

```python
# tests/e2e/web/test_web_interface.py
import pytest
from playwright.sync_api import sync_playwright


class TestWebInterface:
    """End-to-end tests for web interface."""
    
    @pytest.fixture(scope="class")
    def browser(self):
        """Set up browser for testing."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            yield browser
            browser.close()
    
    @pytest.fixture
    def page(self, browser):
        """Create new page for each test."""
        page = browser.new_page()
        yield page
        page.close()
    
    def test_user_can_create_detector_through_web_interface(self, page):
        """Test detector creation through web interface."""
        # Navigate to application
        page.goto("http://localhost:8000/app")
        
        # Click on "Create Detector" button
        page.click("text=Create Detector")
        
        # Fill in detector form
        page.fill("#detector-name", "Web Test Detector")
        page.select_option("#algorithm-select", "IsolationForest")
        page.fill("#contamination-rate", "0.1")
        
        # Submit form
        page.click("button[type=submit]")
        
        # Verify detector was created
        page.wait_for_selector("text=Detector created successfully")
        
        # Check that detector appears in list
        page.click("text=View Detectors")
        page.wait_for_selector("text=Web Test Detector")
    
    def test_user_can_upload_and_analyze_dataset(self, page, sample_dataset_file):
        """Test dataset upload and analysis workflow."""
        # Navigate to application
        page.goto("http://localhost:8000/app")
        
        # Upload dataset
        page.click("text=Upload Dataset")
        page.set_input_files("#dataset-file", str(sample_dataset_file))
        page.fill("#dataset-name", "Web Test Dataset")
        page.click("button[type=submit]")
        
        # Verify upload success
        page.wait_for_selector("text=Dataset uploaded successfully")
        
        # Run analysis
        page.click("text=Analyze Dataset")
        page.wait_for_selector(".analysis-results")
        
        # Verify analysis results are displayed
        assert page.locator(".analysis-results").is_visible()
        assert page.locator("text=Data points").is_visible()
        assert page.locator("text=Features").is_visible()
```

## 📊 Test Data Management

### Test Data Factories

```python
# tests/helpers/factories.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Optional

from pynomaly.domain.entities import Detector, Dataset, Anomaly
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate


class DetectorFactory:
    """Factory for creating test detectors."""
    
    @staticmethod
    def create(
        name: Optional[str] = None,
        algorithm_name: str = "IsolationForest",
        hyperparameters: Optional[dict] = None,
        is_fitted: bool = False
    ) -> Detector:
        """Create a test detector."""
        return Detector(
            id=uuid4(),
            name=name or f"test_detector_{uuid4().hex[:8]}",
            algorithm_name=algorithm_name,
            hyperparameters=hyperparameters or {"contamination": 0.1},
            created_at=datetime.utcnow(),
            is_fitted=is_fitted
        )
    
    @staticmethod
    def create_fitted_isolation_forest() -> Detector:
        """Create a fitted Isolation Forest detector."""
        return DetectorFactory.create(
            algorithm_name="IsolationForest",
            is_fitted=True
        )


class DatasetFactory:
    """Factory for creating test datasets."""
    
    @staticmethod
    def create_normal_dataset(
        n_samples: int = 100,
        n_features: int = 3,
        random_state: Optional[int] = 42
    ) -> Dataset:
        """Create a dataset with normal data."""
        if random_state:
            np.random.seed(random_state)
        
        data = np.random.normal(0, 1, (n_samples, n_features))
        return Dataset.from_array(
            data, 
            name=f"normal_dataset_{n_samples}x{n_features}"
        )
    
    @staticmethod
    def create_anomalous_dataset(
        n_normal: int = 90,
        n_anomalies: int = 10,
        n_features: int = 3,
        anomaly_factor: float = 3.0,
        random_state: Optional[int] = 42
    ) -> Dataset:
        """Create a dataset with known anomalies."""
        if random_state:
            np.random.seed(random_state)
        
        # Normal data
        normal_data = np.random.normal(0, 1, (n_normal, n_features))
        
        # Anomalous data (shifted by anomaly_factor)
        anomalous_data = np.random.normal(
            anomaly_factor, 1, (n_anomalies, n_features)
        )
        
        # Combine
        data = np.vstack([normal_data, anomalous_data])
        
        return Dataset.from_array(
            data,
            name=f"anomalous_dataset_{n_normal + n_anomalies}x{n_features}"
        )
    
    @staticmethod
    def create_time_series_dataset(
        n_points: int = 1000,
        seasonality: int = 24,
        anomaly_points: Optional[list] = None
    ) -> Dataset:
        """Create a time series dataset with optional anomalies."""
        np.random.seed(42)
        
        # Generate time series with trend and seasonality
        t = np.arange(n_points)
        trend = 0.001 * t
        seasonal = 2 * np.sin(2 * np.pi * t / seasonality)
        noise = np.random.normal(0, 0.5, n_points)
        
        values = trend + seasonal + noise
        
        # Add anomalies at specified points
        if anomaly_points:
            for point in anomaly_points:
                if 0 <= point < n_points:
                    values[point] += np.random.normal(10, 2)
        
        # Create DataFrame with timestamp
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='H'),
            'value': values
        })
        
        return Dataset.from_dataframe(df, name="time_series_dataset")


class AnomalyFactory:
    """Factory for creating test anomalies."""
    
    @staticmethod
    def create(
        score: float = 0.8,
        data_point: Optional[dict] = None,
        detected_at: Optional[datetime] = None
    ) -> Anomaly:
        """Create a test anomaly."""
        return Anomaly(
            id=uuid4(),
            score=AnomalyScore(score),
            detected_at=detected_at or datetime.utcnow(),
            data_point=data_point or {"feature1": 1.0, "feature2": 2.0}
        )
    
    @staticmethod
    def create_batch(
        count: int = 10,
        score_range: tuple = (0.5, 1.0)
    ) -> list[Anomaly]:
        """Create a batch of test anomalies."""
        anomalies = []
        for i in range(count):
            score = np.random.uniform(*score_range)
            data_point = {
                "feature1": np.random.randn(),
                "feature2": np.random.randn(),
                "index": i
            }
            anomalies.append(AnomalyFactory.create(score, data_point))
        
        return anomalies
```

### Mock Objects

```python
# tests/helpers/mocks.py
from unittest.mock import Mock, MagicMock
import numpy as np

from pynomaly.domain.protocols import (
    DetectorProtocol,
    DatasetRepositoryProtocol,
    DetectorRepositoryProtocol
)


class MockDetector(Mock):
    """Mock detector for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(spec=DetectorProtocol, *args, **kwargs)
        self.is_fitted = False
        self._scores = None
    
    def fit(self, data):
        """Mock fit method."""
        self.is_fitted = True
        return self
    
    def predict(self, data):
        """Mock predict method."""
        if not self.is_fitted:
            raise ValueError("Detector not fitted")
        
        n_samples = len(data) if hasattr(data, '__len__') else 100
        # Return some outliers
        predictions = np.zeros(n_samples)
        predictions[-int(n_samples * 0.1):] = 1  # Last 10% are anomalies
        return predictions
    
    def score_samples(self, data):
        """Mock score_samples method."""
        if not self.is_fitted:
            raise ValueError("Detector not fitted")
        
        n_samples = len(data) if hasattr(data, '__len__') else 100
        # Generate realistic anomaly scores
        scores = np.random.beta(2, 8, n_samples)  # Most scores low, few high
        # Make last 10% clearly anomalous
        scores[-int(n_samples * 0.1):] = np.random.uniform(0.8, 1.0, int(n_samples * 0.1))
        return scores


class MockRepository(Mock):
    """Mock repository for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._storage = {}
    
    def save(self, entity):
        """Mock save method."""
        self._storage[entity.id] = entity
        return entity
    
    def find_by_id(self, entity_id):
        """Mock find_by_id method."""
        return self._storage.get(entity_id)
    
    def find_all(self):
        """Mock find_all method."""
        return list(self._storage.values())
    
    def delete(self, entity_id):
        """Mock delete method."""
        return self._storage.pop(entity_id, None)


def create_mock_detector_service():
    """Create a mock detector service."""
    service = Mock()
    service.create_detector.return_value = MockDetector()
    service.fit_detector.return_value = True
    service.detect_anomalies.return_value = []
    return service
```

## ⚡ Performance Testing

### Benchmark Tests

```python
# tests/performance/test_detection_performance.py
import pytest
import time
import numpy as np
from memory_profiler import profile

from pynomaly.infrastructure.adapters import SklearnAdapter
from tests.helpers.factories import DatasetFactory


class TestDetectionPerformance:
    """Performance tests for anomaly detection."""
    
    @pytest.mark.benchmark
    def test_isolation_forest_performance_on_large_dataset(self, benchmark):
        """Benchmark Isolation Forest on large dataset."""
        # Arrange
        dataset = DatasetFactory.create_normal_dataset(
            n_samples=10000, 
            n_features=20
        )
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            contamination=0.1,
            n_estimators=100,
            random_state=42
        )
        
        # Act & Assert
        def run_detection():
            detector.fit(dataset.data)
            return detector.score_samples(dataset.data)
        
        result = benchmark(run_detection)
        
        # Verify reasonable performance
        assert len(result) == 10000
        assert benchmark.stats['mean'] < 5.0  # Should complete in <5 seconds
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000, 50000])
    def test_scalability_with_dataset_size(self, n_samples):
        """Test how performance scales with dataset size."""
        # Arrange
        dataset = DatasetFactory.create_normal_dataset(
            n_samples=n_samples,
            n_features=10
        )
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            contamination=0.1,
            n_estimators=50
        )
        
        # Act
        start_time = time.time()
        detector.fit(dataset.data)
        scores = detector.score_samples(dataset.data)
        end_time = time.time()
        
        # Assert
        execution_time = end_time - start_time
        time_per_sample = execution_time / n_samples
        
        # Should process at least 1000 samples per second
        assert time_per_sample < 0.001, f"Too slow: {time_per_sample:.6f}s per sample"
        assert len(scores) == n_samples
    
    @profile
    def test_memory_usage_during_detection(self):
        """Test memory usage during detection."""
        # This test uses memory_profiler to track memory usage
        dataset = DatasetFactory.create_normal_dataset(
            n_samples=50000,
            n_features=20
        )
        
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            contamination=0.1
        )
        
        # Fit and detect
        detector.fit(dataset.data)
        scores = detector.score_samples(dataset.data)
        
        # Memory usage is tracked by the @profile decorator
        assert len(scores) == 50000
```

### Load Testing

```python
# tests/performance/test_api_load.py
import asyncio
import aiohttp
import pytest
import time
from concurrent.futures import ThreadPoolExecutor


class TestAPILoad:
    """Load tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_concurrent_detection_requests(self):
        """Test API under concurrent load."""
        base_url = "http://localhost:8000"
        
        async def make_detection_request(session, request_id):
            """Make a single detection request."""
            async with session.post(
                f"{base_url}/api/v1/detect",
                json={
                    "detector_id": "test_detector",
                    "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                }
            ) as response:
                return {
                    "request_id": request_id,
                    "status": response.status,
                    "time": time.time()
                }
        
        # Create concurrent requests
        num_requests = 50
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                make_detection_request(session, i) 
                for i in range(num_requests)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Analyze results
        successful_requests = [
            r for r in results 
            if not isinstance(r, Exception) and r["status"] == 200
        ]
        
        # Assertions
        total_time = end_time - start_time
        success_rate = len(successful_requests) / num_requests
        
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert total_time < 30, f"Total time too high: {total_time:.2f}s"
        
        # Calculate throughput
        throughput = num_requests / total_time
        assert throughput >= 5, f"Throughput too low: {throughput:.2f} req/s"
```

## 📈 Test Coverage

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/pynomaly"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/migrations/*",
    "*/venv/*",
    "*/virtualenv/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
show_missing = true
precision = 2

[tool.coverage.html]
directory = "htmlcov"
```

### Coverage Commands

```bash
# Run tests with coverage
pytest tests/ --cov=src/pynomaly --cov-report=html --cov-report=term

# Generate coverage report
coverage run -m pytest tests/
coverage report
coverage html

# Check coverage thresholds
pytest tests/ --cov=src/pynomaly --cov-fail-under=85
```

### Coverage Analysis

```python
# scripts/analyze_coverage.py
#!/usr/bin/env python3
"""Analyze test coverage and identify gaps."""

import json
import sys
from pathlib import Path


def analyze_coverage_report(coverage_file: Path) -> dict:
    """Analyze coverage report and identify gaps."""
    with open(coverage_file) as f:
        coverage_data = json.load(f)
    
    analysis = {
        'total_statements': 0,
        'covered_statements': 0,
        'coverage_percentage': 0,
        'files_by_coverage': [],
        'low_coverage_files': [],
    }
    
    for filename, file_data in coverage_data['files'].items():
        if 'src/pynomaly' in filename:
            statements = len(file_data['executed_lines'])
            missing = len(file_data['missing_lines'])
            total = statements + missing
            
            if total > 0:
                coverage = (statements / total) * 100
                
                file_info = {
                    'file': filename,
                    'coverage': coverage,
                    'statements': statements,
                    'missing': missing,
                    'total': total
                }
                
                analysis['files_by_coverage'].append(file_info)
                
                if coverage < 80:  # Flag low coverage files
                    analysis['low_coverage_files'].append(file_info)
                
                analysis['total_statements'] += total
                analysis['covered_statements'] += statements
    
    if analysis['total_statements'] > 0:
        analysis['coverage_percentage'] = (
            analysis['covered_statements'] / analysis['total_statements']
        ) * 100
    
    # Sort by coverage percentage
    analysis['files_by_coverage'].sort(key=lambda x: x['coverage'])
    
    return analysis


def main():
    """Main function."""
    coverage_file = Path('.coverage.json')
    
    if not coverage_file.exists():
        print("Coverage file not found. Run 'coverage json' first.")
        sys.exit(1)
    
    analysis = analyze_coverage_report(coverage_file)
    
    print(f"Overall Coverage: {analysis['coverage_percentage']:.2f}%")
    print(f"Total Statements: {analysis['total_statements']}")
    print(f"Covered Statements: {analysis['covered_statements']}")
    print()
    
    if analysis['low_coverage_files']:
        print("Files with low coverage (<80%):")
        for file_info in analysis['low_coverage_files']:
            print(f"  {file_info['file']}: {file_info['coverage']:.1f}%")
        print()
    
    print("Top 10 files needing attention:")
    for file_info in analysis['files_by_coverage'][:10]:
        print(f"  {file_info['file']}: {file_info['coverage']:.1f}%")


if __name__ == "__main__":
    main()
```

## 🔄 Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Lint with ruff
      run: |
        ruff check src/ tests/
        ruff format --check src/ tests/
    
    - name: Type check with mypy
      run: |
        mypy src/pynomaly/
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src/pynomaly --cov-report=xml --cov-report=term
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  integration-tests:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --tb=short
    
    - name: Run E2E tests
      run: |
        pytest tests/e2e/ -v --tb=short --timeout=300
```

### Quality Gates

```python
# scripts/quality_gates.py
#!/usr/bin/env python3
"""Quality gate checks for CI/CD pipeline."""

import sys
import subprocess
import json
from pathlib import Path


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def check_test_coverage() -> bool:
    """Check if test coverage meets minimum threshold."""
    print("Checking test coverage...")
    
    exit_code, stdout, stderr = run_command([
        "pytest", "tests/", "--cov=src/pynomaly", 
        "--cov-report=json", "--cov-fail-under=85"
    ])
    
    if exit_code == 0:
        print("✅ Test coverage check passed")
        return True
    else:
        print("❌ Test coverage check failed")
        print(stderr)
        return False


def check_code_quality() -> bool:
    """Check code quality with ruff."""
    print("Checking code quality...")
    
    # Check formatting
    exit_code, _, stderr = run_command([
        "ruff", "format", "--check", "src/", "tests/"
    ])
    
    if exit_code != 0:
        print("❌ Code formatting check failed")
        print(stderr)
        return False
    
    # Check linting
    exit_code, _, stderr = run_command([
        "ruff", "check", "src/", "tests/"
    ])
    
    if exit_code != 0:
        print("❌ Code linting check failed")
        print(stderr)
        return False
    
    print("✅ Code quality check passed")
    return True


def check_type_annotations() -> bool:
    """Check type annotations with mypy."""
    print("Checking type annotations...")
    
    exit_code, _, stderr = run_command([
        "mypy", "src/pynomaly/"
    ])
    
    if exit_code == 0:
        print("✅ Type annotation check passed")
        return True
    else:
        print("❌ Type annotation check failed")
        print(stderr)
        return False


def check_security() -> bool:
    """Check for security issues."""
    print("Checking security...")
    
    exit_code, _, stderr = run_command([
        "bandit", "-r", "src/pynomaly/", "-f", "json"
    ])
    
    # Bandit returns 1 for issues found, 0 for no issues
    if exit_code <= 1:  # Allow minor issues
        print("✅ Security check passed")
        return True
    else:
        print("❌ Security check failed")
        print(stderr)
        return False


def main():
    """Run all quality gate checks."""
    print("Running quality gate checks...\n")
    
    checks = [
        ("Test Coverage", check_test_coverage),
        ("Code Quality", check_code_quality),
        ("Type Annotations", check_type_annotations),
        ("Security", check_security),
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
            failed_checks.append(check_name)
        print()
    
    if failed_checks:
        print(f"❌ Quality gate failed. Failed checks: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        print("✅ All quality gate checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

---

## 📝 Summary

These testing guidelines ensure:

1. **Comprehensive Coverage**: Unit, integration, and E2E tests
2. **Quality Assurance**: Automated quality gates and coverage thresholds
3. **Performance Monitoring**: Benchmark and load tests
4. **Maintainability**: Well-organized test code with factories and helpers
5. **CI/CD Integration**: Automated testing in continuous integration

Remember to:
- Write tests before implementing features (TDD)
- Keep tests fast, isolated, and deterministic
- Use appropriate test types for different scenarios
- Maintain high test coverage, especially for critical paths
- Regularly review and update tests as code evolves

---

*Last updated: 2025-01-14*