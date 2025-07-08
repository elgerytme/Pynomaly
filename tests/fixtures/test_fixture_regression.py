import pytest
import pandas as pd
from dependency_injector import containers


def test_container_fixture(container):
    """Test container fixture in isolation."""
    assert container is not None
    assert isinstance(container, containers.DynamicContainer)


def test_db_engine_fixture(db_engine):
    """Test db_engine fixture in isolation."""
    assert db_engine is not None
    assert "sqlite" in str(db_engine.url)


def test_sample_dataset_fixture(sample_dataset):
    """Test sample_dataset fixture in isolation."""
    assert sample_dataset is not None
    assert sample_dataset.name == "Test Dataset"
    assert len(sample_dataset.data) > 0
    assert sample_dataset.data.shape[1] > 0
    assert sample_dataset.target_column == "target"
    assert "test" in sample_dataset.metadata


def test_sample_data_fixture(sample_data):
    """Test sample_data fixture in isolation."""
    assert sample_data is not None
    assert isinstance(sample_data, pd.DataFrame)
    assert len(sample_data) > 0
    assert "target" in sample_data.columns
    assert sample_data["target"].nunique() == 2
    assert sample_data["target"].dtype in [int, "int64"]


def test_sample_detector_fixture(sample_detector):
    """Test sample_detector fixture in isolation."""
    assert sample_detector is not None
    assert sample_detector.name == "Test Detector"
    assert sample_detector.algorithm_name == "IsolationForest"
    assert "contamination" in sample_detector.parameters
    assert sample_detector.parameters["contamination"] == 0.05
    assert "test" in sample_detector.metadata


def test_db_session_fixture(db_session):
    """Test db_session fixture in isolation."""
    assert db_session is not None
    # Test that the session is usable
    assert db_session.is_active


def test_settings_fixture(test_settings):
    """Test settings fixture in isolation."""
    assert test_settings is not None
    assert test_settings.app.name in ["pynomaly-test", "Pynomaly"]
    assert test_settings.app.environment in ["test", "development"]
    assert isinstance(test_settings.app.debug, bool)
    assert isinstance(test_settings.auth_enabled, bool)
    assert test_settings.database_url == "sqlite:///:memory:"


def test_temp_dir_fixture(temp_dir):
    """Test temp_dir fixture in isolation."""
    import os
    assert temp_dir is not None
    assert os.path.exists(temp_dir)
    assert os.path.isdir(temp_dir)


def test_sample_csv_file_fixture(sample_csv_file):
    """Test sample_csv_file fixture in isolation."""
    import os
    assert sample_csv_file is not None
    assert os.path.exists(sample_csv_file)
    assert sample_csv_file.endswith(".csv")


def test_mock_model_fixture(mock_model):
    """Test mock_model fixture in isolation."""
    assert mock_model is not None
    assert hasattr(mock_model, "fit")
    assert hasattr(mock_model, "predict")
    assert hasattr(mock_model, "decision_function")
    # Test that the mock is callable
    assert callable(mock_model.fit)
    assert callable(mock_model.predict)
    assert callable(mock_model.decision_function)


def test_large_dataset_fixture(large_dataset):
    """Test large_dataset fixture in isolation."""
    assert large_dataset is not None
    assert large_dataset.name == "Large Test Dataset"
    assert len(large_dataset.data) == 10000
    assert large_dataset.data.shape[1] == 20
    assert "performance" in large_dataset.metadata


def test_test_detection_result_fixture(test_detection_result):
    """Test test_detection_result fixture in isolation."""
    assert test_detection_result is not None
    assert test_detection_result.scores is not None
    assert len(test_detection_result.scores) == 100
    assert "test" in test_detection_result.metadata


def test_performance_data_fixture(performance_data):
    """Test performance_data fixture in isolation."""
    assert performance_data is not None
    assert len(performance_data) == 100000
    assert performance_data.shape[1] == 20
    assert performance_data.columns[0] == "feature_0"
    assert performance_data.columns[-1] == "feature_19"


def test_malicious_inputs_fixture(malicious_inputs):
    """Test malicious_inputs fixture in isolation."""
    assert malicious_inputs is not None
    assert isinstance(malicious_inputs, list)
    assert len(malicious_inputs) > 0
    assert "<script>alert('xss')</script>" in malicious_inputs
    assert "'; DROP TABLE users; --" in malicious_inputs


def test_trained_detector_fixture(trained_detector):
    """Test trained_detector fixture in isolation."""
    assert trained_detector is not None
    assert trained_detector.is_fitted is True
    assert "training_samples" in trained_detector.metadata

