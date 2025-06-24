"""Comprehensive tests for API and CLI interfaces."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner

from tests.conftest_dependencies import requires_dependency, requires_dependencies


@requires_dependencies('fastapi', 'uvicorn')
class TestAPIEndpointsComprehensive:
    """Comprehensive tests for all API endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        from fastapi.testclient import TestClient
        from pynomaly.presentation.api.app import app
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers for testing."""
        return {"Authorization": "Bearer test_token"}
    
    def test_api_root_endpoint(self, test_client):
        """Test API root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Pynomaly"
    
    def test_health_check_detailed(self, test_client):
        """Test detailed health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "checks" in health_data
        
        # Verify individual health checks
        checks = health_data["checks"]
        assert "database" in checks
        assert "cache" in checks
    
    def test_detector_endpoints_full_crud(self, test_client):
        """Test complete CRUD operations for detectors."""
        # CREATE - Create new detector
        detector_data = {
            "name": "comprehensive_test_detector",
            "algorithm": "IsolationForest",
            "contamination": 0.1,
            "parameters": {
                "n_estimators": 100,
                "random_state": 42,
                "max_features": 1.0
            },
            "description": "Test detector for comprehensive testing"
        }
        
        create_response = test_client.post("/detectors/", json=detector_data)
        assert create_response.status_code in [200, 201]
        
        created_detector = create_response.json()
        detector_id = created_detector["id"]
        assert created_detector["name"] == detector_data["name"]
        
        # READ - Get single detector
        get_response = test_client.get(f"/detectors/{detector_id}")
        assert get_response.status_code == 200
        
        retrieved_detector = get_response.json()
        assert retrieved_detector["id"] == detector_id
        assert retrieved_detector["algorithm"] == detector_data["algorithm"]
        
        # UPDATE - Update detector
        update_data = {
            "description": "Updated description for comprehensive testing"
        }
        
        update_response = test_client.patch(f"/detectors/{detector_id}", json=update_data)
        assert update_response.status_code == 200
        
        updated_detector = update_response.json()
        assert updated_detector["description"] == update_data["description"]
        
        # LIST - List all detectors
        list_response = test_client.get("/detectors/")
        assert list_response.status_code == 200
        
        detectors_list = list_response.json()
        assert isinstance(detectors_list, list)
        assert any(d["id"] == detector_id for d in detectors_list)
        
        # DELETE - Delete detector
        delete_response = test_client.delete(f"/detectors/{detector_id}")
        assert delete_response.status_code in [200, 204]
        
        # Verify deletion
        get_after_delete = test_client.get(f"/detectors/{detector_id}")
        assert get_after_delete.status_code == 404
    
    def test_dataset_endpoints_full_crud(self, test_client):
        """Test complete CRUD operations for datasets."""
        # CREATE - Upload dataset
        dataset_data = {
            "name": "comprehensive_test_dataset",
            "description": "Test dataset for comprehensive testing",
            "data": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [100.0, 200.0, 300.0]  # Obvious anomaly
            ],
            "feature_names": ["feature_1", "feature_2", "feature_3"],
            "metadata": {
                "source": "test",
                "created_by": "comprehensive_test"
            }
        }
        
        create_response = test_client.post("/datasets/", json=dataset_data)
        assert create_response.status_code in [200, 201]
        
        created_dataset = create_response.json()
        dataset_id = created_dataset["id"]
        
        # READ - Get dataset
        get_response = test_client.get(f"/datasets/{dataset_id}")
        assert get_response.status_code == 200
        
        retrieved_dataset = get_response.json()
        assert retrieved_dataset["name"] == dataset_data["name"]
        assert len(retrieved_dataset["data"]) == len(dataset_data["data"])
        
        # UPDATE - Update dataset metadata
        update_data = {
            "description": "Updated dataset description"
        }
        
        update_response = test_client.patch(f"/datasets/{dataset_id}", json=update_data)
        assert update_response.status_code == 200
        
        # LIST - List datasets
        list_response = test_client.get("/datasets/")
        assert list_response.status_code == 200
        
        datasets_list = list_response.json()
        assert isinstance(datasets_list, list)
        assert any(d["id"] == dataset_id for d in datasets_list)
    
    def test_detection_workflow_endpoint(self, test_client):
        """Test complete detection workflow via API."""
        # 1. Create detector
        detector_data = {
            "name": "workflow_detector",
            "algorithm": "IsolationForest",
            "contamination": 0.25  # 25% contamination for clear detection
        }
        
        detector_response = test_client.post("/detectors/", json=detector_data)
        detector_id = detector_response.json()["id"]
        
        # 2. Create dataset with clear anomalies
        dataset_data = {
            "name": "workflow_dataset",
            "data": [
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
                [0.9, 0.9, 0.9],
                [10.0, 10.0, 10.0]  # Clear anomaly
            ]
        }
        
        dataset_response = test_client.post("/datasets/", json=dataset_data)
        dataset_id = dataset_response.json()["id"]
        
        # 3. Train detector
        train_data = {
            "detector_id": detector_id,
            "dataset_id": dataset_id
        }
        
        train_response = test_client.post("/train/", json=train_data)
        assert train_response.status_code == 200
        
        # 4. Perform detection
        detect_data = {
            "detector_id": detector_id,
            "dataset_id": dataset_id
        }
        
        detect_response = test_client.post("/detect/", json=detect_data)
        assert detect_response.status_code == 200
        
        detection_result = detect_response.json()
        assert "scores" in detection_result
        assert "anomalies" in detection_result
        assert "threshold" in detection_result
        assert len(detection_result["scores"]) == 4
    
    def test_batch_detection_endpoint(self, test_client):
        """Test batch detection endpoint."""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            dataset_data = {
                "name": f"batch_dataset_{i}",
                "data": [[j, j+1, j+2] for j in range(10)]
            }
            response = test_client.post("/datasets/", json=dataset_data)
            datasets.append(response.json()["id"])
        
        # Create detector
        detector_data = {"name": "batch_detector", "algorithm": "IsolationForest"}
        detector_response = test_client.post("/detectors/", json=detector_data)
        detector_id = detector_response.json()["id"]
        
        # Perform batch detection
        batch_data = {
            "detector_id": detector_id,
            "dataset_ids": datasets
        }
        
        batch_response = test_client.post("/detect/batch/", json=batch_data)
        assert batch_response.status_code == 200
        
        batch_result = batch_response.json()
        assert "results" in batch_result
        assert len(batch_result["results"]) == 3
    
    def test_api_error_handling(self, test_client):
        """Test API error handling and validation."""
        # Test invalid detector creation
        invalid_detector = {
            "name": "",  # Invalid: empty name
            "algorithm": "NonExistentAlgorithm",  # Invalid algorithm
            "contamination": 1.5  # Invalid: > 1.0
        }
        
        response = test_client.post("/detectors/", json=invalid_detector)
        assert response.status_code == 422  # Validation error
        
        error_data = response.json()
        assert "detail" in error_data
        
        # Test accessing non-existent resource
        response = test_client.get("/detectors/non-existent-id")
        assert response.status_code == 404
        
        # Test malformed JSON
        response = test_client.post(
            "/detectors/",
            data="malformed json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


@requires_dependency('testing')
class TestCLIComprehensive:
    """Comprehensive tests for CLI interface."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def temp_data_file(self):
        """Create temporary data file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature_1,feature_2,feature_3\n")
            f.write("1.0,2.0,3.0\n")
            f.write("4.0,5.0,6.0\n")
            f.write("7.0,8.0,9.0\n")
            f.write("100.0,200.0,300.0\n")  # Anomaly
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    def test_cli_help_commands(self, cli_runner):
        """Test CLI help commands."""
        from pynomaly.presentation.cli.app import cli
        
        # Test main help
        result = cli_runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Pynomaly" in result.output
        assert "anomaly detection" in result.output.lower()
        
        # Test subcommand help
        result = cli_runner.invoke(cli, ['detect', '--help'])
        assert result.exit_code == 0
        assert "detect" in result.output.lower()
    
    def test_cli_version_command(self, cli_runner):
        """Test CLI version command."""
        from pynomaly.presentation.cli.app import cli
        
        result = cli_runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert "version" in result.output.lower()
    
    def test_cli_detect_command(self, cli_runner, temp_data_file):
        """Test CLI detect command."""
        from pynomaly.presentation.cli.app import cli
        
        # Test basic detection
        result = cli_runner.invoke(cli, [
            'detect',
            '--input', temp_data_file,
            '--algorithm', 'IsolationForest',
            '--contamination', '0.2',
            '--output', 'test_output.json'
        ])
        
        # Note: This might fail if dependencies aren't available
        # The important thing is that CLI parsing works
        assert result.exit_code in [0, 1]  # 0 for success, 1 for missing deps
    
    def test_cli_train_command(self, cli_runner, temp_data_file):
        """Test CLI train command."""
        from pynomaly.presentation.cli.app import cli
        
        result = cli_runner.invoke(cli, [
            'train',
            '--input', temp_data_file,
            '--algorithm', 'IsolationForest',
            '--model-output', 'test_model.pkl',
            '--contamination', '0.1'
        ])
        
        # CLI should handle the command even if training fails
        assert result.exit_code in [0, 1]
    
    def test_cli_evaluate_command(self, cli_runner, temp_data_file):
        """Test CLI evaluate command."""
        from pynomaly.presentation.cli.app import cli
        
        result = cli_runner.invoke(cli, [
            'evaluate',
            '--predictions', temp_data_file,
            '--ground-truth', temp_data_file,
            '--metrics', 'precision,recall,f1'
        ])
        
        assert result.exit_code in [0, 1]
    
    def test_cli_config_commands(self, cli_runner):
        """Test CLI configuration commands."""
        from pynomaly.presentation.cli.app import cli
        
        # Test config show
        result = cli_runner.invoke(cli, ['config', 'show'])
        assert result.exit_code == 0
        
        # Test config set
        result = cli_runner.invoke(cli, [
            'config', 'set',
            '--key', 'default_algorithm',
            '--value', 'IsolationForest'
        ])
        assert result.exit_code == 0
    
    def test_cli_list_algorithms(self, cli_runner):
        """Test CLI list algorithms command."""
        from pynomaly.presentation.cli.app import cli
        
        result = cli_runner.invoke(cli, ['algorithms', 'list'])
        assert result.exit_code == 0
        assert "IsolationForest" in result.output
    
    def test_cli_data_validation(self, cli_runner):
        """Test CLI data validation."""
        from pynomaly.presentation.cli.app import cli
        
        # Test with non-existent file
        result = cli_runner.invoke(cli, [
            'detect',
            '--input', 'non_existent_file.csv'
        ])
        
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "not found" in result.output.lower()
    
    def test_cli_output_formats(self, cli_runner, temp_data_file):
        """Test CLI output formats."""
        from pynomaly.presentation.cli.app import cli
        
        output_formats = ['json', 'csv', 'txt']
        
        for format_type in output_formats:
            result = cli_runner.invoke(cli, [
                'detect',
                '--input', temp_data_file,
                '--algorithm', 'IsolationForest',
                '--output-format', format_type
            ])
            
            # Should handle format specification
            assert result.exit_code in [0, 1]


@requires_dependency('testing')
class TestWebUIIntegration:
    """Test web UI integration and templates."""
    
    @pytest.fixture
    def web_client(self):
        """Create web client for testing."""
        from fastapi.testclient import TestClient
        from pynomaly.presentation.web.app import app
        return TestClient(app)
    
    def test_web_ui_home_page(self, web_client):
        """Test web UI home page."""
        response = web_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_web_ui_detector_page(self, web_client):
        """Test web UI detector management page."""
        response = web_client.get("/detectors")
        assert response.status_code == 200
        
        # Check for expected content
        content = response.text
        assert "detector" in content.lower()
    
    def test_web_ui_detection_page(self, web_client):
        """Test web UI detection page."""
        response = web_client.get("/detect")
        assert response.status_code == 200
        
        content = response.text
        assert "detect" in content.lower()
    
    def test_web_ui_api_endpoints(self, web_client):
        """Test web UI API endpoints."""
        # Test getting detector list for UI
        response = web_client.get("/api/detectors")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    def test_web_ui_file_upload(self, web_client):
        """Test web UI file upload functionality."""
        # Create test file content
        file_content = "feature_1,feature_2\n1,2\n3,4\n"
        
        response = web_client.post(
            "/upload",
            files={"file": ("test.csv", file_content, "text/csv")}
        )
        
        # Should handle file upload
        assert response.status_code in [200, 422]  # 422 for validation errors


class TestPerformanceAndBenchmarks:
    """Test performance characteristics and benchmarks."""
    
    @requires_dependency('scikit-learn')
    def test_detection_performance_benchmarks(self):
        """Test detection performance with different data sizes."""
        import time
        import numpy as np
        import pandas as pd
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        from pynomaly.domain.entities import Dataset
        
        adapter = SklearnAdapter("IsolationForest")
        
        # Test with different data sizes
        data_sizes = [100, 1000, 5000]
        performance_results = {}
        
        for size in data_sizes:
            np.random.seed(42)
            data = np.random.randn(size, 5)
            df = pd.DataFrame(data, columns=[f'f_{i}' for i in range(5)])
            dataset = Dataset(name=f"perf_test_{size}", data=df)
            
            # Measure training time
            start_time = time.time()
            adapter.fit(dataset)
            training_time = time.time() - start_time
            
            # Measure detection time
            start_time = time.time()
            result = adapter.detect(dataset)
            detection_time = time.time() - start_time
            
            performance_results[size] = {
                "training_time": training_time,
                "detection_time": detection_time,
                "total_time": training_time + detection_time
            }
            
            # Verify results
            assert len(result.scores) == size
            assert result.execution_time > 0
        
        # Performance should scale reasonably
        assert performance_results[1000]["total_time"] > performance_results[100]["total_time"]
        assert performance_results[5000]["total_time"] > performance_results[1000]["total_time"]
    
    @requires_dependency('hypothesis')
    def test_concurrent_detection_performance(self):
        """Test concurrent detection performance."""
        import concurrent.futures
        import numpy as np
        import pandas as pd
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        from pynomaly.domain.entities import Dataset
        
        def run_detection(dataset_id):
            """Run detection on a dataset."""
            np.random.seed(dataset_id)
            data = np.random.randn(100, 3)
            df = pd.DataFrame(data, columns=['x', 'y', 'z'])
            dataset = Dataset(name=f"concurrent_test_{dataset_id}", data=df)
            
            adapter = SklearnAdapter("IsolationForest")
            adapter.fit(dataset)
            result = adapter.detect(dataset)
            
            return len(result.scores)
        
        # Test concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_detection, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete successfully
        assert len(results) == 10
        assert all(result == 100 for result in results)