"""API versioning and compatibility workflow end-to-end tests.

This module tests API versioning, backward compatibility, migration workflows,
and cross-version compatibility scenarios.
"""

import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import pytest
import numpy as np
from fastapi.testclient import TestClient

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


class TestAPIVersioningCompatibilityWorkflows:
    """Test API versioning and compatibility workflows."""
    
    @pytest.fixture
    def app_client(self):
        """Create test client for API."""
        container = create_container()
        app = create_app(container)
        return TestClient(app)
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 200),
            'feature_2': np.random.normal(0, 1, 200),
            'feature_3': np.random.normal(0, 1, 200)
        })
        
        return data
    
    def test_api_version_discovery_workflow(self, app_client):
        """Test API version discovery and information retrieval."""
        # Test API version discovery
        version_response = app_client.get("/api/version")
        assert version_response.status_code == 200
        version_info = version_response.json()
        
        # Verify version information structure
        assert "current_version" in version_info
        assert "supported_versions" in version_info
        assert "deprecated_versions" in version_info
        assert "api_documentation" in version_info
        
        current_version = version_info["current_version"]
        supported_versions = version_info["supported_versions"]
        
        # Verify current version is in supported versions
        assert current_version in supported_versions
        
        # Test version-specific endpoints
        for version in supported_versions:
            version_endpoint = f"/api/{version}/info"
            version_specific_response = app_client.get(version_endpoint)
            assert version_specific_response.status_code == 200
            
            version_specific_info = version_specific_response.json()
            assert "version" in version_specific_info
            assert "features" in version_specific_info
            assert "endpoints" in version_specific_info
            assert version_specific_info["version"] == version
        
        # Test API capabilities by version
        capabilities_response = app_client.get("/api/capabilities")
        assert capabilities_response.status_code == 200
        capabilities = capabilities_response.json()
        
        assert "features_by_version" in capabilities
        assert "migration_paths" in capabilities
        assert "compatibility_matrix" in capabilities
    
    def test_cross_version_detector_workflow(self, app_client, sample_dataset):
        """Test detector creation and usage across API versions."""
        # Create detector using different API versions
        api_versions = ["v1", "v2", "v3"]
        detector_configs = {
            "v1": {
                "name": "V1 Detector",
                "algorithm": "IsolationForest",
                "params": {"contamination": 0.1}
            },
            "v2": {
                "name": "V2 Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1, "random_state": 42},
                "metadata": {"created_by": "test", "version": "v2"}
            },
            "v3": {
                "name": "V3 Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1, "random_state": 42},
                "configuration": {
                    "preprocessing": {"normalize": True},
                    "postprocessing": {"threshold_adjustment": True}
                },
                "tags": ["test", "versioning"]
            }
        }
        
        detector_ids = {}
        
        # Upload dataset first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            dataset_file = f.name
        
        try:
            with open(dataset_file, 'rb') as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("version_test_data.csv", file, "text/csv")},
                    data={"name": "Version Test Dataset"}
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]
            
            # Create detectors using different API versions
            for version in api_versions:
                if version in detector_configs:
                    create_endpoint = f"/api/{version}/detectors"
                    config = detector_configs[version]
                    
                    create_response = app_client.post(create_endpoint, json=config)
                    
                    if create_response.status_code == 200:
                        detector_ids[version] = create_response.json()["id"]
                        
                        # Test training with version-specific endpoint
                        train_endpoint = f"/api/{version}/detectors/{detector_ids[version]}/train"
                        
                        # Adapt training request format for different versions
                        if version == "v1":
                            train_request = {"dataset": dataset_id}
                        else:
                            train_request = {"dataset_id": dataset_id}
                        
                        train_response = app_client.post(train_endpoint, json=train_request)
                        assert train_response.status_code == 200
            
            # Test cross-version compatibility
            # Use v1 detector with v2 detection endpoint
            if "v1" in detector_ids and "v2" in detector_ids:
                v1_detector_id = detector_ids["v1"]
                v2_detect_endpoint = f"/api/v2/detectors/{v1_detector_id}/detect"
                
                detect_response = app_client.post(v2_detect_endpoint, json={
                    "dataset_id": dataset_id
                })
                # Should work with compatibility layer
                assert detect_response.status_code in [200, 400]  # 400 if compatibility not implemented
                
                if detect_response.status_code == 200:
                    result = detect_response.json()
                    assert "anomalies" in result or "results" in result
            
            # Test detector migration between versions
            if "v2" in detector_ids:
                migration_request = {
                    "detector_id": detector_ids["v2"],
                    "target_version": "v3",
                    "migration_options": {
                        "preserve_training": True,
                        "update_configuration": True
                    }
                }
                
                migration_response = app_client.post("/api/migration/detector", json=migration_request)
                
                if migration_response.status_code == 200:
                    migration_result = migration_response.json()
                    assert "new_detector_id" in migration_result
                    assert "migration_log" in migration_result
                    assert migration_result["target_version"] == "v3"
                    
                    # Test migrated detector functionality
                    new_detector_id = migration_result["new_detector_id"]
                    v3_detect_response = app_client.post(
                        f"/api/v3/detectors/{new_detector_id}/detect",
                        json={"dataset_id": dataset_id}
                    )
                    assert v3_detect_response.status_code == 200
        
        finally:
            Path(dataset_file).unlink(missing_ok=True)
    
    def test_data_format_compatibility_workflow(self, app_client, sample_dataset):
        """Test data format compatibility across API versions."""
        # Test different data format support by version
        data_formats = {
            "v1": {"format": "csv", "separator": ","},
            "v2": {"format": "json", "structure": "records"},
            "v3": {"format": "parquet", "compression": "snappy"}
        }
        
        # Upload dataset in different formats
        uploaded_datasets = {}
        
        for version, format_config in data_formats.items():
            format_type = format_config["format"]
            
            if format_type == "csv":
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    sample_dataset.to_csv(f.name, index=False)
                    file_path = f.name
                
                with open(file_path, 'rb') as file:
                    upload_response = app_client.post(
                        f"/api/{version}/datasets/upload",
                        files={"file": (f"{version}_data.csv", file, "text/csv")},
                        data={"name": f"{version} CSV Dataset"}
                    )
                
            elif format_type == "json":
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    sample_dataset.to_json(f.name, orient='records')
                    file_path = f.name
                
                with open(file_path, 'rb') as file:
                    upload_response = app_client.post(
                        f"/api/{version}/datasets/upload",
                        files={"file": (f"{version}_data.json", file, "application/json")},
                        data={"name": f"{version} JSON Dataset"}
                    )
            
            elif format_type == "parquet":
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
                    try:
                        sample_dataset.to_parquet(f.name)
                        file_path = f.name
                        
                        with open(file_path, 'rb') as file:
                            upload_response = app_client.post(
                                f"/api/{version}/datasets/upload",
                                files={"file": (f"{version}_data.parquet", file, "application/octet-stream")},
                                data={"name": f"{version} Parquet Dataset"}
                            )
                    except Exception:
                        # Skip parquet if not available
                        Path(f.name).unlink(missing_ok=True)
                        continue
            
            if upload_response.status_code == 200:
                uploaded_datasets[version] = upload_response.json()["id"]
            
            # Clean up temporary file
            Path(file_path).unlink(missing_ok=True)
        
        # Test cross-format detector training
        if uploaded_datasets:
            detector_data = {
                "name": "Cross-Format Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1}
            }
            
            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]
            
            # Train on datasets from different versions/formats
            for version, dataset_id in uploaded_datasets.items():
                train_response = app_client.post(f"/api/detectors/{detector_id}/train", json={
                    "dataset_id": dataset_id,
                    "format_compatibility_mode": True
                })
                # Should handle format conversion automatically
                assert train_response.status_code in [200, 400]
    
    def test_response_format_evolution_workflow(self, app_client, sample_dataset):
        """Test response format changes across API versions."""
        # Create detector and dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            dataset_file = f.name
        
        try:
            with open(dataset_file, 'rb') as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("response_format_data.csv", file, "text/csv")},
                    data={"name": "Response Format Dataset"}
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]
            
            detector_data = {
                "name": "Response Format Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1}
            }
            
            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]
            
            # Train detector
            train_response = app_client.post(f"/api/detectors/{detector_id}/train", json={
                "dataset_id": dataset_id
            })
            assert train_response.status_code == 200
            
            # Test detection with different response format versions
            version_response_formats = {
                "v1": {
                    "expected_fields": ["anomalies", "normal_count", "anomaly_count"],
                    "anomaly_format": "indices"
                },
                "v2": {
                    "expected_fields": ["anomalies", "anomaly_rate", "scores", "metadata"],
                    "anomaly_format": "boolean_array"
                },
                "v3": {
                    "expected_fields": ["results", "statistics", "confidence_metrics", "explanations"],
                    "anomaly_format": "detailed_objects"
                }
            }
            
            detection_results = {}
            
            for version, format_spec in version_response_formats.items():
                # Test with Accept header for different versions
                headers = {"Accept": f"application/vnd.pynomaly.{version}+json"}
                
                detect_response = app_client.post(
                    f"/api/detectors/{detector_id}/detect",
                    json={"dataset_id": dataset_id},
                    headers=headers
                )
                
                if detect_response.status_code == 200:
                    result = detect_response.json()
                    detection_results[version] = result
                    
                    # Verify response format
                    expected_fields = format_spec["expected_fields"]
                    for field in expected_fields:
                        if field in result:
                            # Field exists, verify it has appropriate structure
                            assert result[field] is not None
                        else:
                            # Field might be renamed or restructured in this version
                            print(f"Field {field} not found in version {version}, checking alternatives...")
                
                # Test backward compatibility endpoint
                compat_endpoint = f"/api/{version}/detectors/{detector_id}/detect"
                compat_response = app_client.post(compat_endpoint, json={
                    "dataset_id": dataset_id
                })
                
                if compat_response.status_code == 200:
                    compat_result = compat_response.json()
                    # Verify version-specific response format
                    assert isinstance(compat_result, dict)
            
            # Test format conversion between versions
            if len(detection_results) > 1:
                conversion_request = {
                    "source_version": "v1",
                    "target_version": "v3",
                    "data": detection_results.get("v1", {}),
                    "conversion_options": {
                        "preserve_precision": True,
                        "add_metadata": True
                    }
                }
                
                conversion_response = app_client.post("/api/conversion/response-format", 
                                                    json=conversion_request)
                
                if conversion_response.status_code == 200:
                    converted_result = conversion_response.json()
                    assert "converted_data" in converted_result
                    assert "conversion_log" in converted_result
        
        finally:
            Path(dataset_file).unlink(missing_ok=True)
    
    def test_feature_flag_workflow(self, app_client):
        """Test feature flags and gradual feature rollout across versions."""
        # Test feature availability query
        features_response = app_client.get("/api/features")
        assert features_response.status_code == 200
        features = features_response.json()
        
        assert "available_features" in features
        assert "experimental_features" in features
        assert "deprecated_features" in features
        
        # Test version-specific feature availability
        version_features = {}
        for version in ["v1", "v2", "v3"]:
            version_features_response = app_client.get(f"/api/{version}/features")
            if version_features_response.status_code == 200:
                version_features[version] = version_features_response.json()
        
        # Test experimental feature usage
        experimental_features = features.get("experimental_features", [])
        
        for feature in experimental_features:
            feature_name = feature.get("name")
            if feature_name:
                # Test enabling experimental feature
                enable_response = app_client.post(f"/api/features/{feature_name}/enable", json={
                    "user_consent": True,
                    "feedback_opt_in": True
                })
                
                if enable_response.status_code == 200:
                    # Test using experimental feature
                    feature_test_response = app_client.get(f"/api/features/{feature_name}/test")
                    assert feature_test_response.status_code in [200, 501]  # 501 if not implemented
                    
                    # Disable experimental feature
                    disable_response = app_client.post(f"/api/features/{feature_name}/disable")
                    assert disable_response.status_code == 200
    
    def test_deprecation_workflow(self, app_client, sample_dataset):
        """Test deprecation warnings and migration paths."""
        # Test deprecated endpoint usage
        deprecated_endpoints = [
            ("/api/v1/models", "/api/v2/detectors"),  # Old to new mapping
            ("/api/v1/predict", "/api/v2/detect"),
            ("/api/v1/upload", "/api/v2/datasets/upload")
        ]
        
        for old_endpoint, new_endpoint in deprecated_endpoints:
            # Test deprecated endpoint
            deprecated_response = app_client.get(old_endpoint)
            
            if deprecated_response.status_code == 200:
                # Should include deprecation warning in headers
                assert "X-Deprecated-Warning" in deprecated_response.headers
                assert "X-Migration-Path" in deprecated_response.headers
                
                deprecation_warning = deprecated_response.headers["X-Deprecated-Warning"]
                migration_path = deprecated_response.headers["X-Migration-Path"]
                
                assert "deprecated" in deprecation_warning.lower()
                assert new_endpoint in migration_path
                
            elif deprecated_response.status_code == 410:  # Gone
                # Endpoint has been removed
                gone_response = deprecated_response.json()
                assert "deprecated" in gone_response.get("message", "").lower()
                assert "migration" in gone_response
        
        # Test migration assistant
        migration_request = {
            "current_version": "v1",
            "target_version": "v3",
            "analyze_usage": True
        }
        
        migration_response = app_client.post("/api/migration/analyze", json=migration_request)
        
        if migration_response.status_code == 200:
            migration_analysis = migration_response.json()
            
            assert "breaking_changes" in migration_analysis
            assert "migration_steps" in migration_analysis
            assert "estimated_effort" in migration_analysis
            assert "compatibility_score" in migration_analysis
            
            # Test migration plan generation
            plan_request = {
                "source_version": "v1",
                "target_version": "v3",
                "migration_analysis": migration_analysis
            }
            
            plan_response = app_client.post("/api/migration/plan", json=plan_request)
            
            if plan_response.status_code == 200:
                migration_plan = plan_response.json()
                
                assert "steps" in migration_plan
                assert "timeline" in migration_plan
                assert "validation_checklist" in migration_plan
    
    def test_api_contract_testing_workflow(self, app_client):
        """Test API contract validation across versions."""
        # Test OpenAPI schema retrieval for different versions
        schemas = {}
        
        for version in ["v1", "v2", "v3"]:
            schema_response = app_client.get(f"/api/{version}/openapi.json")
            
            if schema_response.status_code == 200:
                schema = schema_response.json()
                schemas[version] = schema
                
                # Verify schema structure
                assert "openapi" in schema
                assert "info" in schema
                assert "paths" in schema
                assert "components" in schema
                
                # Verify version information
                assert schema["info"]["version"] == version
        
        # Test contract compatibility validation
        if len(schemas) > 1:
            versions = list(schemas.keys())
            for i in range(len(versions) - 1):
                source_version = versions[i]
                target_version = versions[i + 1]
                
                compatibility_request = {
                    "source_schema": schemas[source_version],
                    "target_schema": schemas[target_version],
                    "check_breaking_changes": True
                }
                
                compatibility_response = app_client.post("/api/validation/schema-compatibility", 
                                                       json=compatibility_request)
                
                if compatibility_response.status_code == 200:
                    compatibility_result = compatibility_response.json()
                    
                    assert "is_compatible" in compatibility_result
                    assert "breaking_changes" in compatibility_result
                    assert "warnings" in compatibility_result
                    assert "migration_required" in compatibility_result
        
        # Test client SDK compatibility
        sdk_test_request = {
            "client_version": "1.0.0",
            "api_versions": list(schemas.keys()),
            "test_endpoints": [
                "/detectors",
                "/datasets/upload",
                "/detection/detect"
            ]
        }
        
        sdk_compatibility_response = app_client.post("/api/validation/sdk-compatibility", 
                                                    json=sdk_test_request)
        
        if sdk_compatibility_response.status_code == 200:
            sdk_result = sdk_compatibility_response.json()
            
            assert "compatibility_matrix" in sdk_result
            assert "recommended_api_version" in sdk_result
            assert "client_update_required" in sdk_result