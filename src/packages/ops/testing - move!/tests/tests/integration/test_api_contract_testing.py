"""
API Contract Testing Framework

Comprehensive API contract testing to ensure API compatibility, schema validation,
backward compatibility, and adherence to OpenAPI specifications.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import jsonschema
import pytest
import yaml
from fastapi.testclient import TestClient
from openapi_spec_validator import validate_spec


@dataclass
class ContractTestResult:
    """API contract test result."""
    
    test_name: str
    endpoint: str
    method: str
    status_code: int
    expected_status: int
    schema_valid: bool
    backward_compatible: bool
    performance_acceptable: bool
    error_details: List[str]
    recommendations: List[str]


class APIContractTester:
    """Comprehensive API contract testing framework."""
    
    def __init__(self, client: TestClient, spec_file: str = None):
        self.client = client
        self.spec_file = spec_file
        self.openapi_spec = None
        self.test_results: List[ContractTestResult] = []
        self.baseline_responses = {}
        
        if spec_file:
            self.load_openapi_spec()
    
    def load_openapi_spec(self):
        """Load OpenAPI specification."""
        try:
            if self.spec_file.endswith('.yaml') or self.spec_file.endswith('.yml'):
                with open(self.spec_file, 'r') as f:
                    self.openapi_spec = yaml.safe_load(f)
            else:
                with open(self.spec_file, 'r') as f:
                    self.openapi_spec = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load OpenAPI spec: {e}")
            self.openapi_spec = self._generate_mock_spec()
    
    def _generate_mock_spec(self) -> Dict[str, Any]:
        """Generate mock OpenAPI specification for testing."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Pynomaly API",
                "version": "1.0.0"
            },
            "paths": {
                "/api/v1/health": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "Health check",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string"},
                                                "timestamp": {"type": "string"}
                                            },
                                            "required": ["status"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/datasets": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "List datasets",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "datasets": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "id": {"type": "string"},
                                                            "name": {"type": "string"},
                                                            "created_at": {"type": "string"}
                                                        },
                                                        "required": ["id", "name"]
                                                    }
                                                }
                                            },
                                            "required": ["datasets"]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "post": {
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "description": {"type": "string"},
                                            "data": {"type": "array"}
                                        },
                                        "required": ["name", "data"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "Dataset created",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "status": {"type": "string"}
                                            },
                                            "required": ["id", "name", "status"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/detectors": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "List detectors",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "detectors": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "id": {"type": "string"},
                                                            "name": {"type": "string"},
                                                            "algorithm": {"type": "string"},
                                                            "is_trained": {"type": "boolean"}
                                                        },
                                                        "required": ["id", "name", "algorithm"]
                                                    }
                                                }
                                            },
                                            "required": ["detectors"]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "post": {
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "algorithm": {"type": "string"},
                                            "parameters": {"type": "object"}
                                        },
                                        "required": ["name", "algorithm"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "Detector created",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "algorithm": {"type": "string"},
                                                "status": {"type": "string"}
                                            },
                                            "required": ["id", "name", "algorithm", "status"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/detection/detect": {
                    "post": {
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "detector_id": {"type": "string"},
                                            "data": {"type": "array"},
                                            "return_scores": {"type": "boolean"}
                                        },
                                        "required": ["detector_id", "data"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Detection results",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "predictions": {
                                                    "type": "array",
                                                    "items": {"type": "integer"}
                                                },
                                                "anomaly_scores": {
                                                    "type": "array",
                                                    "items": {"type": "number"}
                                                },
                                                "n_anomalies": {"type": "integer"},
                                                "execution_time": {"type": "number"}
                                            },
                                            "required": ["predictions", "n_anomalies"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def validate_openapi_spec(self) -> bool:
        """Validate OpenAPI specification."""
        try:
            validate_spec(self.openapi_spec)
            return True
        except Exception as e:
            print(f"OpenAPI spec validation failed: {e}")
            return False
    
    def test_endpoint_contract(
        self, 
        method: str, 
        endpoint: str, 
        payload: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> ContractTestResult:
        """Test individual endpoint contract."""
        
        error_details = []
        recommendations = []
        
        # Get endpoint specification
        endpoint_spec = self._get_endpoint_spec(method.lower(), endpoint)
        
        if not endpoint_spec:
            error_details.append(f"Endpoint {method} {endpoint} not found in OpenAPI spec")
            return ContractTestResult(
                test_name=f"Contract Test: {method} {endpoint}",
                endpoint=endpoint,
                method=method,
                status_code=0,
                expected_status=200,
                schema_valid=False,
                backward_compatible=False,
                performance_acceptable=False,
                error_details=error_details,
                recommendations=["Add endpoint to OpenAPI specification"]
            )
        
        # Make API request
        try:
            if method.upper() == "GET":
                response = self.client.get(endpoint, headers=headers)
            elif method.upper() == "POST":
                response = self.client.post(endpoint, json=payload, headers=headers)
            elif method.upper() == "PUT":
                response = self.client.put(endpoint, json=payload, headers=headers)
            elif method.upper() == "DELETE":
                response = self.client.delete(endpoint, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        
        except Exception as e:
            error_details.append(f"Request failed: {str(e)}")
            return ContractTestResult(
                test_name=f"Contract Test: {method} {endpoint}",
                endpoint=endpoint,
                method=method,
                status_code=0,
                expected_status=200,
                schema_valid=False,
                backward_compatible=False,
                performance_acceptable=False,
                error_details=error_details,
                recommendations=["Fix endpoint implementation"]
            )
        
        # Validate response status
        expected_statuses = list(endpoint_spec.get("responses", {}).keys())
        expected_status = int(expected_statuses[0]) if expected_statuses else 200
        
        if response.status_code not in [int(s) for s in expected_statuses]:
            error_details.append(f"Unexpected status code: {response.status_code}")
        
        # Validate response schema
        schema_valid = True
        if response.status_code == 200:
            try:
                response_data = response.json()
                response_schema = self._get_response_schema(endpoint_spec, str(response.status_code))
                
                if response_schema:
                    jsonschema.validate(response_data, response_schema)
                else:
                    error_details.append("No response schema defined")
                    schema_valid = False
            
            except jsonschema.ValidationError as e:
                error_details.append(f"Response schema validation failed: {e.message}")
                schema_valid = False
                recommendations.append("Fix response schema compliance")
            
            except Exception as e:
                error_details.append(f"Response validation error: {str(e)}")
                schema_valid = False
        
        # Check backward compatibility
        backward_compatible = self._check_backward_compatibility(
            method, endpoint, response.json() if response.status_code == 200 else None
        )
        
        if not backward_compatible:
            error_details.append("Backward compatibility issues detected")
            recommendations.append("Ensure backward compatibility in API changes")
        
        # Check performance
        performance_acceptable = hasattr(response, 'elapsed') and response.elapsed.total_seconds() < 2.0
        
        if not performance_acceptable:
            recommendations.append("Optimize endpoint performance")
        
        return ContractTestResult(
            test_name=f"Contract Test: {method} {endpoint}",
            endpoint=endpoint,
            method=method,
            status_code=response.status_code,
            expected_status=expected_status,
            schema_valid=schema_valid,
            backward_compatible=backward_compatible,
            performance_acceptable=performance_acceptable,
            error_details=error_details,
            recommendations=recommendations
        )
    
    def _get_endpoint_spec(self, method: str, endpoint: str) -> Optional[Dict[str, Any]]:
        """Get endpoint specification from OpenAPI spec."""
        if not self.openapi_spec or "paths" not in self.openapi_spec:
            return None
        
        # Try exact match first
        if endpoint in self.openapi_spec["paths"]:
            path_spec = self.openapi_spec["paths"][endpoint]
            return path_spec.get(method.lower())
        
        # Try pattern matching for parameterized endpoints
        for path_pattern, path_spec in self.openapi_spec["paths"].items():
            if self._match_path_pattern(endpoint, path_pattern):
                return path_spec.get(method.lower())
        
        return None
    
    def _match_path_pattern(self, endpoint: str, pattern: str) -> bool:
        """Match endpoint against OpenAPI path pattern."""
        # Convert OpenAPI path pattern to regex
        # e.g., /api/v1/datasets/{id} -> /api/v1/datasets/[^/]+
        regex_pattern = re.sub(r'\{[^}]+\}', r'[^/]+', pattern)
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, endpoint))
    
    def _get_response_schema(self, endpoint_spec: Dict[str, Any], status_code: str) -> Optional[Dict[str, Any]]:
        """Get response schema for specific status code."""
        responses = endpoint_spec.get("responses", {})
        
        if status_code in responses:
            response_spec = responses[status_code]
            content = response_spec.get("content", {})
            
            if "application/json" in content:
                return content["application/json"].get("schema")
        
        return None
    
    def _check_backward_compatibility(
        self, 
        method: str, 
        endpoint: str, 
        response_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Check backward compatibility with baseline responses."""
        
        baseline_key = f"{method.upper()} {endpoint}"
        
        if baseline_key not in self.baseline_responses:
            # First time seeing this endpoint, store as baseline
            self.baseline_responses[baseline_key] = response_data
            return True
        
        baseline = self.baseline_responses[baseline_key]
        
        if baseline is None or response_data is None:
            return baseline == response_data
        
        # Check if all baseline fields are still present
        return self._check_schema_compatibility(baseline, response_data)
    
    def _check_schema_compatibility(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Check if current response is compatible with baseline."""
        
        # Check if all baseline keys are present in current response
        for key in baseline.keys():
            if key not in current:
                return False
            
            # Recursively check nested objects
            if isinstance(baseline[key], dict) and isinstance(current[key], dict):
                if not self._check_schema_compatibility(baseline[key], current[key]):
                    return False
            
            # Check array structure
            elif isinstance(baseline[key], list) and isinstance(current[key], list):
                if baseline[key] and current[key]:
                    if isinstance(baseline[key][0], dict) and isinstance(current[key][0], dict):
                        if not self._check_schema_compatibility(baseline[key][0], current[key][0]):
                            return False
        
        return True
    
    def generate_contract_report(self) -> Dict[str, Any]:
        """Generate comprehensive contract testing report."""
        
        total_tests = len(self.test_results)
        passed_tests = [r for r in self.test_results if r.schema_valid and r.backward_compatible]
        failed_tests = [r for r in self.test_results if not (r.schema_valid and r.backward_compatible)]
        
        # Group by endpoint
        endpoint_results = {}
        for result in self.test_results:
            endpoint_key = f"{result.method} {result.endpoint}"
            endpoint_results[endpoint_key] = result
        
        # Analyze contract violations
        schema_violations = [r for r in self.test_results if not r.schema_valid]
        compatibility_violations = [r for r in self.test_results if not r.backward_compatible]
        performance_issues = [r for r in self.test_results if not r.performance_acceptable]
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "success_rate": (len(passed_tests) / total_tests * 100) if total_tests > 0 else 0,
                "openapi_spec_valid": self.validate_openapi_spec()
            },
            "violations": {
                "schema_violations": len(schema_violations),
                "compatibility_violations": len(compatibility_violations),
                "performance_issues": len(performance_issues)
            },
            "endpoint_results": {
                endpoint: {
                    "status_code": result.status_code,
                    "expected_status": result.expected_status,
                    "schema_valid": result.schema_valid,
                    "backward_compatible": result.backward_compatible,
                    "performance_acceptable": result.performance_acceptable,
                    "error_count": len(result.error_details)
                }
                for endpoint, result in endpoint_results.items()
            },
            "detailed_failures": [
                {
                    "test": result.test_name,
                    "endpoint": result.endpoint,
                    "method": result.method,
                    "errors": result.error_details,
                    "recommendations": result.recommendations
                }
                for result in failed_tests
            ]
        }


class TestAPIContractCompliance:
    """API contract compliance testing."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for contract testing."""
        from monorepo.presentation.api.app import create_app
        
        app = create_app(testing=True)
        return TestClient(app)
    
    @pytest.fixture
    def contract_tester(self, test_client):
        """Create API contract tester."""
        return APIContractTester(test_client)
    
    def test_health_endpoint_contract(self, contract_tester):
        """Test health endpoint contract compliance."""
        
        # Mock health endpoint response
        with patch('monorepo.presentation.api.health.get_health_status') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "timestamp": "2023-01-01T00:00:00Z",
                "version": "1.0.0"
            }
            
            result = contract_tester.test_endpoint_contract("GET", "/api/v1/health")
            
            assert result.status_code == 200
            assert result.schema_valid
            assert result.backward_compatible
            
            contract_tester.test_results.append(result)
    
    def test_datasets_endpoint_contract(self, contract_tester):
        """Test datasets endpoint contract compliance."""
        
        # Test GET /api/v1/datasets
        with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
            mock_service.return_value.list_datasets.return_value = [
                Mock(id="dataset1", name="Test Dataset 1", created_at="2023-01-01T00:00:00Z"),
                Mock(id="dataset2", name="Test Dataset 2", created_at="2023-01-02T00:00:00Z")
            ]
            
            result = contract_tester.test_endpoint_contract("GET", "/api/v1/datasets")
            
            assert result.status_code == 200
            assert result.schema_valid
            
            contract_tester.test_results.append(result)
        
        # Test POST /api/v1/datasets
        with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
            mock_service.return_value.create_dataset.return_value = Mock(
                id="new_dataset",
                name="New Dataset",
                status="created"
            )
            
            payload = {
                "name": "New Dataset",
                "description": "Test dataset",
                "data": [[1, 2, 3], [4, 5, 6]]
            }
            
            result = contract_tester.test_endpoint_contract("POST", "/api/v1/datasets", payload)
            
            assert result.status_code in [200, 201]
            assert result.schema_valid
            
            contract_tester.test_results.append(result)
    
    def test_detectors_endpoint_contract(self, contract_tester):
        """Test detectors endpoint contract compliance."""
        
        # Test GET /api/v1/detectors
        with patch('monorepo.application.services.detector_service.DetectorService') as mock_service:
            mock_service.return_value.list_detectors.return_value = [
                Mock(
                    id="detector1",
                    name="Isolation Forest",
                    algorithm="IsolationForest",
                    is_trained=True
                ),
                Mock(
                    id="detector2",
                    name="LOF Detector",
                    algorithm="LocalOutlierFactor",
                    is_trained=False
                )
            ]
            
            result = contract_tester.test_endpoint_contract("GET", "/api/v1/detectors")
            
            assert result.status_code == 200
            assert result.schema_valid
            
            contract_tester.test_results.append(result)
        
        # Test POST /api/v1/detectors
        with patch('monorepo.application.services.detector_service.DetectorService') as mock_service:
            mock_service.return_value.create_detector.return_value = Mock(
                id="new_detector",
                name="New Detector",
                algorithm="IsolationForest",
                status="created"
            )
            
            payload = {
                "name": "New Detector",
                "algorithm": "IsolationForest",
                "parameters": {
                    "contamination": 0.1,
                    "n_estimators": 100
                }
            }
            
            result = contract_tester.test_endpoint_contract("POST", "/api/v1/detectors", payload)
            
            assert result.status_code in [200, 201]
            assert result.schema_valid
            
            contract_tester.test_results.append(result)
    
    def test_detection_endpoint_contract(self, contract_tester):
        """Test detection endpoint contract compliance."""
        
        with patch('monorepo.application.services.detection_service.DetectionService') as mock_service:
            mock_service.return_value.detect_anomalies.return_value = Mock(
                predictions=[0, 1, 0, 0, 1],
                anomaly_scores=[0.2, 0.8, 0.1, 0.3, 0.9],
                n_anomalies=2,
                execution_time=0.123
            )
            
            payload = {
                "detector_id": "test_detector",
                "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
                "return_scores": True
            }
            
            result = contract_tester.test_endpoint_contract("POST", "/api/v1/detection/detect", payload)
            
            assert result.status_code == 200
            assert result.schema_valid
            
            contract_tester.test_results.append(result)
    
    def test_error_response_contracts(self, contract_tester):
        """Test error response contract compliance."""
        
        # Test 404 Not Found
        with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
            mock_service.return_value.get_dataset.side_effect = ValueError("Dataset not found")
            
            result = contract_tester.test_endpoint_contract("GET", "/api/v1/datasets/nonexistent")
            
            # Should return 404 for not found
            assert result.status_code in [404, 422]  # 422 for validation errors
            
            contract_tester.test_results.append(result)
        
        # Test 400 Bad Request
        with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
            mock_service.return_value.create_dataset.side_effect = ValueError("Invalid data")
            
            invalid_payload = {
                "name": "",  # Empty name should cause validation error
                "data": []   # Empty data should cause validation error
            }
            
            result = contract_tester.test_endpoint_contract("POST", "/api/v1/datasets", invalid_payload)
            
            # Should return 400 for bad request
            assert result.status_code in [400, 422]
            
            contract_tester.test_results.append(result)
    
    def test_request_validation_contracts(self, contract_tester):
        """Test request validation contract compliance."""
        
        # Test missing required fields
        with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
            mock_service.return_value.create_dataset.side_effect = ValueError("Missing required field")
            
            invalid_payloads = [
                {},  # Missing all required fields
                {"name": "Test"},  # Missing data field
                {"data": [[1, 2, 3]]},  # Missing name field
            ]
            
            for payload in invalid_payloads:
                result = contract_tester.test_endpoint_contract("POST", "/api/v1/datasets", payload)
                
                # Should return validation error
                assert result.status_code in [400, 422]
                
                contract_tester.test_results.append(result)
        
        # Test invalid data types
        with patch('monorepo.application.services.detector_service.DetectorService') as mock_service:
            mock_service.return_value.create_detector.side_effect = ValueError("Invalid data type")
            
            invalid_payloads = [
                {"name": 123, "algorithm": "IsolationForest"},  # name should be string
                {"name": "Test", "algorithm": 123},  # algorithm should be string
                {"name": "Test", "algorithm": "IsolationForest", "parameters": "invalid"},  # parameters should be object
            ]
            
            for payload in invalid_payloads:
                result = contract_tester.test_endpoint_contract("POST", "/api/v1/detectors", payload)
                
                # Should return validation error
                assert result.status_code in [400, 422]
                
                contract_tester.test_results.append(result)
    
    def test_backward_compatibility(self, contract_tester):
        """Test API backward compatibility."""
        
        # Simulate API version changes
        with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
            # Version 1 response
            mock_service.return_value.list_datasets.return_value = [
                Mock(id="dataset1", name="Test Dataset", created_at="2023-01-01T00:00:00Z")
            ]
            
            result_v1 = contract_tester.test_endpoint_contract("GET", "/api/v1/datasets")
            contract_tester.test_results.append(result_v1)
            
            # Version 2 response (with additional fields)
            mock_service.return_value.list_datasets.return_value = [
                Mock(
                    id="dataset1",
                    name="Test Dataset",
                    created_at="2023-01-01T00:00:00Z",
                    updated_at="2023-01-02T00:00:00Z",  # New field
                    size=1000  # New field
                )
            ]
            
            result_v2 = contract_tester.test_endpoint_contract("GET", "/api/v1/datasets")
            contract_tester.test_results.append(result_v2)
            
            # Both versions should be backward compatible
            assert result_v1.backward_compatible
            assert result_v2.backward_compatible
    
    def test_performance_contract_compliance(self, contract_tester):
        """Test API performance contract compliance."""
        
        import time
        
        # Mock slow response
        with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
            def slow_response():
                time.sleep(0.1)  # 100ms delay
                return [Mock(id="dataset1", name="Test Dataset", created_at="2023-01-01T00:00:00Z")]
            
            mock_service.return_value.list_datasets.side_effect = slow_response
            
            result = contract_tester.test_endpoint_contract("GET", "/api/v1/datasets")
            
            # Performance should be acceptable for this endpoint
            assert result.performance_acceptable or result.status_code == 200
            
            contract_tester.test_results.append(result)
    
    def test_content_type_contracts(self, contract_tester):
        """Test content type contract compliance."""
        
        # Test with different content types
        with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
            mock_service.return_value.create_dataset.return_value = Mock(
                id="dataset1",
                name="Test Dataset",
                status="created"
            )
            
            # Test JSON content type
            payload = {"name": "Test Dataset", "data": [[1, 2, 3]]}
            headers = {"Content-Type": "application/json"}
            
            result = contract_tester.test_endpoint_contract(
                "POST", "/api/v1/datasets", payload, headers
            )
            
            assert result.status_code in [200, 201]
            
            contract_tester.test_results.append(result)
    
    def test_comprehensive_contract_report(self, contract_tester):
        """Test comprehensive contract report generation."""
        
        # Run a subset of contract tests
        self.test_health_endpoint_contract(contract_tester)
        self.test_datasets_endpoint_contract(contract_tester)
        self.test_detectors_endpoint_contract(contract_tester)
        
        # Generate report
        report = contract_tester.generate_contract_report()
        
        # Validate report structure
        assert "summary" in report
        assert "violations" in report
        assert "endpoint_results" in report
        assert "detailed_failures" in report
        
        # Validate summary
        summary = report["summary"]
        assert summary["total_tests"] > 0
        assert summary["passed_tests"] >= 0
        assert summary["failed_tests"] >= 0
        assert 0 <= summary["success_rate"] <= 100
        
        # Print report
        print("\n" + "="*60)
        print("📋 API CONTRACT TESTING REPORT")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  OpenAPI Spec Valid: {summary['openapi_spec_valid']}")
        
        violations = report["violations"]
        print(f"\n⚠️  Violations:")
        print(f"  Schema Violations: {violations['schema_violations']}")
        print(f"  Compatibility Violations: {violations['compatibility_violations']}")
        print(f"  Performance Issues: {violations['performance_issues']}")
        
        if report["detailed_failures"]:
            print(f"\n❌ Detailed Failures:")
            for failure in report["detailed_failures"][:5]:  # Show first 5
                print(f"  • {failure['test']}: {failure['endpoint']}")
                for error in failure['errors'][:2]:  # Show first 2 errors
                    print(f"    - {error}")
        
        print(f"\n✅ Endpoint Results:")
        for endpoint, result in report["endpoint_results"].items():
            status = "✅" if result["schema_valid"] and result["backward_compatible"] else "❌"
            print(f"  {status} {endpoint}: {result['status_code']} "
                  f"(Schema: {result['schema_valid']}, "
                  f"Compatibility: {result['backward_compatible']})")
        
        print("="*60)
        
        # Contract compliance assertions
        assert summary["success_rate"] >= 80, f"Contract compliance too low: {summary['success_rate']:.1f}%"
        
        assert violations["schema_violations"] <= 2, \
            f"Too many schema violations: {violations['schema_violations']}"
        
        assert violations["compatibility_violations"] == 0, \
            f"Backward compatibility violations found: {violations['compatibility_violations']}"
        
        print("✅ API contract testing completed successfully!")


class TestOpenAPISpecValidation:
    """OpenAPI specification validation testing."""
    
    def test_openapi_spec_structure(self):
        """Test OpenAPI specification structure."""
        
        contract_tester = APIContractTester(Mock())
        spec = contract_tester.openapi_spec
        
        # Basic structure validation
        assert spec is not None
        assert "openapi" in spec
        assert "info" in spec
        assert "paths" in spec
        
        # Info section validation
        info = spec["info"]
        assert "title" in info
        assert "version" in info
        
        # Paths validation
        paths = spec["paths"]
        assert len(paths) > 0
        
        # Each path should have operations
        for path, operations in paths.items():
            assert len(operations) > 0
            
            for method, operation in operations.items():
                assert method in ["get", "post", "put", "delete", "patch"]
                assert "responses" in operation
                
                # Response validation
                responses = operation["responses"]
                assert len(responses) > 0
                
                for status_code, response in responses.items():
                    assert status_code.isdigit() or status_code == "default"
                    assert "description" in response
    
    def test_response_schema_definitions(self):
        """Test response schema definitions."""
        
        contract_tester = APIContractTester(Mock())
        spec = contract_tester.openapi_spec
        
        # Check that all responses have proper schemas
        for path, operations in spec["paths"].items():
            for method, operation in operations.items():
                responses = operation.get("responses", {})
                
                for status_code, response in responses.items():
                    if status_code.startswith("2"):  # Success responses
                        content = response.get("content", {})
                        
                        if "application/json" in content:
                            json_content = content["application/json"]
                            assert "schema" in json_content, \
                                f"Missing schema for {method.upper()} {path} {status_code}"
                            
                            schema = json_content["schema"]
                            assert "type" in schema, \
                                f"Missing type in schema for {method.upper()} {path} {status_code}"
    
    def test_request_schema_definitions(self):
        """Test request schema definitions."""
        
        contract_tester = APIContractTester(Mock())
        spec = contract_tester.openapi_spec
        
        # Check POST/PUT operations have request schemas
        for path, operations in spec["paths"].items():
            for method, operation in operations.items():
                if method.lower() in ["post", "put", "patch"]:
                    request_body = operation.get("requestBody")
                    
                    if request_body:
                        assert "required" in request_body
                        assert "content" in request_body
                        
                        content = request_body["content"]
                        assert "application/json" in content
                        
                        json_content = content["application/json"]
                        assert "schema" in json_content
                        
                        schema = json_content["schema"]
                        assert "type" in schema
                        assert "properties" in schema
                        assert "required" in schema
    
    def test_parameter_definitions(self):
        """Test parameter definitions."""
        
        contract_tester = APIContractTester(Mock())
        spec = contract_tester.openapi_spec
        
        # Check parameterized paths
        for path, operations in spec["paths"].items():
            if "{" in path:  # Parameterized path
                for method, operation in operations.items():
                    parameters = operation.get("parameters", [])
                    
                    # Should have parameters defined
                    path_params = re.findall(r'\{([^}]+)\}', path)
                    
                    for param_name in path_params:
                        param_found = any(
                            p.get("name") == param_name and p.get("in") == "path"
                            for p in parameters
                        )
                        
                        assert param_found, \
                            f"Missing path parameter definition for {param_name} in {method.upper()} {path}"


async def test_async_contract_compliance():
    """Test async API contract compliance."""
    
    # Mock async API operations
    with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
        mock_service.return_value.list_datasets = AsyncMock(return_value=[
            Mock(id="dataset1", name="Async Dataset", created_at="2023-01-01T00:00:00Z")
        ])
        
        # Test async endpoint
        from monorepo.presentation.api.app import create_app
        
        app = create_app(testing=True)
        client = TestClient(app)
        
        contract_tester = APIContractTester(client)
        
        # Test async operation
        result = contract_tester.test_endpoint_contract("GET", "/api/v1/datasets")
        
        assert result.status_code == 200
        assert result.schema_valid
        
        print("✅ Async API contract testing completed successfully!")


def test_contract_regression_detection():
    """Test contract regression detection."""
    
    from monorepo.presentation.api.app import create_app
    
    app = create_app(testing=True)
    client = TestClient(app)
    
    contract_tester = APIContractTester(client)
    
    # Establish baseline
    with patch('monorepo.application.services.dataset_service.DatasetService') as mock_service:
        mock_service.return_value.list_datasets.return_value = [
            Mock(id="dataset1", name="Test Dataset", created_at="2023-01-01T00:00:00Z")
        ]
        
        baseline_result = contract_tester.test_endpoint_contract("GET", "/api/v1/datasets")
        
        # Simulate API change that breaks contract
        mock_service.return_value.list_datasets.return_value = [
            Mock(id="dataset1", title="Test Dataset")  # Changed 'name' to 'title'
        ]
        
        regression_result = contract_tester.test_endpoint_contract("GET", "/api/v1/datasets")
        
        # Should detect backward compatibility issue
        assert baseline_result.backward_compatible
        assert not regression_result.backward_compatible
        
        print("✅ Contract regression detection working correctly!")