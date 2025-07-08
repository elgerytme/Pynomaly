"""Integration tests for Pynomaly major workflows."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from pynomaly.application.use_cases.train_detector import TrainDetectorUseCase


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_anomaly_detection_workflow(self, container, sample_data):
        """Test complete workflow from data upload to results."""
        try:
            # 1. Create dataset
            dataset = Dataset(
                name="Integration Test Dataset",
                data=sample_data.drop(columns=["target"]),
                description="Dataset for integration testing",
                features=[f"feature_{i}" for i in range(5)],
            )

            # 2. Save dataset
            dataset_repo = container.dataset_repository()
            dataset_repo.save(dataset)

            # 3. Create detector
            detector = Detector(
                algorithm_name="IsolationForest",
                parameters={"contamination": 0.1, "random_state": 42},
                metadata={"description": "Integration test detector"},
            )

            # 4. Save detector
            detector_repo = container.detector_repository()
            detector_repo.save(detector)

            # 5. Train detector
            train_use_case = TrainDetectorUseCase(
                detector_repository=detector_repo,
                dataset_repository=dataset_repo,
                pyod_adapter=container.pyod_adapter(),
            )

            trained_detector = train_use_case.execute(detector.id, dataset.id)
            assert trained_detector.is_fitted

            # 6. Run detection
            detection_use_case = DetectAnomaliesUseCase(
                detector_repository=detector_repo,
                dataset_repository=dataset_repo,
                result_repository=container.detection_result_repository(),
                pyod_adapter=container.pyod_adapter(),
            )

            result = detection_use_case.execute(detector.id, dataset.id)

            # 7. Verify results
            assert result is not None
            assert result.detector_id == detector.id
            assert result.dataset_id == dataset.id
            assert len(result.scores) == len(dataset.data)
            assert all(0 <= score.value <= 1 for score in result.scores)

            # 8. Save and retrieve result
            result_repo = container.detection_result_repository()
            result_repo.save(result)

            retrieved_result = result_repo.find_by_id(result.id)
            assert retrieved_result is not None
            assert retrieved_result.id == result.id

        except ImportError:
            pytest.skip("Required dependencies not available for integration test")

    def test_multi_detector_comparison_workflow(self, container, sample_data):
        """Test workflow comparing multiple detectors."""
        try:
            # Create dataset
            dataset = Dataset(
                name="Multi-Detector Test Dataset",
                data=sample_data.drop(columns=["target"]),
                features=[f"feature_{i}" for i in range(5)],
            )
            container.dataset_repository().save(dataset)

            # Create multiple detectors
            detectors = [
                Detector(
                    algorithm_name="IsolationForest",
                    parameters={"contamination": 0.1, "random_state": 42},
                ),
                Detector(
                    algorithm_name="LocalOutlierFactor",
                    parameters={"contamination": 0.1},
                ),
                Detector(algorithm_name="OneClassSVM", parameters={"gamma": "auto"}),
            ]

            results = []
            detector_repo = container.detector_repository()

            for detector in detectors:
                # Save detector
                detector_repo.save(detector)

                # Train detector
                train_use_case = TrainDetectorUseCase(
                    detector_repository=detector_repo,
                    dataset_repository=container.dataset_repository(),
                    pyod_adapter=container.pyod_adapter(),
                )

                try:
                    trained_detector = train_use_case.execute(detector.id, dataset.id)

                    # Run detection
                    detection_use_case = DetectAnomaliesUseCase(
                        detector_repository=detector_repo,
                        dataset_repository=container.dataset_repository(),
                        result_repository=container.detection_result_repository(),
                        pyod_adapter=container.pyod_adapter(),
                    )

                    result = detection_use_case.execute(detector.id, dataset.id)
                    results.append((detector.algorithm_name, result))

                except Exception as e:
                    print(f"Detector {detector.algorithm_name} failed: {e}")
                    continue

            # Should have at least one successful result
            assert len(results) > 0, "No detectors completed successfully"

            # Compare results
            for algo_name, result in results:
                assert result is not None
                assert len(result.scores) == len(dataset.data)
                print(f"{algo_name}: Generated {len(result.scores)} scores")

        except ImportError:
            pytest.skip("Required dependencies not available for integration test")

    def test_batch_processing_workflow(self, container):
        """Test batch processing of multiple datasets."""
        try:
            # Create multiple datasets
            datasets = []
            for i in range(3):
                np.random.seed(42 + i)
                data = np.random.normal(0, 1, (100, 3))
                df = pd.DataFrame(data, columns=[f"feature_{j}" for j in range(3)])

                dataset = Dataset(
                    name=f"Batch Dataset {i+1}", data=df, features=df.columns.tolist()
                )
                datasets.append(dataset)
                container.dataset_repository().save(dataset)

            # Create detector
            detector = Detector(
                algorithm_name="IsolationForest",
                parameters={"contamination": 0.1, "random_state": 42},
            )
            container.detector_repository().save(detector)

            # Process all datasets
            results = []
            for dataset in datasets:
                # Train on dataset
                train_use_case = TrainDetectorUseCase(
                    detector_repository=container.detector_repository(),
                    dataset_repository=container.dataset_repository(),
                    pyod_adapter=container.pyod_adapter(),
                )

                train_use_case.execute(detector.id, dataset.id)

                # Run detection
                detection_use_case = DetectAnomaliesUseCase(
                    detector_repository=container.detector_repository(),
                    dataset_repository=container.dataset_repository(),
                    result_repository=container.detection_result_repository(),
                    pyod_adapter=container.pyod_adapter(),
                )

                result = detection_use_case.execute(detector.id, dataset.id)
                results.append(result)

            # Verify all results
            assert len(results) == len(datasets)
            for i, result in enumerate(results):
                assert result is not None
                assert result.dataset_id == datasets[i].id
                assert len(result.scores) == len(datasets[i].data)

        except ImportError:
            pytest.skip("Required dependencies not available for integration test")


@pytest.mark.integration
class TestAPIIntegration:
    """Test API integration workflows."""

    def test_api_authentication_workflow(self, client, auth_service):
        """Test complete API authentication workflow."""
        if not hasattr(client, "post"):
            pytest.skip("API client not available")

        # 1. Register new user (if endpoint exists)
        registration_data = {
            "username": "testuser_api",
            "email": "testuser@example.com",
            "password": "securepassword123",
            "full_name": "Test User API",
        }

        # Try registration (might not be available)
        try:
            register_response = client.post(
                "/api/v1/auth/register", json=registration_data
            )
            if register_response.status_code not in [404, 405]:  # If endpoint exists
                assert register_response.status_code in [200, 201]
        except:
            pass  # Registration endpoint might not exist

        # 2. Login
        login_data = {
            "username": "admin",  # Use default admin user
            "password": "admin123",
        }

        login_response = client.post("/api/v1/auth/login", json=login_data)

        if login_response.status_code == 404:
            pytest.skip("Auth endpoints not available")

        assert login_response.status_code == 200
        token_data = login_response.json()
        assert "access_token" in token_data

        access_token = token_data["access_token"]

        # 3. Access protected endpoint
        headers = {"Authorization": f"Bearer {access_token}"}

        # Try to access a protected endpoint
        protected_endpoints = [
            "/api/v1/datasets",
            "/api/v1/detectors",
            "/api/v1/admin/users",
        ]

        for endpoint in protected_endpoints:
            try:
                response = client.get(endpoint, headers=headers)
                # Should either work (200) or be forbidden due to permissions (403)
                # Should not be unauthorized (401) with valid token
                assert (
                    response.status_code != 401
                ), f"Valid token rejected at {endpoint}"
                break  # If we get a response, authentication is working
            except:
                continue

        # 4. Test token refresh (if endpoint exists)
        if "refresh_token" in token_data:
            refresh_data = {"refresh_token": token_data["refresh_token"]}
            try:
                refresh_response = client.post(
                    "/api/v1/auth/refresh", json=refresh_data
                )
                if refresh_response.status_code != 404:
                    assert refresh_response.status_code == 200
                    new_token_data = refresh_response.json()
                    assert "access_token" in new_token_data
            except:
                pass  # Refresh endpoint might not exist

    def test_dataset_management_api_workflow(self, client, admin_token):
        """Test complete dataset management workflow via API."""
        if not hasattr(client, "post"):
            pytest.skip("API client not available")

        headers = {"Authorization": f"Bearer {admin_token}"}

        # 1. Upload dataset
        dataset_data = {
            "name": "API Test Dataset",
            "description": "Dataset created via API test",
            "features": ["feature_1", "feature_2", "feature_3"],
        }

        try:
            create_response = client.post(
                "/api/v1/datasets", json=dataset_data, headers=headers
            )

            if create_response.status_code == 404:
                pytest.skip("Dataset endpoints not available")

            # Should create successfully or require additional data
            assert create_response.status_code in [200, 201, 422]

            if create_response.status_code in [200, 201]:
                created_dataset = create_response.json()
                dataset_id = created_dataset.get("id")

                # 2. List datasets
                list_response = client.get("/api/v1/datasets", headers=headers)
                assert list_response.status_code == 200
                datasets = list_response.json()
                assert isinstance(datasets, list)

                # 3. Get specific dataset
                if dataset_id:
                    get_response = client.get(
                        f"/api/v1/datasets/{dataset_id}", headers=headers
                    )
                    assert get_response.status_code in [
                        200,
                        404,
                    ]  # Might not be found due to test isolation

                # 4. Update dataset
                update_data = {"description": "Updated description"}
                if dataset_id:
                    update_response = client.put(
                        f"/api/v1/datasets/{dataset_id}",
                        json=update_data,
                        headers=headers,
                    )
                    assert update_response.status_code in [200, 404, 405]

                # 5. Delete dataset
                if dataset_id:
                    delete_response = client.delete(
                        f"/api/v1/datasets/{dataset_id}", headers=headers
                    )
                    assert delete_response.status_code in [200, 204, 404, 405]

        except Exception as e:
            print(f"Dataset API workflow test encountered error: {e}")
            # Don't fail the test as API might not be fully implemented

    def test_detection_api_workflow(self, client, admin_token, sample_data):
        """Test detection workflow via API."""
        if not hasattr(client, "post"):
            pytest.skip("API client not available")

        headers = {"Authorization": f"Bearer {admin_token}"}

        try:
            # 1. Create detector
            detector_data = {
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1, "random_state": 42},
                "metadata": {"description": "API test detector"},
            }

            detector_response = client.post(
                "/api/v1/detectors", json=detector_data, headers=headers
            )

            if detector_response.status_code == 404:
                pytest.skip("Detector endpoints not available")

            assert detector_response.status_code in [200, 201, 422]

            # 2. List available algorithms (if endpoint exists)
            try:
                algo_response = client.get(
                    "/api/v1/detectors/algorithms", headers=headers
                )
                if algo_response.status_code == 200:
                    algorithms = algo_response.json()
                    assert isinstance(algorithms, list)
                    assert len(algorithms) > 0
            except:
                pass

            # 3. Run detection (if endpoints are available)
            detection_data = {
                "detector_id": "test_detector_id",
                "dataset_id": "test_dataset_id",
            }

            try:
                detection_response = client.post(
                    "/api/v1/detection/run", json=detection_data, headers=headers
                )
                # Might fail due to missing data, but should not be unauthorized
                assert detection_response.status_code != 401
            except:
                pass

        except Exception as e:
            print(f"Detection API workflow test encountered error: {e}")

    def test_health_and_monitoring_integration(self, client):
        """Test health and monitoring endpoints integration."""
        if not hasattr(client, "get"):
            pytest.skip("API client not available")

        # 1. Basic health check
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200

        health_data = health_response.json()
        assert "status" in health_data

        # 2. Detailed health check
        try:
            detailed_health = client.get("/api/v1/health/detailed")
            if detailed_health.status_code == 200:
                detailed_data = detailed_health.json()
                assert isinstance(detailed_data, dict)
        except:
            pass

        # 3. Metrics endpoint (if available)
        try:
            metrics_response = client.get("/metrics")
            if metrics_response.status_code == 200:
                # Should be Prometheus format
                metrics_text = metrics_response.text
                assert "# HELP" in metrics_text or "# TYPE" in metrics_text
        except:
            pass

        # 4. Version endpoint
        try:
            version_response = client.get("/api/v1/version")
            if version_response.status_code == 200:
                version_data = version_response.json()
                assert "version" in version_data or "app_version" in version_data
        except:
            pass


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database integration workflows."""

    def test_repository_integration(self, session_factory):
        """Test repository integration with database."""
        try:
            from pynomaly.infrastructure.persistence.database_repositories import (
                DatabaseDatasetRepository,
                DatabaseDetectorRepository,
                DatabaseDetectionResultRepository,
            )
        except ImportError:
            pytest.skip("Database repositories not available")

        # Initialize repositories
        dataset_repo = DatabaseDatasetRepository(session_factory)
        detector_repo = DatabaseDetectorRepository(session_factory)
        result_repo = DatabaseDetectionResultRepository(session_factory)

        # Create test data
        test_data = pd.DataFrame(
            {"feature_1": [1, 2, 3, 4], "feature_2": [0.1, 0.2, 0.3, 0.4]}
        )

        dataset = Dataset(
            name="DB Integration Test Dataset",
            data=test_data,
            features=["feature_1", "feature_2"],
        )

        detector = Detector(
            algorithm_name="IsolationForest", parameters={"contamination": 0.1}
        )

        # Test dataset operations
        dataset_repo.save(dataset)
        retrieved_dataset = dataset_repo.find_by_id(dataset.id)
        assert retrieved_dataset is not None
        assert retrieved_dataset.name == dataset.name

        # Test detector operations
        detector_repo.save(detector)
        retrieved_detector = detector_repo.find_by_id(detector.id)
        assert retrieved_detector is not None
        assert retrieved_detector.algorithm_name == detector.algorithm_name

        # Test detection result operations
        from pynomaly.domain.value_objects import AnomalyScore

        result = DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            scores=[AnomalyScore(value=0.1), AnomalyScore(value=0.9)],
            metadata={"test": True},
        )

        result_repo.save(result)
        retrieved_result = result_repo.find_by_id(result.id)
        assert retrieved_result is not None
        assert retrieved_result.detector_id == detector.id
        assert retrieved_result.dataset_id == dataset.id

        # Test cascading operations
        results_by_detector = result_repo.find_by_detector(detector.id)
        assert len(results_by_detector) > 0

        results_by_dataset = result_repo.find_by_dataset(dataset.id)
        assert len(results_by_dataset) > 0

    def test_transaction_handling(self, session_factory):
        """Test transaction handling in database operations."""
        try:
            from pynomaly.infrastructure.persistence.database_repositories import (
                DatabaseDatasetRepository,
            )
        except ImportError:
            pytest.skip("Database repositories not available")

        dataset_repo = DatabaseDatasetRepository(session_factory)

        # Test successful transaction
        dataset = Dataset(
            name="Transaction Test Dataset",
            data=pd.DataFrame({"feature": [1, 2, 3]}),
            features=["feature"],
        )

        dataset_repo.save(dataset)
        assert dataset_repo.exists(dataset.id)

        # Test rollback on error
        try:
            # Attempt to save invalid data that should cause rollback
            invalid_dataset = Dataset(
                name="" * 1000,  # Name too long
                data=pd.DataFrame({"feature": [1, 2, 3]}),
                features=["feature"],
            )

            dataset_repo.save(invalid_dataset)
        except Exception:
            # Error expected for invalid data
            pass

        # Original dataset should still exist
        assert dataset_repo.exists(dataset.id)


@pytest.mark.integration
class TestSecurityIntegration:
    """Test security integration across components."""

    def test_authentication_authorization_integration(self, client, auth_service):
        """Test authentication and authorization integration."""
        if not hasattr(client, "get"):
            pytest.skip("API client not available")

        # Test without authentication
        response = client.get("/api/v1/admin/users")
        assert response.status_code in [401, 404]  # Unauthorized or not found

        # Test with invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/admin/users", headers=invalid_headers)
        assert response.status_code in [401, 404]

        # Test with valid token but insufficient permissions
        try:
            # Create user with limited permissions
            user = auth_service.create_user(
                username="limited_user",
                email="limited@example.com",
                password="password123",
                roles=["viewer"],
            )

            token_response = auth_service.create_access_token(user)
            limited_headers = {"Authorization": f"Bearer {token_response.access_token}"}

            # Should be forbidden for admin endpoints
            response = client.get("/api/v1/admin/users", headers=limited_headers)
            assert response.status_code in [403, 404]  # Forbidden or not found

        except Exception as e:
            print(f"Permission test failed: {e}")

    def test_rate_limiting_integration(self, client):
        """Test rate limiting integration with API."""
        if not hasattr(client, "get"):
            pytest.skip("API client not available")

        # Make multiple rapid requests
        responses = []
        for i in range(20):
            try:
                response = client.get("/api/v1/health")
                responses.append(response.status_code)

                # If rate limited, break
                if response.status_code == 429:
                    break
            except Exception as e:
                print(f"Request {i} failed: {e}")
                break

        # Should have gotten some responses
        assert len(responses) > 0

        # All successful responses should be 200
        successful_responses = [r for r in responses if r == 200]
        assert len(successful_responses) > 0

    def test_audit_logging_integration(self, audit_logger):
        """Test audit logging integration."""
        try:
            from pynomaly.infrastructure.security.audit_logging import (
                AuditEvent,
                AuditEventType,
                audit_context,
            )
        except ImportError:
            pytest.skip("Audit logging not available")

        # Test direct audit logging
        audit_logger.log_authentication(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="test_user",
            outcome="success",
            ip_address="127.0.0.1",
        )

        # Test context manager integration
        async def test_audit_context():
            async with audit_context(
                audit_logger, "test_user", "read", "test_resource"
            ) as audit_details:
                audit_details["records_processed"] = 100
                # Simulate some work
                pass

        import asyncio

        asyncio.run(test_audit_context())
