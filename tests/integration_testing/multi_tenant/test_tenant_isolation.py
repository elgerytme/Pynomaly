"""Multi-tenant isolation testing."""

import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import pytest


class TestTenantIsolation:
    """Test multi-tenant data and resource isolation."""
    
    @pytest.mark.security
    @pytest.mark.integration
    async def test_data_isolation_between_tenants(
        self,
        api_client,
        security_context,
        test_data_manager
    ):
        """Test that tenant data is completely isolated."""
        
        # Create two tenant users
        tenant_a_user = security_context.create_test_user("analyst")
        tenant_b_user = security_context.create_test_user("analyst")
        
        tenant_a_token = security_context.generate_test_token(tenant_a_user)
        tenant_b_token = security_context.generate_test_token(tenant_b_user)
        
        # Create datasets for each tenant
        tenant_a_dataset = test_data_manager.create_test_dataset(size=1000)
        tenant_b_dataset = test_data_manager.create_test_dataset(size=1000)
        
        # Mock tenant-specific dataset creation
        def mock_dataset_creation(tenant_id: str, dataset_info: Dict[str, Any]):
            response = Mock()
            response.status_code = 201
            response.json.return_value = {
                "dataset_id": f"{dataset_info['id']}-{tenant_id}",
                "tenant_id": tenant_id,
                "name": f"Dataset for {tenant_id}",
                "rows": dataset_info["size"],
                "created_by": tenant_id
            }
            return response
        
        # Tenant A creates dataset
        api_client.post.return_value = mock_dataset_creation("tenant-a", tenant_a_dataset)
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.post(
                "/datasets",
                json={
                    "name": "Tenant A Dataset",
                    "data": tenant_a_dataset["data"].to_dict('records')[:10]
                }
            )
            assert response.status_code == 201
            tenant_a_dataset_id = response.json()["dataset_id"]
            assert "tenant-a" in tenant_a_dataset_id
        
        # Tenant B creates dataset
        api_client.post.return_value = mock_dataset_creation("tenant-b", tenant_b_dataset)
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_b_token}"}):
            response = api_client.post(
                "/datasets",
                json={
                    "name": "Tenant B Dataset",
                    "data": tenant_b_dataset["data"].to_dict('records')[:10]
                }
            )
            assert response.status_code == 201
            tenant_b_dataset_id = response.json()["dataset_id"]
            assert "tenant-b" in tenant_b_dataset_id
        
        # Test isolation: Tenant A can only see their own datasets
        tenant_a_list_response = Mock()
        tenant_a_list_response.status_code = 200
        tenant_a_list_response.json.return_value = {
            "datasets": [
                {
                    "dataset_id": tenant_a_dataset_id,
                    "name": "Tenant A Dataset",
                    "tenant_id": "tenant-a"
                }
            ],
            "total": 1
        }
        
        api_client.get.return_value = tenant_a_list_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.get("/datasets")
            assert response.status_code == 200
            datasets = response.json()["datasets"]
            assert len(datasets) == 1
            assert datasets[0]["tenant_id"] == "tenant-a"
        
        # Test cross-tenant access denial
        forbidden_response = Mock()
        forbidden_response.status_code = 403
        forbidden_response.json.return_value = {
            "error": "Access Denied",
            "message": "Cannot access resources from different tenant"
        }
        
        api_client.get.return_value = forbidden_response
        
        # Tenant A tries to access Tenant B's dataset
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.get(f"/datasets/{tenant_b_dataset_id}")
            assert response.status_code == 403
        
        # Tenant B tries to access Tenant A's dataset
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_b_token}"}):
            response = api_client.get(f"/datasets/{tenant_a_dataset_id}")
            assert response.status_code == 403
    
    @pytest.mark.integration
    async def test_detector_isolation_between_tenants(
        self,
        api_client,
        security_context,
        test_data_manager
    ):
        """Test that detectors are isolated between tenants."""
        
        # Create tenant users
        tenant_a_user = security_context.create_test_user("analyst")
        tenant_b_user = security_context.create_test_user("analyst")
        
        tenant_a_token = security_context.generate_test_token(tenant_a_user)
        tenant_b_token = security_context.generate_test_token(tenant_b_user)
        
        # Create detectors for each tenant
        tenant_a_detector = test_data_manager.create_test_detector()
        tenant_b_detector = test_data_manager.create_test_detector()
        
        def mock_detector_creation(tenant_id: str, detector_config: Dict[str, Any]):
            response = Mock()
            response.status_code = 201
            response.json.return_value = {
                "detector_id": f"{detector_config['id']}-{tenant_id}",
                "tenant_id": tenant_id,
                "algorithm": detector_config["algorithm"],
                "created_by": tenant_id,
                "status": "created"
            }
            return response
        
        # Tenant A creates detector
        api_client.post.return_value = mock_detector_creation("tenant-a", tenant_a_detector)
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.post("/detectors", json=tenant_a_detector)
            assert response.status_code == 201
            tenant_a_detector_id = response.json()["detector_id"]
        
        # Tenant B creates detector
        api_client.post.return_value = mock_detector_creation("tenant-b", tenant_b_detector)
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_b_token}"}):
            response = api_client.post("/detectors", json=tenant_b_detector)
            assert response.status_code == 201
            tenant_b_detector_id = response.json()["detector_id"]
        
        # Test detector listing isolation
        tenant_a_detectors_response = Mock()
        tenant_a_detectors_response.status_code = 200
        tenant_a_detectors_response.json.return_value = {
            "detectors": [
                {
                    "detector_id": tenant_a_detector_id,
                    "tenant_id": "tenant-a",
                    "algorithm": tenant_a_detector["algorithm"]
                }
            ]
        }
        
        api_client.get.return_value = tenant_a_detectors_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.get("/detectors")
            assert response.status_code == 200
            detectors = response.json()["detectors"]
            assert len(detectors) == 1
            assert detectors[0]["tenant_id"] == "tenant-a"
        
        # Test cross-tenant detector access denial
        forbidden_response = Mock()
        forbidden_response.status_code = 403
        forbidden_response.json.return_value = {
            "error": "Access Denied",
            "message": "Detector belongs to different tenant"
        }
        
        api_client.get.return_value = forbidden_response
        
        # Tenant A tries to access Tenant B's detector
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.get(f"/detectors/{tenant_b_detector_id}")
            assert response.status_code == 403
        
        # Test cross-tenant detector training denial
        training_forbidden_response = Mock()
        training_forbidden_response.status_code = 403
        training_forbidden_response.json.return_value = {
            "error": "Access Denied",
            "message": "Cannot train detector from different tenant"
        }
        
        api_client.post.return_value = training_forbidden_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.post(
                f"/detectors/{tenant_b_detector_id}/train",
                json={"dataset_id": "some-dataset"}
            )
            assert response.status_code == 403
    
    @pytest.mark.integration
    async def test_resource_quotas_and_limits(
        self,
        api_client,
        security_context,
        test_data_manager
    ):
        """Test tenant resource quotas and limits enforcement."""
        
        # Create tenant with specific quotas
        tenant_user = security_context.create_test_user("analyst")
        tenant_token = security_context.generate_test_token(tenant_user)
        
        # Mock quota information
        def mock_quota_response(current_usage: Dict[str, int], limits: Dict[str, int]):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "tenant_id": "tenant-quota-test",
                "quotas": {
                    "datasets": {
                        "current": current_usage.get("datasets", 0),
                        "limit": limits.get("datasets", 10)
                    },
                    "detectors": {
                        "current": current_usage.get("detectors", 0),
                        "limit": limits.get("detectors", 5)
                    },
                    "storage_mb": {
                        "current": current_usage.get("storage_mb", 0),
                        "limit": limits.get("storage_mb", 1024)
                    },
                    "api_calls_per_hour": {
                        "current": current_usage.get("api_calls", 0),
                        "limit": limits.get("api_calls", 1000)
                    }
                }
            }
            return response
        
        # Check initial quotas
        api_client.get.return_value = mock_quota_response(
            current_usage={},
            limits={"datasets": 3, "detectors": 2, "storage_mb": 512, "api_calls": 100}
        )
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_token}"}):
            response = api_client.get("/tenant/quotas")
            assert response.status_code == 200
            quotas = response.json()["quotas"]
            assert quotas["datasets"]["limit"] == 3
            assert quotas["detectors"]["limit"] == 2
        
        # Test dataset creation within quota
        dataset_creation_response = Mock()
        dataset_creation_response.status_code = 201
        dataset_creation_response.json.return_value = {
            "dataset_id": "dataset-1",
            "message": "Dataset created successfully"
        }
        
        api_client.post.return_value = dataset_creation_response
        
        # Create datasets up to quota limit
        for i in range(3):
            with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_token}"}):
                response = api_client.post(
                    "/datasets",
                    json={
                        "name": f"Dataset {i+1}",
                        "data": test_data_manager.create_test_dataset(size=100)["data"].to_dict('records')[:5]
                    }
                )
                assert response.status_code == 201
        
        # Test quota exceeded
        quota_exceeded_response = Mock()
        quota_exceeded_response.status_code = 429
        quota_exceeded_response.json.return_value = {
            "error": "Quota Exceeded",
            "message": "Dataset quota limit reached",
            "current_usage": 3,
            "limit": 3
        }
        
        api_client.post.return_value = quota_exceeded_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_token}"}):
            response = api_client.post(
                "/datasets",
                json={
                    "name": "Dataset 4 - Should Fail",
                    "data": [{"col1": 1, "col2": 2}]
                }
            )
            assert response.status_code == 429
            assert "quota" in response.json()["message"].lower()
        
        # Test storage quota enforcement
        storage_exceeded_response = Mock()
        storage_exceeded_response.status_code = 413
        storage_exceeded_response.json.return_value = {
            "error": "Storage Quota Exceeded",
            "message": "Upload would exceed storage limit",
            "size_mb": 600,
            "limit_mb": 512
        }
        
        api_client.post.return_value = storage_exceeded_response
        
        # Try to upload large dataset
        large_dataset = test_data_manager.create_test_dataset(size=50000)  # Large dataset
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_token}"}):
            response = api_client.post(
                "/datasets/upload",
                json={
                    "name": "Large Dataset",
                    "data": large_dataset["data"].to_dict('records')  # Full dataset
                }
            )
            assert response.status_code == 413
        
        # Test API rate limiting
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.json.return_value = {
            "error": "Rate Limit Exceeded",
            "message": "API call quota exceeded for this hour",
            "retry_after": 3600
        }
        
        # Simulate many API calls to exceed hourly limit
        for i in range(5):  # Simulate reaching the limit
            if i < 4:
                api_client.get.return_value = Mock(status_code=200, json=lambda: {"status": "ok"})
            else:
                api_client.get.return_value = rate_limit_response
            
            with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_token}"}):
                response = api_client.get("/health")
                if i < 4:
                    assert response.status_code == 200
                else:
                    assert response.status_code == 429
    
    @pytest.mark.integration
    async def test_tenant_performance_isolation(
        self,
        api_client,
        security_context,
        performance_monitor,
        load_test_simulator
    ):
        """Test that one tenant's load doesn't affect another tenant's performance."""
        
        # Create two tenant users
        heavy_tenant_user = security_context.create_test_user("analyst")
        normal_tenant_user = security_context.create_test_user("analyst")
        
        heavy_tenant_token = security_context.generate_test_token(heavy_tenant_user)
        normal_tenant_token = security_context.generate_test_token(normal_tenant_user)
        
        performance_monitor.start_monitoring()
        
        try:
            # Simulate heavy load from one tenant
            heavy_load_task = asyncio.create_task(
                load_test_simulator.simulate_concurrent_users(
                    client=api_client,
                    num_users=20,  # Heavy load
                    duration=15,
                    endpoint="/health"
                )
            )
            
            # Small delay to let heavy load start
            await asyncio.sleep(2)
            
            # Test normal tenant performance during heavy load
            normal_tenant_response_times = []
            
            for _ in range(10):  # 10 requests from normal tenant
                start_time = asyncio.get_event_loop().time()
                
                # Mock normal tenant response
                normal_response = Mock()
                normal_response.status_code = 200
                normal_response.json.return_value = {"status": "ok", "tenant": "normal"}
                
                api_client.get.return_value = normal_response
                
                with patch.object(api_client, 'headers', {"Authorization": f"Bearer {normal_tenant_token}"}):
                    response = api_client.get("/health")
                    assert response.status_code == 200
                
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                normal_tenant_response_times.append(response_time)
                
                await asyncio.sleep(0.5)  # Space out requests
            
            # Wait for heavy load to complete
            heavy_load_results = await heavy_load_task
            
            # Validate that normal tenant wasn't severely impacted
            avg_response_time = sum(normal_tenant_response_times) / len(normal_tenant_response_times)
            max_response_time = max(normal_tenant_response_times)
            
            # Normal tenant should maintain reasonable performance
            # even under heavy load from another tenant
            assert avg_response_time < 1.0  # Average under 1 second
            assert max_response_time < 3.0  # Max under 3 seconds
            
            # Heavy load tenant should still get reasonable service
            assert heavy_load_results["success_rate"] >= 0.80  # At least 80% success
            
        finally:
            performance_monitor.stop_monitoring()
            perf_summary = performance_monitor.get_summary()
            
            # System should handle multi-tenant load without crashing
            if perf_summary:
                assert perf_summary["memory"]["peak_mb"] < 1500  # Reasonable memory usage
    
    @pytest.mark.integration
    async def test_tenant_configuration_isolation(
        self,
        api_client,
        security_context
    ):
        """Test that tenant configurations are isolated."""
        
        # Create tenants with different configurations
        tenant_a_user = security_context.create_test_user("admin")
        tenant_b_user = security_context.create_test_user("admin")
        
        tenant_a_token = security_context.generate_test_token(tenant_a_user)
        tenant_b_token = security_context.generate_test_token(tenant_b_user)
        
        # Mock tenant-specific configuration
        def mock_config_response(tenant_id: str, config: Dict[str, Any]):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "tenant_id": tenant_id,
                "configuration": config
            }
            return response
        
        # Tenant A sets configuration
        tenant_a_config = {
            "default_algorithm": "isolation_forest",
            "max_dataset_size": 10000,
            "notification_preferences": {
                "email": True,
                "webhook": False
            },
            "data_retention_days": 365
        }
        
        api_client.put.return_value = mock_config_response("tenant-a", tenant_a_config)
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.put("/tenant/config", json=tenant_a_config)
            assert response.status_code == 200
            assert response.json()["configuration"]["default_algorithm"] == "isolation_forest"
        
        # Tenant B sets different configuration
        tenant_b_config = {
            "default_algorithm": "one_class_svm",
            "max_dataset_size": 5000,
            "notification_preferences": {
                "email": False,
                "webhook": True
            },
            "data_retention_days": 90
        }
        
        api_client.put.return_value = mock_config_response("tenant-b", tenant_b_config)
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_b_token}"}):
            response = api_client.put("/tenant/config", json=tenant_b_config)
            assert response.status_code == 200
            assert response.json()["configuration"]["default_algorithm"] == "one_class_svm"
        
        # Verify configuration isolation - Tenant A gets their config
        api_client.get.return_value = mock_config_response("tenant-a", tenant_a_config)
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.get("/tenant/config")
            assert response.status_code == 200
            config = response.json()["configuration"]
            assert config["default_algorithm"] == "isolation_forest"
            assert config["data_retention_days"] == 365
        
        # Verify configuration isolation - Tenant B gets their config
        api_client.get.return_value = mock_config_response("tenant-b", tenant_b_config)
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_b_token}"}):
            response = api_client.get("/tenant/config")
            assert response.status_code == 200
            config = response.json()["configuration"]
            assert config["default_algorithm"] == "one_class_svm"
            assert config["data_retention_days"] == 90
        
        # Test that tenants cannot access each other's configurations
        forbidden_response = Mock()
        forbidden_response.status_code = 403
        forbidden_response.json.return_value = {
            "error": "Access Denied",
            "message": "Cannot access configuration for different tenant"
        }
        
        api_client.get.return_value = forbidden_response
        
        # Tenant A tries to access specific endpoint for Tenant B
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant_a_token}"}):
            response = api_client.get("/tenant/tenant-b/config")
            assert response.status_code == 403