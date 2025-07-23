"""Integration tests for MLOps external service integrations."""

import pytest
import asyncio
import json
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, UTC

from test_utilities.integration_test_base import IntegrationTestBase
from test_utilities.fixtures.external_services import (
    mock_kubernetes_client,
    mock_docker_client,
    mock_monitoring_service,
    mock_artifact_store
)


class TestMLOpsExternalServiceIntegrations(IntegrationTestBase):
    """Integration tests for MLOps external service dependencies."""

    @pytest.fixture
    def sample_deployment_config(self):
        """Sample deployment configuration."""
        return {
            "deployment_id": str(uuid4()),
            "model_id": str(uuid4()),
            "model_version": "1.0.0",
            "deployment_name": "test-model-deployment",
            "environment": "production",
            "scaling_config": {
                "min_replicas": 1,
                "max_replicas": 10,
                "target_cpu_utilization": 70
            },
            "resource_requirements": {
                "cpu": "500m",
                "memory": "1Gi",
                "gpu": 0
            }
        }

    @pytest.fixture
    def sample_pipeline_config(self):
        """Sample ML pipeline configuration."""
        return {
            "pipeline_id": str(uuid4()),
            "name": "test-ml-pipeline",
            "stages": [
                {"name": "data_preprocessing", "image": "preprocess:latest"},
                {"name": "model_training", "image": "train:latest"},
                {"name": "model_evaluation", "image": "evaluate:latest"}
            ],
            "schedule": "0 2 * * *",  # Daily at 2 AM
            "retry_policy": {"max_retries": 3, "backoff_factor": 2}
        }

    @pytest.mark.asyncio
    async def test_kubernetes_deployment_integration(
        self,
        mock_kubernetes_client: AsyncMock,
        sample_deployment_config: Dict[str, Any]
    ):
        """Test Kubernetes deployment integration."""
        # Mock Kubernetes API responses
        mock_kubernetes_client.create_deployment.return_value = {
            "metadata": {
                "name": sample_deployment_config["deployment_name"],
                "namespace": "mlops"
            },
            "status": {"replicas": 1, "ready_replicas": 1}
        }
        
        mock_kubernetes_client.create_service.return_value = {
            "metadata": {"name": f"{sample_deployment_config['deployment_name']}-service"},
            "spec": {"ports": [{"port": 8080, "target_port": 8080}]}
        }
        
        mock_kubernetes_client.get_deployment_status.return_value = {
            "status": "Running",
            "replicas": 1,
            "ready_replicas": 1,
            "conditions": [{"type": "Available", "status": "True"}]
        }
        
        from mlops.infrastructure.external.kubernetes_adapter import KubernetesAdapter
        
        adapter = KubernetesAdapter(client=mock_kubernetes_client)
        
        # Test deployment creation
        deployment_result = await adapter.deploy_model(
            deployment_config=sample_deployment_config
        )
        
        assert deployment_result["success"] is True
        assert deployment_result["deployment_name"] == sample_deployment_config["deployment_name"]
        assert "service_endpoint" in deployment_result
        
        # Test deployment status check
        status = await adapter.get_deployment_status(
            deployment_name=sample_deployment_config["deployment_name"]
        )
        
        assert status["status"] == "Running"
        assert status["ready_replicas"] == 1
        
        # Verify Kubernetes API calls
        mock_kubernetes_client.create_deployment.assert_called_once()
        mock_kubernetes_client.create_service.assert_called_once()
        mock_kubernetes_client.get_deployment_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_docker_container_management(
        self,
        mock_docker_client: AsyncMock,
        sample_deployment_config: Dict[str, Any]
    ):
        """Test Docker container management integration."""
        container_id = "container_123"
        
        # Mock Docker API responses
        mock_docker_client.build_image.return_value = {
            "image_id": "sha256:test_image_id",
            "image_tag": f"model-{sample_deployment_config['model_id']}:latest"
        }
        
        mock_docker_client.run_container.return_value = {
            "container_id": container_id,
            "status": "running"
        }
        
        mock_docker_client.get_container_logs.return_value = [
            "Model loading completed",
            "Server started on port 8080",
            "Ready to accept requests"
        ]
        
        from mlops.infrastructure.external.docker_adapter import DockerAdapter
        
        adapter = DockerAdapter(client=mock_docker_client)
        
        # Test image building
        build_result = await adapter.build_model_image(
            model_id=sample_deployment_config["model_id"],
            dockerfile_path="/path/to/Dockerfile",
            build_context="/path/to/context"
        )
        
        assert build_result["success"] is True
        assert "image_id" in build_result
        assert "image_tag" in build_result
        
        # Test container running
        run_result = await adapter.run_model_container(
            image_tag=build_result["image_tag"],
            container_name=f"model-{sample_deployment_config['model_id']}",
            environment_vars={"MODEL_VERSION": sample_deployment_config["model_version"]},
            resource_limits=sample_deployment_config["resource_requirements"]
        )
        
        assert run_result["success"] is True
        assert run_result["container_id"] == container_id
        
        # Test log retrieval
        logs = await adapter.get_container_logs(container_id)
        
        assert len(logs) == 3
        assert "Model loading completed" in logs[0]
        
        # Verify Docker API calls
        mock_docker_client.build_image.assert_called_once()
        mock_docker_client.run_container.assert_called_once()
        mock_docker_client.get_container_logs.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_service_integration(
        self,
        mock_monitoring_service: AsyncMock,
        sample_deployment_config: Dict[str, Any]
    ):
        """Test monitoring service integration."""
        # Mock monitoring service responses
        mock_monitoring_service.create_alert_rule.return_value = {
            "rule_id": "alert_rule_123",
            "status": "active"
        }
        
        mock_monitoring_service.send_metrics.return_value = {
            "success": True,
            "metrics_sent": 5
        }
        
        mock_monitoring_service.query_metrics.return_value = {
            "metrics": [
                {"timestamp": "2024-01-01T10:00:00Z", "value": 0.95, "metric": "model_accuracy"},
                {"timestamp": "2024-01-01T10:01:00Z", "value": 150.0, "metric": "response_time_ms"},
                {"timestamp": "2024-01-01T10:02:00Z", "value": 25.0, "metric": "requests_per_minute"}
            ]
        }
        
        from mlops.infrastructure.external.monitoring_adapter import MonitoringAdapter
        
        adapter = MonitoringAdapter(client=mock_monitoring_service)
        
        # Test alert rule creation
        alert_result = await adapter.create_model_performance_alert(
            deployment_id=sample_deployment_config["deployment_id"],
            model_id=sample_deployment_config["model_id"],
            alert_conditions={
                "accuracy_threshold": 0.90,
                "response_time_threshold": 200,
                "error_rate_threshold": 0.05
            }
        )
        
        assert alert_result["success"] is True
        assert "rule_id" in alert_result
        
        # Test metrics sending
        metrics_data = [
            {"metric_name": "model_accuracy", "value": 0.95, "timestamp": datetime.now(UTC)},
            {"metric_name": "response_time_ms", "value": 150.0, "timestamp": datetime.now(UTC)},
            {"metric_name": "requests_per_minute", "value": 25.0, "timestamp": datetime.now(UTC)}
        ]
        
        metrics_result = await adapter.send_model_metrics(
            deployment_id=sample_deployment_config["deployment_id"],
            metrics=metrics_data
        )
        
        assert metrics_result["success"] is True
        assert metrics_result["metrics_sent"] == 5
        
        # Test metrics querying
        query_result = await adapter.query_model_metrics(
            deployment_id=sample_deployment_config["deployment_id"],
            time_range="1h",
            metrics=["model_accuracy", "response_time_ms", "requests_per_minute"]
        )
        
        assert len(query_result["metrics"]) == 3
        assert query_result["metrics"][0]["metric"] == "model_accuracy"
        
        # Verify monitoring service calls
        mock_monitoring_service.create_alert_rule.assert_called_once()
        mock_monitoring_service.send_metrics.assert_called_once()
        mock_monitoring_service.query_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_artifact_store_integration(
        self,
        mock_artifact_store: AsyncMock,
        sample_deployment_config: Dict[str, Any]
    ):
        """Test artifact store integration."""
        # Mock artifact store responses
        mock_artifact_store.upload_artifact.return_value = {
            "artifact_id": "artifact_123",
            "storage_path": f"s3://mlops-artifacts/models/{sample_deployment_config['model_id']}/model.pkl",
            "checksum": "sha256:test_checksum"
        }
        
        mock_artifact_store.download_artifact.return_value = {
            "data": b"serialized_model_data",
            "metadata": {"size": 1024, "content_type": "application/octet-stream"}
        }
        
        mock_artifact_store.list_artifacts.return_value = {
            "artifacts": [
                {"id": "artifact_123", "name": "model.pkl", "size": 1024},
                {"id": "artifact_124", "name": "config.json", "size": 256}
            ]
        }
        
        from mlops.infrastructure.external.artifact_store_adapter import ArtifactStoreAdapter
        
        adapter = ArtifactStoreAdapter(client=mock_artifact_store)
        
        # Test artifact upload
        upload_result = await adapter.upload_model_artifact(
            model_id=sample_deployment_config["model_id"],
            artifact_name="model.pkl",
            artifact_data=b"serialized_model_data",
            metadata={"version": sample_deployment_config["model_version"]}
        )
        
        assert upload_result["success"] is True
        assert "artifact_id" in upload_result
        assert "storage_path" in upload_result
        
        # Test artifact download
        download_result = await adapter.download_model_artifact(
            model_id=sample_deployment_config["model_id"],
            artifact_name="model.pkl"
        )
        
        assert download_result["success"] is True
        assert download_result["data"] == b"serialized_model_data"
        
        # Test artifact listing
        list_result = await adapter.list_model_artifacts(
            model_id=sample_deployment_config["model_id"]
        )
        
        assert len(list_result["artifacts"]) == 2
        assert list_result["artifacts"][0]["name"] == "model.pkl"
        
        # Verify artifact store calls
        mock_artifact_store.upload_artifact.assert_called_once()
        mock_artifact_store.download_artifact.assert_called_once()
        mock_artifact_store.list_artifacts.assert_called_once()

    @pytest.mark.asyncio
    async def test_ci_cd_pipeline_integration(
        self,
        mock_kubernetes_client: AsyncMock,
        sample_pipeline_config: Dict[str, Any]
    ):
        """Test CI/CD pipeline integration."""
        # Mock pipeline execution responses
        job_id = "job_123"
        
        mock_kubernetes_client.create_job.return_value = {
            "job_id": job_id,
            "status": "Running"
        }
        
        mock_kubernetes_client.get_job_status.return_value = {
            "status": "Succeeded",
            "start_time": "2024-01-01T10:00:00Z",
            "completion_time": "2024-01-01T10:30:00Z",
            "duration": 1800
        }
        
        mock_kubernetes_client.get_job_logs.return_value = [
            "Pipeline started",
            "Data preprocessing completed successfully",
            "Model training in progress...",
            "Model training completed",
            "Model evaluation completed",
            "Pipeline finished successfully"
        ]
        
        from mlops.infrastructure.external.pipeline_orchestrator import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator(k8s_client=mock_kubernetes_client)
        
        # Test pipeline execution
        execution_result = await orchestrator.execute_pipeline(
            pipeline_config=sample_pipeline_config
        )
        
        assert execution_result["success"] is True
        assert execution_result["job_id"] == job_id
        
        # Test pipeline status monitoring
        status = await orchestrator.get_pipeline_status(job_id)
        
        assert status["status"] == "Succeeded"
        assert status["duration"] == 1800
        
        # Test log retrieval
        logs = await orchestrator.get_pipeline_logs(job_id)
        
        assert len(logs) == 6
        assert "Pipeline started" in logs[0]
        assert "Pipeline finished successfully" in logs[-1]
        
        # Verify orchestrator calls
        mock_kubernetes_client.create_job.assert_called_once()
        mock_kubernetes_client.get_job_status.assert_called_once()
        mock_kubernetes_client.get_job_logs.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_registry_integration(
        self,
        mock_artifact_store: AsyncMock
    ):
        """Test model registry integration."""
        model_id = str(uuid4())
        
        # Mock model registry responses
        mock_artifact_store.register_model.return_value = {
            "model_id": model_id,
            "registry_url": f"https://registry.mlops.com/models/{model_id}",
            "status": "registered"
        }
        
        mock_artifact_store.get_model_metadata.return_value = {
            "model_id": model_id,
            "name": "fraud_detection_model",
            "version": "2.1.0",
            "algorithm": "RandomForest",
            "performance_metrics": {"accuracy": 0.95, "precision": 0.92, "recall": 0.89},
            "training_date": "2024-01-01T10:00:00Z",
            "status": "production"
        }
        
        mock_artifact_store.list_model_versions.return_value = {
            "versions": [
                {"version": "2.1.0", "status": "production", "created_at": "2024-01-01T10:00:00Z"},
                {"version": "2.0.0", "status": "archived", "created_at": "2023-12-01T10:00:00Z"},
                {"version": "1.9.0", "status": "archived", "created_at": "2023-11-01T10:00:00Z"}
            ]
        }
        
        from mlops.infrastructure.external.model_registry_adapter import ModelRegistryAdapter
        
        adapter = ModelRegistryAdapter(client=mock_artifact_store)
        
        # Test model registration
        registration_result = await adapter.register_model(
            model_name="fraud_detection_model",
            version="2.1.0",
            model_path="/path/to/model.pkl",
            metadata={
                "algorithm": "RandomForest",
                "training_dataset": "fraud_data_v2",
                "performance_metrics": {"accuracy": 0.95, "precision": 0.92}
            }
        )
        
        assert registration_result["success"] is True
        assert registration_result["model_id"] == model_id
        
        # Test model metadata retrieval
        metadata = await adapter.get_model_metadata(model_id)
        
        assert metadata["name"] == "fraud_detection_model"
        assert metadata["version"] == "2.1.0"
        assert metadata["performance_metrics"]["accuracy"] == 0.95
        
        # Test model version listing
        versions = await adapter.list_model_versions(model_id)
        
        assert len(versions["versions"]) == 3
        assert versions["versions"][0]["status"] == "production"
        
        # Verify registry calls
        mock_artifact_store.register_model.assert_called_once()
        mock_artifact_store.get_model_metadata.assert_called_once()
        mock_artifact_store.list_model_versions.assert_called_once()

    @pytest.mark.asyncio
    async def test_feature_store_integration(self):
        """Test feature store integration."""
        with patch("mlops.infrastructure.external.feature_store_client.FeatureStoreClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            # Mock feature store responses
            mock_client_instance.get_features.return_value = {
                "features": {
                    "user_age": 35,
                    "account_balance": 1500.50,
                    "transaction_count_30d": 25,
                    "avg_transaction_amount": 67.89
                },
                "feature_timestamp": "2024-01-01T10:00:00Z"
            }
            
            mock_client_instance.register_feature_group.return_value = {
                "feature_group_id": "fg_123",
                "status": "active"
            }
            
            from mlops.infrastructure.external.feature_store_adapter import FeatureStoreAdapter
            
            adapter = FeatureStoreAdapter()
            
            # Test feature retrieval
            features = await adapter.get_user_features(
                user_id="user_123",
                feature_names=["user_age", "account_balance", "transaction_count_30d"]
            )
            
            assert "features" in features
            assert features["features"]["user_age"] == 35
            assert features["features"]["account_balance"] == 1500.50
            
            # Test feature group registration
            registration_result = await adapter.register_feature_group(
                name="user_transaction_features",
                features=["user_age", "account_balance", "transaction_count_30d", "avg_transaction_amount"],
                data_source="user_transactions_table"
            )
            
            assert registration_result["success"] is True
            assert "feature_group_id" in registration_result

    @pytest.mark.asyncio
    async def test_autoscaling_integration(
        self,
        mock_kubernetes_client: AsyncMock,
        sample_deployment_config: Dict[str, Any]
    ):
        """Test autoscaling integration."""
        # Mock autoscaling responses
        mock_kubernetes_client.create_hpa.return_value = {
            "hpa_name": f"{sample_deployment_config['deployment_name']}-hpa",
            "status": "active"
        }
        
        mock_kubernetes_client.get_hpa_status.return_value = {
            "current_replicas": 3,
            "desired_replicas": 3,
            "current_cpu_utilization": 65,
            "target_cpu_utilization": 70
        }
        
        mock_kubernetes_client.update_hpa.return_value = {
            "success": True,
            "updated_fields": ["max_replicas"]
        }
        
        from mlops.infrastructure.external.autoscaling_adapter import AutoscalingAdapter
        
        adapter = AutoscalingAdapter(k8s_client=mock_kubernetes_client)
        
        # Test HPA creation
        hpa_result = await adapter.create_horizontal_pod_autoscaler(
            deployment_name=sample_deployment_config["deployment_name"],
            min_replicas=sample_deployment_config["scaling_config"]["min_replicas"],
            max_replicas=sample_deployment_config["scaling_config"]["max_replicas"],
            target_cpu_utilization=sample_deployment_config["scaling_config"]["target_cpu_utilization"]
        )
        
        assert hpa_result["success"] is True
        assert "hpa_name" in hpa_result
        
        # Test HPA status monitoring
        status = await adapter.get_autoscaling_status(
            deployment_name=sample_deployment_config["deployment_name"]
        )
        
        assert status["current_replicas"] == 3
        assert status["current_cpu_utilization"] == 65
        
        # Test HPA configuration update
        update_result = await adapter.update_autoscaling_config(
            deployment_name=sample_deployment_config["deployment_name"],
            max_replicas=15  # Increase max replicas
        )
        
        assert update_result["success"] is True
        
        # Verify autoscaling calls
        mock_kubernetes_client.create_hpa.assert_called_once()
        mock_kubernetes_client.get_hpa_status.assert_called_once()
        mock_kubernetes_client.update_hpa.assert_called_once()

    @pytest.mark.asyncio
    async def test_secrets_management_integration(self):
        """Test secrets management integration."""
        with patch("mlops.infrastructure.external.secrets_manager.SecretsManager") as mock_secrets:
            mock_secrets_instance = AsyncMock()
            mock_secrets.return_value = mock_secrets_instance
            
            # Mock secrets manager responses
            mock_secrets_instance.get_secret.return_value = {
                "secret_value": "super_secret_api_key",
                "version": "1",
                "last_updated": "2024-01-01T10:00:00Z"
            }
            
            mock_secrets_instance.create_secret.return_value = {
                "secret_id": "secret_123",
                "status": "created"
            }
            
            from mlops.infrastructure.external.secrets_adapter import SecretsAdapter
            
            adapter = SecretsAdapter()
            
            # Test secret retrieval
            secret = await adapter.get_model_api_key(model_id="model_123")
            
            assert secret["value"] == "super_secret_api_key"
            assert secret["version"] == "1"
            
            # Test secret creation
            creation_result = await adapter.create_deployment_secret(
                deployment_id="deployment_123",
                secret_name="database_password",
                secret_value="db_password_123"
            )
            
            assert creation_result["success"] is True
            assert "secret_id" in creation_result

    @pytest.mark.asyncio
    async def test_load_balancer_integration(
        self,
        mock_kubernetes_client: AsyncMock,
        sample_deployment_config: Dict[str, Any]
    ):
        """Test load balancer integration."""
        # Mock load balancer responses
        mock_kubernetes_client.create_load_balancer.return_value = {
            "load_balancer_name": f"{sample_deployment_config['deployment_name']}-lb",
            "external_ip": "203.0.113.1",
            "status": "active"
        }
        
        mock_kubernetes_client.get_load_balancer_metrics.return_value = {
            "requests_per_second": 150.5,
            "active_connections": 45,
            "error_rate": 0.02,
            "average_response_time": 125.0
        }
        
        from mlops.infrastructure.external.load_balancer_adapter import LoadBalancerAdapter
        
        adapter = LoadBalancerAdapter(k8s_client=mock_kubernetes_client)
        
        # Test load balancer creation
        lb_result = await adapter.create_model_load_balancer(
            deployment_name=sample_deployment_config["deployment_name"],
            service_port=8080,
            health_check_path="/health"
        )
        
        assert lb_result["success"] is True
        assert lb_result["external_ip"] == "203.0.113.1"
        
        # Test load balancer metrics
        metrics = await adapter.get_load_balancer_metrics(
            load_balancer_name=f"{sample_deployment_config['deployment_name']}-lb"
        )
        
        assert metrics["requests_per_second"] == 150.5
        assert metrics["error_rate"] == 0.02
        
        # Verify load balancer calls
        mock_kubernetes_client.create_load_balancer.assert_called_once()
        mock_kubernetes_client.get_load_balancer_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_mesh_integration(self):
        """Test service mesh integration."""
        with patch("mlops.infrastructure.external.istio_client.IstioClient") as mock_istio:
            mock_istio_instance = AsyncMock()
            mock_istio.return_value = mock_istio_instance
            
            # Mock Istio responses
            mock_istio_instance.create_virtual_service.return_value = {
                "virtual_service_id": "vs_123",
                "status": "active"
            }
            
            mock_istio_instance.create_destination_rule.return_value = {
                "destination_rule_id": "dr_123",
                "status": "active"
            }
            
            mock_istio_instance.get_traffic_metrics.return_value = {
                "success_rate": 0.998,
                "p99_latency": 150.0,
                "requests_per_second": 200.0,
                "error_distribution": {"5xx": 0.001, "4xx": 0.001}
            }
            
            from mlops.infrastructure.external.service_mesh_adapter import ServiceMeshAdapter
            
            adapter = ServiceMeshAdapter()
            
            # Test virtual service creation
            vs_result = await adapter.create_model_virtual_service(
                model_name="fraud_detection_model",
                service_name="fraud-detection-service",
                routing_rules=[
                    {"match": {"headers": {"version": "v2"}}, "route": {"destination": "fraud-detection-v2"}},
                    {"route": {"destination": "fraud-detection-v1"}}  # Default route
                ]
            )
            
            assert vs_result["success"] is True
            assert "virtual_service_id" in vs_result
            
            # Test destination rule creation
            dr_result = await adapter.create_destination_rule(
                service_name="fraud-detection-service",
                load_balancing_policy="ROUND_ROBIN",
                circuit_breaker_config={
                    "max_connections": 100,
                    "max_pending_requests": 50,
                    "max_requests": 200
                }
            )
            
            assert dr_result["success"] is True
            assert "destination_rule_id" in dr_result
            
            # Test traffic metrics
            metrics = await adapter.get_service_traffic_metrics(
                service_name="fraud-detection-service",
                time_range="1h"
            )
            
            assert metrics["success_rate"] == 0.998
            assert metrics["p99_latency"] == 150.0

    @pytest.mark.asyncio
    async def test_backup_and_disaster_recovery(
        self,
        mock_artifact_store: AsyncMock,
        sample_deployment_config: Dict[str, Any]
    ):
        """Test backup and disaster recovery integration."""
        # Mock backup responses
        mock_artifact_store.create_backup.return_value = {
            "backup_id": "backup_123",
            "backup_location": "s3://mlops-backups/deployments/deployment_123/backup_123.tar.gz",
            "backup_size": 1024 * 1024 * 50,  # 50MB
            "status": "completed"
        }
        
        mock_artifact_store.restore_from_backup.return_value = {
            "restore_id": "restore_123",
            "status": "completed",
            "restored_components": ["deployment", "service", "configmap", "secrets"]
        }
        
        from mlops.infrastructure.external.backup_adapter import BackupAdapter
        
        adapter = BackupAdapter(storage_client=mock_artifact_store)
        
        # Test deployment backup
        backup_result = await adapter.backup_deployment(
            deployment_id=sample_deployment_config["deployment_id"],
            backup_type="full",
            include_data=True
        )
        
        assert backup_result["success"] is True
        assert "backup_id" in backup_result
        assert backup_result["backup_size"] > 0
        
        # Test disaster recovery
        restore_result = await adapter.restore_deployment(
            deployment_id=sample_deployment_config["deployment_id"],
            backup_id="backup_123",
            target_environment="disaster_recovery"
        )
        
        assert restore_result["success"] is True
        assert len(restore_result["restored_components"]) == 4
        
        # Verify backup calls
        mock_artifact_store.create_backup.assert_called_once()
        mock_artifact_store.restore_from_backup.assert_called_once()

    @pytest.mark.asyncio
    async def test_integration_error_handling_and_retries(
        self,
        mock_kubernetes_client: AsyncMock
    ):
        """Test error handling and retry mechanisms for external integrations."""
        # Simulate intermittent failures
        mock_kubernetes_client.create_deployment.side_effect = [
            Exception("Connection timeout"),
            Exception("API rate limit exceeded"),
            {"metadata": {"name": "test-deployment"}, "status": {"replicas": 1}}
        ]
        
        from mlops.infrastructure.external.kubernetes_adapter import KubernetesAdapter
        from mlops.infrastructure.external.retry_decorator import with_exponential_backoff
        
        adapter = KubernetesAdapter(client=mock_kubernetes_client)
        
        # Apply retry decorator
        adapter.deploy_model = with_exponential_backoff(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0
        )(adapter.deploy_model)
        
        # Should succeed after retries
        result = await adapter.deploy_model({
            "deployment_name": "test-deployment",
            "model_id": str(uuid4()),
            "model_version": "1.0.0"
        })
        
        assert result["success"] is True
        assert mock_kubernetes_client.create_deployment.call_count == 3

    @pytest.mark.asyncio
    async def test_multi_cloud_deployment_integration(
        self,
        mock_kubernetes_client: AsyncMock,
        sample_deployment_config: Dict[str, Any]
    ):
        """Test multi-cloud deployment integration."""
        # Mock multi-cloud responses
        mock_kubernetes_client.deploy_to_cluster.return_value = {
            "cluster_id": "cluster_123",
            "cloud_provider": "aws",
            "region": "us-west-2",
            "deployment_status": "active"
        }
        
        from mlops.infrastructure.external.multi_cloud_adapter import MultiCloudAdapter
        
        adapter = MultiCloudAdapter(k8s_client=mock_kubernetes_client)
        
        # Test multi-cloud deployment
        deployment_result = await adapter.deploy_to_multiple_clouds(
            deployment_config=sample_deployment_config,
            target_clouds=[
                {"provider": "aws", "region": "us-west-2", "cluster": "prod-cluster-1"},
                {"provider": "gcp", "region": "us-central1", "cluster": "prod-cluster-2"},
                {"provider": "azure", "region": "eastus", "cluster": "prod-cluster-3"}
            ]
        )
        
        assert deployment_result["success"] is True
        assert len(deployment_result["deployments"]) == 3
        
        # Verify multi-cloud calls
        assert mock_kubernetes_client.deploy_to_cluster.call_count == 3