"""Comprehensive tests for deployment infrastructure."""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from pynomaly.application.services.deployment_orchestration_service import (
    DeploymentNotFoundError,
    DeploymentOrchestrationService,
)
from pynomaly.application.services.model_registry_service import ModelRegistryService
from pynomaly.domain.entities.deployment import (
    Deployment,
    DeploymentConfig,
    DeploymentStatus,
    DeploymentStrategy,
    Environment,
    HealthMetrics,
    RollbackCriteria,
    StrategyType,
)
from pynomaly.infrastructure.serving.model_server import ModelServer


class TestDeploymentEntities:
    """Test deployment domain entities."""

    def test_deployment_creation(self):
        """Test deployment entity creation."""
        config = DeploymentConfig(replicas=3)
        strategy = DeploymentStrategy(strategy_type=StrategyType.ROLLING)

        deployment = Deployment(
            model_version_id=uuid4(),
            environment=Environment.STAGING,
            deployment_config=config,
            strategy=strategy,
            created_by="test-user",
        )

        assert deployment.environment == Environment.STAGING
        assert deployment.status == DeploymentStatus.PENDING
        assert deployment.deployment_config.replicas == 3
        assert deployment.strategy.strategy_type == StrategyType.ROLLING
        assert deployment.namespace == "pynomaly-staging"
        assert not deployment.is_deployed
        assert deployment.health_score >= 0.0

    def test_deployment_validation(self):
        """Test deployment validation."""
        config = DeploymentConfig()
        strategy = DeploymentStrategy(strategy_type=StrategyType.BLUE_GREEN)

        # Test invalid environment type
        with pytest.raises(TypeError):
            Deployment(
                model_version_id=uuid4(),
                environment="invalid",  # Should be Environment enum
                deployment_config=config,
                strategy=strategy,
                created_by="test-user",
            )

        # Test empty created_by
        with pytest.raises(ValueError):
            Deployment(
                model_version_id=uuid4(),
                environment=Environment.PRODUCTION,
                deployment_config=config,
                strategy=strategy,
                created_by="",
            )

    def test_health_metrics(self):
        """Test health metrics functionality."""
        metrics = HealthMetrics(
            cpu_usage=75.0, memory_usage=60.0, error_rate=2.0, response_time_p95=500.0
        )

        assert metrics.is_healthy()  # All within thresholds
        assert metrics.get_health_score() > 0.5

        # Test unhealthy metrics
        unhealthy_metrics = HealthMetrics(
            cpu_usage=90.0,  # Over threshold
            memory_usage=95.0,  # Over threshold
            error_rate=10.0,  # Over threshold
            response_time_p95=2000.0,  # Over threshold
        )

        assert not unhealthy_metrics.is_healthy()
        assert unhealthy_metrics.get_health_score() < 0.5

    def test_rollback_criteria(self):
        """Test rollback criteria evaluation."""
        criteria = RollbackCriteria(
            max_error_rate=5.0, max_response_time=1000.0, min_success_rate=95.0
        )

        # Healthy metrics - no rollback
        healthy_metrics = HealthMetrics(error_rate=2.0, response_time_p95=500.0)
        assert not criteria.should_rollback(healthy_metrics, 1000)

        # High error rate - should rollback
        unhealthy_metrics = HealthMetrics(error_rate=8.0, response_time_p95=500.0)
        assert criteria.should_rollback(unhealthy_metrics, 1000)

        # Not enough requests - no rollback
        assert not criteria.should_rollback(unhealthy_metrics, 50)

    def test_deployment_strategy_configurations(self):
        """Test deployment strategy default configurations."""
        # Canary strategy
        canary_strategy = DeploymentStrategy(strategy_type=StrategyType.CANARY)
        assert "initial_traffic_percentage" in canary_strategy.configuration
        assert canary_strategy.configuration["initial_traffic_percentage"] == 10

        # Blue-green strategy
        bg_strategy = DeploymentStrategy(strategy_type=StrategyType.BLUE_GREEN)
        assert "smoke_test_duration_minutes" in bg_strategy.configuration

        # Rolling strategy
        rolling_strategy = DeploymentStrategy(strategy_type=StrategyType.ROLLING)
        assert "max_unavailable" in rolling_strategy.configuration

        # Test canary traffic calculation
        canary_strategy = DeploymentStrategy(strategy_type=StrategyType.CANARY)

        # Initial percentage
        initial_duration = timedelta(minutes=5)
        percentage = canary_strategy.get_canary_traffic_percentage(initial_duration)
        assert percentage == 10  # Initial percentage

        # After one interval
        after_interval = timedelta(minutes=15)
        percentage = canary_strategy.get_canary_traffic_percentage(after_interval)
        assert percentage == 30  # 10 + 20 (increment)

    def test_deployment_lifecycle(self):
        """Test deployment lifecycle methods."""
        config = DeploymentConfig()
        strategy = DeploymentStrategy(strategy_type=StrategyType.DIRECT)

        deployment = Deployment(
            model_version_id=uuid4(),
            environment=Environment.PRODUCTION,
            deployment_config=config,
            strategy=strategy,
            created_by="test-user",
        )

        # Initial state
        assert deployment.status == DeploymentStatus.PENDING
        assert not deployment.is_deployed
        assert deployment.deployed_at is None

        # Mark as deployed
        deployment.mark_deployed()
        assert deployment.status == DeploymentStatus.DEPLOYED
        assert deployment.is_deployed
        assert deployment.deployed_at is not None
        assert deployment.traffic_percentage == 100

        # Test failure
        deployment.mark_failed("Test error")
        assert deployment.status == DeploymentStatus.FAILED
        assert "error_message" in deployment.metadata

        # Test rollback process
        deployment.start_rollback("Performance issues")
        assert deployment.status == DeploymentStatus.ROLLING_BACK
        assert "rollback_reason" in deployment.metadata

        deployment.complete_rollback()
        assert deployment.status == DeploymentStatus.ROLLED_BACK


class TestDeploymentOrchestrationService:
    """Test deployment orchestration service."""

    @pytest.fixture
    def mock_model_registry_service(self):
        """Mock model registry service."""
        return Mock(spec=ModelRegistryService)

    @pytest.fixture
    def temp_storage_path(self):
        """Temporary storage path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def deployment_service(self, mock_model_registry_service, temp_storage_path):
        """Deployment orchestration service fixture."""
        return DeploymentOrchestrationService(
            model_registry_service=mock_model_registry_service,
            storage_path=temp_storage_path,
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self, deployment_service):
        """Test service initialization."""
        assert deployment_service.storage_path.exists()
        assert isinstance(deployment_service.active_deployments, dict)
        assert isinstance(deployment_service.deployment_history, list)

    @pytest.mark.asyncio
    async def test_deploy_model_success(self, deployment_service):
        """Test successful model deployment."""
        model_version_id = uuid4()
        config = DeploymentConfig(replicas=2)
        strategy = DeploymentStrategy(strategy_type=StrategyType.DIRECT)

        deployment = await deployment_service.deploy_model(
            model_version_id=model_version_id,
            target_environment=Environment.STAGING,
            strategy=strategy,
            deployment_config=config,
            user="test-user",
        )

        assert deployment.model_version_id == model_version_id
        assert deployment.environment == Environment.STAGING
        assert deployment.status == DeploymentStatus.DEPLOYED
        assert deployment.id in deployment_service.active_deployments

        # Verify deployment was saved
        deployment_files = list(
            deployment_service.storage_path.glob("deployment_*.json")
        )
        assert len(deployment_files) == 1

    @pytest.mark.asyncio
    async def test_deploy_model_with_existing_deployment(self, deployment_service):
        """Test deployment when existing deployment exists."""
        model_version_id_1 = uuid4()
        model_version_id_2 = uuid4()

        # First deployment
        deployment1 = await deployment_service.deploy_model(
            model_version_id=model_version_id_1,
            target_environment=Environment.STAGING,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="test-user",
        )

        # Second deployment (should archive first)
        deployment2 = await deployment_service.deploy_model(
            model_version_id=model_version_id_2,
            target_environment=Environment.STAGING,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="test-user",
        )

        # Check that first deployment was archived
        assert deployment1.id not in deployment_service.active_deployments
        assert deployment2.id in deployment_service.active_deployments
        assert deployment2.rollback_version_id == model_version_id_1

    @pytest.mark.asyncio
    async def test_rollback_deployment(self, deployment_service):
        """Test deployment rollback."""
        # Create initial deployment
        model_version_id_1 = uuid4()
        deployment1 = await deployment_service.deploy_model(
            model_version_id=model_version_id_1,
            target_environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="test-user",
        )

        # Create second deployment
        model_version_id_2 = uuid4()
        deployment2 = await deployment_service.deploy_model(
            model_version_id=model_version_id_2,
            target_environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="test-user",
        )

        # Rollback to previous version
        rollback_deployment = await deployment_service.rollback_deployment(
            deployment_id=deployment2.id,
            reason="Performance regression",
            user="test-user",
        )

        assert rollback_deployment is not None
        assert rollback_deployment.model_version_id == model_version_id_1
        assert deployment2.status == DeploymentStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_rollback_without_previous_version(self, deployment_service):
        """Test rollback when no previous version exists."""
        deployment = await deployment_service.deploy_model(
            model_version_id=uuid4(),
            target_environment=Environment.STAGING,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="test-user",
        )

        # Attempt rollback (should return None)
        rollback_deployment = await deployment_service.rollback_deployment(
            deployment_id=deployment.id, reason="Test rollback", user="test-user"
        )

        assert rollback_deployment is None
        assert deployment.status == DeploymentStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_promote_to_production(self, deployment_service):
        """Test promotion from staging to production."""
        # Create staging deployment
        staging_deployment = await deployment_service.deploy_model(
            model_version_id=uuid4(),
            target_environment=Environment.STAGING,
            strategy=DeploymentStrategy(strategy_type=StrategyType.CANARY),
            user="test-user",
        )

        # Promote to production
        production_deployment = await deployment_service.promote_to_production(
            deployment_id=staging_deployment.id,
            approval_metadata={
                "approved_by": "manager",
                "notes": "Performance validated",
            },
            user="test-user",
        )

        assert production_deployment.environment == Environment.PRODUCTION
        assert (
            production_deployment.model_version_id
            == staging_deployment.model_version_id
        )
        assert "promoted_from" in production_deployment.metadata
        assert production_deployment.metadata["promoted_from"] == str(
            staging_deployment.id
        )

    @pytest.mark.asyncio
    async def test_list_deployments_with_filters(self, deployment_service):
        """Test listing deployments with filters."""
        # Create deployments in different environments
        staging_deployment = await deployment_service.deploy_model(
            model_version_id=uuid4(),
            target_environment=Environment.STAGING,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="test-user",
        )

        production_deployment = await deployment_service.deploy_model(
            model_version_id=uuid4(),
            target_environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy(strategy_type=StrategyType.BLUE_GREEN),
            user="test-user",
        )

        # Test environment filter
        staging_deployments = await deployment_service.list_deployments(
            environment=Environment.STAGING
        )
        assert len(staging_deployments) == 1
        assert staging_deployments[0].id == staging_deployment.id

        # Test status filter
        deployed_deployments = await deployment_service.list_deployments(
            status=DeploymentStatus.DEPLOYED
        )
        assert len(deployed_deployments) == 2

        # Test limit
        limited_deployments = await deployment_service.list_deployments(limit=1)
        assert len(limited_deployments) == 1

    @pytest.mark.asyncio
    async def test_get_environment_status(self, deployment_service):
        """Test environment status retrieval."""
        # Test empty environment
        status = await deployment_service.get_environment_status(
            Environment.DEVELOPMENT
        )
        assert not status["has_active_deployment"]
        assert status["deployment_count"] == 0

        # Create deployment
        await deployment_service.deploy_model(
            model_version_id=uuid4(),
            target_environment=Environment.DEVELOPMENT,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="test-user",
        )

        # Test with active deployment
        status = await deployment_service.get_environment_status(
            Environment.DEVELOPMENT
        )
        assert status["has_active_deployment"]
        assert status["deployment_count"] == 1
        assert "deployment_id" in status

    @pytest.mark.asyncio
    async def test_deployment_not_found_error(self, deployment_service):
        """Test deployment not found error."""
        with pytest.raises(DeploymentNotFoundError):
            await deployment_service.get_deployment(uuid4())

    @pytest.mark.asyncio
    async def test_health_metrics_update(self, deployment_service):
        """Test health metrics update and monitoring."""
        deployment = await deployment_service.deploy_model(
            model_version_id=uuid4(),
            target_environment=Environment.STAGING,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="test-user",
        )

        # Update health metrics
        new_metrics = HealthMetrics(
            cpu_usage=80.0, memory_usage=70.0, error_rate=3.0, response_time_p95=600.0
        )

        await deployment_service.update_deployment_health(deployment.id, new_metrics)

        # Verify metrics were updated
        updated_deployment = await deployment_service.get_deployment(deployment.id)
        assert updated_deployment.health_metrics.cpu_usage == 80.0
        assert updated_deployment.health_metrics.memory_usage == 70.0


class TestModelServer:
    """Test model serving infrastructure."""

    @pytest.fixture
    def mock_deployment_service(self):
        """Mock deployment service."""
        service = Mock(spec=DeploymentOrchestrationService)
        service.list_deployments = AsyncMock(return_value=[])
        return service

    @pytest.fixture
    def mock_model_registry_service(self):
        """Mock model registry service."""
        return Mock(spec=ModelRegistryService)

    @pytest.fixture
    def model_server(self, mock_deployment_service, mock_model_registry_service):
        """Model server fixture."""
        return ModelServer(
            deployment_service=mock_deployment_service,
            model_registry_service=mock_model_registry_service,
            environment=Environment.PRODUCTION,
        )

    def test_model_server_initialization(self, model_server):
        """Test model server initialization."""
        assert model_server.environment == Environment.PRODUCTION
        assert isinstance(model_server.loaded_models, dict)
        assert isinstance(model_server.model_metadata, dict)
        assert model_server.app is not None
        assert model_server.start_time is not None

    @pytest.mark.asyncio
    async def test_model_loading_and_caching(self, model_server):
        """Test model loading and LRU cache."""
        model_id = str(uuid4())

        # Load model
        await model_server._load_model(model_id)

        assert model_id in model_server.loaded_models
        assert model_id in model_server.model_metadata
        assert model_id in model_server.model_access_times

        # Test cache eviction when exceeding cache size
        model_server.model_cache_size = 1

        # Load second model (should evict first)
        model_id_2 = str(uuid4())
        await model_server._load_model(model_id_2)

        # First model should be evicted
        assert model_id not in model_server.loaded_models
        assert model_id_2 in model_server.loaded_models

    @pytest.mark.asyncio
    async def test_prediction_functionality(self, model_server):
        """Test prediction functionality."""
        model_id = str(uuid4())
        await model_server._load_model(model_id)

        model = model_server.loaded_models[model_id]
        model_info = model_server.model_metadata[model_id]

        # Test single prediction
        data = {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}

        result = await model_server._make_prediction(
            model=model,
            data=data,
            model_info=model_info,
            return_confidence=True,
            return_explanation=True,
        )

        assert "score" in result
        assert "is_anomaly" in result
        assert "confidence" in result
        assert "explanation" in result
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["is_anomaly"], bool)

    def test_memory_usage_calculation(self, model_server):
        """Test memory usage calculation."""
        # Test with psutil not available
        memory_usage = model_server._get_memory_usage()
        assert memory_usage >= 0.0

    @pytest.mark.asyncio
    async def test_model_eviction_lru(self, model_server):
        """Test LRU model eviction."""
        model_server.model_cache_size = 2

        # Load three models
        model_id_1 = str(uuid4())
        model_id_2 = str(uuid4())
        model_id_3 = str(uuid4())

        await model_server._load_model(model_id_1)
        await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        await model_server._load_model(model_id_2)

        # Access first model to make it more recently used
        model_server.model_access_times[model_id_1] = datetime.utcnow()

        # Load third model (should evict model_id_2, not model_id_1)
        await model_server._load_model(model_id_3)

        assert model_id_1 in model_server.loaded_models  # Still loaded
        assert model_id_2 not in model_server.loaded_models  # Evicted
        assert model_id_3 in model_server.loaded_models  # Newly loaded


class TestDeploymentCLI:
    """Test deployment CLI functionality."""

    @pytest.fixture
    def cli_runner(self):
        """CLI runner fixture."""
        from typer.testing import CliRunner

        return CliRunner()

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from pynomaly.presentation.cli.deployment import app

        assert app is not None

    @pytest.mark.asyncio
    async def test_deployment_service_creation(self):
        """Test deployment service creation in CLI."""
        from pynomaly.presentation.cli.deployment import get_deployment_service

        service = get_deployment_service()
        assert isinstance(service, DeploymentOrchestrationService)


class TestDeploymentIntegration:
    """Integration tests for deployment infrastructure."""

    @pytest.fixture
    def temp_storage_path(self):
        """Temporary storage path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_end_to_end_deployment_workflow(self, temp_storage_path):
        """Test complete deployment workflow."""
        # Initialize services
        model_registry_service = ModelRegistryService(
            storage_path=temp_storage_path / "registry"
        )
        deployment_service = DeploymentOrchestrationService(
            model_registry_service=model_registry_service,
            storage_path=temp_storage_path / "deployments",
        )

        model_version_id = uuid4()

        # 1. Deploy to staging
        staging_deployment = await deployment_service.deploy_model(
            model_version_id=model_version_id,
            target_environment=Environment.STAGING,
            strategy=DeploymentStrategy(strategy_type=StrategyType.CANARY),
            user="integration-test",
        )

        assert staging_deployment.environment == Environment.STAGING
        assert staging_deployment.is_deployed

        # 2. Update health metrics
        good_metrics = HealthMetrics(
            cpu_usage=50.0, memory_usage=40.0, error_rate=1.0, response_time_p95=200.0
        )

        await deployment_service.update_deployment_health(
            staging_deployment.id, good_metrics
        )

        # 3. Promote to production
        production_deployment = await deployment_service.promote_to_production(
            deployment_id=staging_deployment.id,
            approval_metadata={"integration_test": True},
            user="integration-test",
        )

        assert production_deployment.environment == Environment.PRODUCTION
        assert production_deployment.model_version_id == model_version_id

        # 4. Simulate performance degradation and rollback
        bad_metrics = HealthMetrics(
            cpu_usage=95.0, memory_usage=90.0, error_rate=15.0, response_time_p95=3000.0
        )

        await deployment_service.update_deployment_health(
            production_deployment.id, bad_metrics
        )

        # Check if automatic rollback was triggered
        updated_deployment = await deployment_service.get_deployment(
            production_deployment.id
        )
        # Note: In this test, automatic rollback might not trigger due to request count threshold

        # 5. Manual rollback
        rollback_deployment = await deployment_service.rollback_deployment(
            deployment_id=production_deployment.id,
            reason="Integration test rollback",
            user="integration-test",
        )

        if rollback_deployment:  # Only if there was a previous version
            assert rollback_deployment.environment == Environment.PRODUCTION
            assert rollback_deployment.status == DeploymentStatus.DEPLOYED

        # 6. Verify deployment history
        all_deployments = await deployment_service.list_deployments()
        assert len(all_deployments) >= 2  # At least staging and production

        # 7. Check environment status
        prod_status = await deployment_service.get_environment_status(
            Environment.PRODUCTION
        )
        assert prod_status["has_active_deployment"]

        staging_status = await deployment_service.get_environment_status(
            Environment.STAGING
        )
        assert staging_status["has_active_deployment"]

    @pytest.mark.asyncio
    async def test_persistence_and_recovery(self, temp_storage_path):
        """Test deployment persistence and service recovery."""
        model_registry_service = ModelRegistryService(
            storage_path=temp_storage_path / "registry"
        )

        # Create first service instance and deploy
        service1 = DeploymentOrchestrationService(
            model_registry_service=model_registry_service,
            storage_path=temp_storage_path / "deployments",
        )

        deployment = await service1.deploy_model(
            model_version_id=uuid4(),
            target_environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
            user="persistence-test",
        )

        deployment_id = deployment.id

        # Create second service instance (simulating restart)
        service2 = DeploymentOrchestrationService(
            model_registry_service=model_registry_service,
            storage_path=temp_storage_path / "deployments",
        )

        # Give time for loading
        await asyncio.sleep(0.1)

        # Verify deployment was loaded
        recovered_deployment = await service2.get_deployment(deployment_id)
        assert recovered_deployment.id == deployment_id
        assert recovered_deployment.environment == Environment.PRODUCTION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
