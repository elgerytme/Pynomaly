#!/usr/bin/env python3
"""Integration test for MLOps hexagonal architecture implementation."""

import asyncio
import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import domain services
from mlops.domain.services.refactored_experiment_service import ExperimentService
from mlops.domain.services.refactored_model_registry_service import ModelRegistryService
from mlops.domain.services.refactored_monitoring_service import MonitoringService

# Import file-based adapters
from mlops.infrastructure.adapters.file_based.experiment_tracking_adapters import (
    FileBasedExperimentTracking,
    FileBasedExperimentRun,
    FileBasedArtifactManagement,
    FileBasedExperimentAnalysis,
    FileBasedMetricsTracking,
    FileBasedExperimentSearch
)
from mlops.infrastructure.adapters.file_based.model_registry_adapters import (
    FileBasedModelRegistry,
    FileBasedModelLifecycle,
    FileBasedModelDeployment,
    FileBasedModelStorage,
    FileBasedModelVersioning,
    FileBasedModelSearch
)
from mlops.infrastructure.adapters.file_based.monitoring_adapters import (
    FileBasedModelPerformanceMonitoring,
    FileBasedInfrastructureMonitoring,
    FileBasedDataQualityMonitoring,
    FileBasedDataDriftMonitoring,
    FileBasedAlerting,
    FileBasedHealthCheck
)

# Import domain interfaces for data types
from mlops.domain.interfaces.experiment_tracking_operations import RunStatus
from mlops.domain.interfaces.model_registry_operations import ModelFramework, ModelStatus, DeploymentStage
from mlops.domain.interfaces.mlops_monitoring_operations import (
    PerformanceMetrics, 
    InfrastructureMetrics, 
    DataQualityMetrics,
    MonitoringAlertSeverity
)


async def test_experiment_service_integration():
    """Test the experiment service with file-based adapters."""
    print("🧪 Testing Experiment Service Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file-based adapters
        experiment_tracking = FileBasedExperimentTracking(f"{temp_dir}/experiments")
        experiment_run = FileBasedExperimentRun(f"{temp_dir}/runs")
        artifact_management = FileBasedArtifactManagement(f"{temp_dir}/artifacts")
        experiment_analysis = FileBasedExperimentAnalysis(experiment_run)
        metrics_tracking = FileBasedMetricsTracking(f"{temp_dir}/metrics")
        experiment_search = FileBasedExperimentSearch(experiment_tracking)
        
        # Create experiment service using dependency injection
        experiment_service = ExperimentService(
            experiment_tracking_port=experiment_tracking,
            experiment_run_port=experiment_run,
            artifact_management_port=artifact_management,
            experiment_analysis_port=experiment_analysis,
            metrics_tracking_port=metrics_tracking,
            experiment_search_port=experiment_search
        )
        
        # Test experiment creation
        experiment_id = await experiment_service.create_experiment(
            name="Test Anomaly Detection Experiment",
            description="Testing hexagonal architecture implementation",
            tags=["test", "anomaly_detection", "architecture"],
            created_by="test_user"
        )
        
        assert experiment_id.startswith("exp_"), f"Expected experiment ID to start with 'exp_', got: {experiment_id}"
        print(f"✅ Created experiment: {experiment_id}")
        
        # Test run creation and execution
        run_id = await experiment_service.start_experiment_run(
            experiment_id=experiment_id,
            detector_name="IsolationForest",
            dataset_name="test_dataset",
            parameters={"contamination": 0.1, "n_estimators": 100}
        )
        
        assert run_id.startswith("run_"), f"Expected run ID to start with 'run_', got: {run_id}"
        print(f"✅ Started run: {run_id}")
        
        # Test metrics logging
        await experiment_service.log_run_metrics(
            run_id=run_id,
            metrics={"accuracy": 0.85, "precision": 0.83, "recall": 0.87, "f1_score": 0.85}
        )
        print("✅ Logged run metrics")
        
        # Test run completion
        final_run = await experiment_service.finish_experiment_run(
            run_id=run_id,
            final_metrics={"accuracy": 0.86, "precision": 0.84, "recall": 0.88, "f1_score": 0.86},
            status=RunStatus.COMPLETED
        )
        
        assert final_run.status == RunStatus.COMPLETED, f"Expected run status COMPLETED, got: {final_run.status}"
        print("✅ Completed run successfully")
        
        # Test experiment search
        search_results = await experiment_service.search_experiments(
            query="anomaly detection",
            limit=10
        )
        
        assert len(search_results) > 0, "Expected to find at least one experiment"
        assert search_results[0].experiment_id == experiment_id, "Expected to find our test experiment"
        print(f"✅ Found {len(search_results)} experiments in search")
        
        # Test best run retrieval
        best_run = await experiment_service.get_best_run(experiment_id, metric="f1_score")
        assert best_run is not None, "Expected to find a best run"
        assert best_run.run_id == run_id, "Expected our run to be the best run"
        print(f"✅ Found best run: {best_run.run_id} with F1-score: {best_run.metrics.f1_score}")


async def test_model_registry_service_integration():
    """Test the model registry service with file-based adapters."""
    print("\n🏛️ Testing Model Registry Service Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file-based adapters
        model_registry = FileBasedModelRegistry(f"{temp_dir}/models")
        model_lifecycle = FileBasedModelLifecycle(model_registry)
        model_deployment = FileBasedModelDeployment(f"{temp_dir}/deployments")
        model_storage = FileBasedModelStorage(f"{temp_dir}/storage")
        model_versioning = FileBasedModelVersioning(model_registry)
        model_search = FileBasedModelSearch(model_registry)
        
        # Create model registry service using dependency injection
        model_service = ModelRegistryService(
            model_registry_port=model_registry,
            model_lifecycle_port=model_lifecycle,
            model_deployment_port=model_deployment,
            model_storage_port=model_storage,
            model_versioning_port=model_versioning,
            model_search_port=model_search
        )
        
        # Create a dummy model file for testing
        model_file = Path(temp_dir) / "test_model.pkl"
        model_file.write_text("dummy model content")
        
        # Test model registration
        model_key = await model_service.register_model(
            model_path=str(model_file),
            model_id="test_anomaly_detector",
            algorithm="IsolationForest",
            framework=ModelFramework.SCIKIT_LEARN,
            created_by="test_user",
            description="Test anomaly detection model",
            tags=["test", "anomaly_detection"],
            hyperparameters={"contamination": 0.1, "n_estimators": 100},
            metrics={"accuracy": 0.87, "precision": 0.85, "recall": 0.89, "f1_score": 0.87}
        )
        
        assert ":" in model_key, f"Expected model key to contain version, got: {model_key}"
        model_id, version = model_key.split(":", 1)
        print(f"✅ Registered model: {model_key}")
        
        # Test model promotion
        promotion_success = await model_service.promote_model(
            model_id=model_id,
            version=version,
            target_stage=ModelStatus.STAGING
        )
        
        assert promotion_success, "Expected model promotion to succeed"
        print(f"✅ Promoted model to staging")
        
        # Test model deployment
        deployment_id = await model_service.deploy_model(
            model_id=model_id,
            version=version,
            stage=DeploymentStage.STAGING,
            replicas=2,
            resources={"cpu": "200m", "memory": "512Mi"}
        )
        
        assert deployment_id is not None, "Expected deployment to succeed"
        assert deployment_id.startswith("deploy_"), f"Expected deployment ID to start with 'deploy_', got: {deployment_id}"
        print(f"✅ Deployed model: {deployment_id}")
        
        # Test model search
        search_results = await model_service.search_models(
            query="anomaly",
            algorithm="IsolationForest",
            limit=10
        )
        
        assert len(search_results) > 0, "Expected to find at least one model"
        assert search_results[0].model_id == model_id, "Expected to find our test model"
        print(f"✅ Found {len(search_results)} models in search")
        
        # Test model health status
        health_status = await model_service.get_model_health_status(
            model_id=model_id,
            version=version,
            include_deployments=True
        )
        
        assert health_status["model_id"] == model_id, "Expected health status for our model"
        assert "overall_health" in health_status, "Expected overall health status"
        print(f"✅ Retrieved model health: {health_status['overall_health']}")


async def test_monitoring_service_integration():
    """Test the monitoring service with file-based adapters."""
    print("\n📊 Testing Monitoring Service Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create file-based adapters
        performance_monitoring = FileBasedModelPerformanceMonitoring(f"{temp_dir}/performance")
        infrastructure_monitoring = FileBasedInfrastructureMonitoring(f"{temp_dir}/infrastructure")
        data_quality_monitoring = FileBasedDataQualityMonitoring(f"{temp_dir}/data_quality")
        data_drift_monitoring = FileBasedDataDriftMonitoring(f"{temp_dir}/drift")
        alerting = FileBasedAlerting(f"{temp_dir}/alerts")
        health_check = FileBasedHealthCheck(f"{temp_dir}/health")
        
        # Create monitoring service using dependency injection
        monitoring_service = MonitoringService(
            performance_monitoring_port=performance_monitoring,
            infrastructure_monitoring_port=infrastructure_monitoring,
            data_quality_monitoring_port=data_quality_monitoring,
            data_drift_monitoring_port=data_drift_monitoring,
            alerting_port=alerting,
            health_check_port=health_check
        )
        
        # Test performance metrics tracking
        await monitoring_service.track_model_performance(
            model_id="test_model",
            deployment_id="test_deployment",
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            latency_p95=150.0,
            throughput=75.0,
            error_rate=0.01
        )
        print("✅ Tracked model performance metrics")
        
        # Test data drift monitoring
        reference_data = {"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]}
        current_data = {"feature1": [1.1, 2.1, 3.1], "feature2": [0.15, 0.25, 0.35]}
        
        drift_result = await monitoring_service.monitor_data_drift(
            model_id="test_model",
            reference_data=reference_data,
            current_data=current_data,
            drift_threshold=0.1,
            auto_alert=True
        )
        
        assert drift_result is not None, "Expected drift detection result"
        print(f"✅ Monitored data drift: detected={drift_result.is_drift_detected}, score={drift_result.drift_score:.3f}")
        
        # Test model health assessment
        health_report = await monitoring_service.assess_model_health(
            model_id="test_model",
            deployment_id="test_deployment",
            include_predictions=True,
            include_infrastructure=True,
            include_drift=True
        )
        
        assert health_report is not None, "Expected health report"
        assert health_report.model_id == "test_model", "Expected health report for our model"
        print(f"✅ Assessed model health: {health_report.overall_health.value}")
        
        # Test monitoring rule creation
        rule_id = await monitoring_service.create_monitoring_rule(
            name="High Error Rate Alert",
            model_id="test_model",
            deployment_id="test_deployment",
            metric_name="error_rate",
            threshold_value=0.05,
            operator=">",
            severity=MonitoringAlertSeverity.HIGH,
            description="Alert when error rate exceeds 5%"
        )
        
        assert rule_id is not None, "Expected monitoring rule creation to succeed"
        print(f"✅ Created monitoring rule: {rule_id}")
        
        # Test performance degradation handling
        degradation_analysis = await monitoring_service.handle_performance_degradation(
            model_id="test_model",
            deployment_id="test_deployment",
            baseline_days=7,
            comparison_days=1,
            degradation_threshold=0.05
        )
        
        assert "degradation_metrics" in degradation_analysis, f"Expected degradation analysis with degradation_metrics, got: {degradation_analysis}"
        assert "overall_health_score" in degradation_analysis["degradation_metrics"], "Expected overall_health_score in degradation_metrics"
        health_score = degradation_analysis["degradation_metrics"]["overall_health_score"]
        print(f"✅ Analyzed performance degradation: health_score={health_score}")
        
        # Test monitoring insights generation
        insights = await monitoring_service.generate_monitoring_insights(
            model_id="test_model",
            time_period=timedelta(days=7),
            include_trends=True,
            include_anomalies=True,
            include_recommendations=True
        )
        
        assert insights["model_id"] == "test_model", "Expected insights for our model"
        assert "summary" in insights, "Expected insights summary"
        print(f"✅ Generated monitoring insights with {len(insights.get('recommendations', []))} recommendations")


async def test_end_to_end_workflow():
    """Test an end-to-end MLOps workflow using all services."""
    print("\n🔄 Testing End-to-End MLOps Workflow...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup all adapters
        exp_tracking = FileBasedExperimentTracking(f"{temp_dir}/experiments")
        exp_run = FileBasedExperimentRun(f"{temp_dir}/runs")
        exp_artifacts = FileBasedArtifactManagement(f"{temp_dir}/artifacts")
        exp_analysis = FileBasedExperimentAnalysis(exp_run)
        exp_metrics = FileBasedMetricsTracking(f"{temp_dir}/metrics")
        exp_search = FileBasedExperimentSearch(exp_tracking)
        
        model_registry = FileBasedModelRegistry(f"{temp_dir}/models")
        model_lifecycle = FileBasedModelLifecycle(model_registry)
        model_deployment = FileBasedModelDeployment(f"{temp_dir}/deployments")
        model_storage = FileBasedModelStorage(f"{temp_dir}/storage")
        model_versioning = FileBasedModelVersioning(model_registry)
        model_search = FileBasedModelSearch(model_registry)
        
        perf_monitoring = FileBasedModelPerformanceMonitoring(f"{temp_dir}/performance")
        infra_monitoring = FileBasedInfrastructureMonitoring(f"{temp_dir}/infrastructure")
        quality_monitoring = FileBasedDataQualityMonitoring(f"{temp_dir}/data_quality")
        drift_monitoring = FileBasedDataDriftMonitoring(f"{temp_dir}/drift")
        alerting = FileBasedAlerting(f"{temp_dir}/alerts")
        health_check = FileBasedHealthCheck(f"{temp_dir}/health")
        
        # Create services
        experiment_service = ExperimentService(
            exp_tracking, exp_run, exp_artifacts, exp_analysis, exp_metrics, exp_search
        )
        
        model_service = ModelRegistryService(
            model_registry, model_lifecycle, model_deployment, 
            model_storage, model_versioning, model_search
        )
        
        monitoring_service = MonitoringService(
            perf_monitoring, infra_monitoring, quality_monitoring,
            drift_monitoring, alerting, health_check
        )
        
        # Step 1: Run an experiment
        print("Step 1: Creating and running experiment...")
        experiment_id = await experiment_service.create_experiment(
            name="E2E Anomaly Detection Experiment",
            description="End-to-end test of MLOps workflow",
            tags=["e2e", "test", "anomaly_detection"],
            created_by="e2e_test"
        )
        
        run_id = await experiment_service.start_experiment_run(
            experiment_id=experiment_id,
            detector_name="IsolationForest",
            dataset_name="e2e_test_dataset",
            parameters={"contamination": 0.1, "n_estimators": 150}
        )
        
        await experiment_service.log_run_metrics(
            run_id=run_id,
            metrics={"accuracy": 0.89, "precision": 0.87, "recall": 0.91, "f1_score": 0.89}
        )
        
        completed_run = await experiment_service.finish_experiment_run(
            run_id=run_id,
            final_metrics={"accuracy": 0.90, "precision": 0.88, "recall": 0.92, "f1_score": 0.90}
        )
        
        # Step 2: Register the model
        print("Step 2: Registering model...")
        model_file = Path(temp_dir) / "e2e_model.pkl"
        model_file.write_text("e2e model content")
        
        model_key = await model_service.register_model(
            model_path=str(model_file),
            model_id="e2e_anomaly_detector",
            algorithm="IsolationForest",
            framework=ModelFramework.SCIKIT_LEARN,
            created_by="e2e_test",
            description="E2E test anomaly detection model",
            tags=["e2e", "production_ready"],
            hyperparameters={"contamination": 0.1, "n_estimators": 150},
            metrics={"accuracy": 0.90, "precision": 0.88, "recall": 0.92, "f1_score": 0.90},
            experiment_id=experiment_id
        )
        
        model_id, version = model_key.split(":", 1)
        
        # Step 3: Promote and deploy the model
        print("Step 3: Promoting and deploying model...")
        await model_service.promote_model(model_id, version, ModelStatus.PRODUCTION)
        
        deployment_id = await model_service.deploy_model(
            model_id=model_id,
            version=version,
            stage=DeploymentStage.PRODUCTION,
            replicas=3,
            resources={"cpu": "500m", "memory": "1Gi"}
        )
        
        # Step 4: Set up monitoring
        print("Step 4: Setting up monitoring...")
        rule_id = await monitoring_service.create_monitoring_rule(
            name="E2E Performance Monitor",
            model_id=model_id,
            deployment_id=deployment_id,
            metric_name="accuracy",
            threshold_value=0.85,
            operator="<",
            severity=MonitoringAlertSeverity.HIGH,
            description="Alert when accuracy drops below 85%"
        )
        
        # Step 5: Simulate production monitoring
        print("Step 5: Simulating production monitoring...")
        await monitoring_service.track_model_performance(
            model_id=model_id,
            deployment_id=deployment_id,
            accuracy=0.88,
            precision=0.86,
            recall=0.90,
            f1_score=0.88,
            latency_p95=120.0,
            throughput=85.0,
            error_rate=0.008
        )
        
        health_report = await monitoring_service.assess_model_health(
            model_id=model_id,
            deployment_id=deployment_id
        )
        
        # Step 6: Generate final report
        print("Step 6: Generating workflow report...")
        workflow_report = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "model_key": model_key,
            "deployment_id": deployment_id,
            "monitoring_rule_id": rule_id,
            "final_accuracy": completed_run.metrics.f1_score,
            "model_health": health_report.overall_health.value,
            "workflow_status": "completed_successfully"
        }
        
        print(f"✅ E2E Workflow completed successfully!")
        print(f"   - Experiment: {experiment_id}")
        print(f"   - Model: {model_key}")
        print(f"   - Deployment: {deployment_id}")
        print(f"   - Final F1-Score: {completed_run.metrics.f1_score}")
        print(f"   - Model Health: {health_report.overall_health.value}")
        
        return workflow_report


async def main():
    """Run all integration tests."""
    print("🚀 Starting MLOps Hexagonal Architecture Integration Tests")
    print("=" * 60)
    
    try:
        # Run individual service tests
        await test_experiment_service_integration()
        await test_model_registry_service_integration()
        await test_monitoring_service_integration()
        
        # Run end-to-end workflow test
        workflow_report = await test_end_to_end_workflow()
        
        # Final summary
        print("\n🎉 All Integration Tests Passed!")
        print("=" * 60)
        print("✅ Experiment Service: Working correctly with dependency injection")
        print("✅ Model Registry Service: Working correctly with dependency injection")
        print("✅ Monitoring Service: Working correctly with dependency injection")
        print("✅ End-to-End Workflow: Complete MLOps pipeline functional")
        print("\n🏗️ Hexagonal Architecture Benefits Demonstrated:")
        print("   • Clean separation of domain logic from infrastructure")
        print("   • Easy swapping of adapters (file-based ↔ database ↔ cloud)")
        print("   • Comprehensive dependency injection throughout")
        print("   • Domain services isolated from external dependencies")
        print("   • Testable architecture with clear boundaries")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)