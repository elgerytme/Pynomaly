#!/usr/bin/env python3
"""Integration test for data quality hexagonal architecture."""

import asyncio
import tempfile
import sys
from pathlib import Path
from datetime import datetime

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import container and configuration
from data_quality.infrastructure.container.container import (
    DataQualityContainer,
    DataQualityContainerConfig
)

# Import domain interfaces
from data_quality.domain.interfaces.data_processing_operations import (
    DataProfilingPort,
    DataValidationPort,
    StatisticalAnalysisPort,
    DataProfilingRequest
)
from data_quality.domain.interfaces.external_system_operations import (
    DataSourcePort,
    NotificationPort,
    MetadataPort
)
from data_quality.domain.interfaces.quality_assessment_operations import (
    RuleEvaluationPort,
    QualityMetricsPort,
    AnomalyDetectionPort
)

# Import domain entities
from data_quality.domain.entities.data_quality_rule import DataQualityRule


async def test_container_configuration():
    """Test the dependency injection container configuration."""
    print("🔧 Testing Container Configuration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create container with file-based configuration
        config = DataQualityContainerConfig(
            enable_file_data_processing=True,
            enable_statistical_analysis=True,
            enable_quality_monitoring=True,
            data_storage_path=temp_dir,
            environment="test"
        )
        
        container = DataQualityContainer(config)
        
        # Test that all required services are registered
        required_interfaces = [
            DataProfilingPort,
            DataValidationPort,
            StatisticalAnalysisPort,
            DataSourcePort,
            NotificationPort,
            MetadataPort,
            RuleEvaluationPort,
            QualityMetricsPort,
            AnomalyDetectionPort
        ]
        
        for interface in required_interfaces:
            assert container.is_registered(interface), f"Interface {interface.__name__} not registered"
            service = container.get(interface)
            assert service is not None, f"Service for {interface.__name__} is None"
        
        # Test configuration summary
        summary = container.get_configuration_summary()
        assert summary["environment"] == "test"
        assert summary["data_processing"]["file_enabled"] is True
        assert len(summary["registered_services"]["singletons"]) > 0
        
        print(f"✅ Container configured with {summary['registered_services']['count']} services")
        return container


async def test_data_profiling_integration():
    """Test data profiling operations through the container."""
    print("\n📊 Testing Data Profiling Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        container = DataQualityContainer(DataQualityContainerConfig(
            data_storage_path=temp_dir,
            environment="test"
        ))
        
        # Get data profiling service through dependency injection
        profiling_service = container.get(DataProfilingPort)
        
        # Create a test data profiling request
        profiling_request = DataProfilingRequest(
            data_source="test_data.csv",
            profile_config={
                "include_distributions": True,
                "include_correlations": False
            },
            include_distributions=True,
            sample_size=1000,
            metadata={"test": True}
        )
        
        # Execute profiling
        profile = await profiling_service.create_data_profile(profiling_request)
        
        # Verify profile creation
        assert profile is not None, "Profile should not be None"
        assert profile.data_source == "test_data.csv", "Data source should match"
        assert profile.row_count > 0, "Row count should be positive"
        assert profile.column_count > 0, "Column count should be positive"
        assert len(profile.column_profiles) > 0, "Should have column profiles"
        
        print(f"✅ Created profile with {profile.row_count} rows and {profile.column_count} columns")
        
        # Test column profiling
        column_profile = await profiling_service.create_column_profile(
            "test_data.csv", "test_column", {}
        )
        
        assert column_profile is not None, "Column profile should not be None"
        assert column_profile.column_name == "test_column", "Column name should match"
        
        print("✅ Column profiling completed successfully")
        
        return profile


async def test_data_validation_integration():
    """Test data validation operations through the container."""
    print("\n🔍 Testing Data Validation Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        container = DataQualityContainer(DataQualityContainerConfig(
            data_storage_path=temp_dir,
            environment="test"
        ))
        
        # Get validation service through dependency injection
        validation_service = container.get(DataValidationPort)
        
        # Create test quality rules
        test_rules = [
            DataQualityRule(
                id="rule_001",
                name="Completeness Check",
                description="Check data completeness",
                rule_type="completeness",
                target_column="id",
                conditions={"min_completeness": 0.95},
                threshold_value=0.95,
                severity="high",
                tags=["completeness"],
                metadata={}
            ),
            DataQualityRule(
                id="rule_002",
                name="Uniqueness Check",
                description="Check data uniqueness",
                rule_type="uniqueness",
                target_column="id",
                conditions={},
                threshold_value=1.0,
                severity="medium",
                tags=["uniqueness"],
                metadata={}
            )
        ]
        
        # Test business rules validation
        validation_results = await validation_service.validate_business_rules(
            "test_data.csv", test_rules
        )
        
        assert len(validation_results) == 2, "Should have 2 validation results"
        
        for result in validation_results:
            assert result.check_id is not None, "Check ID should not be None"
            assert result.rule_id is not None, "Rule ID should not be None"
            assert isinstance(result.passed, bool), "Passed should be boolean"
            assert isinstance(result.score, float), "Score should be float"
            assert result.executed_at is not None, "Execution time should not be None"
        
        print(f"✅ Validated {len(validation_results)} rules successfully")
        
        # Test completeness checking
        completeness_result = await validation_service.check_data_completeness(
            "test_data.csv", ["id", "name", "value"]
        )
        
        assert "overall_completeness" in completeness_result, "Should have overall completeness"
        assert "column_completeness" in completeness_result, "Should have column completeness"
        
        print("✅ Data completeness check completed successfully")
        
        return validation_results


async def test_quality_assessment_integration():
    """Test quality assessment operations through the container."""
    print("\n🎯 Testing Quality Assessment Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        container = DataQualityContainer(DataQualityContainerConfig(
            data_storage_path=temp_dir,
            environment="test"
        ))
        
        # Get quality assessment services through dependency injection
        rule_evaluation_service = container.get(RuleEvaluationPort)
        quality_metrics_service = container.get(QualityMetricsPort)
        anomaly_detection_service = container.get(AnomalyDetectionPort)
        
        # Test rule evaluation
        test_rule = DataQualityRule(
            id="eval_rule_001",
            name="Test Rule",
            description="Test rule evaluation",
            rule_type="completeness",
            target_column="test_column",
            conditions={},
            threshold_value=0.9,
            severity="medium",
            tags=["test"],
            metadata={}
        )
        
        rule_result = await rule_evaluation_service.evaluate_rule("test_data.csv", test_rule)
        
        assert rule_result is not None, "Rule result should not be None"
        assert rule_result.rule_id == test_rule.id, "Rule ID should match"
        assert isinstance(rule_result.passed, bool), "Passed should be boolean"
        assert isinstance(rule_result.score, float), "Score should be float"
        
        print("✅ Rule evaluation completed successfully")
        
        # Test quality metrics calculation
        quality_score = await quality_metrics_service.calculate_quality_score(
            "test_data.csv", {"include_all_metrics": True}
        )
        
        assert isinstance(quality_score, float), "Quality score should be float"
        assert 0.0 <= quality_score <= 1.0, "Quality score should be between 0 and 1"
        
        print(f"✅ Calculated quality score: {quality_score}")
        
        # Test completeness metrics
        completeness_metrics = await quality_metrics_service.calculate_completeness_metrics(
            "test_data.csv", ["id", "name"]
        )
        
        assert len(completeness_metrics) > 0, "Should have completeness metrics"
        
        for metric in completeness_metrics:
            assert metric.name is not None, "Metric name should not be None"
            assert metric.value is not None, "Metric value should not be None"
        
        print(f"✅ Calculated {len(completeness_metrics)} completeness metrics")
        
        # Test anomaly detection
        from data_quality.domain.interfaces.quality_assessment_operations import (
            AnomalyDetectionConfig, AnomalyType
        )
        
        anomaly_config = AnomalyDetectionConfig(
            anomaly_types=[AnomalyType.STATISTICAL, AnomalyType.VOLUME],
            sensitivity=0.95
        )
        
        anomalies = await anomaly_detection_service.detect_statistical_anomalies(
            "test_data.csv", anomaly_config
        )
        
        assert isinstance(anomalies, list), "Anomalies should be a list"
        
        print(f"✅ Detected {len(anomalies)} anomalies")
        
        return {
            "rule_result": rule_result,
            "quality_score": quality_score,
            "completeness_metrics": completeness_metrics,
            "anomalies": anomalies
        }


async def test_external_system_integration():
    """Test external system operations through the container."""
    print("\n🌐 Testing External System Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        container = DataQualityContainer(DataQualityContainerConfig(
            data_storage_path=temp_dir,
            environment="test"
        ))
        
        # Get external system services through dependency injection
        data_source_service = container.get(DataSourcePort)
        notification_service = container.get(NotificationPort)
        metadata_service = container.get(MetadataPort)
        
        # Test data source operations
        from data_quality.domain.interfaces.external_system_operations import (
            DataSourceConfig, DataSourceType
        )
        
        data_source_config = DataSourceConfig(
            source_type=DataSourceType.CSV,
            connection_params={"file_path": "test.csv"},
            timeout_seconds=30
        )
        
        connection_id = await data_source_service.connect_to_source(data_source_config)
        assert connection_id is not None, "Connection ID should not be None"
        
        test_connection = await data_source_service.test_connection(data_source_config)
        assert test_connection is True, "Test connection should succeed"
        
        print("✅ Data source operations completed successfully")
        
        # Test notification operations
        from data_quality.domain.interfaces.external_system_operations import (
            NotificationConfig, NotificationChannel
        )
        
        notification_config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            recipients=["test@example.com"],
            priority="normal"
        )
        
        notification_sent = await notification_service.send_notification(
            notification_config, "Test Subject", "Test Message"
        )
        assert notification_sent is True, "Notification should be sent"
        
        alert_sent = await notification_service.send_alert(
            notification_config, "warning", "Test alert message"
        )
        assert alert_sent is True, "Alert should be sent"
        
        print("✅ Notification operations completed successfully")
        
        # Test metadata operations
        metadata_stored = await metadata_service.store_metadata(
            "profile", "test_profile_001", {"test": True, "created_by": "test"}
        )
        assert metadata_stored is True, "Metadata should be stored"
        
        retrieved_metadata = await metadata_service.retrieve_metadata(
            "profile", "test_profile_001"
        )
        assert retrieved_metadata is not None, "Metadata should be retrieved"
        assert retrieved_metadata["entity_type"] == "profile", "Entity type should match"
        
        print("✅ Metadata operations completed successfully")
        
        return {
            "connection_id": connection_id,
            "notification_sent": notification_sent,
            "alert_sent": alert_sent,
            "metadata": retrieved_metadata
        }


async def test_end_to_end_workflow():
    """Test complete end-to-end data quality workflow."""
    print("\n🔄 Testing End-to-End Data Quality Workflow...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        container = DataQualityContainer(DataQualityContainerConfig(
            data_storage_path=temp_dir,
            environment="integration_test"
        ))
        
        workflow_results = {}
        
        # Step 1: Data Profiling
        print("Step 1: Creating data profile...")
        profiling_service = container.get(DataProfilingPort)
        
        profiling_request = DataProfilingRequest(
            data_source="workflow_test_data.csv",
            profile_config={"comprehensive": True},
            include_distributions=True,
            metadata={"workflow": "end_to_end_test"}
        )
        
        profile = await profiling_service.create_data_profile(profiling_request)
        workflow_results["profile"] = profile
        print(f"✅ Profile created: {profile.id}")
        
        # Step 2: Rule-based Validation
        print("Step 2: Executing validation rules...")
        validation_service = container.get(DataValidationPort)
        
        validation_rules = [
            DataQualityRule(
                id="workflow_rule_001",
                name="Workflow Completeness",
                description="Ensure data completeness for workflow",
                rule_type="completeness",
                target_column="id",
                conditions={"min_completeness": 0.95},
                threshold_value=0.95,
                severity="high",
                tags=["workflow"],
                metadata={}
            )
        ]
        
        validation_results = await validation_service.validate_business_rules(
            "workflow_test_data.csv", validation_rules
        )
        workflow_results["validation"] = validation_results
        print(f"✅ Validated {len(validation_results)} rules")
        
        # Step 3: Quality Metrics Calculation
        print("Step 3: Calculating quality metrics...")
        metrics_service = container.get(QualityMetricsPort)
        
        overall_quality = await metrics_service.calculate_quality_score(
            "workflow_test_data.csv", {"comprehensive": True}
        )
        
        completeness_metrics = await metrics_service.calculate_completeness_metrics(
            "workflow_test_data.csv"
        )
        
        workflow_results["quality_metrics"] = {
            "overall_quality": overall_quality,
            "completeness_metrics": completeness_metrics
        }
        print(f"✅ Quality score: {overall_quality}")
        
        # Step 4: Anomaly Detection
        print("Step 4: Detecting anomalies...")
        anomaly_service = container.get(AnomalyDetectionPort)
        
        from data_quality.domain.interfaces.quality_assessment_operations import (
            AnomalyDetectionConfig, AnomalyType
        )
        
        anomaly_config = AnomalyDetectionConfig(
            anomaly_types=[AnomalyType.STATISTICAL, AnomalyType.VOLUME],
            sensitivity=0.9
        )
        
        statistical_anomalies = await anomaly_service.detect_statistical_anomalies(
            "workflow_test_data.csv", anomaly_config
        )
        
        volume_anomalies = await anomaly_service.detect_volume_anomalies(
            "workflow_test_data.csv"
        )
        
        workflow_results["anomalies"] = {
            "statistical": statistical_anomalies,
            "volume": volume_anomalies
        }
        print(f"✅ Detected {len(statistical_anomalies)} statistical and {len(volume_anomalies)} volume anomalies")
        
        # Step 5: Metadata and Lineage Tracking
        print("Step 5: Tracking metadata and lineage...")
        metadata_service = container.get(MetadataPort)
        
        await metadata_service.store_metadata(
            "workflow", "end_to_end_test", {
                "profile_id": profile.id,
                "validation_rules": len(validation_rules),
                "quality_score": overall_quality,
                "anomalies_detected": len(statistical_anomalies) + len(volume_anomalies),
                "completed_at": datetime.now().isoformat()
            }
        )
        
        print("✅ Metadata and lineage tracked")
        
        # Step 6: Generate Summary Report
        print("Step 6: Generating workflow summary...")
        
        workflow_summary = {
            "workflow_id": "end_to_end_test",
            "data_source": "workflow_test_data.csv",
            "completed_at": datetime.now().isoformat(),
            "profile_summary": {
                "profile_id": profile.id,
                "row_count": profile.row_count,
                "column_count": profile.column_count,
                "data_types": len(profile.data_types)
            },
            "validation_summary": {
                "rules_executed": len(validation_results),
                "rules_passed": sum(1 for r in validation_results if r.passed),
                "overall_validation_score": sum(r.score for r in validation_results) / len(validation_results)
            },
            "quality_summary": {
                "overall_quality_score": overall_quality,
                "completeness_metrics_count": len(completeness_metrics)
            },
            "anomaly_summary": {
                "statistical_anomalies": len(statistical_anomalies),
                "volume_anomalies": len(volume_anomalies),
                "total_anomalies": len(statistical_anomalies) + len(volume_anomalies)
            },
            "workflow_status": "completed_successfully"
        }
        
        workflow_results["summary"] = workflow_summary
        
        print("✅ End-to-End Workflow completed successfully!")
        print(f"   - Profile ID: {workflow_summary['profile_summary']['profile_id']}")
        print(f"   - Validation Score: {workflow_summary['validation_summary']['overall_validation_score']:.2f}")
        print(f"   - Quality Score: {workflow_summary['quality_summary']['overall_quality_score']:.2f}")
        print(f"   - Anomalies Detected: {workflow_summary['anomaly_summary']['total_anomalies']}")
        
        return workflow_results


async def main():
    """Run all integration tests."""
    print("🚀 Starting Data Quality Hexagonal Architecture Integration Tests")
    print("=" * 80)
    
    try:
        # Test container configuration
        container = await test_container_configuration()
        
        # Test individual service integrations
        profile = await test_data_profiling_integration()
        validation_results = await test_data_validation_integration()
        quality_assessment = await test_quality_assessment_integration()
        external_systems = await test_external_system_integration()
        
        # Test end-to-end workflow
        workflow_results = await test_end_to_end_workflow()
        
        # Final summary
        print("\n🎉 All Integration Tests Passed!")
        print("=" * 80)
        print("✅ Container Configuration: Working correctly with dependency injection")
        print("✅ Data Profiling: Working correctly with file-based adapters")
        print("✅ Data Validation: Working correctly with rule evaluation")
        print("✅ Quality Assessment: Working correctly with metrics and anomaly detection")
        print("✅ External Systems: Working correctly with stubs and file adapters")
        print("✅ End-to-End Workflow: Complete data quality pipeline functional")
        print("\n🏗️ Hexagonal Architecture Benefits Demonstrated:")
        print("   • Clean separation of domain logic from infrastructure")
        print("   • Easy swapping of adapters (file-based ↔ database ↔ cloud)")
        print("   • Comprehensive dependency injection throughout")
        print("   • Domain services isolated from external dependencies")
        print("   • Testable architecture with clear boundaries")
        print("   • Graceful fallback to stub implementations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)