"""Example demonstrating the integrated alerting workflow for model performance degradation detection.

This example shows how to use the ModelPerformanceDegradationDetector with the new
alerting workflow integration, including:
- Creating alerts when degradation is detected
- Using different notification channels
- Correlating alerts with the intelligent alert service
- Viewing alert statistics and management
"""

import asyncio
from datetime import datetime
from typing import List

from pynomaly.application.services.model_performance_degradation_detector import (
    ModelPerformanceDegradationDetector,
    DegradationDetectorConfig,
    DetectionAlgorithm,
)
from pynomaly.application.services.performance_alert_service import PerformanceAlertService
from pynomaly.application.services.intelligent_alert_service import IntelligentAlertService
from pynomaly.domain.entities.model_performance import (
    ModelPerformanceMetrics,
    ModelPerformanceBaseline,
)
from pynomaly.domain.entities.alert import (
    NotificationChannel,
    AlertSeverity,
    AlertSource,
    AlertStatus,
)


async def main():
    """Main function demonstrating the integrated alerting workflow."""
    
    print("=== Model Performance Degradation Alerting Demo ===")
    print()
    
    # Setup: Create detector with different algorithms
    configs = [
        DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
            delta=0.1,
        ),
        DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.STATISTICAL,
            confidence=0.95,
            statistical_method="z_score",
        ),
        DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.ML_BASED,
            # ml_model would be provided in real usage
        ),
    ]
    
    # Create performance alert service
    intelligent_alert_service = IntelligentAlertService()
    performance_alert_service = PerformanceAlertService(intelligent_alert_service)
    
    # Test models and their performance data
    test_models = [
        {
            "model_id": "fraud_detection_v1",
            "model_name": "Fraud Detection Model v1.0",
            "current_metrics": ModelPerformanceMetrics(
                accuracy=0.85,
                precision=0.82,
                recall=0.78,
                f1=0.80,
                timestamp=datetime.utcnow(),
                model_id="fraud_detection_v1",
                dataset_id="fraud_validation_set_2024",
            ),
            "baseline": ModelPerformanceBaseline(
                model_id="fraud_detection_v1",
                version="v1.0",
                mean=0.92,
                std=0.02,
            ),
            "notification_channels": [
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
            ],
        },
        {
            "model_id": "recommendation_engine_v2",
            "model_name": "Recommendation Engine v2.1",
            "current_metrics": ModelPerformanceMetrics(
                accuracy=0.88,
                precision=0.85,
                recall=0.82,
                f1=0.83,
                timestamp=datetime.utcnow(),
                model_id="recommendation_engine_v2",
                dataset_id="recommendation_test_set_2024",
            ),
            "baseline": ModelPerformanceBaseline(
                model_id="recommendation_engine_v2",
                version="v2.1",
                mean=0.89,
                std=0.01,
            ),
            "notification_channels": [
                NotificationChannel.EMAIL,
                NotificationChannel.WEBHOOK,
            ],
        },
        {
            "model_id": "churn_prediction_v1",
            "model_name": "Customer Churn Prediction v1.0",
            "current_metrics": ModelPerformanceMetrics(
                accuracy=0.75,
                precision=0.70,
                recall=0.65,
                f1=0.67,
                timestamp=datetime.utcnow(),
                model_id="churn_prediction_v1",
                dataset_id="churn_validation_set_2024",
            ),
            "baseline": ModelPerformanceBaseline(
                model_id="churn_prediction_v1",
                version="v1.0",
                mean=0.85,
                std=0.03,
            ),
            "notification_channels": [
                NotificationChannel.EMAIL,
                NotificationChannel.PAGERDUTY,
            ],
        },
    ]
    
    # Test different detection algorithms
    for i, config in enumerate(configs):
        print(f"--- Testing {config.algorithm.value} Algorithm ---")
        
        detector = ModelPerformanceDegradationDetector(config)
        
        for model_data in test_models:
            print(f"\nProcessing model: {model_data['model_name']}")
            
            try:
                # Run detection with integrated alerting
                degradation_result = await detector.detect(
                    current_metrics=model_data["current_metrics"],
                    baseline=model_data["baseline"],
                    model_id=model_data["model_id"],
                    model_name=model_data["model_name"],
                    notification_channels=model_data["notification_channels"],
                )
                
                # Display results
                print(f"  Degradation detected: {degradation_result.degrade_flag}")
                print(f"  Overall severity: {degradation_result.overall_severity:.3f}")
                print(f"  Affected metrics: {len(degradation_result.affected_metrics)}")
                
                if degradation_result.affected_metrics:
                    for metric in degradation_result.affected_metrics:
                        print(f"    - {metric.metric_name}: "
                              f"{metric.current_value:.3f} -> "
                              f"{metric.baseline_value:.3f} "
                              f"({metric.relative_deviation:.1f}% deviation)")
                
                # If degradation was detected, show alert creation
                if degradation_result.degrade_flag:
                    print(f"  ✓ Alert created and notifications sent via: "
                          f"{[ch.value for ch in model_data['notification_channels']]}")
                
            except Exception as e:
                print(f"  ✗ Error processing model: {e}")
        
        print()
    
    # Demonstrate alert management capabilities
    print("--- Alert Management Demo ---")
    
    # Get performance alerts
    performance_alerts = await performance_alert_service.get_performance_alerts(
        severity=AlertSeverity.HIGH,
        status=AlertStatus.ACTIVE,
        limit=10
    )
    
    print(f"Active high-severity performance alerts: {len(performance_alerts)}")
    for alert in performance_alerts:
        print(f"  - {alert.name} (Model: {alert.metadata.get('model_name', 'Unknown')})")
    
    # Get alert statistics
    print("\n--- Alert Statistics ---")
    stats = await performance_alert_service.get_performance_alert_statistics(days=7)
    
    print(f"Total performance alerts (last 7 days): {stats['total_performance_alerts']}")
    print(f"Models with degradation: {stats['models_with_degradation']}")
    print(f"Severity distribution: {stats['severity_distribution']}")
    
    if stats['most_affected_metrics']:
        print("Most affected metrics:")
        for metric, count in list(stats['most_affected_metrics'].items())[:5]:
            print(f"  - {metric}: {count} alerts")
    
    if stats['detection_algorithms']:
        print("Detection algorithms used:")
        for algorithm, count in stats['detection_algorithms'].items():
            print(f"  - {algorithm}: {count} alerts")
    
    # Demonstrate creating specific metric alerts
    print("\n--- Metric-Specific Alert Demo ---")
    
    from pynomaly.application.services.model_performance_degradation_detector import DegradationDetails
    
    # Create a specific metric degradation alert
    metric_degradation = DegradationDetails(
        metric_name="accuracy",
        current_value=0.75,
        baseline_value=0.90,
        deviation=0.0,  # Will be calculated
        relative_deviation=0.0,  # Will be calculated
        statistical_significance={"p_value": 0.001, "z_score": -3.2}
    )
    
    metric_alert = await performance_alert_service.create_metric_specific_alert(
        degradation_detail=metric_degradation,
        model_id="demo_model",
        model_name="Demo Model for Metric Alert",
        additional_context={
            "environment": "production",
            "data_source": "live_traffic",
            "monitoring_system": "mlops_platform"
        }
    )
    
    print(f"Created metric-specific alert: {metric_alert.name}")
    print(f"Alert ID: {metric_alert.id}")
    print(f"Severity: {metric_alert.severity.value}")
    print(f"Tags: {metric_alert.tags}")
    
    # Demonstrate intelligent alert correlation
    print("\n--- Intelligent Alert Correlation Demo ---")
    
    # Get analytics from the intelligent alert service
    analytics = await intelligent_alert_service.get_alert_analytics(days=1)
    
    print(f"Total alerts processed: {analytics['performance_metrics']['total_processed']}")
    print(f"Suppression rate: {analytics['performance_metrics']['suppression_rate']:.2%}")
    print(f"Correlated alerts: {analytics['correlation_stats']['correlated_alerts']}")
    
    if analytics['correlation_stats']['avg_correlation_strength'] > 0:
        print(f"Average correlation strength: {analytics['correlation_stats']['avg_correlation_strength']:.3f}")
    
    # Show noise reduction effectiveness
    noise_stats = analytics['noise_reduction_stats']
    print(f"ML classified as noise: {noise_stats['ml_classified_noise']}")
    print(f"Signal-to-noise ratio: {noise_stats['signal_to_noise_ratio']:.2f}")
    
    print("\n=== Demo Complete ===")


async def demonstrate_alert_lifecycle():
    """Demonstrate the complete alert lifecycle with notifications."""
    
    print("=== Alert Lifecycle Demo ===")
    
    # Create services
    intelligent_alert_service = IntelligentAlertService()
    performance_alert_service = PerformanceAlertService(intelligent_alert_service)
    
    # Create detector
    config = DegradationDetectorConfig(
        algorithm=DetectionAlgorithm.STATISTICAL,
        confidence=0.95,
        statistical_method="z_score",
    )
    detector = ModelPerformanceDegradationDetector(config)
    
    # Simulate degradation detection
    current_metrics = ModelPerformanceMetrics(
        accuracy=0.70,
        precision=0.68,
        recall=0.65,
        f1=0.66,
        timestamp=datetime.utcnow(),
        model_id="lifecycle_demo_model",
        dataset_id="lifecycle_test_set",
    )
    
    baseline = ModelPerformanceBaseline(
        model_id="lifecycle_demo_model",
        version="v1.0",
        mean=0.85,
        std=0.02,
    )
    
    # Run detection
    print("1. Running degradation detection...")
    degradation_result = await detector.detect(
        current_metrics=current_metrics,
        baseline=baseline,
        model_id="lifecycle_demo_model",
        model_name="Lifecycle Demo Model",
        notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
    )
    
    if degradation_result.degrade_flag:
        print("   ✓ Degradation detected - Alert created")
        
        # Get the created alert
        alerts = await performance_alert_service.get_performance_alerts(
            model_name="Lifecycle Demo Model",
            limit=1
        )
        
        if alerts:
            alert = alerts[0]
            print(f"   Alert ID: {alert.id}")
            print(f"   Severity: {alert.severity.value}")
            print(f"   Status: {alert.status.value}")
            
            # Acknowledge the alert
            print("2. Acknowledging alert...")
            await intelligent_alert_service.acknowledge_alert(
                alert_id=alert.id,
                acknowledged_by="ml_engineer@company.com",
                note="Investigating root cause"
            )
            print("   ✓ Alert acknowledged")
            
            # Resolve the alert
            print("3. Resolving alert...")
            await intelligent_alert_service.resolve_alert(
                alert_id=alert.id,
                resolved_by="ml_engineer@company.com",
                resolution_note="Model retrained with updated data",
                quality_score=0.8
            )
            print("   ✓ Alert resolved")
            
            # Show alert timeline
            print("4. Alert timeline:")
            timeline = alert.get_timeline()
            for event in timeline:
                print(f"   {event['timestamp']}: {event['event']} by {event['user']}")
    else:
        print("   No degradation detected")
    
    print("=== Lifecycle Demo Complete ===")


if __name__ == "__main__":
    # Run the main demo
    asyncio.run(main())
    
    print("\n" + "="*50 + "\n")
    
    # Run the lifecycle demo
    asyncio.run(demonstrate_alert_lifecycle())
