"""
Comprehensive Data Observability Example

This example demonstrates the complete functionality of the Data Observability Package,
including lineage tracking, pipeline health monitoring, data catalog management,
and predictive quality monitoring.
"""

import json
import time
from datetime import datetime, timedelta
from uuid import uuid4

from monorepo.packages.data_observability import (
    DataObservabilityFacade,
    DataObservabilityContainer,
    DataAssetType,
    DataFormat,
    PredictionType
)


def main():
    """Run comprehensive data observability example."""
    print("=== Data Observability Package Example ===\n")
    
    # Initialize the container and facade
    container = DataObservabilityContainer()
    facade = container.observability_facade()
    
    # Part 1: Data Catalog Management
    print("1. Data Catalog Management")
    print("-" * 30)
    
    # Register data assets
    raw_data = facade.register_data_asset(
        name="raw_customer_data",
        asset_type=DataAssetType.FILE,
        location="s3://data-lake/raw/customers/",
        data_format=DataFormat.JSON,
        description="Raw customer data from various sources",
        owner="data-engineering@company.com",
        domain="customer"
    )
    print(f"Registered raw data asset: {raw_data.name}")
    
    processed_data = facade.register_data_asset(
        name="processed_customer_data",
        asset_type=DataAssetType.TABLE,
        location="warehouse.customers",
        data_format=DataFormat.PARQUET,
        description="Processed and cleaned customer data",
        owner="data-engineering@company.com",
        domain="customer"
    )
    print(f"Registered processed data asset: {processed_data.name}")
    
    analytics_data = facade.register_data_asset(
        name="customer_analytics",
        asset_type=DataAssetType.VIEW,
        location="warehouse.customer_analytics",
        data_format=DataFormat.SQL,
        description="Customer analytics and metrics",
        owner="analytics@company.com",
        domain="analytics"
    )
    print(f"Registered analytics asset: {analytics_data.name}")
    
    # Part 2: Data Lineage Tracking
    print("\n2. Data Lineage Tracking")
    print("-" * 30)
    
    # Track data transformations
    facade.track_data_transformation(
        source_id=raw_data.id,
        target_id=processed_data.id,
        transformation_type="data_cleaning",
        transformation_details={
            "operations": ["remove_duplicates", "validate_emails", "standardize_addresses"],
            "quality_checks": ["completeness", "validity", "consistency"]
        }
    )
    print("Tracked transformation: raw_data -> processed_data")
    
    facade.track_data_transformation(
        source_id=processed_data.id,
        target_id=analytics_data.id,
        transformation_type="aggregation",
        transformation_details={
            "metrics": ["total_customers", "avg_order_value", "customer_lifetime_value"],
            "group_by": ["region", "customer_segment", "date"]
        }
    )
    print("Tracked transformation: processed_data -> analytics_data")
    
    # Analyze impact
    impact = facade.analyze_impact(processed_data.id, "downstream")
    print(f"Impact analysis: {len(impact.get('affected_nodes', []))} assets affected by changes to processed_data")
    
    # Find data path
    path = facade.find_data_path(raw_data.id, analytics_data.id)
    print(f"Data path from raw to analytics: {len(path)} steps")
    
    # Part 3: Pipeline Health Monitoring
    print("\n3. Pipeline Health Monitoring")
    print("-" * 30)
    
    # Create a pipeline and monitor its health
    pipeline_id = uuid4()
    
    # Simulate pipeline executions with different health states
    for i in range(5):
        # Simulate degrading performance
        execution_time = 30000 + (i * 5000)  # Increasing execution time
        memory_usage = 256 + (i * 50)        # Increasing memory usage
        error_rate = 0.001 + (i * 0.002)     # Increasing error rate
        
        health = facade.monitor_pipeline_health(
            pipeline_id=pipeline_id,
            metrics={
                "execution_time_ms": execution_time,
                "memory_usage_mb": memory_usage,
                "rows_processed": 1000000,
                "error_rate": error_rate,
                "cpu_utilization": 0.6 + (i * 0.1)
            },
            context={
                "pipeline_name": "customer_data_processing",
                "version": "1.2.0",
                "environment": "production"
            }
        )
        
        print(f"Pipeline execution {i+1}: Health score = {health.health_score:.3f}")
        time.sleep(1)  # Simulate time between executions
    
    # Get pipeline health summary
    summary = facade.get_pipeline_health_summary(pipeline_id)
    print(f"Pipeline health summary: {summary['status']} (score: {summary['health_score']:.3f})")
    
    # Check for alerts
    alerts = facade.get_pipeline_alerts(pipeline_id)
    print(f"Active pipeline alerts: {len(alerts)}")
    
    # Part 4: Quality Prediction and Monitoring
    print("\n4. Quality Prediction and Monitoring")
    print("-" * 30)
    
    # Add quality metrics over time
    base_time = datetime.utcnow() - timedelta(days=7)
    for i in range(50):
        timestamp = base_time + timedelta(hours=i * 3)
        
        # Simulate declining quality
        completeness = 0.95 - (i * 0.002)  # Gradually decreasing completeness
        accuracy = 0.92 - (i * 0.001)      # Gradually decreasing accuracy
        
        facade.add_quality_metric(
            asset_id=processed_data.id,
            metric_type="completeness",
            value=completeness,
            timestamp=timestamp
        )
        
        facade.add_quality_metric(
            asset_id=processed_data.id,
            metric_type="accuracy",
            value=accuracy,
            timestamp=timestamp
        )
    
    print("Added 50 quality metric data points")
    
    # Analyze quality trends
    trend = facade.analyze_quality_trends(
        asset_id=processed_data.id,
        metric_type="completeness",
        days=7
    )
    print(f"Quality trend analysis: {trend.direction} (R² = {trend.r_squared:.3f})")
    
    # Create quality predictions
    prediction = facade.predict_quality_issues(
        asset_id=processed_data.id,
        prediction_type=PredictionType.QUALITY_DEGRADATION,
        target_time=datetime.utcnow() + timedelta(hours=24)
    )
    print(f"Quality prediction: {prediction.predicted_value:.3f} (confidence: {prediction.confidence})")
    
    # Create quality forecast
    forecast = facade.forecast_quality_metrics(
        asset_id=processed_data.id,
        metric_type="completeness",
        horizon_hours=48,
        resolution_hours=6
    )
    print(f"Quality forecast: {len(forecast.forecasted_values)} data points over 48 hours")
    
    # Part 5: Cross-Service Operations
    print("\n5. Cross-Service Operations")
    print("-" * 30)
    
    # Get comprehensive asset view
    asset_view = facade.get_comprehensive_asset_view(processed_data.id)
    print(f"Comprehensive view of {processed_data.name}:")
    print(f"  - Connected assets: {asset_view['lineage']['total_connected_assets']}")
    print(f"  - Quality score: {asset_view['asset_info']['quality_score']}")
    print(f"  - Active quality alerts: {asset_view['quality']['active_alerts']}")
    
    # Get data health dashboard
    dashboard = facade.get_data_health_dashboard()
    print(f"\nData Health Dashboard:")
    print(f"  - Total assets: {dashboard['catalog']['total_assets']}")
    print(f"  - Total pipelines: {dashboard['pipeline_health']['total_pipelines']}")
    print(f"  - Quality alerts: {dashboard['quality_predictions']['total_active_alerts']}")
    
    # Investigate data issue
    investigation = facade.investigate_data_issue(
        asset_id=processed_data.id,
        issue_type="quality_degradation",
        severity="medium"
    )
    print(f"\nData Issue Investigation:")
    print(f"  - {investigation['investigation_summary']}")
    print(f"  - Recommendations: {len(investigation['recommendations'])}")
    for rec in investigation['recommendations']:
        print(f"    • {rec}")
    
    # Part 6: Usage Tracking and Analytics
    print("\n6. Usage Tracking and Analytics")
    print("-" * 30)
    
    # Simulate usage tracking
    users = ["analyst1", "analyst2", "data_scientist", "dashboard_service"]
    for user in users:
        for _ in range(5):
            facade.track_asset_usage(
                asset_id=processed_data.id,
                user_id=user,
                usage_type="read",
                application="analytics_platform",
                duration_ms=1500
            )
    
    print(f"Tracked usage for {len(users)} users")
    
    # Get usage analytics
    usage_analytics = facade.catalog_service.get_usage_analytics(processed_data.id)
    print(f"Usage analytics: {usage_analytics['total_accesses']} total accesses")
    print(f"Unique users: {usage_analytics['unique_users']}")
    
    # Part 7: Discovery and Recommendations
    print("\n7. Discovery and Recommendations")
    print("-" * 30)
    
    # Discover assets
    discovered = facade.discover_data_assets("customer analytics")
    print(f"Discovered {len(discovered)} assets matching 'customer analytics'")
    
    # Get recommendations
    recommendations = facade.get_asset_recommendations(processed_data.id)
    print(f"Found {len(recommendations)} similar assets")
    for asset, similarity in recommendations[:3]:  # Show top 3
        print(f"  - {asset.name} (similarity: {similarity:.3f})")
    
    print("\n=== Example Complete ===")
    print("\nThis example demonstrated:")
    print("• Asset registration and catalog management")
    print("• Data lineage tracking and impact analysis")
    print("• Pipeline health monitoring and alerting")
    print("• Quality prediction and trend analysis")
    print("• Cross-service operations and investigations")
    print("• Usage tracking and analytics")
    print("• Asset discovery and recommendations")


if __name__ == "__main__":
    main()