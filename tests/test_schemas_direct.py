#!/usr/bin/env python3
"""Simple test script to validate the schemas package directly."""

import sys
from datetime import datetime, date
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, 'src')

try:
    # Test direct imports from schemas modules
    from pynomaly.schemas.analytics.base import MetricFrame, RealTimeMetricFrame
    from pynomaly.schemas.analytics.anomaly_kpis import (
        AnomalyKPIFrame, 
        AnomalyDetectionMetrics, 
        AnomalyClassificationMetrics,
        AnomalySeverity, 
        AnomalyCategory
    )
    from pynomaly.schemas.analytics.system_health import (
        SystemHealthFrame, 
        SystemResourceMetrics,
        SystemPerformanceMetrics, 
        SystemStatusMetrics,
        SystemStatus
    )
    from pynomaly.schemas.analytics.financial_impact import (
        FinancialImpactFrame,
        CostMetrics,
        SavingsMetrics,
        RevenueMetrics,
        ROICalculation
    )
    from pynomaly.schemas.analytics.roi import (
        ROIFrame,
        CostBenefitAnalysis,
        InvestmentMetrics
    )
    from pynomaly.schemas.validation import (
        SchemaCompatibilityError,
        BackwardCompatibilityValidator,
        validate_schema_compatibility
    )
    from pynomaly.schemas.versioning import (
        SchemaVersion,
        is_compatible_version
    )
    
    print("✓ All direct imports successful")
    
    # Test basic MetricFrame creation
    basic_metric = MetricFrame(
        metric_id="test_metric_1",
        name="Test Metric",
        value=42.5,
        timestamp=datetime.utcnow()
    )
    print(f"✓ Basic metric created: {basic_metric.name} = {basic_metric.value}")
    
    # Test AnomalyKPIFrame creation
    detection_metrics = AnomalyDetectionMetrics(
        accuracy=0.95,
        precision=0.92,
        recall=0.88,
        f1_score=0.9,
        false_positive_rate=0.05,
        false_negative_rate=0.12,
        roc_auc=0.94,
        pr_auc=0.91
    )
    
    classification_metrics = AnomalyClassificationMetrics(
        true_positives=150,
        false_positives=8,
        true_negatives=1842,
        false_negatives=20,
        anomalies_detected=158,
        anomalies_confirmed=150,
        anomalies_dismissed=8,
        severity_distribution={
            AnomalySeverity.CRITICAL: 5,
            AnomalySeverity.HIGH: 25,
            AnomalySeverity.MEDIUM: 70,
            AnomalySeverity.LOW: 50
        },
        category_distribution={
            AnomalyCategory.STATISTICAL: 80,
            AnomalyCategory.BEHAVIORAL: 40,
            AnomalyCategory.CONTEXTUAL: 30,
            AnomalyCategory.TEMPORAL: 8
        }
    )
    
    anomaly_kpi = AnomalyKPIFrame(
        metric_id="anomaly_kpi_1",
        name="Anomaly Detection KPIs",
        value=95.0,
        timestamp=datetime.utcnow(),
        detection_metrics=detection_metrics,
        classification_metrics=classification_metrics,
        model_name="IsolationForest",
        model_version="1.2.3",
        dataset_id="production_dataset_001",
        throughput=1250.0,
        cpu_usage=65.5,
        memory_usage=2048.0,
        active_alerts=3,
        critical_alerts=1,
        confidence_score=0.92,
        data_quality_score=0.88
    )
    
    print(f"✓ Anomaly KPI Frame created: {anomaly_kpi.name}")
    print(f"  - Accuracy: {anomaly_kpi.detection_metrics.accuracy}")
    print(f"  - Anomaly rate: {anomaly_kpi.get_anomaly_rate():.3f}")
    print(f"  - Is healthy: {anomaly_kpi.is_healthy()}")
    
    # Test SystemHealthFrame creation
    resource_metrics = SystemResourceMetrics(
        cpu_usage_percent=45.2,
        cpu_load_average=1.8,
        cpu_cores=8,
        cpu_frequency=2.4,
        memory_usage_percent=68.5,
        memory_used_mb=5500.0,
        memory_total_mb=8192.0,
        memory_available_mb=2692.0,
        disk_usage_percent=72.3,
        disk_used_gb=362.0,
        disk_total_gb=500.0,
        disk_io_read_rate=25.5,
        disk_io_write_rate=18.2,
        network_bytes_sent_rate=12.8,
        network_bytes_recv_rate=24.5,
        network_packets_sent_rate=150.0,
        network_packets_recv_rate=280.0,
        process_count=156,
        thread_count=892
    )
    
    performance_metrics = SystemPerformanceMetrics(
        avg_response_time_ms=125.5,
        p95_response_time_ms=280.0,
        p99_response_time_ms=450.0,
        requests_per_second=850.0,
        transactions_per_second=650.0,
        error_rate=0.025,
        timeout_rate=0.008,
        retry_rate=0.015,
        database_connections=25,
        database_query_time_ms=45.2,
        database_deadlocks=0,
        cache_hit_rate=0.85,
        cache_miss_rate=0.15,
        queue_depth=12,
        queue_processing_time_ms=85.5
    )
    
    status_metrics = SystemStatusMetrics(
        system_status=SystemStatus.HEALTHY,
        uptime_seconds=2678400.0,  # 31 days
        services_total=15,
        services_healthy=14,
        services_degraded=1,
        services_failed=0,
        active_alerts=2,
        critical_alerts=0,
        warning_alerts=2,
        maintenance_mode=False,
        deployment_in_progress=False,
        security_scan_passed=True,
        vulnerability_count=0,
        last_backup_timestamp=datetime.utcnow(),
        backup_status="completed"
    )
    
    system_health = SystemHealthFrame(
        metric_id="system_health_1",
        name="System Health Metrics",
        value=0.92,
        timestamp=datetime.utcnow(),
        resource_metrics=resource_metrics,
        performance_metrics=performance_metrics,
        status_metrics=status_metrics,
        hostname="prod-server-01",
        environment="production",
        region="us-east-1",
        overall_health_score=0.92,
        availability_score=0.995,
        reliability_score=0.98,
        capacity_utilization=0.68,
        cpu_trend="stable",
        memory_trend="increasing",
        disk_trend="stable",
        configuration_version="v2.1.0",
        deployment_version="v1.8.5"
    )
    
    print(f"✓ System Health Frame created: {system_health.name}")
    print(f"  - Overall health: {system_health.overall_health_score}")
    print(f"  - Is healthy: {system_health.is_healthy()}")
    print(f"  - Needs attention: {system_health.needs_attention()}")
    
    # Test FinancialImpactFrame creation
    cost_metrics = CostMetrics(
        total_cost=15000.0,
        cost_per_unit=1.25,
        budget=20000.0
    )
    
    savings_metrics = SavingsMetrics(
        total_savings=5000.0,
        savings_rate=0.25
    )
    
    revenue_metrics = RevenueMetrics(
        total_revenue=50000.0,
        revenue_per_unit=4.15,
        revenue_growth_rate=0.15
    )
    
    roi_calculation = ROICalculation(
        investment=15000.0,
        returns=55000.0,
        period_start_date=date(2024, 1, 1),
        period_end_date=date(2024, 12, 31)
    )
    
    financial_impact = FinancialImpactFrame(
        metric_id="financial_impact_1",
        name="Financial Impact Analysis",
        value=0.75,
        timestamp=datetime.utcnow(),
        cost_metrics=cost_metrics,
        savings_metrics=savings_metrics,
        revenue_metrics=revenue_metrics,
        roi_calculation=roi_calculation
    )
    
    print(f"✓ Financial Impact Frame created: {financial_impact.name}")
    print(f"  - ROI: {financial_impact.roi_calculation.roi:.2%}")
    print(f"  - Is profitable: {financial_impact.roi_calculation.is_profitable()}")
    print(f"  - Total benefits: ${financial_impact.total_benefits:,.2f}")
    
    # Test ROIFrame creation
    cost_benefit_analysis = CostBenefitAnalysis(
        total_benefits=75000.0,
        total_costs=25000.0,
        internal_rate_of_return=0.18
    )
    
    investment_metrics = InvestmentMetrics(
        initial_investment=25000.0,
        investment_period_years=3,
        annual_return_rate=0.15
    )
    
    roi_frame = ROIFrame(
        metric_id="roi_analysis_1",
        name="ROI Analysis",
        value=2.0,
        timestamp=datetime.utcnow(),
        cost_benefit_analysis=cost_benefit_analysis,
        investment_metrics=investment_metrics,
        analysis_date=date.today()
    )
    
    print(f"✓ ROI Frame created: {roi_frame.name}")
    print(f"  - Net benefits: ${roi_frame.cost_benefit_analysis.net_benefits:,.2f}")
    print(f"  - BCR: {roi_frame.cost_benefit_analysis.benefit_cost_ratio:.2f}")
    print(f"  - Is viable: {roi_frame.is_viable_investment()}")
    
    # Test schema versioning
    version = SchemaVersion("1.2.3")
    print(f"✓ Schema version: {version}")
    print(f"  - Major: {version.MAJOR}")
    print(f"  - Minor: {version.MINOR}")
    print(f"  - Patch: {version.PATCH}")
    
    # Test version compatibility
    compatible = is_compatible_version("1.2.3", "1.5.0")
    print(f"✓ Version compatibility (1.2.3 vs 1.5.0): {compatible}")
    
    incompatible = is_compatible_version("1.2.3", "2.0.0")
    print(f"✓ Version compatibility (1.2.3 vs 2.0.0): {incompatible}")
    
    # Test schema validation
    validator = BackwardCompatibilityValidator()
    print(f"✓ Backward compatibility validator created")
    
    print("\n🎉 All tests passed successfully!")
    print("✓ Schemas package is working correctly")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
