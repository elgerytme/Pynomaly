#!/usr/bin/env python3
"""Test script to validate Pydantic v2 updates."""

import warnings
from datetime import datetime

# Enable all warnings to catch deprecations
warnings.filterwarnings("error", category=DeprecationWarning)

# Import modules individually to catch errors
try:
    from pynomaly.schemas.analytics.financial_impact import CostMetrics, ROICalculation, FinancialImpactFrame
    print("✓ Financial impact imports successful")
except ImportError as e:
    print(f"✗ Financial impact imports failed: {e}")
    raise

try:
    from pynomaly.schemas.analytics.anomaly_kpis import AnomalyDetectionMetrics, AnomalyKPIFrame
    print("✓ Anomaly KPIs imports successful")
except ImportError as e:
    print(f"✗ Anomaly KPIs imports failed: {e}")
    raise

try:
    from pynomaly.schemas.analytics.base import RealTimeMetricFrame
    print("✓ Base schema imports successful")
except ImportError as e:
    print(f"✗ Base schema imports failed: {e}")
    raise

try:
    from pynomaly.schemas.analytics.system_health import SystemHealthFrame, SystemResourceMetrics, SystemPerformanceMetrics, SystemStatusMetrics
    print("✓ System health imports successful")
except ImportError as e:
    print(f"✗ System health imports failed: {e}")
    raise

try:
    from pynomaly.presentation.sdk.models import BaseSDKModel, AnomalyScore
    print("✓ SDK models imports successful")
except ImportError as e:
    print(f"✗ SDK models imports failed: {e}")
    raise

try:
    from pynomaly.presentation.api.docs.response_models import SuccessResponse, ErrorResponse, PaginationResponse
    print("✓ Response models imports successful")
except ImportError as e:
    print(f"✗ Response models imports failed: {e}")
    raise

def test_financial_impact():
    """Test financial impact models."""
    print("Testing financial impact models...")
    
    # Test CostMetrics
    cost_metrics = CostMetrics(
        total_cost=1000.0,
        cost_per_unit=10.0,
        budget=1200.0
    )
    assert cost_metrics.total_cost == 1000.0
    assert cost_metrics.is_within_budget()
    print("✓ CostMetrics working")
    
    # Test ROICalculation with model validator
    roi_calc = ROICalculation(
        investment=1000.0,
        returns=1200.0
    )
    assert roi_calc.roi == 0.2  # 20% ROI
    assert roi_calc.is_profitable()
    print("✓ ROICalculation working")

def test_anomaly_kpis():
    """Test anomaly KPI models."""
    print("\nTesting anomaly KPI models...")
    
    # Test AnomalyDetectionMetrics
    metrics = AnomalyDetectionMetrics(
        accuracy=0.95,
        precision=0.90,
        recall=0.80,
        f1_score=0.85,  # This should be close to 2 * (0.90 * 0.80) / (0.90 + 0.80) = 0.847
        false_positive_rate=0.05,
        false_negative_rate=0.10
    )
    assert metrics.accuracy == 0.95
    print("✓ AnomalyDetectionMetrics working")

def test_system_health():
    """Test system health models."""
    print("\nTesting system health models...")
    
    # Test SystemResourceMetrics
    resource_metrics = SystemResourceMetrics(
        cpu_usage_percent=50.0,
        cpu_load_average=1.5,
        cpu_cores=4,
        memory_usage_percent=60.0,
        memory_used_mb=8000.0,
        memory_total_mb=16000.0,
        memory_available_mb=8000.0,
        disk_usage_percent=40.0,
        disk_used_gb=200.0,
        disk_total_gb=500.0,
        disk_io_read_rate=10.0,
        disk_io_write_rate=5.0,
        network_bytes_sent_rate=1.0,
        network_bytes_recv_rate=2.0,
        network_packets_sent_rate=100.0,
        network_packets_recv_rate=200.0,
        process_count=150,
        thread_count=500
    )
    assert resource_metrics.cpu_usage_percent == 50.0
    print("✓ SystemResourceMetrics working")

def test_sdk_models():
    """Test SDK models."""
    print("\nTesting SDK models...")
    
    # Test AnomalyScore
    score = AnomalyScore(
        value=0.85,
        confidence=0.90,
        percentile=95.0
    )
    assert score.value == 0.85
    print("✓ AnomalyScore working")

def test_response_models():
    """Test API response models."""
    print("\nTesting response models...")
    
    # Test SuccessResponse
    success_resp = SuccessResponse[dict](
        data={"message": "success"},
        message="Operation completed"
    )
    assert success_resp.success is True
    assert success_resp.data == {"message": "success"}
    print("✓ SuccessResponse working")
    
    # Test ErrorResponse
    error_resp = ErrorResponse(
        error="Something went wrong",
        error_code="ERROR_CODE"
    )
    assert error_resp.success is False
    assert error_resp.error == "Something went wrong"
    print("✓ ErrorResponse working")

def main():
    """Run all tests."""
    print("Testing Pydantic v2 updates...")
    
    try:
        test_financial_impact()
        test_anomaly_kpis()
        test_system_health()
        test_sdk_models()
        test_response_models()
        print("\n✅ All tests passed! Pydantic v2 updates successful.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
