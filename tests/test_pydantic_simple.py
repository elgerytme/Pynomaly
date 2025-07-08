#!/usr/bin/env python3
"""Simple test for Pydantic v2 updates."""

import warnings
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Enable all warnings to catch deprecations
warnings.filterwarnings("error", category=DeprecationWarning)

def test_direct_imports():
    """Test direct imports of the modules we updated."""
    print("Testing direct imports...")
    
    # Test financial impact
    from pynomaly.schemas.analytics.financial_impact import CostMetrics, ROICalculation
    print("✓ Financial impact imports successful")
    
    # Test anomaly KPIs
    from pynomaly.schemas.analytics.anomaly_kpis import AnomalyDetectionMetrics
    print("✓ Anomaly KPIs imports successful")
    
    # Test base schemas
    from pynomaly.schemas.analytics.base import RealTimeMetricFrame
    print("✓ Base schema imports successful")
    
    # Test system health
    from pynomaly.schemas.analytics.system_health import SystemResourceMetrics
    print("✓ System health imports successful")
    
    print("\nAll imports successful!")

def test_model_creation():
    """Test creating models to verify they work."""
    print("\nTesting model creation...")
    
    from pynomaly.schemas.analytics.financial_impact import CostMetrics, ROICalculation
    
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
    
    # Test system resource metrics
    from pynomaly.schemas.analytics.system_health import SystemResourceMetrics
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
    
    print("\nAll model creation tests passed!")

def main():
    """Run all tests."""
    print("Testing Pydantic v2 updates...")
    
    try:
        test_direct_imports()
        test_model_creation()
        print("\n✅ All tests passed! Pydantic v2 updates successful.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
