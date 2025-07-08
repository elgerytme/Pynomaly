#!/usr/bin/env python3
"""Test schema modules directly."""

import warnings
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Enable all warnings to catch deprecations
warnings.filterwarnings("error", category=DeprecationWarning)

# Test the specific modules we updated
print("Testing specific Pydantic schema modules...")

try:
    # Test by importing the module with all its pydantic code
    import pynomaly.schemas.analytics.financial_impact
    print("✓ Financial impact module imported successfully")
    
    # Test instantiation
    from pynomaly.schemas.analytics.financial_impact import CostMetrics, ROICalculation
    cost_metrics = CostMetrics(total_cost=1000.0, cost_per_unit=10.0, budget=1200.0)
    roi_calc = ROICalculation(investment=1000.0, returns=1200.0)
    print("✓ Financial impact models created successfully")
    
except Exception as e:
    print(f"❌ Financial impact test failed: {e}")
    raise

try:
    import pynomaly.schemas.analytics.anomaly_kpis
    print("✓ Anomaly KPIs module imported successfully")
    
    from pynomaly.schemas.analytics.anomaly_kpis import AnomalyDetectionMetrics
    metrics = AnomalyDetectionMetrics(
        accuracy=0.95,
        precision=0.90,
        recall=0.80,
        f1_score=0.85,
        false_positive_rate=0.05,
        false_negative_rate=0.10
    )
    print("✓ Anomaly KPIs models created successfully")
    
except Exception as e:
    print(f"❌ Anomaly KPIs test failed: {e}")
    raise

try:
    import pynomaly.schemas.analytics.base
    print("✓ Base schema module imported successfully")
    
except Exception as e:
    print(f"❌ Base schema test failed: {e}")
    raise

try:
    import pynomaly.schemas.analytics.system_health
    print("✓ System health module imported successfully")
    
except Exception as e:
    print(f"❌ System health test failed: {e}")
    raise

print("\n✅ All schema modules passed Pydantic v2 validation!")
