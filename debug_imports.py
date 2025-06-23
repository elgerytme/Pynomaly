#!/usr/bin/env python3
"""Debug import issues."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_individual_imports():
    """Test each import individually."""
    
    try:
        from pynomaly.domain.exceptions import InvalidValueError
        print("✅ InvalidValueError imported")
    except Exception as e:
        print(f"❌ InvalidValueError failed: {e}")
        return
    
    try:
        from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
        print("✅ ConfidenceInterval imported")
    except Exception as e:
        print(f"❌ ConfidenceInterval failed: {e}")
        return
    
    try:
        from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
        print("✅ AnomalyScore imported")
    except Exception as e:
        print(f"❌ AnomalyScore failed: {e}")
        return
    
    try:
        from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
        print("✅ ContaminationRate imported")
    except Exception as e:
        print(f"❌ ContaminationRate failed: {e}")
        return
    
    try:
        from pynomaly.domain.value_objects.threshold_config import ThresholdConfig
        print("✅ ThresholdConfig imported")
    except Exception as e:
        print(f"❌ ThresholdConfig failed: {e}")
        return
    
    try:
        from pynomaly.domain.value_objects import AnomalyScore, ConfidenceInterval, ContaminationRate, ThresholdConfig
        print("✅ All value objects imported via __init__")
    except Exception as e:
        print(f"❌ __init__ import failed: {e}")
        return

if __name__ == "__main__":
    test_individual_imports()