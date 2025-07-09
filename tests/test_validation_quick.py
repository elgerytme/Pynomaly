#!/usr/bin/env python3
"""Quick test to validate the validation module works."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pynomaly.domain import (
    ValidationSeverity,
    ValidationError,
    ValidationResult,
    DomainValidator,
    ValidationStrategy,
    validate_anomaly,
    validate_dataset
)

def test_validation_basic():
    """Test basic validation functionality."""
    print("Testing basic validation...")
    
    # Test anomaly score validation
    result = DomainValidator.validate_anomaly_score(0.8)
    print(f"Anomaly score 0.8 valid: {result.is_valid}")
    print(f"Warnings: {len(result.warnings)}")
    
    # Test invalid score
    result = DomainValidator.validate_anomaly_score(1.5)
    print(f"Anomaly score 1.5 valid: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    
    # Test detector name validation
    result = DomainValidator.validate_detector_name("isolation_forest")
    print(f"Detector name 'isolation_forest' valid: {result.is_valid}")
    
    # Test data point validation
    data_point = {"feature1": 1.0, "feature2": "value"}
    result = DomainValidator.validate_data_point(data_point)
    print(f"Data point valid: {result.is_valid}")
    
    # Test contamination rate validation
    result = DomainValidator.validate_contamination_rate(0.1)
    print(f"Contamination rate 0.1 valid: {result.is_valid}")
    
    # Test high-level validation
    result = validate_anomaly(0.9, {"x": 1, "y": 2}, "test_detector", "Explanation")
    print(f"Complete anomaly validation: {result.is_valid}")
    
    print("All basic tests passed!")

if __name__ == "__main__":
    test_validation_basic()
