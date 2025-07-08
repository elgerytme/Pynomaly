#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test classifiers directly
from pynomaly.domain.services.threshold_severity_classifier import ThresholdSeverityClassifier
from pynomaly.domain.services.statistical_severity_classifier import StatisticalSeverityClassifier  
from pynomaly.domain.services.rule_based_type_classifier import RuleBasedTypeClassifier

def test_threshold_classifier():
    print("Testing ThresholdSeverityClassifier...")
    classifier = ThresholdSeverityClassifier()
    
    # Test basic classification
    assert classifier.classify_single(0.2) == "low"
    assert classifier.classify_single(0.5) == "medium" 
    assert classifier.classify_single(0.8) == "high"
    assert classifier.classify_single(0.95) == "critical"
    
    print("✓ ThresholdSeverityClassifier tests passed")

def test_statistical_classifier():
    print("Testing StatisticalSeverityClassifier...")
    classifier = StatisticalSeverityClassifier()
    
    # Test batch classification
    scores = [0.5, 1.5, 2.5, 3.5]
    results = classifier.classify_batch(scores)
    expected = ["low", "medium", "high", "critical"]
    assert results == expected, f"Expected {expected}, got {results}"
    
    print("✓ StatisticalSeverityClassifier tests passed")

def test_rule_based_classifier():
    print("Testing RuleBasedTypeClassifier...")
    classifier = RuleBasedTypeClassifier()
    
    # Test basic classification
    result = classifier.classify_single(0.8)
    assert result in classifier.rules.keys() or result == classifier.default_type
    
    # Test with context
    context = {"previous_scores": [0.1, 0.1, 0.1, 0.1, 0.1]}
    result = classifier.classify_single(0.9, context=context)
    # Should classify as spike due to sudden increase
    assert result == "spike", f"Expected spike, got {result}"
    
    print("✓ RuleBasedTypeClassifier tests passed")

if __name__ == "__main__":
    try:
        test_threshold_classifier()
        test_statistical_classifier()
        test_rule_based_classifier()
        print("\n🎉 All classifier tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
