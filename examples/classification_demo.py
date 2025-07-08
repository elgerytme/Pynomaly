#!/usr/bin/env python3
"""
Anomaly Classification Demo

This script demonstrates the complete Pynomaly anomaly classification workflow,
showcasing the two-dimensional taxonomy system for categorizing anomalies by
severity and type.

Usage:
    python examples/classification_demo.py
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pynomaly.application.services.anomaly_classification_service import AnomalyClassificationService
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.services.anomaly_classifiers import DefaultSeverityClassifier


def main():
    """Main demonstration function."""
    print("🔍 Pynomaly Anomaly Classification Demo")
    print("=" * 50)
    
    # Initialize the classification service
    service = AnomalyClassificationService()
    
    # Demo 1: Basic Classification
    print("\n📊 Demo 1: Basic Classification")
    print("-" * 30)
    
    # Create sample anomalies with different characteristics
    anomalies = [
        Anomaly(
            id="critical-fraud",
            score=AnomalyScore(0.95),
            data_point={"amount": 50000, "location": "foreign"},
            metadata={"source": "financial_transaction"}
        ),
        Anomaly(
            id="sensor-failure",
            score=AnomalyScore(0.85),
            data_point={"temperature": 150.0},
            metadata={"sensor_id": "temp_01"}
        ),
        Anomaly(
            id="network-anomaly",
            score=AnomalyScore(0.65),
            data_point={"bytes_sent": 1000, "bytes_received": 800, "duration": 30},
            metadata={"temporal_context": True}
        ),
        Anomaly(
            id="minor-deviation",
            score=AnomalyScore(0.35),
            data_point={"cpu_usage": 0.75},
            metadata={"threshold": 0.8}
        )
    ]
    
    # Classify all anomalies
    for anomaly in anomalies:
        service.classify(anomaly)
        
        severity = anomaly.metadata.get('severity')
        type_category = anomaly.metadata.get('type')
        
        print(f"🔸 {anomaly.id}:")
        print(f"   Score: {anomaly.score.value:.2f}")
        print(f"   Severity: {severity.upper()}")
        print(f"   Type: {type_category.title()}")
        print(f"   Features: {len(anomaly.data_point)}")
        print()
    
    # Demo 2: Custom Severity Thresholds
    print("\n🎯 Demo 2: Custom Severity Thresholds")
    print("-" * 40)
    
    # Financial services with stricter thresholds
    financial_classifier = DefaultSeverityClassifier({
        'critical': 0.95,  # Only highest scores are critical
        'high': 0.85,      # Tighter high threshold
        'medium': 0.7,     # Raised medium threshold
        'low': 0.5         # Higher low threshold
    })
    
    financial_service = AnomalyClassificationService(
        severity_classifier=financial_classifier
    )
    
    # Test transaction with score 0.82
    transaction = Anomaly(
        id="suspicious-transaction",
        score=AnomalyScore(0.82),
        data_point={"amount": 25000, "time": "2am", "location": "ATM"}
    )
    
    # Compare default vs custom classification
    service.classify(transaction)
    default_severity = transaction.metadata.get('severity')
    
    financial_service.classify(transaction)
    custom_severity = transaction.metadata.get('severity')
    
    print(f"💳 Transaction Analysis (Score: {transaction.score.value:.2f})")
    print(f"   Default severity: {default_severity.upper()}")
    print(f"   Financial severity: {custom_severity.upper()}")
    print(f"   Impact: {'⚠️ Requires immediate review' if custom_severity == 'high' else '✅ Standard monitoring'}")
    
    # Demo 3: Batch Processing
    print("\n⚡ Demo 3: Batch Processing Performance")
    print("-" * 40)
    
    import time
    import random
    
    # Generate batch of anomalies
    batch_size = 100
    batch_anomalies = []
    
    for i in range(batch_size):
        anomaly = Anomaly(
            id=f"batch-{i:03d}",
            score=AnomalyScore(random.uniform(0.0, 1.0)),
            data_point={
                f"feature_{j}": random.uniform(-1, 1) 
                for j in range(random.randint(1, 5))
            }
        )
        batch_anomalies.append(anomaly)
    
    # Enable batch processing
    batch_service = AnomalyClassificationService()
    batch_service.use_batch_processing_classifiers()
    
    # Measure processing time
    start_time = time.time()
    
    for anomaly in batch_anomalies:
        batch_service.classify(anomaly)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"📊 Processed {batch_size} anomalies in {processing_time:.3f} seconds")
    print(f"⏱️ Average time per anomaly: {processing_time/batch_size*1000:.2f} ms")
    
    # Analyze results
    severity_counts = {}
    for anomaly in batch_anomalies:
        severity = anomaly.metadata.get('severity')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print("\n📈 Severity Distribution:")
    for severity in ['critical', 'high', 'medium', 'low']:
        count = severity_counts.get(severity, 0)
        percentage = (count / batch_size) * 100
        bar = "█" * int(percentage // 5)  # Simple bar chart
        print(f"   {severity.capitalize():8}: {count:3d} ({percentage:5.1f}%) {bar}")
    
    # Clean up
    batch_service.clear_classifier_cache()
    
    print("\n✅ Classification demo completed successfully!")
    print("\n📚 Next Steps:")
    print("   • Explore the comprehensive notebook: examples/notebooks/classification_comprehensive_example.ipynb")
    print("   • Read the taxonomy documentation: docs/reference/taxonomy/anomaly-classification-taxonomy.md")
    print("   • Check the ADR: docs/developer-guides/architecture/adr/ADR-010-anomaly-classification-taxonomy.md")


if __name__ == "__main__":
    main()
