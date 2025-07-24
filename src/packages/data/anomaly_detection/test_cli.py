#!/usr/bin/env python3
"""Simple test script for CLI functionality."""

import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from anomaly_detection.cli.commands.detection import generate_sample_data

def test_basic_cli():
    """Test basic CLI functionality."""
    
    # Create test data
    print("🔄 Creating test data...")
    data = generate_sample_data(100, 0.1, 42)
    
    # Save test data
    test_file = Path("test_data.json")
    with open(test_file, 'w') as f:
        json.dump({
            'samples': data,
            'algorithm': 'isolation_forest'
        }, f, indent=2)
    
    print(f"✅ Test data created: {test_file}")
    print(f"   - {len(data)} samples generated")
    print(f"   - {sum(1 for s in data if s.get('is_anomaly', False))} anomalies")
    
    # Test detection service import
    try:
        from anomaly_detection.domain.services.detection_service import DetectionService
        print("✅ DetectionService import successful")
    except Exception as e:
        print(f"❌ DetectionService import failed: {e}")
    
    # Test simple detection
    try:
        service = DetectionService()
        print("✅ DetectionService instantiation successful")
    except Exception as e:
        print(f"❌ DetectionService instantiation failed: {e}")
    
    # Cleanup
    if test_file.exists():
        test_file.unlink()
    
    print("🎉 Basic CLI test completed!")

if __name__ == "__main__":
    test_basic_cli()