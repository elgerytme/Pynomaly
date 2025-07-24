"""Simple test to verify basic functionality works."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Source directory exists: {src_dir.exists()}")

try:
    print("\n1. Testing basic import...")
    from anomaly_detection.domain.services.detection_service import DetectionService
    print("✅ DetectionService imported successfully")
    
    print("\n2. Testing detection service creation...")
    service = DetectionService()
    print("✅ DetectionService created successfully")
    
    print("\n3. Testing simple detection...")
    # Create simple test data
    np.random.seed(42)
    test_data = np.random.rand(50, 3)
    
    result = service.detect_anomalies(
        data=test_data,
        algorithm="iforest",
        contamination=0.1
    )
    
    print(f"✅ Detection completed - found {result.anomaly_count} anomalies in {result.total_samples} samples")
    
    print("\n🎉 Basic functionality is working!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()