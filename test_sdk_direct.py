#!/usr/bin/env python3
"""Direct test of SDK models without async dependencies."""

import sys
import os

def test_direct_imports():
    """Test direct imports of SDK components."""
    try:
        print(f"🐍 Python version: {sys.version}")
        print(f"📁 Current working directory: {os.getcwd()}")
        
        # Test basic package import
        import pynomaly
        print("✅ Package import successful")
        
        # Test core components
        from pynomaly.domain.entities.anomaly import Anomaly
        print("✅ Domain entities import successful")
        
        from pynomaly.application.services.detection_service import DetectionService
        print("✅ Application services import successful")
        
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        print("✅ Infrastructure adapters import successful")
        
        # Test SDK models directly (bypassing __init__.py)
        from pynomaly.presentation.sdk.models import (
            DetectionConfig, 
            DatasetConfig,
            AnomalyScore,
            BaseSDKModel
        )
        print("✅ SDK models direct import successful")
        
        # Test model creation
        config = DetectionConfig(
            algorithm="isolation_forest",
            contamination=0.1
        )
        print(f"✅ DetectionConfig created: {config.algorithm}")
        
        dataset_config = DatasetConfig(
            format="csv",
            path="/tmp/test.csv"
        )
        print(f"✅ DatasetConfig created: {dataset_config.format}")
        
        score = AnomalyScore(value=0.8, confidence=0.9)
        print(f"✅ AnomalyScore created: {score.value}")
        
        # Test sync client if available
        try:
            from pynomaly.presentation.sdk.client import PynomaliClient
            print("✅ Sync client import successful")
            
            # Test client initialization (may fail due to missing deps)
            try:
                client = PynomaliClient(base_url="http://localhost:8000")
                print("✅ Sync client initialization successful")
            except Exception as e:
                print(f"⚠️  Sync client initialization failed: {e}")
        except Exception as e:
            print(f"⚠️  Sync client import failed: {e}")
        
        print("🎉 Direct SDK test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Direct imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_imports()
    sys.exit(0 if success else 1)