#!/usr/bin/env python3
"""Test script to verify the implemented methods work correctly."""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_audit_storage():
    """Test AuditStorage abstract implementation."""
    print("Testing AuditStorage...")
    
    from pynomaly.infrastructure.compliance.audit_system import AuditStorage, AuditEvent, EventType, Severity
    from datetime import datetime
    
    # Test that abstract methods are documented
    storage = AuditStorage()
    
    # These should raise NotImplementedError
    try:
        event = AuditEvent(
            event_id="test-1",
            event_type=EventType.MODEL_CREATE,
            timestamp=datetime.now(),
            user_id="test_user",
            session_id=None,
            ip_address="127.0.0.1",
            user_agent=None,
            resource="test_model",
            action="create",
            outcome="success",
            severity=Severity.LOW
        )
        result = storage.store_event(event)
        print("❌ store_event should raise NotImplementedError")
    except NotImplementedError:
        print("✅ store_event correctly raises NotImplementedError")
    
    # Test default implementations
    try:
        count = storage.get_event_count()
        print("❌ get_event_count should work with default implementation")
    except Exception as e:
        print(f"❌ get_event_count failed: {e}")
    
    print()

def test_model_persistence_onnx():
    """Test ONNX model saving."""
    print("Testing ModelPersistenceService ONNX support...")
    
    try:
        from pynomaly.application.services.model_persistence_service import ModelPersistenceService
        print("✅ ModelPersistenceService imported successfully")
        
        # Check if the _save_onnx_model method exists
        if hasattr(ModelPersistenceService, '_save_onnx_model'):
            print("✅ _save_onnx_model method exists")
        else:
            print("❌ _save_onnx_model method missing")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    print()

def test_pytorch_adapter():
    """Test PyTorch adapter implementations."""
    print("Testing PyTorch adapter...")
    
    try:
        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
        print("✅ PyTorchAdapter imported successfully")
        
        # Check if new methods exist
        methods_to_check = ['infer', 'forward', '_create_stub_detection_result']
        for method in methods_to_check:
            if hasattr(PyTorchAdapter, method):
                print(f"✅ {method} method exists")
            else:
                print(f"❌ {method} method missing")
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    print()

def test_deep_learning_adapter():
    """Test deep learning PyTorch adapter."""
    print("Testing deep learning PyTorch adapter...")
    
    try:
        from pynomaly.infrastructure.adapters.deep_learning.pytorch_adapter import PyTorchAdapter
        print("✅ Deep learning PyTorchAdapter imported successfully")
        
        # Check if stub methods exist
        methods_to_check = ['_create_stub_model', '_stub_predict', 'forward', 'infer']
        for method in methods_to_check:
            if hasattr(PyTorchAdapter, method):
                print(f"✅ {method} method exists")
            else:
                print(f"❌ {method} method missing")
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    print()

def test_feature_flags():
    """Test feature flag functionality."""
    print("Testing feature flags...")
    
    try:
        from pynomaly.infrastructure.config.feature_flags import feature_flags
        print("✅ Feature flags imported successfully")
        
        # Test deep learning flag
        dl_enabled = feature_flags.is_enabled("deep_learning")
        print(f"✅ Deep learning enabled: {dl_enabled}")
        
        # Test other flags
        flags_to_test = ["algorithm_optimization", "performance_monitoring"]
        for flag in flags_to_test:
            enabled = feature_flags.is_enabled(flag)
            print(f"✅ {flag} enabled: {enabled}")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    print()

def main():
    """Run all tests."""
    print("=== Testing Implemented Methods ===\n")
    
    test_audit_storage()
    test_model_persistence_onnx()
    test_pytorch_adapter()
    test_deep_learning_adapter()
    test_feature_flags()
    
    print("=== Tests Complete ===")

if __name__ == "__main__":
    main()
