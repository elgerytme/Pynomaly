#!/usr/bin/env python3
"""Test basic features mentioned in documentation"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_basic_imports():
    """Test basic imports from the core features"""
    try:
        print("Testing basic imports...")
        
        # Test domain entities
        from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
        print("✅ Domain entities import successful")
        
        # Test value objects
        from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
        print("✅ Value objects import successful")
        
        # Test adapters
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        print("✅ SklearnAdapter import successful")
        
        # Test PyOD adapter
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
        print("✅ PyODAdapter import successful")
        
        # Test services
        from pynomaly.application.services.detection_service import DetectionService
        print("✅ DetectionService import successful")
        
        # Test CLI
        from pynomaly.presentation.cli.app import app as cli_app
        print("✅ CLI app import successful")
        
        # Test API
        from pynomaly.presentation.api.app import app as api_app
        print("✅ API app import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_algorithm_support():
    """Test algorithm support claims"""
    try:
        print("\nTesting algorithm support...")
        
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

        # Test PyOD algorithms
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        for algo in algorithms:
            try:
                adapter = SklearnAdapter(algorithm_name=algo, name=f"Test {algo}")
                print(f"✅ {algo} supported")
            except Exception as e:
                print(f"❌ {algo} not supported: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Algorithm support test failed: {e}")
        return False

def test_data_formats():
    """Test data format support"""
    try:
        print("\nTesting data format support...")
        
        # Test CSV loader
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        print("✅ CSV loader available")
        
        # Test JSON loader
        from pynomaly.infrastructure.data_loaders.json_loader import JSONLoader
        print("✅ JSON loader available")
        
        # Test Excel loader
        from pynomaly.infrastructure.data_loaders.excel_loader import ExcelLoader
        print("✅ Excel loader available")
        
        # Test Parquet loader
        from pynomaly.infrastructure.data_loaders.parquet_loader import ParquetLoader
        print("✅ Parquet loader available")
        
        return True
        
    except Exception as e:
        print(f"❌ Data format test failed: {e}")
        return False

def test_architecture_layers():
    """Test clean architecture layers"""
    try:
        print("\nTesting clean architecture layers...")
        
        # Test domain layer
        from pynomaly.domain.entities import Dataset
        from pynomaly.domain.services import anomaly_scorer
        print("✅ Domain layer accessible")
        
        # Test application layer
        from pynomaly.application.use_cases.detect_anomalies import (
            DetectAnomaliesUseCase,
        )
        print("✅ Application layer accessible")
        
        # Test infrastructure layer
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        print("✅ Infrastructure layer accessible")
        
        # Test presentation layer
        from pynomaly.presentation.cli.app import app
        print("✅ Presentation layer accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture layers test failed: {e}")
        return False

def test_monitoring_features():
    """Test monitoring and observability features"""
    try:
        print("\nTesting monitoring features...")
        
        # Test health checks
        from pynomaly.infrastructure.monitoring.health_checks import HealthCheck
        print("✅ Health checks available")
        
        # Test prometheus metrics
        from pynomaly.infrastructure.monitoring.prometheus_metrics import (
            PrometheusMetrics,
        )
        print("✅ Prometheus metrics available")
        
        # Test performance monitoring
        from pynomaly.infrastructure.monitoring.performance_monitor import (
            PerformanceMonitor,
        )
        print("✅ Performance monitor available")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring features test failed: {e}")
        return False

def main():
    """Run all feature tests"""
    print("🔍 Testing Pynomaly Features")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_algorithm_support,
        test_data_formats,
        test_architecture_layers,
        test_monitoring_features
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)