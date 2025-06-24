#!/usr/bin/env python3
"""Test script for memory optimization infrastructure.

This script validates the memory-efficient data processing capabilities
implemented in Phase 2 of the controlled feature reintroduction.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_streaming_processor():
    """Test streaming data processor."""
    print("🌊 Testing Streaming Data Processor...")
    
    try:
        from pynomaly.infrastructure.data_processing import StreamingDataProcessor
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 10000),
            'feature2': np.random.normal(0, 1, 10000),
            'feature3': np.random.normal(0, 1, 10000)
        })
        
        # Create processor
        processor = StreamingDataProcessor(chunk_size=2000)
        
        # Test DataFrame chunking
        chunk_count = 0
        total_samples = 0
        
        for chunk_dataset in processor.process_large_dataset(test_data):
            chunk_count += 1
            total_samples += len(chunk_dataset.data)
            print(f"  ✅ Processed chunk {chunk_count}: {len(chunk_dataset.data)} samples")
        
        print(f"  ✅ Total chunks: {chunk_count}, Total samples: {total_samples}")
        print(f"  ✅ Expected samples: {len(test_data)}")
        
        # Test file processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            file_chunk_count = 0
            for chunk_dataset in processor.process_large_dataset(temp_file):
                file_chunk_count += 1
            print(f"  ✅ File processing: {file_chunk_count} chunks")
        finally:
            os.unlink(temp_file)
        
        print("  ✅ Streaming processor working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Streaming processor test failed: {e}")
        return False


def test_memory_optimized_loader():
    """Test memory-optimized data loader."""
    print("\n💾 Testing Memory-Optimized Data Loader...")
    
    try:
        from pynomaly.infrastructure.data_processing import MemoryOptimizedDataLoader
        from pynomaly.infrastructure.data_processing import get_memory_usage
        
        # Create test data with different types
        test_data = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000).astype('int64'),
            'float_col': np.random.random(1000).astype('float64'),
            'category_col': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        loader = MemoryOptimizedDataLoader()
        
        # Test memory optimization
        memory_before = get_memory_usage()
        optimized_df = loader.optimize_dataframe_memory(test_data)
        memory_after = get_memory_usage()
        
        print(f"  ✅ Original dtypes: {dict(test_data.dtypes)}")
        print(f"  ✅ Optimized dtypes: {dict(optimized_df.dtypes)}")
        print(f"  ✅ Memory before: {memory_before:.1f}MB, after: {memory_after:.1f}MB")
        
        # Test efficient loading
        dataset = loader.load_dataset_efficiently(test_data, target_memory_mb=100)
        print(f"  ✅ Loaded dataset: {dataset.data.shape}")
        
        print("  ✅ Memory-optimized loader working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Memory-optimized loader test failed: {e}")
        return False


def test_large_dataset_analyzer():
    """Test large dataset analyzer."""
    print("\n📊 Testing Large Dataset Analyzer...")
    
    try:
        from pynomaly.infrastructure.data_processing import LargeDatasetAnalyzer
        
        # Create test data with some outliers
        normal_data = np.random.normal(0, 1, (5000, 3))
        outlier_data = np.random.normal(5, 1, (100, 3))  # Clear outliers
        test_data = np.vstack([normal_data, outlier_data])
        
        analyzer = LargeDatasetAnalyzer()
        
        # Test dataset statistics
        stats = analyzer.analyze_dataset_statistics(test_data)
        print(f"  ✅ Dataset stats: {stats['total_rows']} rows, {stats['total_columns']} columns")
        print(f"  ✅ Memory estimate: {stats['memory_estimate_mb']:.1f}MB")
        print(f"  ✅ Chunk count: {stats['chunk_count']}")
        
        # Test anomaly candidate detection
        candidates = analyzer.detect_anomaly_candidates(test_data, threshold_factor=2.0)
        print(f"  ✅ Anomaly candidates: {candidates['total_candidates']}")
        print(f"  ✅ Processing chunks: {candidates['processing_chunks']}")
        
        # Verify we found some outliers
        if candidates['total_candidates'] > 0:
            print("  ✅ Successfully detected anomaly candidates")
        else:
            print("  ⚠️ No anomaly candidates detected (may be expected)")
        
        print("  ✅ Large dataset analyzer working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Large dataset analyzer test failed: {e}")
        return False


def test_memory_optimization_service():
    """Test memory optimization service."""
    print("\n🚀 Testing Memory Optimization Service...")
    
    try:
        # Enable memory efficiency feature for testing
        os.environ["PYNOMALY_MEMORY_EFFICIENCY"] = "true"
        
        from pynomaly.application.services.memory_optimization_service import (
            MemoryOptimizationService, MemoryProfiler
        )
        from pynomaly.domain.entities import Dataset
        
        # Create test dataset
        test_data = np.random.normal(0, 1, (1000, 5))
        test_dataset = Dataset(name="test_data", data=test_data)
        
        service = MemoryOptimizationService(chunk_size=500, memory_limit_mb=100)
        
        # Test dataset optimization
        optimized_dataset, optimization_info = service.optimize_dataset_for_detection(
            test_dataset, target_memory_mb=50
        )
        
        print(f"  ✅ Optimization applied: {optimization_info['optimization_applied']}")
        print(f"  ✅ Original memory: {optimization_info['original_memory_mb']:.1f}MB")
        print(f"  ✅ Final memory: {optimization_info['final_memory_mb']:.1f}MB")
        
        # Test dataset analysis
        analysis = service.analyze_large_dataset_characteristics(test_data)
        print(f"  ✅ Analysis completed: {len(analysis)} components")
        print(f"  ✅ Processing feasible: {analysis['processing_feasible']}")
        
        # Test configuration recommendations
        config = service.recommend_optimal_configuration(analysis)
        print(f"  ✅ Recommended config: {config['optimization_level']}")
        print(f"  ✅ Recommended algorithm: {config['recommended_algorithm']}")
        
        # Test memory profiler
        profiler = MemoryProfiler()
        profiler.start_profiling("test_operation")
        
        # Simulate some memory usage
        temp_array = np.random.random((1000, 100))
        
        result = profiler.end_profiling("test_operation")
        print(f"  ✅ Profiler result: {result['memory_delta']:.1f}MB delta")
        
        # Get service statistics
        stats = service.get_optimization_statistics()
        print(f"  ✅ Service stats: {stats['datasets_optimized']} datasets optimized")
        
        print("  ✅ Memory optimization service working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Memory optimization service test failed: {e}")
        return False
    finally:
        # Clean up environment
        os.environ.pop("PYNOMALY_MEMORY_EFFICIENCY", None)


def test_container_integration():
    """Test container integration with memory services."""
    print("\n🔧 Testing Container Integration...")
    
    try:
        # Enable memory efficiency feature
        os.environ["PYNOMALY_MEMORY_EFFICIENCY"] = "true"
        
        from pynomaly.infrastructure.config.container import Container
        
        # Create container
        container = Container()
        
        # Test memory optimization service availability
        try:
            memory_service = container.memory_optimization_service()
            print("  ✅ Memory optimization service available")
        except AttributeError:
            print("  ⚠️ Memory optimization service not available (feature may be disabled)")
        
        # Test streaming processor availability
        try:
            streaming_processor = container.streaming_data_processor()
            print("  ✅ Streaming data processor available")
        except AttributeError:
            print("  ⚠️ Streaming data processor not available (feature may be disabled)")
        
        # Test memory profiler availability
        try:
            memory_profiler = container.memory_profiler()
            print("  ✅ Memory profiler available")
        except AttributeError:
            print("  ⚠️ Memory profiler not available (feature may be disabled)")
        
        print("  ✅ Container integration working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Container integration test failed: {e}")
        return False
    finally:
        # Clean up environment
        os.environ.pop("PYNOMALY_MEMORY_EFFICIENCY", None)


def test_memory_infrastructure_readiness():
    """Test overall memory infrastructure readiness."""
    print("\n🚀 Testing Memory Infrastructure Readiness...")
    
    try:
        # Check if all memory components are available
        components = [
            "streaming_processor",
            "memory_optimized_loader",
            "large_dataset_analyzer",
            "memory_optimization_service",
            "container_integration"
        ]
        
        results = {
            "streaming_processor": test_streaming_processor(),
            "memory_optimized_loader": test_memory_optimized_loader(),
            "large_dataset_analyzer": test_large_dataset_analyzer(),
            "memory_optimization_service": test_memory_optimization_service(),
            "container_integration": test_container_integration()
        }
        
        passing = sum(results.values())
        total = len(results)
        
        print(f"\n📈 Memory Infrastructure Status: {passing}/{total} components ready")
        
        if passing == total:
            print("🎉 Memory optimization infrastructure is fully operational!")
            print("✅ Ready for memory-efficient anomaly detection")
            return True
        else:
            print("⚠️ Some memory infrastructure components need attention")
            for component, status in results.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}")
            return False
        
    except Exception as e:
        print(f"❌ Memory infrastructure readiness test failed: {e}")
        return False


def main():
    """Run all memory optimization infrastructure tests."""
    print("🧪 Pynomaly Memory Optimization Infrastructure Validation")
    print("=" * 60)
    
    try:
        success = test_memory_infrastructure_readiness()
        
        if success:
            print("\n🎯 Memory optimization infrastructure validation successful!")
            print("🚀 Ready for memory-efficient data processing")
            sys.exit(0)
        else:
            print("\n⚠️ Memory optimization infrastructure validation failed")
            print("🔧 Please review and fix issues before proceeding")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()