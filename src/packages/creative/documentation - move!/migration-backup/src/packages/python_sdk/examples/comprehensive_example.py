#!/usr/bin/env python3
"""
Comprehensive Example: Pynomaly Python SDK

This example demonstrates all major features of the Pynomaly Python SDK
including anomaly detection, data quality assessment, model evaluation,
batch processing, and error handling.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the SDK to the path (for development)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import create_client, PynomagyClient, ClientConfig
from application.dto.detection_dto import DetectionResponseDTO


class PynomagySDKDemo:
    """
    Comprehensive demonstration of the Pynomaly Python SDK.
    
    This class shows practical usage patterns and best practices
    for using the SDK in real-world applications.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the demo with optional API key."""
        self.api_key = api_key or os.getenv('PYNOMALY_API_KEY', 'demo-key')
        self.client = None
        self.demo_data = self._generate_demo_data()
        
    def _generate_demo_data(self) -> Dict[str, Any]:
        """Generate sample datasets for demonstration."""
        np.random.seed(42)  # For reproducible results
        
        # Normal data with some outliers
        normal_data = np.random.normal(0, 1, (100, 2))
        outliers = np.random.normal(5, 1, (10, 2))
        anomaly_data = np.vstack([normal_data, outliers])
        
        # Create DataFrame with mixed data types
        mixed_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'score': np.random.uniform(0, 100, 100),
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D')
        })
        
        # Add some missing values and duplicates for quality assessment
        mixed_data.loc[5:7, 'feature1'] = None
        mixed_data.loc[20:22, 'category'] = None
        mixed_data.loc[50, :] = mixed_data.loc[49, :]  # Duplicate row
        
        # Time series data
        time_series = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='H'),
            'value': np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.1, 200)
        })
        
        # Add some anomalies to time series
        time_series.loc[50:55, 'value'] += 3  # Spike
        time_series.loc[150:155, 'value'] -= 2  # Dip
        
        return {
            'simple_anomaly': anomaly_data,
            'mixed_data': mixed_data,
            'time_series': time_series,
            'large_dataset': np.random.randn(1000, 5),
            'small_dataset': np.random.randn(20, 2)
        }
    
    async def demonstrate_basic_anomaly_detection(self):
        """Demonstrate basic anomaly detection functionality."""
        print("\n" + "="*60)
        print("📊 BASIC ANOMALY DETECTION DEMO")
        print("="*60)
        
        data = self.demo_data['simple_anomaly']
        print(f"Dataset shape: {data.shape}")
        print(f"Data sample:\n{data[:5]}")
        
        # Test different algorithms
        algorithms = [
            ("isolation_forest", {"contamination": 0.1}),
            ("lof", {"n_neighbors": 20}),
            ("one_class_svm", {"nu": 0.1})
        ]
        
        results = {}
        for algorithm, params in algorithms:
            print(f"\n🔍 Testing {algorithm}...")
            
            start_time = time.time()
            try:
                result = await self.client.detect_anomalies(
                    data=data,
                    algorithm=algorithm,
                    algorithm_params=params,
                    metadata={"demo": "basic_detection"}
                )
                
                execution_time = time.time() - start_time
                anomaly_count = sum(result.anomaly_labels)
                
                results[algorithm] = {
                    'anomaly_count': anomaly_count,
                    'execution_time': execution_time,
                    'api_execution_time': result.execution_time
                }
                
                print(f"  ✅ Success: {anomaly_count} anomalies detected")
                print(f"  ⏱️ Local time: {execution_time:.3f}s")
                print(f"  ⏱️ API time: {result.execution_time:.3f}s")
                print(f"  📊 Anomaly rate: {anomaly_count/len(data):.2%}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[algorithm] = {'error': str(e)}
        
        # Summary
        print(f"\n📋 SUMMARY - Basic Anomaly Detection:")
        for algo, result in results.items():
            if 'error' not in result:
                print(f"  {algo}: {result['anomaly_count']} anomalies ({result['execution_time']:.3f}s)")
            else:
                print(f"  {algo}: Failed - {result['error']}")
    
    async def demonstrate_batch_processing(self):
        """Demonstrate batch processing capabilities."""
        print("\n" + "="*60)
        print("🔄 BATCH PROCESSING DEMO")
        print("="*60)
        
        # Create multiple datasets of different sizes
        datasets = [
            {
                "data": self.demo_data['small_dataset'],
                "algorithm": "isolation_forest",
                "algorithm_params": {"contamination": 0.1}
            },
            {
                "data": self.demo_data['simple_anomaly'][:50],
                "algorithm": "lof",
                "algorithm_params": {"n_neighbors": 10}
            },
            {
                "data": self.demo_data['large_dataset'][:100],
                "algorithm": "one_class_svm",
                "algorithm_params": {"nu": 0.05}
            }
        ]
        
        print(f"Processing {len(datasets)} datasets...")
        
        # Progress tracking
        progress_data = []
        def progress_callback(completed: int, total: int, result: DetectionResponseDTO):
            progress_data.append({
                'completed': completed,
                'total': total,
                'algorithm': result.algorithm_used,
                'anomalies': sum(result.anomaly_labels),
                'execution_time': result.execution_time
            })
            print(f"  📈 Progress: {completed}/{total} - {result.algorithm_used} "
                  f"({sum(result.anomaly_labels)} anomalies, {result.execution_time:.3f}s)")
        
        start_time = time.time()
        try:
            results = await self.client.detect_anomalies_batch(
                datasets=datasets,
                max_concurrent=2,
                progress_callback=progress_callback
            )
            
            total_time = time.time() - start_time
            
            print(f"\n✅ Batch processing completed in {total_time:.3f}s")
            print(f"📊 Results summary:")
            
            total_anomalies = 0
            for i, result in enumerate(results):
                anomaly_count = sum(result.anomaly_labels)
                total_anomalies += anomaly_count
                print(f"  Dataset {i+1}: {anomaly_count} anomalies "
                      f"({result.algorithm_used}, {result.execution_time:.3f}s)")
            
            print(f"🎯 Total anomalies detected: {total_anomalies}")
            print(f"⚡ Average time per dataset: {total_time/len(datasets):.3f}s")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
    
    async def demonstrate_data_quality_assessment(self):
        """Demonstrate data quality assessment features."""
        print("\n" + "="*60)
        print("🔍 DATA QUALITY ASSESSMENT DEMO")
        print("="*60)
        
        data = self.demo_data['mixed_data']
        print(f"Dataset shape: {data.shape}")
        print(f"Dataset info:\n{data.info()}")
        print(f"Dataset sample:\n{data.head()}")
        
        # Test different quality metrics
        quality_metrics = [
            ["completeness", "uniqueness"],
            ["validity", "consistency"],
            ["completeness", "uniqueness", "validity", "consistency"]
        ]
        
        for metrics in quality_metrics:
            print(f"\n🔍 Testing metrics: {metrics}")
            
            try:
                start_time = time.time()
                quality_report = await self.client.assess_data_quality(
                    data=data,
                    quality_metrics=metrics,
                    use_cache=True
                )
                execution_time = time.time() - start_time
                
                print(f"  ✅ Quality assessment completed in {execution_time:.3f}s")
                print("  📊 Quality Metrics:")
                for metric, score in quality_report.items():
                    if isinstance(score, (int, float)):
                        print(f"    {metric.capitalize()}: {score:.2%}")
                    else:
                        print(f"    {metric.capitalize()}: {score}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    async def demonstrate_model_performance_evaluation(self):
        """Demonstrate model performance evaluation."""
        print("\n" + "="*60)
        print("📈 MODEL PERFORMANCE EVALUATION DEMO")
        print("="*60)
        
        # Create synthetic test data with ground truth
        np.random.seed(42)
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'actual_label': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
            'predicted_label': np.random.choice([0, 1], 100, p=[0.8, 0.2])
        })
        
        print(f"Test dataset shape: {test_data.shape}")
        print(f"Class distribution: {test_data['actual_label'].value_counts().to_dict()}")
        
        # Test different metric sets
        metric_sets = [
            ["accuracy", "precision", "recall"],
            ["f1_score", "auc_roc"],
            ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        ]
        
        model_ids = ["anomaly_detector_v1", "quality_classifier_v2", "performance_model_v3"]
        
        for model_id in model_ids:
            print(f"\n🤖 Evaluating model: {model_id}")
            
            for metrics in metric_sets:
                print(f"  📊 Metrics: {metrics}")
                
                try:
                    start_time = time.time()
                    performance = await self.client.evaluate_model_performance(
                        model_id=model_id,
                        test_data=test_data,
                        metrics=metrics,
                        use_cache=True
                    )
                    execution_time = time.time() - start_time
                    
                    print(f"    ✅ Evaluation completed in {execution_time:.3f}s")
                    for metric, value in performance.items():
                        if isinstance(value, (int, float)):
                            print(f"    {metric.upper()}: {value:.3f}")
                        else:
                            print(f"    {metric.upper()}: {value}")
                
                except Exception as e:
                    print(f"    ❌ Error: {e}")
    
    async def demonstrate_utility_functions(self):
        """Demonstrate utility functions and metadata operations."""
        print("\n" + "="*60)
        print("🔧 UTILITY FUNCTIONS DEMO")
        print("="*60)
        
        # Test API health check
        print("🏥 Testing API health check...")
        try:
            health = await self.client.health_check()
            print(f"  ✅ API Status: {health.get('status', 'unknown')}")
            print(f"  📅 Timestamp: {health.get('timestamp', 'unknown')}")
            print(f"  🔢 Version: {health.get('version', 'unknown')}")
        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
        
        # Test algorithm listing
        print("\n🧮 Testing algorithm listing...")
        try:
            algorithms = await self.client.list_available_algorithms()
            print(f"  ✅ Found {len(algorithms)} algorithms:")
            for algo in algorithms[:5]:  # Show first 5
                print(f"    - {algo.get('name', 'unknown')}: {algo.get('description', 'N/A')}")
        except Exception as e:
            print(f"  ❌ Algorithm listing failed: {e}")
        
        # Test algorithm info
        print("\n🔍 Testing algorithm information...")
        test_algorithms = ["isolation_forest", "lof", "one_class_svm"]
        
        for algo in test_algorithms:
            try:
                info = await self.client.get_algorithm_info(algo)
                print(f"  ✅ {algo}:")
                print(f"    Description: {info.get('description', 'N/A')}")
                params = info.get('parameters', {})
                if params:
                    print(f"    Parameters: {list(params.keys())}")
            except Exception as e:
                print(f"  ❌ Info for {algo} failed: {e}")
    
    async def demonstrate_caching_and_performance(self):
        """Demonstrate caching capabilities and performance optimization."""
        print("\n" + "="*60)
        print("⚡ CACHING AND PERFORMANCE DEMO")
        print("="*60)
        
        # Clear cache to start fresh
        self.client.clear_cache()
        print("🗑️ Cache cleared")
        
        # Test cache performance
        test_data = self.demo_data['simple_anomaly'][:50]
        
        print(f"📊 Cache stats before: {self.client.get_cache_stats()}")
        
        # First request (cache miss)
        print("\n🔄 First request (cache miss)...")
        start_time = time.time()
        try:
            result1 = await self.client.detect_anomalies(
                data=test_data,
                algorithm="isolation_forest",
                use_cache=True
            )
            first_request_time = time.time() - start_time
            print(f"  ⏱️ Time: {first_request_time:.3f}s")
            print(f"  🎯 Anomalies: {sum(result1.anomaly_labels)}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print(f"📊 Cache stats after first request: {self.client.get_cache_stats()}")
        
        # Second request (cache hit)
        print("\n🔄 Second request (cache hit)...")
        start_time = time.time()
        try:
            result2 = await self.client.detect_anomalies(
                data=test_data,
                algorithm="isolation_forest",
                use_cache=True
            )
            second_request_time = time.time() - start_time
            print(f"  ⏱️ Time: {second_request_time:.3f}s")
            print(f"  🎯 Anomalies: {sum(result2.anomaly_labels)}")
            
            # Performance comparison
            speedup = first_request_time / second_request_time if second_request_time > 0 else float('inf')
            print(f"  🚀 Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print(f"📊 Cache stats after second request: {self.client.get_cache_stats()}")
        
        # Test cache with different parameters (cache miss)
        print("\n🔄 Third request with different parameters (cache miss)...")
        start_time = time.time()
        try:
            result3 = await self.client.detect_anomalies(
                data=test_data,
                algorithm="isolation_forest",
                algorithm_params={"contamination": 0.2},  # Different parameters
                use_cache=True
            )
            third_request_time = time.time() - start_time
            print(f"  ⏱️ Time: {third_request_time:.3f}s")
            print(f"  🎯 Anomalies: {sum(result3.anomaly_labels)}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print(f"📊 Final cache stats: {self.client.get_cache_stats()}")
    
    async def demonstrate_error_handling(self):
        """Demonstrate error handling and retry mechanisms."""
        print("\n" + "="*60)
        print("🚨 ERROR HANDLING DEMO")
        print("="*60)
        
        # Test various error scenarios
        error_scenarios = [
            ("Invalid data type", "invalid_data_string"),
            ("Empty data", []),
            ("Malformed data", [[1, 2, 3], [4, 5]]),  # Inconsistent dimensions
            ("Invalid algorithm", "nonexistent_algorithm"),
        ]
        
        for scenario_name, test_data in error_scenarios:
            print(f"\n🧪 Testing: {scenario_name}")
            
            try:
                if scenario_name == "Invalid algorithm":
                    result = await self.client.detect_anomalies(
                        data=[[1, 2], [3, 4]],
                        algorithm=test_data,
                        use_cache=False
                    )
                else:
                    result = await self.client.detect_anomalies(
                        data=test_data,
                        algorithm="isolation_forest",
                        use_cache=False
                    )
                
                print(f"  ⚠️ Unexpected success: {result.request_id}")
                
            except Exception as e:
                print(f"  ✅ Expected error caught: {type(e).__name__}: {e}")
    
    async def run_comprehensive_demo(self):
        """Run the complete demonstration."""
        print("🚀 PYNOMALY PYTHON SDK - COMPREHENSIVE DEMO")
        print("="*60)
        print(f"🔑 API Key: {self.api_key[:10]}..." if self.api_key else "🔑 No API key")
        print(f"📊 Demo datasets: {list(self.demo_data.keys())}")
        
        # Configure client
        config = ClientConfig(
            api_key=self.api_key,
            base_url="https://api.monorepo.com/v1",
            timeout=30,
            max_retries=3,
            enable_caching=True,
            cache_ttl=300,
            debug=True
        )
        
        self.client = PynomagyClient(config)
        
        try:
            async with self.client:
                # Run all demonstrations
                await self.demonstrate_basic_anomaly_detection()
                await self.demonstrate_batch_processing()
                await self.demonstrate_data_quality_assessment()
                await self.demonstrate_model_performance_evaluation()
                await self.demonstrate_utility_functions()
                await self.demonstrate_caching_and_performance()
                await self.demonstrate_error_handling()
                
                print("\n" + "="*60)
                print("🎉 DEMO COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f"📊 Final cache stats: {self.client.get_cache_stats()}")
                
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.exception("Demo failed with exception")


async def main():
    """Main entry point for the demonstration."""
    # Get API key from environment or use demo key
    api_key = os.getenv('PYNOMALY_API_KEY', 'demo-key-for-testing')
    
    # Create and run demo
    demo = PynomagySDKDemo(api_key=api_key)
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())