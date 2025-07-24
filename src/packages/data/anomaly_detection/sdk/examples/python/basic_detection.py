#!/usr/bin/env python3
"""
Basic Anomaly Detection Example using Python SDK

This example demonstrates the basic usage of the Python SDK for anomaly detection.
"""

import asyncio
import numpy as np
from anomaly_detection_sdk import AnomalyDetectionClient, AsyncAnomalyDetectionClient, AlgorithmType


def generate_sample_data():
    """Generate sample data with some anomalies."""
    np.random.seed(42)
    
    # Normal data points
    normal_data = np.random.normal(0, 1, (100, 2))
    
    # Anomalous data points
    anomalous_data = np.random.normal(5, 1, (10, 2))
    
    # Combine data
    data = np.vstack([normal_data, anomalous_data])
    
    # Shuffle the data
    indices = np.random.permutation(len(data))
    return data[indices].tolist()


def sync_example():
    """Synchronous client example."""
    print("=== Synchronous Client Example ===")
    
    # Initialize the client
    client = AnomalyDetectionClient(
        base_url="http://localhost:8000",
        timeout=30.0
    )
    
    try:
        # Generate sample data
        data = generate_sample_data()
        print(f"Generated {len(data)} data points")
        
        # Detect anomalies using Isolation Forest
        print("\nDetecting anomalies with Isolation Forest...")
        result = client.detect_anomalies(
            data=data,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            parameters={'contamination': 0.1},
            return_explanations=False
        )
        
        print(f"Detection completed in {result.execution_time:.3f} seconds")
        print(f"Found {result.anomaly_count} anomalies out of {result.total_points} points")
        
        # Print details of detected anomalies
        for i, anomaly in enumerate(result.anomalies[:5]):  # Show first 5
            print(f"  Anomaly {i+1}: Index={anomaly.index}, Score={anomaly.score:.4f}")
            
        if len(result.anomalies) > 5:
            print(f"  ... and {len(result.anomalies) - 5} more")
        
        # Try different algorithms
        algorithms = [
            AlgorithmType.LOCAL_OUTLIER_FACTOR,
            AlgorithmType.ONE_CLASS_SVM,
            AlgorithmType.ENSEMBLE
        ]
        
        print("\nComparing different algorithms:")
        for algorithm in algorithms:
            try:
                result = client.detect_anomalies(
                    data=data[:50],  # Use smaller subset for faster comparison
                    algorithm=algorithm
                )
                print(f"  {algorithm}: {result.anomaly_count} anomalies "
                      f"({result.execution_time:.3f}s)")
            except Exception as e:
                print(f"  {algorithm}: Error - {e}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()


async def async_example():
    """Asynchronous client example."""
    print("\n=== Asynchronous Client Example ===")
    
    # Initialize the async client
    async with AsyncAnomalyDetectionClient(
        base_url="http://localhost:8000",
        timeout=30.0
    ) as client:
        try:
            # Generate sample data
            data = generate_sample_data()
            print(f"Generated {len(data)} data points")
            
            # Detect anomalies asynchronously
            print("\nDetecting anomalies asynchronously...")
            result = await client.detect_anomalies(
                data=data,
                algorithm=AlgorithmType.ISOLATION_FOREST,
                parameters={'contamination': 0.1}
            )
            
            print(f"Async detection completed in {result.execution_time:.3f} seconds")
            print(f"Found {result.anomaly_count} anomalies")
            
            # Run multiple detections concurrently
            print("\nRunning multiple detections concurrently...")
            tasks = []
            
            algorithms = [
                AlgorithmType.ISOLATION_FOREST,
                AlgorithmType.LOCAL_OUTLIER_FACTOR,
                AlgorithmType.ONE_CLASS_SVM
            ]
            
            for algorithm in algorithms:
                task = client.detect_anomalies(
                    data=data[:50],
                    algorithm=algorithm
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for algorithm, result in zip(algorithms, results):
                if isinstance(result, Exception):
                    print(f"  {algorithm}: Error - {result}")
                else:
                    print(f"  {algorithm}: {result.anomaly_count} anomalies "
                          f"({result.execution_time:.3f}s)")
        
        except Exception as e:
            print(f"Error: {e}")


def health_check_example():
    """Health check example."""
    print("\n=== Health Check Example ===")
    
    client = AnomalyDetectionClient(base_url="http://localhost:8000")
    
    try:
        # Check service health
        health = client.get_health()
        
        print(f"Service Status: {health.status}")
        print(f"Version: {health.version}")
        print(f"Uptime: {health.uptime:.1f} seconds")
        
        if health.components:
            print("Components:")
            for component, status in health.components.items():
                print(f"  {component}: {status}")
        
        # Get metrics
        metrics = client.get_metrics()
        print(f"\nService Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Health check failed: {e}")
    finally:
        client.close()


def batch_processing_example():
    """Batch processing example."""
    print("\n=== Batch Processing Example ===")
    
    client = AnomalyDetectionClient(base_url="http://localhost:8000")
    
    try:
        # Generate larger dataset for batch processing
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (500, 3))
        anomalous_data = np.random.normal(4, 1, (50, 3))
        data = np.vstack([normal_data, anomalous_data])
        np.random.shuffle(data)
        
        from anomaly_detection_sdk.models import BatchProcessingRequest
        
        # Create batch request
        batch_request = BatchProcessingRequest(
            data=data.tolist(),
            algorithm=AlgorithmType.ISOLATION_FOREST,
            parameters={'contamination': 0.1},
            return_explanations=True
        )
        
        print(f"Processing batch of {len(data)} points...")
        result = client.batch_detect(batch_request)
        
        print(f"Batch processing completed in {result.execution_time:.3f} seconds")
        print(f"Found {result.anomaly_count} anomalies")
        
        # Analyze results
        scores = [anomaly.score for anomaly in result.anomalies]
        if scores:
            print(f"Anomaly scores: min={min(scores):.4f}, "
                  f"max={max(scores):.4f}, "
                  f"avg={sum(scores)/len(scores):.4f}")
    
    except Exception as e:
        print(f"Batch processing error: {e}")
    finally:
        client.close()


def main():
    """Run all examples."""
    print("Anomaly Detection Python SDK Examples")
    print("=" * 50)
    
    # Run synchronous example
    sync_example()
    
    # Run asynchronous example
    asyncio.run(async_example())
    
    # Run health check example
    health_check_example()
    
    # Run batch processing example
    batch_processing_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()