"""
Pynomaly SDK Usage Examples

Comprehensive examples demonstrating how to use the Pynomaly SDK
for various anomaly detection scenarios and workflows.
"""

import asyncio
import numpy as np
from typing import List
from pathlib import Path

from . import PynomaliClient, AsyncPynomaliClient
from .models import AlgorithmType, DataFormat
from .config import SDKConfig


def basic_synchronous_example():
    """Basic synchronous SDK usage example."""
    
    print("=== Basic Synchronous Example ===")
    
    # Initialize client
    client = PynomaliClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    )
    
    try:
        # Check connection
        if not client.validate_connection():
            print("Failed to connect to Pynomaly API")
            return
        
        # Create sample data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (1000, 5))
        anomalous_data = np.random.normal(3, 1, (50, 5))
        all_data = np.vstack([normal_data, anomalous_data])
        
        print(f"Generated dataset with {len(all_data)} samples and {all_data.shape[1]} features")
        
        # Create dataset
        dataset = client.create_dataset(
            data_source=all_data.tolist(),
            name="Example Dataset",
            description="Generated data for demonstration",
            feature_names=[f"feature_{i}" for i in range(5)]
        )
        print(f"Created dataset: {dataset.id}")
        
        # Train detector
        detector = client.train_detector(
            dataset_id=dataset.id,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            name="Example Detector",
            parameters={"n_estimators": 100, "contamination": 0.05},
            wait_for_completion=True
        )
        print(f"Trained detector: {detector.id}")
        
        # Detect anomalies on new data
        test_data = np.random.normal(0, 1, (100, 5))
        test_data[0] = [5, 5, 5, 5, 5]  # Add obvious anomaly
        
        result = client.detect_anomalies(
            detector_id=detector.id,
            data=test_data.tolist(),
            return_scores=True,
            return_explanations=True
        )
        
        print(f"Detection results: {result.num_anomalies}/{result.num_samples} anomalies")
        print(f"Anomaly rate: {result.anomaly_rate:.2%}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        # List highest scoring anomalies
        sorted_scores = sorted(
            enumerate(result.anomaly_scores),
            key=lambda x: x[1].value,
            reverse=True
        )
        
        print("\nTop 5 highest anomaly scores:")
        for i, (idx, score) in enumerate(sorted_scores[:5]):
            print(f"  {i+1}. Sample {idx}: {score.value:.3f}")
    
    finally:
        client.close()


async def basic_asynchronous_example():
    """Basic asynchronous SDK usage example."""
    
    print("\n=== Basic Asynchronous Example ===")
    
    async with AsyncPynomaliClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    ) as client:
        
        # Check connection
        if not await client.validate_connection():
            print("Failed to connect to Pynomaly API")
            return
        
        # Create multiple datasets concurrently
        datasets = []
        dataset_tasks = []
        
        for i in range(3):
            data = np.random.normal(i, 1, (500, 3)).tolist()
            task = client.create_dataset(
                data_source=data,
                name=f"Async Dataset {i+1}",
                feature_names=["x", "y", "z"]
            )
            dataset_tasks.append(task)
        
        datasets = await asyncio.gather(*dataset_tasks)
        print(f"Created {len(datasets)} datasets concurrently")
        
        # Train detectors on all datasets concurrently
        detector_tasks = []
        
        for i, dataset in enumerate(datasets):
            task = client.train_detector(
                dataset_id=dataset.id,
                algorithm=AlgorithmType.LOCAL_OUTLIER_FACTOR,
                name=f"Async Detector {i+1}",
                wait_for_completion=True
            )
            detector_tasks.append(task)
        
        detectors = await asyncio.gather(*detector_tasks)
        print(f"Trained {len(detectors)} detectors concurrently")
        
        # Perform concurrent detection
        test_batches = [
            np.random.normal(0, 1, (50, 3)).tolist(),
            np.random.normal(1, 1, (50, 3)).tolist(),
            np.random.normal(2, 1, (50, 3)).tolist()
        ]
        
        results = await client.batch_detect_concurrent(
            detector_id=detectors[0].id,
            data_sources=test_batches,
            max_concurrent=3
        )
        
        print(f"Completed {len(results)} detections concurrently")
        for i, result in enumerate(results):
            print(f"  Batch {i+1}: {result.num_anomalies}/{result.num_samples} anomalies")


def experiment_comparison_example():
    """Example of algorithm comparison experiment."""
    
    print("\n=== Algorithm Comparison Experiment ===")
    
    client = PynomaliClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    )
    
    try:
        # Create benchmark dataset
        np.random.seed(123)
        normal_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 800)
        anomaly_data = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 50)
        data = np.vstack([normal_data, anomaly_data])
        
        dataset = client.create_dataset(
            data_source=data.tolist(),
            name="Benchmark Dataset",
            description="2D dataset for algorithm comparison",
            feature_names=["x", "y"]
        )
        
        # Run experiment comparing multiple algorithms
        algorithms = [
            AlgorithmType.ISOLATION_FOREST,
            AlgorithmType.LOCAL_OUTLIER_FACTOR,
            AlgorithmType.ONE_CLASS_SVM
        ]
        
        experiment = client.create_experiment(
            name="Algorithm Comparison",
            dataset_id=dataset.id,
            algorithms=algorithms,
            description="Comparing anomaly detection algorithms",
            parameters={
                "IsolationForest": {"n_estimators": 100},
                "LocalOutlierFactor": {"n_neighbors": 20},
                "OneClassSVM": {"nu": 0.05}
            }
        )
        
        print(f"Created experiment: {experiment.id}")
        print(f"Best algorithm: {experiment.best_algorithm}")
        
        # Print results for each algorithm
        print("\nAlgorithm Performance:")
        for algorithm, metrics in experiment.results.items():
            print(f"  {algorithm}:")
            print(f"    Accuracy: {metrics.accuracy:.3f}")
            print(f"    Precision: {metrics.precision:.3f}")
            print(f"    Recall: {metrics.recall:.3f}")
            print(f"    F1 Score: {metrics.f1_score:.3f}")
    
    finally:
        client.close()


async def streaming_detection_example():
    """Example of streaming anomaly detection."""
    
    print("\n=== Streaming Detection Example ===")
    
    async with AsyncPynomaliClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    ) as client:
        
        # Create and train detector (simplified)
        training_data = np.random.normal(0, 1, (1000, 2)).tolist()
        dataset = await client.create_dataset(
            data_source=training_data,
            name="Streaming Training Data",
            feature_names=["sensor1", "sensor2"]
        )
        
        detector = await client.train_detector(
            dataset_id=dataset.id,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            name="Streaming Detector"
        )
        
        # Simulate streaming data
        async def generate_streaming_data():
            """Generate simulated streaming data."""
            for batch_idx in range(10):  # 10 batches
                batch_size = 20
                
                # Mostly normal data with occasional anomalies
                batch_data = []
                for i in range(batch_size):
                    if np.random.random() < 0.1:  # 10% anomalies
                        point = np.random.normal(5, 1, 2).tolist()  # Anomaly
                    else:
                        point = np.random.normal(0, 1, 2).tolist()  # Normal
                    batch_data.append(point)
                
                yield batch_data
                await asyncio.sleep(0.1)  # Simulate real-time delay
        
        # Process streaming data
        print("Processing streaming data...")
        anomaly_count = 0
        total_samples = 0
        
        async for result in client.stream_detection(
            detector_id=detector.id,
            data_stream=generate_streaming_data(),
            buffer_size=50,
            return_scores=True
        ):
            anomaly_count += result.num_anomalies
            total_samples += result.num_samples
            
            print(f"Batch: {result.num_anomalies}/{result.num_samples} anomalies "
                  f"(Total: {anomaly_count}/{total_samples})")
        
        print(f"Streaming completed. Overall anomaly rate: {anomaly_count/total_samples:.2%}")


def configuration_example():
    """Example of different configuration methods."""
    
    print("\n=== Configuration Examples ===")
    
    # Method 1: Direct parameters
    client1 = PynomaliClient(
        base_url="http://localhost:8000",
        api_key="direct-api-key"
    )
    print("Client 1: Direct configuration")
    
    # Method 2: Configuration object
    config = SDKConfig(
        base_url="http://api.example.com",
        api_key="config-api-key",
        auth_type="api_key"
    )
    config.client.timeout = 60.0
    config.client.max_retries = 5
    
    client2 = PynomaliClient(config=config)
    print("Client 2: Configuration object")
    
    # Method 3: Environment variables
    # Set environment variables:
    # export PYNOMALY_BASE_URL="http://localhost:8000"
    # export PYNOMALY_API_KEY="env-api-key"
    
    client3 = PynomaliClient()  # Will load from environment
    print("Client 3: Environment configuration")
    
    # Method 4: Configuration file
    config_path = Path("pynomaly_config.json")
    if config_path.exists():
        client4 = PynomaliClient(config_path=str(config_path))
        print("Client 4: File configuration")
    
    # Clean up
    for client in [client1, client2, client3]:
        client.close()


def batch_processing_example():
    """Example of batch processing multiple datasets."""
    
    print("\n=== Batch Processing Example ===")
    
    client = PynomaliClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    )
    
    try:
        # Create training dataset
        training_data = np.random.normal(0, 1, (1000, 4)).tolist()
        training_dataset = client.create_dataset(
            data_source=training_data,
            name="Batch Training Data",
            feature_names=["a", "b", "c", "d"]
        )
        
        # Train detector
        detector = client.train_detector(
            dataset_id=training_dataset.id,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            name="Batch Detector"
        )
        
        # Create multiple test datasets
        test_datasets = []
        for i in range(5):
            test_data = np.random.normal(i*0.5, 1, (200, 4)).tolist()
            test_dataset = client.create_dataset(
                data_source=test_data,
                name=f"Test Dataset {i+1}",
                feature_names=["a", "b", "c", "d"]
            )
            test_datasets.append(test_dataset.id)
        
        # Batch detection on all test datasets
        results = client.batch_detect(
            detector_id=detector.id,
            data_sources=test_datasets,
            return_scores=True
        )
        
        print(f"Batch processing completed for {len(results)} datasets:")
        for i, result in enumerate(results):
            print(f"  Dataset {i+1}: {result.num_anomalies}/{result.num_samples} anomalies "
                  f"({result.anomaly_rate:.2%})")
    
    finally:
        client.close()


def main():
    """Run all examples."""
    
    print("Pynomaly SDK Examples")
    print("=" * 50)
    
    try:
        # Synchronous examples
        basic_synchronous_example()
        experiment_comparison_example()
        configuration_example()
        batch_processing_example()
        
        # Asynchronous examples
        asyncio.run(basic_asynchronous_example())
        asyncio.run(streaming_detection_example())
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        print("Make sure the Pynomaly API is running and accessible")


if __name__ == "__main__":
    main()