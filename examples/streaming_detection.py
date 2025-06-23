#!/usr/bin/env python3
"""
Streaming Anomaly Detection Example
==================================

This example demonstrates real-time anomaly detection on streaming data.
Perfect for monitoring systems, IoT sensors, or financial transactions.
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, Any
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pynomaly.infrastructure.config import create_container
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.value_objects import ContaminationRate


class StreamingDataGenerator:
    """Simulates streaming sensor data with occasional anomalies."""
    
    def __init__(self, anomaly_rate: float = 0.05):
        self.anomaly_rate = anomaly_rate
        self.base_values = {
            'temperature': 22.5,
            'pressure': 101.3,
            'humidity': 45.0,
            'vibration': 0.05,
            'rotation_speed': 1500
        }
    
    async def generate_stream(self, duration_seconds: int = 60) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming data points every second."""
        start_time = time.time()
        
        while (time.time() - start_time) < duration_seconds:
            # Generate normal or anomalous data
            is_anomaly = random.random() < self.anomaly_rate
            
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'is_simulated_anomaly': is_anomaly  # Ground truth for evaluation
            }
            
            # Generate sensor readings
            for sensor, base_value in self.base_values.items():
                if is_anomaly:
                    # Create anomalous readings
                    if sensor == 'temperature':
                        data_point[sensor] = base_value + random.uniform(15, 25)  # Heat spike
                    elif sensor == 'pressure':
                        data_point[sensor] = base_value - random.uniform(20, 40)  # Pressure drop
                    elif sensor == 'vibration':
                        data_point[sensor] = base_value + random.uniform(0.5, 1.0)  # High vibration
                    elif sensor == 'rotation_speed':
                        data_point[sensor] = base_value - random.uniform(300, 700)  # Speed drop
                    else:
                        data_point[sensor] = base_value + random.uniform(10, 30)
                else:
                    # Generate normal readings with small variations
                    noise = random.uniform(-0.1, 0.1) * base_value
                    data_point[sensor] = base_value + noise
            
            yield data_point
            await asyncio.sleep(1)  # Stream every second


async def setup_streaming_detector():
    """Set up detector for streaming data."""
    container = create_container()
    
    # Create a training dataset
    print("📊 Creating training dataset...")
    training_data = []
    generator = StreamingDataGenerator(anomaly_rate=0.0)  # Normal data only
    
    # Generate 100 normal data points for training
    count = 0
    async for data_point in generator.generate_stream(duration_seconds=30):
        # Remove metadata for training
        clean_point = {k: v for k, v in data_point.items() 
                      if k not in ['timestamp', 'is_simulated_anomaly']}
        training_data.append(clean_point)
        count += 1
        if count >= 100:
            break
    
    # Create dataset
    dataset_service = container.dataset_service()
    dataset = await dataset_service.create_from_data(
        data=training_data,
        name="Sensor Training Data",
        description="Normal sensor readings for training"
    )
    
    # Create detector
    print("🔧 Creating IsolationForest detector...")
    detection_service = container.detection_service()
    detector = await detection_service.create_detector(
        name="Streaming Sensor Monitor",
        algorithm="IsolationForest",
        parameters={
            "contamination": 0.1,
            "n_estimators": 50,
            "random_state": 42
        }
    )
    
    # Train detector
    print("🎯 Training detector...")
    await detection_service.train_detector(detector.id, dataset.id)
    
    return container, detector.id, dataset.id


async def run_streaming_detection():
    """Run real-time anomaly detection on streaming data."""
    print("🚀 Starting Streaming Anomaly Detection Demo")
    print("=" * 50)
    
    # Setup
    container, detector_id, dataset_id = await setup_streaming_detector()
    detection_service = container.detection_service()
    
    # Streaming detection
    print("\n📡 Starting real-time detection (60 seconds)...")
    print("Format: [Timestamp] Status | Confidence | Values")
    print("-" * 80)
    
    stream_generator = StreamingDataGenerator(anomaly_rate=0.1)  # 10% anomalies
    anomaly_count = 0
    total_points = 0
    correct_predictions = 0
    
    async for data_point in stream_generator.generate_stream(duration_seconds=60):
        total_points += 1
        
        # Prepare data for detection (remove metadata)
        detection_data = {k: v for k, v in data_point.items() 
                         if k not in ['timestamp', 'is_simulated_anomaly']}
        
        # Run detection
        try:
            result = await detection_service.detect_single(detector_id, detection_data)
            
            # Determine if our prediction matches ground truth
            predicted_anomaly = result.is_anomaly
            actual_anomaly = data_point['is_simulated_anomaly']
            
            if predicted_anomaly == actual_anomaly:
                correct_predictions += 1
            
            # Display result
            status = "🚨 ANOMALY" if result.is_anomaly else "✅ NORMAL "
            confidence = f"{result.anomaly_score:.3f}"
            temp = data_point['temperature']
            pressure = data_point['pressure']
            vibration = data_point['vibration']
            
            timestamp = data_point['timestamp'].split('T')[1][:8]  # Just time
            print(f"[{timestamp}] {status} | {confidence} | "
                  f"T:{temp:5.1f}°C P:{pressure:6.1f} V:{vibration:.3f}")
            
            if result.is_anomaly:
                anomaly_count += 1
                
                # Show explanation if available
                if hasattr(result, 'explanation') and result.explanation:
                    print(f"           └─ Explanation: {result.explanation}")
        
        except Exception as e:
            print(f"           ❌ Detection error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Streaming Detection Summary")
    print("=" * 50)
    print(f"Total data points processed: {total_points}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Anomaly rate: {anomaly_count/total_points:.1%}")
    print(f"Accuracy: {correct_predictions/total_points:.1%}")
    
    # Get latest results from repository
    result_repo = container.detection_result_repository()
    recent_results = result_repo.get_recent(limit=5)
    
    if recent_results:
        print(f"\n🔍 Last 5 Detection Results:")
        for i, result in enumerate(recent_results, 1):
            status = "ANOMALY" if result.is_anomaly else "NORMAL"
            print(f"  {i}. {status} (score: {result.anomaly_score:.3f})")


async def streaming_with_ensemble():
    """Advanced example using ensemble of detectors for streaming."""
    print("\n🎭 Advanced: Ensemble Streaming Detection")
    print("=" * 50)
    
    container = create_container()
    detection_service = container.detection_service()
    
    # Create multiple detectors
    algorithms = ["IsolationForest", "LOF", "OCSVM"]
    detector_ids = []
    
    # Setup training data (reusing previous approach)
    training_data = []
    generator = StreamingDataGenerator(anomaly_rate=0.0)
    count = 0
    async for data_point in generator.generate_stream(duration_seconds=15):
        clean_point = {k: v for k, v in data_point.items() 
                      if k not in ['timestamp', 'is_simulated_anomaly']}
        training_data.append(clean_point)
        count += 1
        if count >= 50:
            break
    
    dataset_service = container.dataset_service()
    dataset = await dataset_service.create_from_data(
        data=training_data,
        name="Ensemble Training Data"
    )
    
    # Create and train multiple detectors
    for algo in algorithms:
        print(f"🔧 Setting up {algo} detector...")
        detector = await detection_service.create_detector(
            name=f"Streaming {algo}",
            algorithm=algo,
            parameters={"contamination": 0.1}
        )
        await detection_service.train_detector(detector.id, dataset.id)
        detector_ids.append(detector.id)
    
    # Ensemble streaming detection
    print("\n📡 Ensemble streaming detection (30 seconds)...")
    print("Format: [Time] IF | LOF | SVM | Ensemble | Values")
    print("-" * 90)
    
    stream_generator = StreamingDataGenerator(anomaly_rate=0.15)
    
    async for data_point in stream_generator.generate_stream(duration_seconds=30):
        detection_data = {k: v for k, v in data_point.items() 
                         if k not in ['timestamp', 'is_simulated_anomaly']}
        
        # Get predictions from all detectors
        predictions = []
        scores = []
        
        for detector_id in detector_ids:
            try:
                result = await detection_service.detect_single(detector_id, detection_data)
                predictions.append(result.is_anomaly)
                scores.append(result.anomaly_score)
            except:
                predictions.append(False)
                scores.append(0.0)
        
        # Ensemble decision (majority vote)
        ensemble_prediction = sum(predictions) > len(predictions) / 2
        ensemble_score = sum(scores) / len(scores)
        
        # Display results
        timestamp = data_point['timestamp'].split('T')[1][:8]
        pred_str = " | ".join(["🚨" if p else "✅" for p in predictions])
        ensemble_str = "🚨 ANOM" if ensemble_prediction else "✅ NORM"
        temp = data_point['temperature']
        
        print(f"[{timestamp}] {pred_str} | {ensemble_str} | T:{temp:5.1f}°C")


if __name__ == "__main__":
    print("🌊 Pynomaly Streaming Detection Examples")
    print("=" * 50)
    
    # Run basic streaming detection
    asyncio.run(run_streaming_detection())
    
    # Run ensemble streaming detection
    asyncio.run(streaming_with_ensemble())
    
    print("\n✅ Streaming detection examples completed!")
    print("\nNext steps:")
    print("- Integrate with Apache Kafka for production streaming")
    print("- Add real-time alerting and notifications")
    print("- Implement model drift detection")
    print("- Scale with multiple worker processes")