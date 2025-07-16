#!/usr/bin/env python3
"""
Pynomaly Real-time Monitoring Example
=====================================

This example demonstrates real-time anomaly detection monitoring
with streaming data simulation and performance tracking.
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import threading
import time
from datetime import datetime
from queue import Empty, Queue

import numpy as np
import pandas as pd

from monorepo.domain.entities import Dataset
from monorepo.domain.value_objects import ContaminationRate
from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class RealTimeMonitor:
    """Real-time anomaly detection monitor."""

    def __init__(self, detector, window_size=100, update_interval=1.0):
        self.detector = detector
        self.window_size = window_size
        self.update_interval = update_interval

        # Data storage
        self.data_queue = Queue()
        self.recent_data = []
        self.detection_history = []

        # Statistics
        self.total_samples = 0
        self.total_anomalies = 0
        self.avg_processing_time = 0

        # Control
        self.running = False
        self.monitor_thread = None

    def add_data_point(self, data_point):
        """Add a new data point for real-time processing."""
        timestamp = datetime.now()
        self.data_queue.put((timestamp, data_point))

    def process_data_stream(self):
        """Process data stream in real-time."""
        print("🔄 Starting real-time monitoring...")

        while self.running:
            try:
                # Collect data points for this interval
                new_points = []
                start_time = time.time()

                # Collect all available data points
                while time.time() - start_time < self.update_interval:
                    try:
                        timestamp, data_point = self.data_queue.get(timeout=0.1)
                        new_points.append((timestamp, data_point))
                    except Empty:
                        continue

                if new_points:
                    self.process_batch(new_points)

            except Exception as e:
                print(f"❌ Error in data stream processing: {e}")

        print("🛑 Real-time monitoring stopped")

    def process_batch(self, data_points):
        """Process a batch of data points."""
        processing_start = time.time()

        # Add to recent data
        for timestamp, data_point in data_points:
            self.recent_data.append({"timestamp": timestamp, "data": data_point})
            self.total_samples += 1

        # Maintain sliding window
        if len(self.recent_data) > self.window_size:
            self.recent_data = self.recent_data[-self.window_size :]

        # Create dataset from recent data
        if len(self.recent_data) >= 10:  # Minimum samples for detection
            data_values = [point["data"] for point in self.recent_data]
            df = pd.DataFrame(data_values)

            dataset = Dataset(
                name=f"RealTime_Window_{datetime.now().strftime('%H%M%S')}", data=df
            )

            try:
                # Run detection
                result = self.detector.detect(dataset)

                # Analyze results
                n_anomalies = len(result.anomalies)
                detection_rate = n_anomalies / len(dataset.data)
                avg_score = np.mean([score.value for score in result.scores])

                # Update statistics
                self.total_anomalies += n_anomalies
                processing_time = time.time() - processing_start
                self.avg_processing_time = (
                    self.avg_processing_time * 0.9 + processing_time * 0.1
                )

                # Store detection result
                detection_result = {
                    "timestamp": datetime.now(),
                    "window_size": len(dataset.data),
                    "anomalies_detected": n_anomalies,
                    "detection_rate": detection_rate,
                    "avg_score": avg_score,
                    "processing_time": processing_time,
                }

                self.detection_history.append(detection_result)

                # Keep history limited
                if len(self.detection_history) > 50:
                    self.detection_history = self.detection_history[-50:]

                # Print real-time update
                current_time = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{current_time}] Processed {len(data_points):2} samples | "
                    f"Window: {len(dataset.data):3} | Anomalies: {n_anomalies:2} | "
                    f"Rate: {detection_rate:.1%} | Score: {avg_score:.3f} | "
                    f"Time: {processing_time:.3f}s"
                )

            except Exception as e:
                print(f"❌ Detection error: {e}")

    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.running:
            print("⚠️ Monitoring already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self.process_data_stream)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def get_statistics(self):
        """Get monitoring statistics."""
        overall_detection_rate = self.total_anomalies / max(self.total_samples, 1)

        recent_stats = {}
        if self.detection_history:
            recent_detections = self.detection_history[-10:]
            recent_stats = {
                "recent_avg_score": np.mean(
                    [d["avg_score"] for d in recent_detections]
                ),
                "recent_detection_rate": np.mean(
                    [d["detection_rate"] for d in recent_detections]
                ),
                "recent_processing_time": np.mean(
                    [d["processing_time"] for d in recent_detections]
                ),
            }

        return {
            "total_samples": self.total_samples,
            "total_anomalies": self.total_anomalies,
            "overall_detection_rate": overall_detection_rate,
            "avg_processing_time": self.avg_processing_time,
            "detection_runs": len(self.detection_history),
            **recent_stats,
        }


class DataStreamSimulator:
    """Simulate streaming data with normal patterns and occasional anomalies."""

    def __init__(self, monitor, duration=30):
        self.monitor = monitor
        self.duration = duration
        self.running = False

    def generate_normal_data(self):
        """Generate normal data point."""
        # Normal pattern: multi-variate gaussian
        return np.random.multivariate_normal(
            mean=[0, 0, 1], cov=[[1, 0.3, 0], [0.3, 1, 0.1], [0, 0.1, 0.8]], size=1
        )[0]

    def generate_anomaly_data(self):
        """Generate anomalous data point."""
        # Anomaly: extreme values
        anomaly_type = np.random.choice(["outlier", "pattern_break"])

        if anomaly_type == "outlier":
            # Extreme outlier
            return np.random.uniform(-5, 5, 3)
        else:
            # Pattern break
            return np.random.multivariate_normal(
                mean=[3, -3, 0], cov=[[2, 0, 0], [0, 2, 0], [0, 0, 1]], size=1
            )[0]

    def simulate_stream(self):
        """Simulate data stream."""
        print(f"📡 Starting data stream simulation for {self.duration} seconds...")

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < self.duration:
            # Generate data point
            if np.random.random() < 0.05:  # 5% anomaly rate
                data_point = self.generate_anomaly_data()
            else:
                data_point = self.generate_normal_data()

            # Send to monitor
            self.monitor.add_data_point(data_point)
            sample_count += 1

            # Variable rate (0.1 to 0.5 seconds between samples)
            time.sleep(np.random.uniform(0.1, 0.5))

        print(f"📡 Stream simulation completed: {sample_count} samples generated")


def run_realtime_monitoring_example():
    """Run the real-time monitoring example."""
    print("🔄 Pynomaly Real-time Monitoring Example")
    print("=" * 50)

    # Create and train detector
    print("\n🤖 Setting up anomaly detector...")
    detector = SklearnAdapter(
        algorithm_name="IsolationForest",
        contamination_rate=ContaminationRate(0.1),
        random_state=42,
        n_estimators=50,  # Faster for real-time
    )

    # Train on initial dataset
    print("🔧 Training detector on baseline data...")
    np.random.seed(42)
    training_data = np.random.multivariate_normal(
        mean=[0, 0, 1], cov=[[1, 0.3, 0], [0.3, 1, 0.1], [0, 0.1, 0.8]], size=200
    )

    training_df = pd.DataFrame(
        training_data, columns=["sensor_1", "sensor_2", "sensor_3"]
    )
    training_dataset = Dataset(name="Training Data", data=training_df)

    detector.fit(training_dataset)
    print("✅ Detector trained successfully")

    # Create real-time monitor
    print("\n📊 Initializing real-time monitor...")
    monitor = RealTimeMonitor(
        detector=detector,
        window_size=50,  # 50-sample sliding window
        update_interval=2.0,  # Process every 2 seconds
    )

    # Start monitoring
    monitor.start_monitoring()
    print("✅ Real-time monitoring started")

    # Simulate data stream
    print("\n📡 Simulating live data stream...")
    simulator = DataStreamSimulator(monitor, duration=20)

    try:
        simulator.simulate_stream()

        # Allow final processing
        time.sleep(3)

    finally:
        # Stop monitoring
        print("\n🛑 Stopping real-time monitoring...")
        monitor.stop_monitoring()

    # Display results
    print("\n📈 Real-time Monitoring Results:")
    print("-" * 40)

    stats = monitor.get_statistics()

    print(f"Total samples processed: {stats['total_samples']}")
    print(f"Total anomalies detected: {stats['total_anomalies']}")
    print(f"Overall detection rate: {stats['overall_detection_rate']:.1%}")
    print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    print(f"Detection runs completed: {stats['detection_runs']}")

    if "recent_avg_score" in stats:
        print(f"Recent average score: {stats['recent_avg_score']:.3f}")
        print(f"Recent detection rate: {stats['recent_detection_rate']:.1%}")
        print(f"Recent processing time: {stats['recent_processing_time']:.3f}s")

    # Show detection timeline
    print("\n📊 Recent Detection History:")
    for i, detection in enumerate(monitor.detection_history[-5:], 1):
        timestamp = detection["timestamp"].strftime("%H:%M:%S")
        print(
            f"   {i}. [{timestamp}] {detection['anomalies_detected']:2} anomalies | "
            f"Rate: {detection['detection_rate']:.1%} | "
            f"Score: {detection['avg_score']:.3f}"
        )

    print("\n🎉 Real-time monitoring example completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- Sliding window anomaly detection")
    print("- Real-time data processing")
    print("- Performance monitoring and statistics")
    print("- Concurrent data ingestion and processing")

    return True


def main():
    """Main function."""
    try:
        success = run_realtime_monitoring_example()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Real-time monitoring example failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
