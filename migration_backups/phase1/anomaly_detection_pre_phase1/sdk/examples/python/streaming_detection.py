#!/usr/bin/env python3
"""
Streaming Anomaly Detection Example using Python SDK

This example demonstrates real-time anomaly detection using WebSocket streaming.
"""

import time
import numpy as np
import threading
from anomaly_detection_sdk import StreamingClient, AlgorithmType
from anomaly_detection_sdk.models import StreamingConfig


class AnomalyDetectionStream:
    """Example streaming anomaly detection class."""
    
    def __init__(self, ws_url="ws://localhost:8000/ws/stream"):
        self.ws_url = ws_url
        self.anomaly_count = 0
        self.total_points = 0
        self.running = False
        
        # Configure streaming
        config = StreamingConfig(
            buffer_size=50,
            detection_threshold=0.6,
            batch_size=5,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            auto_retrain=False
        )
        
        # Initialize streaming client
        self.client = StreamingClient(
            ws_url=ws_url,
            config=config,
            auto_reconnect=True,
            reconnect_delay=3.0
        )
        
        # Set up event handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Set up event handlers for the streaming client."""
        
        @self.client.on_connect
        def on_connect():
            print("✅ Connected to streaming service")
            self.running = True
        
        @self.client.on_disconnect
        def on_disconnect():
            print("❌ Disconnected from streaming service")
            self.running = False
        
        @self.client.on_anomaly
        def on_anomaly(anomaly_data):
            self.anomaly_count += 1
            print(f"🚨 ANOMALY DETECTED:")
            print(f"   Index: {anomaly_data.index}")
            print(f"   Score: {anomaly_data.score:.4f}")
            print(f"   Data Point: {anomaly_data.data_point}")
            if anomaly_data.confidence:
                print(f"   Confidence: {anomaly_data.confidence:.4f}")
            print(f"   Total anomalies so far: {self.anomaly_count}")
            print("-" * 50)
        
        @self.client.on_error
        def on_error(error):
            print(f"❌ Streaming error: {error}")
    
    def start_streaming(self):
        """Start the streaming client."""
        print("Starting streaming anomaly detection...")
        self.client.start()
    
    def stop_streaming(self):
        """Stop the streaming client."""
        print("Stopping streaming...")
        self.client.stop()
        self.running = False
    
    def send_data_point(self, data_point):
        """Send a single data point for detection."""
        try:
            self.client.send_data(data_point)
            self.total_points += 1
        except Exception as e:
            print(f"Error sending data: {e}")
    
    def generate_and_send_data(self, duration=30, interval=1.0):
        """Generate and send sample data for a specified duration."""
        print(f"Generating sample data for {duration} seconds...")
        print("Normal data will be around [0, 0], anomalies around [5, 5]")
        print("-" * 50)
        
        np.random.seed(42)
        start_time = time.time()
        
        while time.time() - start_time < duration and self.running:
            # Generate mostly normal data with occasional anomalies
            if np.random.random() < 0.1:  # 10% chance of anomaly
                # Anomalous data point
                data_point = [
                    np.random.normal(5, 0.5),
                    np.random.normal(5, 0.5)
                ]
                print(f"📤 Sending anomalous point: {[round(x, 3) for x in data_point]}")
            else:
                # Normal data point
                data_point = [
                    np.random.normal(0, 1),
                    np.random.normal(0, 1)
                ]
                print(f"📤 Sending normal point: {[round(x, 3) for x in data_point]}")
            
            self.send_data_point(data_point)
            time.sleep(interval)
        
        print(f"\n✅ Sent {self.total_points} data points")
        print(f"✅ Detected {self.anomaly_count} anomalies")


def interactive_streaming_example():
    """Interactive streaming example where user can input data."""
    print("=== Interactive Streaming Example ===")
    
    stream = AnomalyDetectionStream()
    stream.start_streaming()
    
    # Wait for connection
    time.sleep(2)
    
    if not stream.running:
        print("Failed to connect to streaming service")
        return
    
    print("\nInteractive mode - Enter data points as comma-separated values")
    print("Example: 1.5,2.3 or 5.0,5.0 (for anomaly)")
    print("Type 'quit' to exit")
    
    try:
        while stream.running:
            user_input = input("\nEnter data point (x,y): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                # Parse input
                values = [float(x.strip()) for x in user_input.split(',')]
                if len(values) != 2:
                    print("Please enter exactly 2 values separated by comma")
                    continue
                
                stream.send_data_point(values)
                print(f"Sent: {values}")
                
            except ValueError:
                print("Invalid input. Please enter numeric values separated by comma")
            except Exception as e:
                print(f"Error: {e}")
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    
    finally:
        stream.stop_streaming()
        print(f"\nSession summary:")
        print(f"Total points sent: {stream.total_points}")
        print(f"Anomalies detected: {stream.anomaly_count}")


def automated_streaming_example():
    """Automated streaming example with predefined data."""
    print("=== Automated Streaming Example ===")
    
    stream = AnomalyDetectionStream()
    stream.start_streaming()
    
    # Wait for connection
    time.sleep(2)
    
    if not stream.running:
        print("Failed to connect to streaming service")
        return
    
    try:
        # Run automated data generation in a separate thread
        data_thread = threading.Thread(
            target=stream.generate_and_send_data,
            args=(20, 0.5)  # 20 seconds, 0.5 second intervals
        )
        data_thread.start()
        
        # Wait for data generation to complete
        data_thread.join()
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    
    finally:
        stream.stop_streaming()


def batch_streaming_example():
    """Example of sending data in batches."""
    print("=== Batch Streaming Example ===")
    
    # Generate batch data
    np.random.seed(42)
    normal_batch = np.random.normal(0, 1, (20, 2))
    anomaly_batch = np.random.normal(5, 0.5, (5, 2))
    
    stream = AnomalyDetectionStream()
    stream.start_streaming()
    
    # Wait for connection
    time.sleep(2)
    
    if not stream.running:
        print("Failed to connect to streaming service")
        return
    
    try:
        print("Sending normal data batch...")
        for i, point in enumerate(normal_batch):
            stream.send_data_point(point.tolist())
            print(f"Sent normal point {i+1}/20: {[round(x, 3) for x in point]}")
            time.sleep(0.2)
        
        print("\nSending anomalous data batch...")
        for i, point in enumerate(anomaly_batch):
            stream.send_data_point(point.tolist())
            print(f"Sent anomaly point {i+1}/5: {[round(x, 3) for x in point]}")
            time.sleep(0.2)
        
        # Wait for processing
        print("\nWaiting for processing to complete...")
        time.sleep(5)
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    
    finally:
        stream.stop_streaming()


def performance_streaming_example():
    """Performance test for streaming detection."""
    print("=== Performance Streaming Example ===")
    
    # Configure for high throughput
    config = StreamingConfig(
        buffer_size=100,
        detection_threshold=0.5,
        batch_size=20,  # Larger batches for efficiency
        algorithm=AlgorithmType.ISOLATION_FOREST
    )
    
    client = StreamingClient(
        ws_url="ws://localhost:8000/ws/stream",
        config=config,
        auto_reconnect=True
    )
    
    # Performance tracking
    anomaly_count = 0
    start_time = None
    points_sent = 0
    
    @client.on_connect
    def on_connect():
        nonlocal start_time
        print("✅ Connected - Starting performance test")
        start_time = time.time()
    
    @client.on_anomaly
    def on_anomaly(anomaly_data):
        nonlocal anomaly_count
        anomaly_count += 1
        if anomaly_count % 10 == 0:
            elapsed = time.time() - start_time
            rate = points_sent / elapsed if elapsed > 0 else 0
            print(f"📊 Performance: {anomaly_count} anomalies detected, "
                  f"{rate:.1f} points/sec")
    
    @client.on_error
    def on_error(error):
        print(f"❌ Error: {error}")
    
    try:
        client.start()
        time.sleep(2)  # Wait for connection
        
        # Generate high volume of data
        np.random.seed(42)
        print("Sending high volume data stream (1000 points)...")
        
        for i in range(1000):
            if i % 100 == 0:
                print(f"Sent {i}/1000 points...")
            
            # Mix of normal and anomalous data
            if np.random.random() < 0.05:  # 5% anomalies
                data_point = np.random.normal(4, 0.5, 2).tolist()
            else:
                data_point = np.random.normal(0, 1, 2).tolist()
            
            client.send_data(data_point)
            points_sent += 1
            
            # Small delay to avoid overwhelming
            time.sleep(0.01)
        
        # Wait for final processing
        print("Waiting for final processing...")
        time.sleep(5)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        throughput = points_sent / total_time
        
        print(f"\n📊 Performance Results:")
        print(f"   Total points: {points_sent}")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Throughput: {throughput:.1f} points/second")
        print(f"   Anomalies detected: {anomaly_count}")
        print(f"   Anomaly rate: {(anomaly_count/points_sent)*100:.2f}%")
    
    except KeyboardInterrupt:
        print("\nPerformance test interrupted")
    
    finally:
        client.stop()


def main():
    """Run streaming examples."""
    print("Anomaly Detection Streaming Examples")
    print("=" * 50)
    
    examples = {
        "1": ("Automated Streaming", automated_streaming_example),
        "2": ("Interactive Streaming", interactive_streaming_example),
        "3": ("Batch Streaming", batch_streaming_example),
        "4": ("Performance Test", performance_streaming_example),
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\nSelect example (1-4) or 'all' to run all: ").strip()
    
    if choice.lower() == 'all':
        for name, example_func in examples.values():
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                example_func()
            except Exception as e:
                print(f"Example failed: {e}")
            print("\nPress Enter to continue to next example...")
            input()
    elif choice in examples:
        name, example_func = examples[choice]
        print(f"\n{'='*20} {name} {'='*20}")
        example_func()
    else:
        print("Invalid choice. Running automated example...")
        automated_streaming_example()
    
    print("\n" + "=" * 50)
    print("Streaming examples completed!")


if __name__ == "__main__":
    main()