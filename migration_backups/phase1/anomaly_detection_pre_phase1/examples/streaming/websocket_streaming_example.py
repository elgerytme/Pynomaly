"""Comprehensive WebSocket streaming examples for real-time anomaly detection."""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import random
import math

from anomaly_detection.streaming.websocket_client import (
    StreamingWebSocketClient,
    StreamingClientConfig,
    StreamingResponse,
    MessageType,
    create_streaming_client,
    stream_samples
)
from anomaly_detection.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SensorDataGenerator:
    """Generates realistic sensor data with configurable anomalies."""
    
    def __init__(self, 
                 n_features: int = 3,
                 base_values: Optional[List[float]] = None,
                 noise_level: float = 0.1,
                 anomaly_probability: float = 0.05):
        self.n_features = n_features
        self.base_values = base_values or [1.0] * n_features
        self.noise_level = noise_level
        self.anomaly_probability = anomaly_probability
        self.time_step = 0
        
    def generate_normal_sample(self) -> List[float]:
        """Generate a normal sensor reading."""
        sample = []
        for i, base in enumerate(self.base_values):
            # Add seasonal component
            seasonal = math.sin(self.time_step * 0.1 + i) * 0.2
            # Add noise
            noise = random.gauss(0, self.noise_level)
            # Add slight trend
            trend = self.time_step * 0.001
            
            value = base + seasonal + noise + trend
            sample.append(value)
        
        self.time_step += 1
        return sample
    
    def generate_anomaly_sample(self) -> List[float]:
        """Generate an anomalous sensor reading."""
        sample = self.generate_normal_sample()
        
        # Different types of anomalies
        anomaly_type = random.choice(['spike', 'dip', 'drift', 'noise'])
        
        if anomaly_type == 'spike':
            # Sudden spike in random feature
            feature_idx = random.randint(0, len(sample) - 1)
            sample[feature_idx] += random.uniform(3, 8)
        elif anomaly_type == 'dip':
            # Sudden dip in random feature
            feature_idx = random.randint(0, len(sample) - 1)
            sample[feature_idx] -= random.uniform(3, 8)
        elif anomaly_type == 'drift':
            # All features drift together
            drift_factor = random.uniform(2, 5)
            sample = [x * drift_factor for x in sample]
        elif anomaly_type == 'noise':
            # High noise in all features
            sample = [x + random.gauss(0, 0.5) for x in sample]
        
        return sample
    
    def generate_sample(self) -> List[float]:
        """Generate a sensor sample (normal or anomalous)."""
        if random.random() < self.anomaly_probability:
            return self.generate_anomaly_sample()
        else:
            return self.generate_normal_sample()
    
    async def generate_stream(self, duration_seconds: int = 60, frequency_hz: float = 1.0):
        """Generate continuous stream of sensor data."""
        interval = 1.0 / frequency_hz
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            yield self.generate_sample()
            await asyncio.sleep(interval)


class StreamingDashboard:
    """Real-time dashboard for monitoring streaming results."""
    
    def __init__(self):
        self.stats = {
            "total_samples": 0,
            "anomalies_detected": 0,
            "normal_samples": 0,
            "processing_times": [],
            "confidence_scores": [],
            "algorithms_used": {},
            "start_time": time.time(),
            "last_anomaly_time": None
        }
        
        self.recent_samples = []
        self.max_recent_samples = 50
    
    def update_stats(self, response: StreamingResponse):
        """Update dashboard statistics with new response."""
        self.stats["total_samples"] += 1
        
        if response.is_anomaly:
            self.stats["anomalies_detected"] += 1
            self.stats["last_anomaly_time"] = time.time()
        else:
            self.stats["normal_samples"] += 1
        
        if response.confidence_score is not None:
            self.stats["confidence_scores"].append(response.confidence_score)
        
        if response.algorithm:
            algo = response.algorithm
            self.stats["algorithms_used"][algo] = self.stats["algorithms_used"].get(algo, 0) + 1
        
        # Track processing time if available
        if response.metadata and "processing_time_ms" in response.metadata:
            self.stats["processing_times"].append(response.metadata["processing_time_ms"])
        
        # Keep track of recent samples
        self.recent_samples.append({
            "timestamp": datetime.now(),
            "is_anomaly": response.is_anomaly,
            "confidence": response.confidence_score,
            "sample_id": response.sample_id
        })
        
        if len(self.recent_samples) > self.max_recent_samples:
            self.recent_samples = self.recent_samples[-self.max_recent_samples:]
    
    def print_dashboard(self):
        """Print current dashboard state."""
        runtime = time.time() - self.stats["start_time"]
        anomaly_rate = (self.stats["anomalies_detected"] / max(self.stats["total_samples"], 1)) * 100
        
        print(f"\n{'='*60}")
        print(f"🚨 REAL-TIME ANOMALY DETECTION DASHBOARD")
        print(f"{'='*60}")
        print(f"⏱️  Runtime: {runtime:.1f}s")
        print(f"📊 Total Samples: {self.stats['total_samples']}")
        print(f"🚨 Anomalies: {self.stats['anomalies_detected']} ({anomaly_rate:.1f}%)")
        print(f"✅ Normal: {self.stats['normal_samples']}")
        
        if self.stats["processing_times"]:
            avg_processing = np.mean(self.stats["processing_times"])
            print(f"⚡ Avg Processing Time: {avg_processing:.2f}ms")
        
        if self.stats["confidence_scores"]:
            avg_confidence = np.mean(self.stats["confidence_scores"])
            print(f"🎯 Avg Confidence: {avg_confidence:.3f}")
        
        if self.stats["last_anomaly_time"]:
            time_since_anomaly = time.time() - self.stats["last_anomaly_time"]
            print(f"🕐 Last Anomaly: {time_since_anomaly:.1f}s ago")
        
        print(f"🔧 Algorithms Used: {dict(self.stats['algorithms_used'])}")
        
        # Show recent anomalies
        recent_anomalies = [s for s in self.recent_samples if s["is_anomaly"]][-5:]
        if recent_anomalies:
            print(f"\n🚨 Recent Anomalies:")
            for anomaly in recent_anomalies:
                time_str = anomaly["timestamp"].strftime("%H:%M:%S")
                conf_str = f"{anomaly['confidence']:.3f}" if anomaly["confidence"] else "N/A"
                print(f"  • {time_str} | Confidence: {conf_str} | ID: {anomaly['sample_id'][:8]}")
        
        print(f"{'='*60}\n")


async def basic_streaming_example():
    """Basic WebSocket streaming example."""
    print("🚀 Starting Basic WebSocket Streaming Example")
    
    # Configuration
    config = StreamingClientConfig(
        url="ws://localhost:8000/api/v1/streaming/enhanced/basic_example",
        session_id="basic_example",
        reconnect_attempts=3,
        ping_interval=30.0
    )
    
    # Create client
    client = StreamingWebSocketClient(config)
    
    try:
        # Connect
        if not await client.connect():
            print("❌ Failed to connect to WebSocket server")
            return
        
        print("✅ Connected to WebSocket server")
        
        # Create data generator
        generator = SensorDataGenerator(
            n_features=3,
            base_values=[20.0, 1013.25, 50.0],  # Temperature, Pressure, Humidity
            anomaly_probability=0.1
        )
        
        # Send samples
        for i in range(20):
            sample = generator.generate_sample()
            print(f"Sending sample {i+1}: {[f'{x:.2f}' for x in sample]}")
            
            # Create callback to handle response
            response_received = asyncio.Event()
            result = None
            
            def response_callback(response: StreamingResponse):
                nonlocal result
                result = response
                response_received.set()
            
            # Send sample
            request_id = await client.send_sample(
                data=sample,
                algorithm="isolation_forest",
                callback=response_callback
            )
            
            # Wait for response
            await response_received.wait()
            
            if result and result.success:
                status = "🚨 ANOMALY" if result.is_anomaly else "✅ Normal"
                confidence = f"{result.confidence_score:.3f}" if result.confidence_score else "N/A"
                print(f"  → {status} | Confidence: {confidence}")
            else:
                print(f"  → ❌ Error processing sample")
            
            await asyncio.sleep(0.5)
        
        print("✅ Basic streaming example completed")
        
    except Exception as e:
        print(f"❌ Error in basic streaming example: {e}")
    finally:
        await client.disconnect()


async def advanced_streaming_example():
    """Advanced streaming example with dashboard and multiple algorithms."""
    print("🚀 Starting Advanced WebSocket Streaming Example")
    
    # Configuration
    config = StreamingClientConfig(
        url="ws://localhost:8000/api/v1/streaming/enhanced/advanced_example",
        session_id="advanced_example",
        reconnect_attempts=5,
        ping_interval=30.0,
        message_timeout=10.0
    )
    
    # Create client and dashboard
    client = StreamingWebSocketClient(config)
    dashboard = StreamingDashboard()
    
    # Add callbacks
    def on_error(response: StreamingResponse):
        print(f"❌ Error: {response.error}")
    
    def on_connection():
        print("🔗 Connected to server")
    
    def on_disconnection():
        print("🔌 Disconnected from server")
    
    client.add_error_callback(on_error)
    client.add_connection_callback(on_connection)
    client.add_disconnection_callback(on_disconnection)
    
    try:
        # Connect
        if not await client.connect():
            print("❌ Failed to connect to WebSocket server")
            return
        
        # Subscribe to alerts
        def alert_callback(response: StreamingResponse):
            alert_data = response.metadata
            if alert_data:
                print(f"⚠️  ALERT: {alert_data.get('title')} - {alert_data.get('message')}")
        
        await client.subscribe("alerts", alert_callback)
        
        # Create data generator
        generator = SensorDataGenerator(
            n_features=4,
            base_values=[25.0, 1013.25, 60.0, 100.0],  # Temp, Pressure, Humidity, Flow
            anomaly_probability=0.08,
            noise_level=0.15
        )
        
        # Test different algorithms
        algorithms = ["isolation_forest", "one_class_svm", "lof"]
        
        print("📊 Starting continuous streaming with dashboard...")
        
        # Start dashboard update task
        async def update_dashboard():
            while True:
                dashboard.print_dashboard()
                await asyncio.sleep(10)  # Update every 10 seconds
        
        dashboard_task = asyncio.create_task(update_dashboard())
        
        try:
            # Stream data for 2 minutes
            stream_duration = 120  # seconds
            async for sample in generator.generate_stream(stream_duration, frequency_hz=2.0):
                algorithm = random.choice(algorithms)
                
                # Create callback to handle response
                response_received = asyncio.Event()
                result = None
                
                def response_callback(response: StreamingResponse):
                    nonlocal result
                    result = response
                    response_received.set()
                
                # Send sample
                await client.send_sample(
                    data=sample,
                    algorithm=algorithm,
                    callback=response_callback
                )
                
                # Wait for response (with timeout)
                try:
                    await asyncio.wait_for(response_received.wait(), timeout=5.0)
                    
                    if result and result.success:
                        dashboard.update_stats(result)
                        
                        if result.is_anomaly:
                            print(f"🚨 ANOMALY DETECTED! Algorithm: {algorithm}, "
                                  f"Confidence: {result.confidence_score:.3f}")
                    
                except asyncio.TimeoutError:
                    print("⏰ Response timeout")
                
        finally:
            dashboard_task.cancel()
        
        # Final dashboard
        dashboard.print_dashboard()
        
        # Get final statistics
        print("📈 Final Client Statistics:")
        client_stats = client.get_stats()
        for key, value in client_stats.items():
            print(f"  {key}: {value}")
        
        print("✅ Advanced streaming example completed")
        
    except Exception as e:
        print(f"❌ Error in advanced streaming example: {e}")
    finally:
        await client.disconnect()


async def batch_streaming_example():
    """Example of batch processing via WebSocket."""
    print("🚀 Starting Batch WebSocket Streaming Example")
    
    # Create client
    client = await create_streaming_client(
        url="ws://localhost:8000/api/v1/streaming/enhanced/batch_example",
        session_id="batch_example"
    )
    
    try:
        # Create data generator
        generator = SensorDataGenerator(
            n_features=2,
            anomaly_probability=0.15
        )
        
        # Generate batch data
        batch_sizes = [5, 10, 15, 20]
        
        for batch_size in batch_sizes:
            print(f"📦 Processing batch of size {batch_size}")
            
            # Generate batch
            batch_data = [generator.generate_sample() for _ in range(batch_size)]
            
            # Create callback
            response_received = asyncio.Event()
            result = None
            
            def batch_callback(response: StreamingResponse):
                nonlocal result
                result = response
                response_received.set()
            
            # Send batch
            await client.send_batch(
                batch_data=batch_data,
                algorithm="isolation_forest",
                callback=batch_callback
            )
            
            # Wait for response
            await response_received.wait()
            
            if result and result.success and result.results:
                anomaly_count = sum(1 for r in result.results if r.get("is_anomaly"))
                print(f"  → Batch processed: {anomaly_count}/{batch_size} anomalies detected")
                
                # Show processing details
                processing_time = result.metadata.get("processing_time_ms", 0)
                print(f"  → Processing time: {processing_time:.2f}ms")
                
                # Show individual results
                for i, sample_result in enumerate(result.results):
                    if sample_result.get("is_anomaly"):
                        confidence = sample_result.get("confidence_score", 0)
                        print(f"    • Sample {i}: 🚨 ANOMALY (confidence: {confidence:.3f})")
            
            await asyncio.sleep(2)
        
        print("✅ Batch streaming example completed")
        
    except Exception as e:
        print(f"❌ Error in batch streaming example: {e}")
    finally:
        await client.disconnect()


async def drift_detection_example():
    """Example of concept drift detection via WebSocket."""
    print("🚀 Starting Drift Detection Example")
    
    client = await create_streaming_client(
        url="ws://localhost:8000/api/v1/streaming/enhanced/drift_example",
        session_id="drift_example"
    )
    
    try:
        # Create data generators with different characteristics
        generator1 = SensorDataGenerator(
            n_features=3,
            base_values=[1.0, 1.0, 1.0],
            noise_level=0.1
        )
        
        generator2 = SensorDataGenerator(
            n_features=3,
            base_values=[3.0, 3.0, 3.0],  # Different base values (concept drift)
            noise_level=0.1
        )
        
        print("📊 Sending initial data (Concept A)...")
        
        # Send data from first distribution
        for i in range(50):
            sample = generator1.generate_sample()
            await client.send_sample(sample, "isolation_forest")
            await asyncio.sleep(0.1)
        
        # Check for drift
        print("🔍 Checking for concept drift...")
        
        drift_response_received = asyncio.Event()
        drift_result = None
        
        def drift_callback(response: StreamingResponse):
            nonlocal drift_result
            drift_result = response
            drift_response_received.set()
        
        await client.check_drift(window_size=30, callback=drift_callback)
        await drift_response_received.wait()
        
        if drift_result:
            print(f"📈 Drift check result: {drift_result.drift_detected}")
            if drift_result.drift_details:
                print(f"   Details: {drift_result.drift_details}")
        
        print("📊 Sending changed data (Concept B - should trigger drift)...")
        
        # Send data from second distribution (should trigger drift)
        for i in range(50):
            sample = generator2.generate_sample()
            await client.send_sample(sample, "isolation_forest")
            await asyncio.sleep(0.1)
        
        # Check for drift again
        print("🔍 Checking for concept drift after distribution change...")
        
        drift_response_received.clear()
        await client.check_drift(window_size=40, callback=drift_callback)
        await drift_response_received.wait()
        
        if drift_result:
            print(f"📈 Drift check result: {drift_result.drift_detected}")
            if drift_result.drift_detected:
                print("🚨 CONCEPT DRIFT DETECTED!")
            if drift_result.drift_details:
                print(f"   Details: {drift_result.drift_details}")
        
        print("✅ Drift detection example completed")
        
    except Exception as e:
        print(f"❌ Error in drift detection example: {e}")
    finally:
        await client.disconnect()


async def subscription_example():
    """Example of pub/sub messaging via WebSocket."""
    print("🚀 Starting Subscription Example")
    
    client = await create_streaming_client(
        url="ws://localhost:8000/api/v1/streaming/enhanced/subscription_example",
        session_id="subscription_example"
    )
    
    try:
        # Set up subscriptions
        def anomaly_callback(response: StreamingResponse):
            if response.is_anomaly:
                print(f"🚨 Anomaly Subscriber: Detected anomaly with confidence {response.confidence_score:.3f}")
        
        def stats_callback(response: StreamingResponse):
            if response.stats:
                streaming_stats = response.stats.get("streaming", {})
                print(f"📊 Stats Subscriber: Buffer size: {streaming_stats.get('buffer_size', 'N/A')}")
        
        def alert_callback(response: StreamingResponse):
            alert_data = response.metadata
            if alert_data:
                print(f"⚠️  Alert Subscriber: {alert_data.get('title')}")
        
        # Subscribe to different topics
        await client.subscribe("anomalies", anomaly_callback)
        await client.subscribe("stats", stats_callback)
        await client.subscribe("alerts", alert_callback)
        
        print("📡 Subscribed to: anomalies, stats, alerts")
        
        # Generate data and periodically request stats
        generator = SensorDataGenerator(anomaly_probability=0.2)
        
        for i in range(30):
            # Send sample
            sample = generator.generate_sample()
            await client.send_sample(sample, "isolation_forest")
            
            # Request stats every 10 samples
            if i % 10 == 0:
                await client.request_stats()
            
            await asyncio.sleep(0.5)
        
        # Unsubscribe
        await client.unsubscribe("stats")
        print("📡 Unsubscribed from stats")
        
        # Send a few more samples
        for i in range(5):
            sample = generator.generate_sample()
            await client.send_sample(sample, "isolation_forest")
            await asyncio.sleep(0.5)
        
        print("✅ Subscription example completed")
        
    except Exception as e:
        print(f"❌ Error in subscription example: {e}")
    finally:
        await client.disconnect()


async def performance_test():
    """Performance test for WebSocket streaming."""
    print("🚀 Starting Performance Test")
    
    client = await create_streaming_client(
        url="ws://localhost:8000/api/v1/streaming/enhanced/performance_test",
        session_id="performance_test"
    )
    
    try:
        generator = SensorDataGenerator(n_features=5, anomaly_probability=0.05)
        
        # Test parameters
        num_samples = 1000
        batch_size = 50
        
        print(f"🏃 Sending {num_samples} samples in batches of {batch_size}")
        
        start_time = time.time()
        results_received = 0
        
        def result_callback(response: StreamingResponse):
            nonlocal results_received
            results_received += 1
        
        # Send samples in batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            batch_data = [generator.generate_sample() for _ in range(current_batch_size)]
            
            await client.send_batch(
                batch_data=batch_data,
                algorithm="isolation_forest",
                callback=result_callback
            )
            
            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.01)
        
        # Wait for all results
        timeout = 30  # seconds
        end_time = time.time() + timeout
        
        while results_received < (num_samples // batch_size) and time.time() < end_time:
            await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        
        print(f"⏱️  Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Samples sent: {num_samples}")
        print(f"   Batches sent: {num_samples // batch_size}")
        print(f"   Results received: {results_received}")
        print(f"   Throughput: {num_samples / total_time:.1f} samples/sec")
        
        # Get client stats
        client_stats = client.get_stats()
        print(f"   Messages sent: {client_stats['messages_sent']}")
        print(f"   Messages received: {client_stats['messages_received']}")
        print(f"   Errors: {client_stats['errors']}")
        
        print("✅ Performance test completed")
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
    finally:
        await client.disconnect()


async def main():
    """Run all WebSocket streaming examples."""
    print("🌟 WebSocket Streaming Examples for Anomaly Detection")
    print("=" * 60)
    
    examples = [
        ("Basic Streaming", basic_streaming_example),
        ("Advanced Streaming with Dashboard", advanced_streaming_example),
        ("Batch Processing", batch_streaming_example),
        ("Concept Drift Detection", drift_detection_example),
        ("Pub/Sub Subscriptions", subscription_example),
        ("Performance Test", performance_test)
    ]
    
    for name, example_func in examples:
        print(f"\n🔄 Running: {name}")
        try:
            await example_func()
            print(f"✅ Completed: {name}")
        except Exception as e:
            print(f"❌ Failed: {name} - {e}")
        
        print("-" * 40)
        await asyncio.sleep(2)  # Brief pause between examples
    
    print("\n🎉 All WebSocket streaming examples completed!")


if __name__ == "__main__":
    asyncio.run(main())