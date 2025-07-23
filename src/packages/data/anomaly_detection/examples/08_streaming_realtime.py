#!/usr/bin/env python3
"""
Streaming & Real-time Processing Examples for Anomaly Detection Package

This example demonstrates real-time anomaly detection including:
- Kafka integration for streaming data processing
- WebSocket-based real-time detection
- Concept drift detection and adaptation
- Sliding window processing
- Real-time alerting systems
- Stream processing with different frameworks
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Callable, AsyncGenerator
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import logging

# Kafka for streaming
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
    print("Kafka available for streaming examples")
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: Kafka not available. Install with: pip install kafka-python")

# WebSocket support
try:
    import websockets
    import websocket
    WEBSOCKET_AVAILABLE = True
    print("WebSocket available for real-time examples")
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: WebSocket libraries not available. Install with: pip install websockets websocket-client")

# MQTT for IoT streaming
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
    print("MQTT available for IoT streaming")
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: MQTT not available. Install with: pip install paho-mqtt")

# Redis for real-time data storage
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available. Install with: pip install redis")

# Threading and async support
from concurrent.futures import ThreadPoolExecutor
import queue

# Import anomaly detection components
try:
    from anomaly_detection import DetectionService, StreamingService
    from anomaly_detection.domain.entities.detection_result import DetectionResult
except ImportError:
    print("Please install the anomaly_detection package first:")
    print("pip install -e .")
    exit(1)


class StreamingDataGenerator:
    """Generate streaming data with various patterns and anomalies."""
    
    def __init__(self, base_pattern: str = 'sinusoidal', noise_level: float = 0.1):
        self.base_pattern = base_pattern
        self.noise_level = noise_level
        self.time_step = 0
        self.anomaly_probability = 0.05
        
    def generate_sample(self, n_features: int = 5) -> Dict[str, Any]:
        """Generate a single data sample."""
        timestamp = datetime.now()
        
        # Base patterns
        if self.base_pattern == 'sinusoidal':
            base_values = [
                np.sin(self.time_step * 0.1 + i) + np.random.normal(0, self.noise_level)
                for i in range(n_features)
            ]
        elif self.base_pattern == 'linear':
            base_values = [
                0.01 * self.time_step + i + np.random.normal(0, self.noise_level)
                for i in range(n_features)
            ]
        else:  # random
            base_values = [
                np.random.normal(0, 1) for _ in range(n_features)
            ]
        
        # Inject anomalies
        is_anomaly = np.random.random() < self.anomaly_probability
        if is_anomaly:
            # Add anomalous behavior
            anomaly_type = np.random.choice(['spike', 'shift', 'drift'])
            if anomaly_type == 'spike':
                base_values[0] += np.random.uniform(3, 5) * np.random.choice([-1, 1])
            elif anomaly_type == 'shift':
                base_values = [val + 2 for val in base_values]
            else:  # drift
                base_values = [val * 1.5 for val in base_values]
        
        self.time_step += 1
        
        return {
            'timestamp': timestamp.isoformat(),
            'features': base_values,
            'is_anomaly': is_anomaly,
            'sample_id': self.time_step
        }
    
    async def generate_stream(self, duration: float = 60, interval: float = 0.5) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate a continuous stream of data."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            yield self.generate_sample()
            await asyncio.sleep(interval)


class KafkaStreamingService:
    """Kafka-based streaming anomaly detection service."""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.detection_service = DetectionService()
        self.producer = None
        self.consumer = None
        
        # Initialize if Kafka is available
        if KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None
                )
                print(f"Kafka producer initialized for {bootstrap_servers}")
            except Exception as e:
                print(f"Failed to initialize Kafka producer: {e}")
                self.producer = None
    
    def produce_stream(self, topic: str, duration: float = 60, interval: float = 1.0):
        """Produce streaming data to Kafka topic."""
        if not KAFKA_AVAILABLE or not self.producer:
            print("Kafka producer not available")
            return
        
        generator = StreamingDataGenerator()
        start_time = time.time()
        sample_count = 0
        
        print(f"Starting to produce data to topic '{topic}'...")
        
        try:
            while time.time() - start_time < duration:
                sample = generator.generate_sample()
                
                # Send to Kafka
                future = self.producer.send(
                    topic, 
                    value=sample,
                    key=f"sample_{sample['sample_id']}"
                )
                sample_count += 1
                
                if sample_count % 10 == 0:
                    print(f"Produced {sample_count} samples...")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("Stopping producer...")
        except Exception as e:
            print(f"Producer error: {e}")
        finally:
            if self.producer:
                self.producer.flush()
                self.producer.close()
                
        print(f"Produced {sample_count} samples total")
    
    def consume_and_detect(self, topic: str, duration: float = 60):
        """Consume from Kafka topic and perform anomaly detection."""
        if not KAFKA_AVAILABLE:
            print("Kafka not available")
            return
        
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                group_id='anomaly_detection_group'
            )
            
            print(f"Starting to consume from topic '{topic}'...")
            
            # Buffer for batch processing
            sample_buffer = []
            batch_size = 10
            anomaly_count = 0
            total_samples = 0
            
            start_time = time.time()
            
            for message in consumer:
                if time.time() - start_time > duration:
                    break
                
                sample = message.value
                sample_buffer.append(sample['features'])
                total_samples += 1
                
                # Process batch
                if len(sample_buffer) >= batch_size:
                    try:
                        # Convert to numpy array
                        X = np.array(sample_buffer)
                        
                        # Detect anomalies
                        result = self.detection_service.detect_anomalies(
                            X, algorithm='iforest', contamination=0.1
                        )
                        
                        # Count anomalies
                        batch_anomalies = np.sum(result.predictions == -1)
                        anomaly_count += batch_anomalies
                        
                        if batch_anomalies > 0:
                            print(f"Batch anomalies detected: {batch_anomalies} out of {batch_size}")
                            
                            # Send alerts for anomalies
                            for i, pred in enumerate(result.predictions):
                                if pred == -1:
                                    alert = {
                                        'timestamp': datetime.now().isoformat(),
                                        'anomaly_score': float(result.anomaly_scores[i]),
                                        'features': sample_buffer[i],
                                        'alert_type': 'kafka_anomaly'
                                    }
                                    self._send_alert(alert)
                        
                        # Clear buffer
                        sample_buffer = []
                        
                    except Exception as e:
                        print(f"Detection error: {e}")
                        sample_buffer = []
            
            print(f"\nConsumer summary:")
            print(f"Total samples processed: {total_samples}")
            print(f"Total anomalies detected: {anomaly_count}")
            print(f"Anomaly rate: {anomaly_count/max(total_samples, 1):.2%}")
            
        except Exception as e:
            print(f"Consumer error: {e}")
        finally:
            if 'consumer' in locals():
                consumer.close()
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send anomaly alert."""
        print(f"🚨 ANOMALY ALERT: {alert['timestamp']} - Score: {alert['anomaly_score']:.3f}")


class WebSocketStreamingService:
    """WebSocket-based real-time anomaly detection service."""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.detection_service = DetectionService()
        self.clients = set()
        self.is_running = False
        
    async def register_client(self, websocket):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
    
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_result(self, result: Dict[str, Any]):
        """Broadcast detection result to all clients."""
        if not self.clients:
            return
        
        message = json.dumps(result)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        await self.register_client(websocket)
        
        try:
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                data = json.loads(message)
                
                if data.get('type') == 'detect':
                    # Perform anomaly detection
                    features = np.array(data['features']).reshape(1, -1)
                    result = self.detection_service.detect_anomalies(
                        features, algorithm='iforest'
                    )
                    
                    response = {
                        'type': 'detection_result',
                        'timestamp': datetime.now().isoformat(),
                        'prediction': int(result.predictions[0]),
                        'anomaly_score': float(result.anomaly_scores[0]),
                        'is_anomaly': result.predictions[0] == -1
                    }
                    
                    await websocket.send(json.dumps(response))
                    
                elif data.get('type') == 'batch_detect':
                    # Batch detection
                    features = np.array(data['features'])
                    result = self.detection_service.detect_anomalies(
                        features, algorithm=data.get('algorithm', 'iforest')
                    )
                    
                    response = {
                        'type': 'batch_result',
                        'timestamp': datetime.now().isoformat(),
                        'predictions': result.predictions.tolist(),
                        'anomaly_scores': result.anomaly_scores.tolist(),
                        'anomaly_count': result.anomaly_count,
                        'anomaly_rate': result.anomaly_rate
                    }
                    
                    await websocket.send(json.dumps(response))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server."""
        if not WEBSOCKET_AVAILABLE:
            print("WebSocket not available")
            return
        
        print(f"Starting WebSocket server on port {self.port}")
        self.is_running = True
        
        async with websockets.serve(self.handle_client, "localhost", self.port):
            print(f"WebSocket server running on ws://localhost:{self.port}")
            
            # Keep server running
            while self.is_running:
                await asyncio.sleep(1)
    
    def stop_server(self):
        """Stop the WebSocket server."""
        self.is_running = False


class ConceptDriftDetector:
    """Detect concept drift in streaming data."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.drift_history = []
        
    def add_sample(self, sample: np.ndarray) -> Dict[str, Any]:
        """Add a new sample and check for drift."""
        self.current_window.append(sample)
        
        # Check for drift if we have enough samples
        if len(self.current_window) == self.window_size and len(self.reference_window) == self.window_size:
            drift_score = self._calculate_drift_score()
            is_drift = drift_score > self.threshold
            
            drift_info = {
                'timestamp': datetime.now(),
                'drift_score': drift_score,
                'is_drift': is_drift,
                'threshold': self.threshold,
                'window_size': self.window_size
            }
            
            self.drift_history.append(drift_info)
            
            # If drift detected, update reference window
            if is_drift:
                print(f"🔄 Concept drift detected! Score: {drift_score:.4f}")
                self.reference_window = deque(self.current_window, maxlen=self.window_size)
                self.current_window.clear()
            
            return drift_info
        
        # Initialize reference window
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(sample)
        
        return {'is_drift': False, 'drift_score': 0.0}
    
    def _calculate_drift_score(self) -> float:
        """Calculate drift score using statistical tests."""
        ref_data = np.array(list(self.reference_window))
        cur_data = np.array(list(self.current_window))
        
        # Simple drift detection using mean difference
        ref_mean = np.mean(ref_data, axis=0)
        cur_mean = np.mean(cur_data, axis=0)
        
        # Normalized difference
        ref_std = np.std(ref_data, axis=0)
        drift_score = np.mean(np.abs(cur_mean - ref_mean) / (ref_std + 1e-8))
        
        return float(drift_score)
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection history."""
        if not self.drift_history:
            return {'total_checks': 0, 'drift_events': 0, 'drift_rate': 0.0}
        
        drift_events = sum(1 for d in self.drift_history if d['is_drift'])
        
        return {
            'total_checks': len(self.drift_history),
            'drift_events': drift_events,
            'drift_rate': drift_events / len(self.drift_history),
            'last_drift': max((d['timestamp'] for d in self.drift_history if d['is_drift']), default=None),
            'avg_drift_score': np.mean([d['drift_score'] for d in self.drift_history])
        }


class RealTimeAnomalyPipeline:
    """Complete real-time anomaly detection pipeline."""
    
    def __init__(self, buffer_size: int = 50):
        self.detection_service = DetectionService()
        self.drift_detector = ConceptDriftDetector()
        self.data_buffer = deque(maxlen=buffer_size)
        self.anomaly_history = deque(maxlen=1000)
        self.alerts_queue = queue.Queue()
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'total_anomalies': 0,
            'total_drift_events': 0,
            'processing_times': deque(maxlen=100)
        }
        
        # Redis connection for caching (if available)
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                print("Redis connected for caching")
            except:
                self.redis_client = None
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample through the pipeline."""
        start_time = time.time()
        
        features = np.array(sample['features']).reshape(1, -1)
        
        # Anomaly detection
        result = self.detection_service.detect_anomalies(
            features, algorithm='iforest', contamination=0.1
        )
        
        is_anomaly = result.predictions[0] == -1
        anomaly_score = result.anomaly_scores[0]
        
        # Concept drift detection
        drift_info = self.drift_detector.add_sample(features[0])
        
        # Update statistics
        self.stats['total_samples'] += 1
        if is_anomaly:
            self.stats['total_anomalies'] += 1
        if drift_info.get('is_drift', False):
            self.stats['total_drift_events'] += 1
        
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        
        # Create result
        pipeline_result = {
            'timestamp': sample.get('timestamp', datetime.now().isoformat()),
            'sample_id': sample.get('sample_id', self.stats['total_samples']),
            'features': sample['features'],
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'drift_info': drift_info,
            'processing_time': processing_time,
            'pipeline_stats': self._get_current_stats()
        }
        
        # Store in history
        self.anomaly_history.append(pipeline_result)
        
        # Generate alerts
        if is_anomaly or drift_info.get('is_drift', False):
            self._generate_alert(pipeline_result)
        
        # Cache result
        if self.redis_client:
            self._cache_result(pipeline_result)
        
        return pipeline_result
    
    def _get_current_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return {
            'total_samples': self.stats['total_samples'],
            'anomaly_rate': self.stats['total_anomalies'] / max(self.stats['total_samples'], 1),
            'drift_events': self.stats['total_drift_events'],
            'avg_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
            'throughput': len(self.stats['processing_times']) / sum(self.stats['processing_times']) if self.stats['processing_times'] else 0
        }
    
    def _generate_alert(self, result: Dict[str, Any]):
        """Generate alert for anomaly or drift."""
        alert_type = []
        if result['is_anomaly']:
            alert_type.append('anomaly')
        if result['drift_info'].get('is_drift', False):
            alert_type.append('drift')
        
        alert = {
            'timestamp': result['timestamp'],
            'alert_type': alert_type,
            'sample_id': result['sample_id'],
            'anomaly_score': result['anomaly_score'],
            'drift_score': result['drift_info'].get('drift_score', 0),
            'severity': 'high' if result['anomaly_score'] > 0.8 else 'medium'
        }
        
        self.alerts_queue.put(alert)
        print(f"🚨 Alert generated: {alert_type} - Sample {result['sample_id']}")
    
    def _cache_result(self, result: Dict[str, Any]):
        """Cache result in Redis."""
        if not self.redis_client:
            return
        
        key = f"anomaly_result:{result['sample_id']}"
        self.redis_client.setex(key, 3600, json.dumps(result, default=str))
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all pending alerts."""
        alerts = []
        while not self.alerts_queue.empty():
            alerts.append(self.alerts_queue.get())
        return alerts


def example_1_kafka_streaming():
    """Example 1: Kafka-based streaming anomaly detection."""
    print("\n" + "="*60)
    print("Example 1: Kafka Streaming Anomaly Detection")
    print("="*60)
    
    if not KAFKA_AVAILABLE:
        print("Kafka not available. Please install: pip install kafka-python")
        print("Also make sure Kafka server is running on localhost:9092")
        return
    
    # Initialize Kafka service
    kafka_service = KafkaStreamingService()
    
    if not kafka_service.producer:
        print("Could not connect to Kafka. Make sure Kafka is running on localhost:9092")
        print("Start Kafka with:")
        print("1. Start Zookeeper: bin/zookeeper-server-start.sh config/zookeeper.properties")
        print("2. Start Kafka: bin/kafka-server-start.sh config/server.properties")
        return
    
    topic_name = "anomaly_detection_stream"
    
    print(f"Using Kafka topic: {topic_name}")
    print("1. Starting producer (produces data for 30 seconds)")
    print("2. Starting consumer (consumes and detects for 35 seconds)")
    
    # Start producer in a separate thread
    producer_thread = threading.Thread(
        target=kafka_service.produce_stream,
        args=(topic_name, 30, 0.5)  # 30 seconds, 0.5 second intervals
    )
    
    # Start consumer in a separate thread
    consumer_thread = threading.Thread(
        target=kafka_service.consume_and_detect,
        args=(topic_name, 35)  # 35 seconds to ensure it catches all messages
    )
    
    # Start both threads
    producer_thread.start()
    time.sleep(2)  # Give producer a head start
    consumer_thread.start()
    
    # Wait for completion
    producer_thread.join()
    consumer_thread.join()
    
    print("Kafka streaming example completed!")


def example_2_websocket_realtime():
    """Example 2: WebSocket real-time anomaly detection."""
    print("\n" + "="*60)
    print("Example 2: WebSocket Real-time Detection")
    print("="*60)
    
    if not WEBSOCKET_AVAILABLE:
        print("WebSocket not available. Please install: pip install websockets")
        return
    
    # Initialize WebSocket service
    websocket_service = WebSocketStreamingService(port=8765)
    
    # Start server in a separate task
    async def run_server():
        await websocket_service.start_server()
    
    # Client simulation
    async def simulate_client():
        await asyncio.sleep(2)  # Wait for server to start
        
        try:
            uri = "ws://localhost:8765"
            print(f"Connecting to WebSocket server at {uri}")
            
            async with websockets.connect(uri) as websocket:
                print("Connected to WebSocket server")
                
                # Generate and send data
                generator = StreamingDataGenerator()
                
                for i in range(20):
                    sample = generator.generate_sample(n_features=4)
                    
                    # Send detection request
                    request = {
                        'type': 'detect',
                        'features': sample['features'],
                        'sample_id': sample['sample_id']
                    }
                    
                    await websocket.send(json.dumps(request))
                    
                    # Receive response
                    response = await websocket.recv()
                    result = json.loads(response)
                    
                    print(f"Sample {sample['sample_id']}: "
                          f"{'ANOMALY' if result['is_anomaly'] else 'NORMAL'} "
                          f"(score: {result['anomaly_score']:.3f})")
                    
                    # Batch detection example
                    if i % 5 == 0 and i > 0:
                        batch_samples = [generator.generate_sample(4)['features'] for _ in range(5)]
                        batch_request = {
                            'type': 'batch_detect',
                            'features': batch_samples,
                            'algorithm': 'iforest'
                        }
                        
                        await websocket.send(json.dumps(batch_request))
                        batch_response = await websocket.recv()
                        batch_result = json.loads(batch_response)
                        
                        print(f"Batch result: {batch_result['anomaly_count']} anomalies "
                              f"out of {len(batch_samples)} samples")
                    
                    await asyncio.sleep(0.5)
                
                print("Client simulation completed")
                
        except Exception as e:
            print(f"Client error: {e}")
    
    # Run server and client
    async def main():
        server_task = asyncio.create_task(run_server())
        client_task = asyncio.create_task(simulate_client())
        
        # Wait for client to complete, then stop server
        await client_task
        websocket_service.stop_server()
        
        try:
            await asyncio.wait_for(server_task, timeout=2.0)
        except asyncio.TimeoutError:
            pass
    
    # Run the example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Example interrupted")


def example_3_concept_drift_detection():
    """Example 3: Concept drift detection in streaming data."""
    print("\n" + "="*60)
    print("Example 3: Concept Drift Detection")
    print("="*60)
    
    # Initialize drift detector
    drift_detector = ConceptDriftDetector(window_size=50, threshold=0.15)
    
    # Generate streaming data with concept drift
    print("Generating streaming data with intentional concept drift...")
    
    # Phase 1: Normal data
    print("\nPhase 1: Normal baseline data")
    for i in range(100):
        sample = np.random.normal(0, 1, 5)  # Normal distribution
        drift_info = drift_detector.add_sample(sample)
        
        if i % 20 == 0:
            print(f"Sample {i}: drift_score = {drift_info.get('drift_score', 0):.4f}")
    
    # Phase 2: Gradual drift
    print("\nPhase 2: Gradual concept drift")
    for i in range(100, 200):
        # Gradually shift the mean
        shift = (i - 100) * 0.02
        sample = np.random.normal(shift, 1, 5)
        drift_info = drift_detector.add_sample(sample)
        
        if i % 20 == 0:
            print(f"Sample {i}: drift_score = {drift_info.get('drift_score', 0):.4f}, "
                  f"drift detected = {drift_info.get('is_drift', False)}")
    
    # Phase 3: Sudden drift
    print("\nPhase 3: Sudden concept drift")
    for i in range(200, 300):
        # Sudden change in distribution
        sample = np.random.normal(3, 2, 5)  # Different mean and variance
        drift_info = drift_detector.add_sample(sample)
        
        if i % 20 == 0:
            print(f"Sample {i}: drift_score = {drift_info.get('drift_score', 0):.4f}, "
                  f"drift detected = {drift_info.get('is_drift', False)}")
    
    # Get drift summary
    summary = drift_detector.get_drift_summary()
    print(f"\nDrift Detection Summary:")
    print(f"Total checks: {summary['total_checks']}")
    print(f"Drift events: {summary['drift_events']}")
    print(f"Drift rate: {summary['drift_rate']:.2%}")
    print(f"Average drift score: {summary['avg_drift_score']:.4f}")
    
    # Visualize drift history
    if drift_detector.drift_history:
        drift_scores = [d['drift_score'] for d in drift_detector.drift_history]
        drift_events = [d['is_drift'] for d in drift_detector.drift_history]
        
        plt.figure(figsize=(12, 6))
        
        # Plot drift scores
        plt.subplot(2, 1, 1)
        plt.plot(drift_scores, 'b-', linewidth=1)
        plt.axhline(y=drift_detector.threshold, color='r', linestyle='--', 
                   label=f'Threshold ({drift_detector.threshold})')
        plt.ylabel('Drift Score')
        plt.title('Concept Drift Detection Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot drift events
        plt.subplot(2, 1, 2)
        drift_y = [1 if d else 0 for d in drift_events]
        plt.plot(drift_y, 'ro-', markersize=4, linewidth=1)
        plt.ylabel('Drift Detected')
        plt.xlabel('Sample Batch')
        plt.title('Drift Events')
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def example_4_realtime_pipeline():
    """Example 4: Complete real-time anomaly detection pipeline."""
    print("\n" + "="*60)
    print("Example 4: Complete Real-time Pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RealTimeAnomalyPipeline(buffer_size=100)
    
    # Generate streaming data
    generator = StreamingDataGenerator(base_pattern='sinusoidal')
    
    print("Processing streaming data through the pipeline...")
    print("Pipeline includes: anomaly detection, drift detection, alerting, and caching")
    
    results = []
    
    # Process samples
    for i in range(150):
        # Generate sample
        sample = generator.generate_sample(n_features=6)
        
        # Process through pipeline
        result = pipeline.process_sample(sample)
        results.append(result)
        
        # Print progress
        if i % 25 == 0:
            stats = result['pipeline_stats']
            print(f"\nSample {i}:")
            print(f"  Anomaly rate: {stats['anomaly_rate']:.2%}")
            print(f"  Drift events: {stats['drift_events']}")
            print(f"  Avg processing time: {stats['avg_processing_time']:.4f}s")
            print(f"  Throughput: {stats['throughput']:.1f} samples/sec")
        
        # Check for alerts
        alerts = pipeline.get_alerts()
        for alert in alerts:
            print(f"  🚨 {', '.join(alert['alert_type']).upper()} alert - "
                  f"Sample {alert['sample_id']} (severity: {alert['severity']})")
        
        # Small delay to simulate real-time processing
        time.sleep(0.01)
    
    # Final statistics
    print(f"\n" + "="*50)
    print("Pipeline Summary:")
    final_stats = results[-1]['pipeline_stats']
    print(f"Total samples processed: {final_stats['total_samples']}")
    print(f"Overall anomaly rate: {final_stats['anomaly_rate']:.2%}")
    print(f"Total drift events: {final_stats['drift_events']}")
    print(f"Average processing time: {final_stats['avg_processing_time']:.4f} seconds")
    print(f"Average throughput: {final_stats['throughput']:.1f} samples/second")
    
    # Get drift summary
    drift_summary = pipeline.drift_detector.get_drift_summary()
    if drift_summary['total_checks'] > 0:
        print(f"Drift detection rate: {drift_summary['drift_rate']:.2%}")
    
    # Visualize results
    timestamps = [datetime.fromisoformat(r['timestamp']) for r in results]
    anomaly_scores = [r['anomaly_score'] for r in results]
    is_anomaly = [r['is_anomaly'] for r in results]
    processing_times = [r['processing_time'] for r in results]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Anomaly scores over time
    axes[0].plot(timestamps, anomaly_scores, 'b-', alpha=0.7, linewidth=1)
    anomaly_times = [t for t, a in zip(timestamps, is_anomaly) if a]
    anomaly_scores_filtered = [s for s, a in zip(anomaly_scores, is_anomaly) if a]
    axes[0].scatter(anomaly_times, anomaly_scores_filtered, color='red', s=50, alpha=0.8)
    axes[0].set_ylabel('Anomaly Score')
    axes[0].set_title('Real-time Anomaly Detection Results')
    axes[0].grid(True, alpha=0.3)
    
    # Anomaly detection over time
    anomaly_binary = [1 if a else 0 for a in is_anomaly]
    axes[1].plot(timestamps, anomaly_binary, 'ro-', markersize=3, linewidth=1)
    axes[1].set_ylabel('Anomaly Detected')
    axes[1].set_title('Anomaly Events')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    # Processing time
    axes[2].plot(timestamps, processing_times, 'g-', linewidth=1)
    axes[2].set_ylabel('Processing Time (s)')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Processing Performance')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_5_mqtt_iot_streaming():
    """Example 5: MQTT-based IoT streaming anomaly detection."""
    print("\n" + "="*60)
    print("Example 5: MQTT IoT Streaming Detection")
    print("="*60)
    
    if not MQTT_AVAILABLE:
        print("MQTT not available. Please install: pip install paho-mqtt")
        print("Also make sure an MQTT broker is running (e.g., Mosquitto)")
        return
    
    # MQTT configuration
    broker_host = "localhost"
    broker_port = 1883
    topic_data = "iot/sensors/data"
    topic_alerts = "iot/anomalies/alerts"
    
    # Detection service
    detection_service = DetectionService()
    anomaly_count = 0
    total_samples = 0
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker at {broker_host}:{broker_port}")
            client.subscribe(topic_data)
            print(f"Subscribed to topic: {topic_data}")
        else:
            print(f"Failed to connect to MQTT broker. Code: {rc}")
    
    def on_message(client, userdata, message):
        nonlocal anomaly_count, total_samples
        
        try:
            # Parse message
            data = json.loads(message.payload.decode())
            features = np.array(data['sensor_values']).reshape(1, -1)
            
            # Detect anomaly
            result = detection_service.detect_anomalies(
                features, algorithm='iforest', contamination=0.1
            )
            
            is_anomaly = result.predictions[0] == -1
            anomaly_score = result.anomaly_scores[0]
            
            total_samples += 1
            if is_anomaly:
                anomaly_count += 1
            
            print(f"Sample {total_samples}: "
                  f"{'ANOMALY' if is_anomaly else 'NORMAL'} "
                  f"(score: {anomaly_score:.3f})")
            
            # Send alert if anomaly detected
            if is_anomaly:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'sensor_id': data.get('sensor_id', 'unknown'),
                    'anomaly_score': float(anomaly_score),
                    'sensor_values': data['sensor_values'],
                    'alert_level': 'high' if anomaly_score > 0.8 else 'medium'
                }
                
                client.publish(topic_alerts, json.dumps(alert))
                print(f"🚨 Alert sent for sensor {alert['sensor_id']}")
            
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def publish_iot_data(client):
        """Simulate IoT sensor data publishing."""
        generator = StreamingDataGenerator(base_pattern='sinusoidal')
        
        for i in range(50):
            sample = generator.generate_sample(n_features=4)
            
            # Create IoT message
            iot_message = {
                'sensor_id': f'sensor_{i % 5}',  # 5 different sensors
                'timestamp': sample['timestamp'],
                'sensor_values': sample['features'],
                'device_type': 'temperature_sensor'
            }
            
            # Publish to MQTT
            client.publish(topic_data, json.dumps(iot_message))
            time.sleep(0.5)
        
        print("Finished publishing IoT data")
    
    # Create MQTT client
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        
        # Connect to broker
        client.connect(broker_host, broker_port, 60)
        
        # Start client loop in background
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        # Publish data in separate thread
        publisher_thread = threading.Thread(target=publish_iot_data, args=(client,))
        publisher_thread.start()
        
        # Wait for data processing
        publisher_thread.join()
        time.sleep(2)  # Wait for final messages
        
        # Stop client
        client.loop_stop()
        client.disconnect()
        
        print(f"\nMQTT IoT Streaming Summary:")
        print(f"Total samples: {total_samples}")
        print(f"Anomalies detected: {anomaly_count}")
        print(f"Anomaly rate: {anomaly_count/max(total_samples, 1):.2%}")
        
    except Exception as e:
        print(f"MQTT error: {e}")
        print("Make sure an MQTT broker is running (e.g., mosquitto -v)")


def main():
    """Run all streaming and real-time examples."""
    print("\n" + "="*60)
    print("STREAMING & REAL-TIME ANOMALY DETECTION")
    print("="*60)
    
    examples = [
        ("Kafka Streaming Detection", example_1_kafka_streaming),
        ("WebSocket Real-time Detection", example_2_websocket_realtime),
        ("Concept Drift Detection", example_3_concept_drift_detection),
        ("Complete Real-time Pipeline", example_4_realtime_pipeline),
        ("MQTT IoT Streaming", example_5_mqtt_iot_streaming)
    ]
    
    while True:
        print("\nAvailable Examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect an example (0-5): "))
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(examples):
                examples[choice-1][1]()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error running example: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()