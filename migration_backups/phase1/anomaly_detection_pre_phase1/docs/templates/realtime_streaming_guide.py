#!/usr/bin/env python3
"""
Real-time Streaming Anomaly Detection Guide
===========================================

Complete guide for implementing real-time anomaly detection with streaming data.
Covers Kafka integration, WebSocket streaming, batch processing, and monitoring.

Usage:
    python realtime_streaming_guide.py

Requirements:
    - anomaly_detection
    - kafka-python (optional)
    - websockets (optional)
    - redis (optional)
    - asyncio
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.domain.services.detection_service import DetectionService


@dataclass
class StreamingConfig:
    """Configuration for streaming detection system."""
    
    # Detection settings
    window_size: int = 1000
    update_frequency: int = 100
    contamination: float = 0.1
    algorithm: str = "isolation_forest"
    
    # Performance settings
    max_batch_size: int = 100
    processing_timeout: int = 30
    buffer_size: int = 10000
    
    # Kafka settings (optional)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "sensor-data"
    kafka_consumer_group: str = "anomaly-detection"
    
    # WebSocket settings
    websocket_port: int = 8765
    max_connections: int = 100
    
    # Redis settings (optional)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Monitoring settings
    enable_monitoring: bool = True
    alert_threshold: float = 0.8
    notification_webhook: Optional[str] = None


@dataclass
class StreamingAlert:
    """Alert for detected anomalies."""
    
    timestamp: datetime
    anomaly_score: float
    sample_data: List[float]
    confidence: float
    algorithm: str
    metadata: Dict[str, Any]


class RealTimeAnomalyDetector:
    """Real-time anomaly detection system with multiple streaming sources."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.streaming_service = StreamingService(
            window_size=config.window_size,
            update_frequency=config.update_frequency
        )
        
        # Buffers and state
        self.data_buffer = deque(maxlen=config.buffer_size)
        self.alerts = deque(maxlen=1000)
        self.connected_clients = set()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_running = False
        
        # Statistics
        self.total_samples = 0
        self.total_anomalies = 0
        self.start_time = datetime.now()
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for streaming system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Real-time anomaly detector initialized")
    
    async def process_sample(self, sample: List[float], metadata: Optional[Dict] = None) -> Optional[StreamingAlert]:
        """
        Process a single sample for anomaly detection.
        
        Args:
            sample: Data sample as list of floats
            metadata: Optional metadata about the sample
            
        Returns:
            StreamingAlert if anomaly detected, None otherwise
        """
        try:
            # Convert to numpy array
            sample_array = np.array(sample, dtype=np.float64)
            
            # Process through streaming service
            result = self.streaming_service.process_sample(sample_array)
            
            self.total_samples += 1
            
            # Check if anomaly detected
            if result.is_anomaly:
                self.total_anomalies += 1
                
                # Create alert
                alert = StreamingAlert(
                    timestamp=datetime.now(),
                    anomaly_score=result.anomaly_score,
                    sample_data=sample,
                    confidence=result.confidence,
                    algorithm=self.config.algorithm,
                    metadata=metadata or {}
                )
                
                self.alerts.append(alert)
                
                # Log alert
                self.logger.warning(f"ANOMALY DETECTED: Score {result.anomaly_score:.3f}, Confidence {result.confidence:.3f}")
                
                # Send notifications
                await self._handle_alert(alert)
                
                return alert
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing sample: {e}")
            return None
    
    async def _handle_alert(self, alert: StreamingAlert):
        """Handle anomaly alert by sending notifications."""
        
        # Send to connected WebSocket clients
        if self.connected_clients:
            alert_message = {
                "type": "anomaly_alert",
                "timestamp": alert.timestamp.isoformat(),
                "score": alert.anomaly_score,
                "confidence": alert.confidence,
                "data": alert.sample_data,
                "metadata": alert.metadata
            }
            
            message = json.dumps(alert_message)
            disconnected_clients = set()
            
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except Exception as e:
                    self.logger.warning(f"Failed to send alert to client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected_clients
        
        # Send webhook notification (if configured)
        if self.config.notification_webhook and alert.anomaly_score > self.config.alert_threshold:
            await self._send_webhook_notification(alert)
    
    async def _send_webhook_notification(self, alert: StreamingAlert):
        """Send webhook notification for high-severity alerts."""
        try:
            import aiohttp
            
            payload = {
                "timestamp": alert.timestamp.isoformat(),
                "severity": "HIGH" if alert.anomaly_score > 0.9 else "MEDIUM",
                "score": alert.anomaly_score,
                "confidence": alert.confidence,
                "algorithm": alert.algorithm,
                "sample_data": alert.sample_data,
                "metadata": alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.notification_webhook,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        self.logger.info("Webhook notification sent successfully")
                    else:
                        self.logger.warning(f"Webhook notification failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error sending webhook notification: {e}")
    
    async def start_kafka_consumer(self):
        """Start Kafka consumer for streaming data."""
        try:
            from kafka import KafkaConsumer
            
            consumer = KafkaConsumer(
                self.config.kafka_topic,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            self.logger.info(f"Started Kafka consumer for topic: {self.config.kafka_topic}")
            
            for message in consumer:
                if not self.is_running:
                    break
                
                try:
                    data = message.value
                    
                    # Extract sample data
                    if isinstance(data, dict):
                        sample = data.get('data', data.get('values', []))
                        metadata = data.get('metadata', {})
                    else:
                        sample = data
                        metadata = {}
                    
                    # Process sample
                    if sample:
                        await self.process_sample(sample, metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error processing Kafka message: {e}")
            
            consumer.close()
            
        except ImportError:
            self.logger.error("kafka-python not installed. Install with: pip install kafka-python")
        except Exception as e:
            self.logger.error(f"Kafka consumer error: {e}")
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time data streaming."""
        self.connected_clients.add(websocket)
        self.logger.info(f"WebSocket client connected. Total clients: {len(self.connected_clients)}")
        
        try:
            # Send welcome message
            welcome_message = {
                "type": "welcome",
                "message": "Connected to anomaly detection stream",
                "config": {
                    "algorithm": self.config.algorithm,
                    "window_size": self.config.window_size,
                    "contamination": self.config.contamination
                }
            }
            await websocket.send(json.dumps(welcome_message))
            
            # Keep connection alive and handle incoming data
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'data':
                        sample = data.get('sample', [])
                        metadata = data.get('metadata', {})
                        
                        if sample:
                            alert = await self.process_sample(sample, metadata)
                            
                            # Send immediate response
                            response = {
                                "type": "detection_result",
                                "anomaly_detected": alert is not None,
                                "score": alert.anomaly_score if alert else 0.0,
                                "timestamp": datetime.now().isoformat()
                            }
                            await websocket.send(json.dumps(response))
                    
                    elif data.get('type') == 'get_stats':
                        stats = self.get_statistics()
                        await websocket.send(json.dumps({
                            "type": "statistics",
                            "data": stats
                        }))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    self.logger.error(f"WebSocket message error: {e}")
        
        except Exception as e:
            self.logger.warning(f"WebSocket connection error: {e}")
        
        finally:
            self.connected_clients.discard(websocket)
            self.logger.info(f"WebSocket client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for real-time streaming."""
        try:
            import websockets
            
            self.logger.info(f"Starting WebSocket server on port {self.config.websocket_port}")
            
            async with websockets.serve(
                self.websocket_handler,
                "localhost",
                self.config.websocket_port,
                max_size=1024*1024,  # 1MB message limit
                max_queue=100
            ):
                await asyncio.Future()  # Keep server running
                
        except ImportError:
            self.logger.error("websockets not installed. Install with: pip install websockets")
        except Exception as e:
            self.logger.error(f"WebSocket server error: {e}")
    
    def generate_sample_data_stream(self) -> AsyncGenerator[List[float], None]:
        """Generate sample streaming data for testing."""
        async def data_generator():
            while self.is_running:
                # Generate normal data (95% of time)
                if np.random.random() < 0.95:
                    sample = np.random.normal(0, 1, 5).tolist()
                else:
                    # Generate anomalous data
                    sample = np.random.normal(5, 2, 5).tolist()
                
                yield sample
                await asyncio.sleep(0.1)  # 10 samples per second
        
        return data_generator()
    
    async def start_sample_data_stream(self):
        """Start generating and processing sample data."""
        self.logger.info("Starting sample data stream...")
        
        async for sample in self.generate_sample_data_stream():
            await self.process_sample(sample, {"source": "sample_generator"})
            
            # Print statistics periodically
            if self.total_samples % 100 == 0:
                stats = self.get_statistics()
                self.logger.info(f"Processed {stats['total_samples']} samples, "
                               f"detected {stats['total_anomalies']} anomalies "
                               f"({stats['anomaly_rate']:.2%})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current streaming statistics."""
        runtime = datetime.now() - self.start_time
        
        return {
            "total_samples": self.total_samples,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": self.total_anomalies / max(self.total_samples, 1),
            "samples_per_second": self.total_samples / max(runtime.total_seconds(), 1),
            "runtime_seconds": runtime.total_seconds(),
            "connected_clients": len(self.connected_clients),
            "recent_alerts": len([a for a in self.alerts if (datetime.now() - a.timestamp).total_seconds() < 3600]),
            "buffer_size": len(self.data_buffer),
            "algorithm": self.config.algorithm,
            "window_size": self.config.window_size
        }
    
    async def start_monitoring(self):
        """Start monitoring and periodic reporting."""
        while self.is_running:
            await asyncio.sleep(60)  # Report every minute
            
            stats = self.get_statistics()
            self.logger.info(f"MONITORING: {stats['samples_per_second']:.1f} samples/sec, "
                           f"{stats['anomaly_rate']:.2%} anomaly rate, "
                           f"{stats['connected_clients']} clients connected")
    
    async def start_all_services(self):
        """Start all streaming services concurrently."""
        self.is_running = True
        
        tasks = [
            asyncio.create_task(self.start_websocket_server()),
            asyncio.create_task(self.start_sample_data_stream()),
            asyncio.create_task(self.start_monitoring())
        ]
        
        # Add Kafka consumer if available
        try:
            import kafka
            tasks.append(asyncio.create_task(self.start_kafka_consumer()))
        except ImportError:
            self.logger.info("Kafka not available, skipping Kafka consumer")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Shutting down streaming services...")
            self.is_running = False


def demonstrate_streaming_patterns():
    """Demonstrate different streaming patterns and use cases."""
    
    print("🌊 Real-time Streaming Anomaly Detection Guide")
    print("=" * 55)
    
    print("\n📊 Streaming Patterns:")
    
    print("\n1. Real-time Processing Pattern:")
    print("""
    Data Source → Stream Processor → Anomaly Detector → Alerts
         ↓              ↓                    ↓            ↓
    Kafka/WS    Buffer & Batch    ML Algorithm    WebHook/WS
    """)
    
    print("\n2. Batch Processing Pattern:")
    print("""
    Data Source → Buffer → Batch Processor → Results Storage
         ↓          ↓            ↓               ↓
    Continuous  Fixed Size   Bulk Detection   Database/Cache
    """)
    
    print("\n3. Hybrid Pattern:")
    print("""
    Data Source → Real-time Check → Background Processing → Long-term Analysis
         ↓              ↓                    ↓                    ↓
    Fast Path     Simple Rules      ML Models         Trend Analysis
    """)
    
    print("\n🔧 Implementation Examples:")
    
    print("\n1. IoT Sensor Monitoring:")
    print("""
    config = StreamingConfig(
        window_size=500,        # 500 sensor readings
        update_frequency=50,    # Update model every 50 samples
        contamination=0.05,     # Expect 5% anomalies
        algorithm="lof",        # Good for sensor data
        alert_threshold=0.9     # High confidence alerts only
    )
    
    detector = RealTimeAnomalyDetector(config)
    await detector.start_all_services()
    """)
    
    print("\n2. Network Traffic Analysis:")
    print("""
    config = StreamingConfig(
        window_size=1000,       # 1000 network packets
        update_frequency=100,   # Update every 100 packets
        contamination=0.1,      # Expect 10% anomalies
        algorithm="isolation_forest",  # Good for mixed data
        kafka_topic="network-traffic"
    )
    """)
    
    print("\n3. Financial Transaction Monitoring:")
    print("""
    config = StreamingConfig(
        window_size=2000,       # 2000 transactions
        update_frequency=200,   # Update every 200 transactions
        contamination=0.01,     # Expect 1% fraud
        algorithm="ensemble",   # Maximum accuracy
        alert_threshold=0.95,   # Very high confidence only
        notification_webhook="https://fraud-alerts.example.com/webhook"
    )
    """)
    
    print("\n📡 Client Connection Examples:")
    
    print("\n1. WebSocket Client (JavaScript):")
    print("""
    const ws = new WebSocket('ws://localhost:8765');
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'anomaly_alert') {
            console.log('ANOMALY DETECTED:', data.score);
        }
    };
    
    // Send data for analysis
    ws.send(JSON.stringify({
        type: 'data',
        sample: [1.2, 3.4, 5.6, 7.8, 9.0],
        metadata: {source: 'sensor_1'}
    }));
    """)
    
    print("\n2. Python Client:")
    print("""
    import asyncio
    import websockets
    import json
    
    async def stream_client():
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            # Send sample data
            await websocket.send(json.dumps({
                "type": "data",
                "sample": [1.0, 2.0, 3.0, 4.0, 5.0]
            }))
            
            # Receive response
            response = await websocket.recv()
            result = json.loads(response)
            print(f"Anomaly detected: {result['anomaly_detected']}")
    
    asyncio.run(stream_client())
    """)
    
    print("\n3. Kafka Producer (Python):")
    print("""
    from kafka import KafkaProducer
    import json
    
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Send sensor data
    producer.send('sensor-data', {
        'data': [1.2, 3.4, 5.6, 7.8, 9.0],
        'metadata': {
            'sensor_id': 'temp_001',
            'timestamp': time.time(),
            'location': 'warehouse_a'
        }
    })
    """)
    
    print("\n⚡ Performance Optimization Tips:")
    
    tips = [
        "• Use appropriate window sizes (larger = more stable, smaller = more responsive)",
        "• Batch process when possible to reduce overhead",
        "• Use connection pooling for database operations",
        "• Implement circuit breakers for external dependencies",
        "• Monitor memory usage and implement data cleanup",
        "• Use async/await for I/O operations",
        "• Consider using Redis for high-speed caching",
        "• Implement backpressure handling for high-volume streams",
        "• Use appropriate serialization formats (msgpack, protobuf)",
        "• Set up proper monitoring and alerting"
    ]
    
    for tip in tips:
        print(f"   {tip}")


async def run_streaming_demo():
    """Run a complete streaming demonstration."""
    
    print("\n🚀 Starting Real-time Streaming Demo...")
    print("Press Ctrl+C to stop")
    
    config = StreamingConfig(
        window_size=200,
        update_frequency=20,
        contamination=0.1,
        websocket_port=8765,
        enable_monitoring=True
    )
    
    detector = RealTimeAnomalyDetector(config)
    
    try:
        await detector.start_all_services()
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")


def main():
    """Main function to run the streaming guide."""
    
    demonstrate_streaming_patterns()
    
    print(f"\n🎯 Ready to start streaming detection?")
    choice = input("Start demo? (y/n): ").lower().strip()
    
    if choice == 'y':
        asyncio.run(run_streaming_demo())
    else:
        print("Demo skipped. Use the examples above to implement your own streaming solution!")


if __name__ == "__main__":
    main()