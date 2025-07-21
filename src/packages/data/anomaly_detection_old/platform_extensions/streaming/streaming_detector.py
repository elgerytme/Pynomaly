"""
Real-Time Streaming Anomaly Detection for Pynomaly Detection
============================================================

Advanced streaming analytics with support for Kafka, Pulsar, Redis Streams,
and real-time anomaly detection with adaptive models.
"""

import logging
import asyncio
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import pulsar
    PULSAR_AVAILABLE = True
except ImportError:
    PULSAR_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import websockets
    import asyncio
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator

# Import our services
from ...simplified_services.core_detection_service import CoreDetectionService

logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for streaming anomaly detection."""
    # Stream processing
    batch_size: int = 100
    window_size: int = 1000
    slide_interval: float = 1.0  # seconds
    max_queue_size: int = 10000
    
    # Anomaly detection
    detection_algorithm: str = "isolation_forest"
    contamination: float = 0.1
    adaptation_rate: float = 0.1
    min_training_samples: int = 100
    
    # Model updating
    update_frequency: int = 100  # Update model every N samples
    model_warm_up_samples: int = 500
    enable_online_learning: bool = True
    
    # Performance
    max_processing_time: float = 0.1  # seconds
    enable_async_processing: bool = True
    num_worker_threads: int = 4
    
    # Alerting
    enable_real_time_alerts: bool = True
    alert_threshold: float = 0.8
    alert_cooldown: float = 60.0  # seconds
    
    # Data preprocessing
    normalize_data: bool = True
    handle_missing_values: bool = True
    feature_columns: Optional[List[str]] = None
    timestamp_column: str = "timestamp"
    
    # Buffering and storage
    buffer_size: int = 10000
    enable_data_persistence: bool = False
    persistence_interval: int = 1000

@dataclass
class StreamingResult:
    """Result of streaming anomaly detection."""
    timestamp: datetime
    data: Dict[str, Any]
    is_anomaly: bool
    anomaly_score: float
    model_confidence: float
    processing_time: float
    buffer_size: int
    model_version: int

class StreamingDetector:
    """Real-time streaming anomaly detector."""
    
    def __init__(self, config: StreamingConfig = None):
        """Initialize streaming detector.
        
        Args:
            config: Streaming configuration
        """
        self.config = config or StreamingConfig()
        self.core_service = CoreDetectionService()
        
        # Model and preprocessing
        self.model = None
        self.scaler = StandardScaler() if self.config.normalize_data else None
        self.model_version = 0
        
        # Streaming buffers
        self.data_buffer = deque(maxlen=self.config.buffer_size)
        self.training_buffer = deque(maxlen=self.config.window_size)
        self.result_buffer = deque(maxlen=1000)
        
        # State management
        self.is_running = False
        self.sample_count = 0
        self.last_model_update = 0
        self.last_alert_time = 0
        
        # Threading
        self.processing_thread = None
        self.lock = threading.RLock()
        
        # Callbacks
        self.anomaly_callbacks = []
        self.data_callbacks = []
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'avg_processing_time': 0,
            'model_updates': 0,
            'alerts_sent': 0
        }
        
        logger.info(f"Streaming detector initialized with {self.config.detection_algorithm}")
    
    def start(self):
        """Start streaming detection."""
        if self.is_running:
            logger.warning("Streaming detector is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Streaming detector started")
    
    def stop(self):
        """Stop streaming detection."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Streaming detector stopped")
    
    def process_sample(self, data: Union[Dict[str, Any], np.ndarray, List]) -> Optional[StreamingResult]:
        """Process a single data sample.
        
        Args:
            data: Input data sample
            
        Returns:
            Streaming result if processed, None if buffered
        """
        start_time = time.time()
        
        try:
            # Convert data to standardized format
            processed_data = self._prepare_sample(data)
            
            # Add to buffer
            with self.lock:
                self.data_buffer.append({
                    'timestamp': datetime.now(),
                    'data': processed_data,
                    'raw_data': data
                })
                self.sample_count += 1
            
            # Process if batch is ready
            if len(self.data_buffer) >= self.config.batch_size:
                return self._process_batch()
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            return None
    
    def process_batch(self, data_batch: List[Union[Dict[str, Any], np.ndarray]]) -> List[StreamingResult]:
        """Process a batch of data samples.
        
        Args:
            data_batch: Batch of data samples
            
        Returns:
            List of streaming results
        """
        results = []
        
        for sample in data_batch:
            result = self.process_sample(sample)
            if result:
                results.append(result)
        
        return results
    
    def add_anomaly_callback(self, callback: Callable[[StreamingResult], None]):
        """Add callback for anomaly detection.
        
        Args:
            callback: Function to call when anomaly is detected
        """
        self.anomaly_callbacks.append(callback)
    
    def add_data_callback(self, callback: Callable[[StreamingResult], None]):
        """Add callback for all data processing.
        
        Args:
            callback: Function to call for each processed sample
        """
        self.data_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'buffer_size': len(self.data_buffer),
                'training_buffer_size': len(self.training_buffer),
                'model_version': self.model_version,
                'is_running': self.is_running,
                'samples_processed': self.sample_count,
                'anomaly_rate': stats['anomalies_detected'] / max(stats['total_processed'], 1)
            })
            return stats
    
    def get_recent_results(self, limit: int = 100) -> List[StreamingResult]:
        """Get recent processing results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent results
        """
        with self.lock:
            return list(self.result_buffer)[-limit:]
    
    def _prepare_sample(self, data: Union[Dict[str, Any], np.ndarray, List]) -> np.ndarray:
        """Prepare data sample for processing."""
        if isinstance(data, dict):
            # Extract feature columns if specified
            if self.config.feature_columns:
                features = [data.get(col, 0) for col in self.config.feature_columns]
            else:
                # Use all numeric values
                features = [v for v in data.values() 
                           if isinstance(v, (int, float)) and not np.isnan(v)]
            
            return np.array(features, dtype=np.float32)
            
        elif isinstance(data, (list, np.ndarray)):
            return np.array(data, dtype=np.float32).flatten()
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _process_batch(self) -> Optional[StreamingResult]:
        """Process current batch of data."""
        start_time = time.time()
        
        with self.lock:
            if len(self.data_buffer) == 0:
                return None
            
            # Get batch data
            batch_data = list(self.data_buffer)
            self.data_buffer.clear()
        
        try:
            # Extract features
            features = []
            timestamps = []
            raw_samples = []
            
            for sample in batch_data:
                features.append(sample['data'])
                timestamps.append(sample['timestamp'])
                raw_samples.append(sample['raw_data'])
            
            features_array = np.array(features)
            
            # Handle missing values
            if self.config.handle_missing_values:
                features_array = self._handle_missing_values(features_array)
            
            # Normalize if required
            if self.scaler and self.config.normalize_data:
                # Update scaler incrementally if we have enough samples
                if len(self.training_buffer) >= self.config.min_training_samples:
                    try:
                        # Partial fit for online learning
                        self.scaler.partial_fit(features_array)
                    except AttributeError:
                        # Fallback for scalers without partial_fit
                        pass
                
                # Transform features
                try:
                    features_array = self.scaler.transform(features_array)
                except Exception as e:
                    logger.warning(f"Scaling failed: {e}")
            
            # Update training buffer
            for feature in features_array:
                self.training_buffer.append(feature)
            
            # Initialize or update model
            if self._should_update_model():
                self._update_model()
            
            # Make predictions if model is available
            if self.model is not None:
                results = self._predict_batch(features_array, timestamps, raw_samples)
                
                # Update statistics
                self.stats['total_processed'] += len(results)
                self.stats['anomalies_detected'] += sum(1 for r in results if r.is_anomaly)
                
                processing_time = time.time() - start_time
                self.stats['avg_processing_time'] = (
                    self.stats['avg_processing_time'] * 0.9 + processing_time * 0.1
                )
                
                # Store results
                for result in results:
                    self.result_buffer.append(result)
                    self._trigger_callbacks(result)
                
                # Return last result for immediate feedback
                return results[-1] if results else None
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
        
        return None
    
    def _should_update_model(self) -> bool:
        """Check if model should be updated."""
        if self.model is None and len(self.training_buffer) >= self.config.min_training_samples:
            return True
        
        if (self.config.enable_online_learning and 
            self.sample_count - self.last_model_update >= self.config.update_frequency and
            len(self.training_buffer) >= self.config.min_training_samples):
            return True
        
        return False
    
    def _update_model(self):
        """Update the anomaly detection model."""
        try:
            # Get training data
            training_data = np.array(list(self.training_buffer))
            
            if len(training_data) < self.config.min_training_samples:
                return
            
            # Train model based on algorithm
            if self.config.detection_algorithm == "isolation_forest":
                self.model = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=42,
                    n_estimators=50  # Reduced for faster training
                )
            else:
                # Use core service for other algorithms
                results = self.core_service.detect_anomalies(
                    training_data,
                    algorithm=self.config.detection_algorithm,
                    contamination=self.config.contamination
                )
                self.model = results['model']
            
            # Fit model
            if hasattr(self.model, 'fit'):
                self.model.fit(training_data)
            
            self.model_version += 1
            self.last_model_update = self.sample_count
            self.stats['model_updates'] += 1
            
            logger.info(f"Model updated (version {self.model_version}) with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def _predict_batch(self, features: np.ndarray, timestamps: List[datetime], 
                      raw_samples: List[Any]) -> List[StreamingResult]:
        """Make predictions on feature batch."""
        results = []
        
        try:
            # Get predictions
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(features)
            else:
                predictions = np.zeros(len(features))
            
            # Get anomaly scores
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(features)
            elif hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(features)
            else:
                scores = np.abs(predictions)
            
            # Normalize scores to [0, 1]
            if len(scores) > 0:
                min_score, max_score = np.min(scores), np.max(scores)
                if max_score > min_score:
                    normalized_scores = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.zeros_like(scores)
            else:
                normalized_scores = scores
            
            # Create results
            for i in range(len(features)):
                is_anomaly = predictions[i] == -1 if hasattr(self.model, 'predict') else normalized_scores[i] > self.config.alert_threshold
                
                result = StreamingResult(
                    timestamp=timestamps[i],
                    data=raw_samples[i],
                    is_anomaly=bool(is_anomaly),
                    anomaly_score=float(normalized_scores[i]),
                    model_confidence=1.0 - abs(normalized_scores[i] - 0.5) * 2,
                    processing_time=time.time() - timestamps[i].timestamp(),
                    buffer_size=len(self.data_buffer),
                    model_version=self.model_version
                )
                
                results.append(result)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
        
        return results
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in data."""
        if not np.any(np.isnan(data)):
            return data
        
        # Simple forward fill for missing values
        data = pd.DataFrame(data).fillna(method='ffill').fillna(0).values
        return data
    
    def _trigger_callbacks(self, result: StreamingResult):
        """Trigger appropriate callbacks for result."""
        try:
            # Data callbacks (always triggered)
            for callback in self.data_callbacks:
                callback(result)
            
            # Anomaly callbacks (only for anomalies)
            if result.is_anomaly:
                current_time = time.time()
                
                # Check cooldown
                if current_time - self.last_alert_time > self.config.alert_cooldown:
                    for callback in self.anomaly_callbacks:
                        callback(result)
                    
                    self.last_alert_time = current_time
                    self.stats['alerts_sent'] += 1
                    
        except Exception as e:
            logger.error(f"Callback execution failed: {e}")
    
    def _processing_loop(self):
        """Main processing loop for continuous streaming."""
        while self.is_running:
            try:
                # Process any buffered data
                if len(self.data_buffer) >= self.config.batch_size:
                    self._process_batch()
                
                # Sleep based on slide interval
                time.sleep(self.config.slide_interval)
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(1.0)  # Prevent tight error loop

class KafkaStreaming:
    """Kafka integration for streaming anomaly detection."""
    
    def __init__(self, detector: StreamingDetector, 
                 bootstrap_servers: str = "localhost:9092",
                 input_topic: str = "anomaly_input",
                 output_topic: str = "anomaly_output",
                 consumer_group: str = "pynomaly_group"):
        """Initialize Kafka streaming.
        
        Args:
            detector: Streaming detector instance
            bootstrap_servers: Kafka bootstrap servers
            input_topic: Input topic name
            output_topic: Output topic name
            consumer_group: Consumer group ID
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is required for Kafka integration")
        
        self.detector = detector
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer_group = consumer_group
        
        # Initialize Kafka components
        self.consumer = None
        self.producer = None
        self.is_running = False
        
    def start(self):
        """Start Kafka streaming."""
        try:
            # Initialize consumer
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            # Start detector
            self.detector.start()
            
            # Add callback for anomaly output
            self.detector.add_anomaly_callback(self._send_anomaly_result)
            
            self.is_running = True
            
            # Start consuming
            threading.Thread(target=self._consume_messages, daemon=True).start()
            
            logger.info(f"Kafka streaming started: {self.input_topic} -> {self.output_topic}")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka streaming: {e}")
            raise
    
    def stop(self):
        """Stop Kafka streaming."""
        self.is_running = False
        self.detector.stop()
        
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        
        logger.info("Kafka streaming stopped")
    
    def _consume_messages(self):
        """Consume messages from Kafka topic."""
        try:
            for message in self.consumer:
                if not self.is_running:
                    break
                
                # Process message
                data = message.value
                result = self.detector.process_sample(data)
                
                # Send result if available
                if result:
                    self._send_result(result)
                    
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
    
    def _send_result(self, result: StreamingResult):
        """Send result to output topic."""
        try:
            result_data = {
                'timestamp': result.timestamp.isoformat(),
                'is_anomaly': result.is_anomaly,
                'anomaly_score': result.anomaly_score,
                'model_confidence': result.model_confidence,
                'processing_time': result.processing_time,
                'model_version': result.model_version,
                'data': result.data
            }
            
            self.producer.send(self.output_topic, result_data)
            
        except Exception as e:
            logger.error(f"Failed to send Kafka message: {e}")
    
    def _send_anomaly_result(self, result: StreamingResult):
        """Send anomaly-specific result to alert topic."""
        try:
            alert_data = {
                'alert_type': 'anomaly_detected',
                'timestamp': result.timestamp.isoformat(),
                'anomaly_score': result.anomaly_score,
                'model_confidence': result.model_confidence,
                'data': result.data,
                'severity': 'high' if result.anomaly_score > 0.8 else 'medium'
            }
            
            # Send to alerts topic
            alert_topic = f"{self.output_topic}_alerts"
            self.producer.send(alert_topic, alert_data)
            
        except Exception as e:
            logger.error(f"Failed to send anomaly alert: {e}")

class PulsarStreaming:
    """Apache Pulsar integration for streaming anomaly detection."""
    
    def __init__(self, detector: StreamingDetector,
                 service_url: str = "pulsar://localhost:6650",
                 input_topic: str = "persistent://public/default/anomaly_input",
                 output_topic: str = "persistent://public/default/anomaly_output",
                 subscription_name: str = "pynomaly_subscription"):
        """Initialize Pulsar streaming.
        
        Args:
            detector: Streaming detector instance
            service_url: Pulsar service URL
            input_topic: Input topic name
            output_topic: Output topic name
            subscription_name: Subscription name
        """
        if not PULSAR_AVAILABLE:
            raise ImportError("pulsar-client is required for Pulsar integration")
        
        self.detector = detector
        self.service_url = service_url
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.subscription_name = subscription_name
        
        self.client = None
        self.consumer = None
        self.producer = None
        self.is_running = False
    
    def start(self):
        """Start Pulsar streaming."""
        try:
            # Initialize client
            self.client = pulsar.Client(self.service_url)
            
            # Initialize consumer
            self.consumer = self.client.subscribe(
                self.input_topic,
                subscription_name=self.subscription_name
            )
            
            # Initialize producer
            self.producer = self.client.create_producer(self.output_topic)
            
            # Start detector
            self.detector.start()
            self.detector.add_anomaly_callback(self._send_anomaly_result)
            
            self.is_running = True
            
            # Start consuming
            threading.Thread(target=self._consume_messages, daemon=True).start()
            
            logger.info(f"Pulsar streaming started: {self.input_topic} -> {self.output_topic}")
            
        except Exception as e:
            logger.error(f"Failed to start Pulsar streaming: {e}")
            raise
    
    def stop(self):
        """Stop Pulsar streaming."""
        self.is_running = False
        self.detector.stop()
        
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        if self.client:
            self.client.close()
        
        logger.info("Pulsar streaming stopped")
    
    def _consume_messages(self):
        """Consume messages from Pulsar topic."""
        try:
            while self.is_running:
                try:
                    # Receive message with timeout
                    msg = self.consumer.receive(timeout_millis=1000)
                    
                    # Parse message data
                    data = json.loads(msg.data().decode('utf-8'))
                    
                    # Process message
                    result = self.detector.process_sample(data)
                    
                    # Send result if available
                    if result:
                        self._send_result(result)
                    
                    # Acknowledge message
                    self.consumer.acknowledge(msg)
                    
                except pulsar.Timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error processing Pulsar message: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Pulsar consumer error: {e}")
    
    def _send_result(self, result: StreamingResult):
        """Send result to output topic."""
        try:
            result_data = {
                'timestamp': result.timestamp.isoformat(),
                'is_anomaly': result.is_anomaly,
                'anomaly_score': result.anomaly_score,
                'model_confidence': result.model_confidence,
                'processing_time': result.processing_time,
                'model_version': result.model_version,
                'data': result.data
            }
            
            self.producer.send(json.dumps(result_data).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Failed to send Pulsar message: {e}")
    
    def _send_anomaly_result(self, result: StreamingResult):
        """Send anomaly-specific result."""
        # Similar to Kafka implementation
        pass