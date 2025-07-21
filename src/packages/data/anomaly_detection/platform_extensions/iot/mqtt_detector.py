"""
MQTT Anomaly Detection for Pynomaly Detection
==============================================

Specialized MQTT integration for IoT anomaly detection with:
- MQTT broker connectivity
- Topic-based data routing
- Real-time message processing
- QoS handling and reliability
"""

import logging
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import numpy as np

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from ...simplified_services.core_detection_service import CoreDetectionService

logger = logging.getLogger(__name__)

@dataclass
class MQTTConfig:
    """MQTT detector configuration."""
    broker_host: str = "localhost"
    broker_port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: str = "pynomaly_detector"
    
    # Topic configuration
    input_topics: List[str] = None
    output_topic: str = "pynomaly/anomalies"
    status_topic: str = "pynomaly/status"
    
    # QoS and reliability
    qos: int = 1
    retain: bool = False
    clean_session: bool = True
    keep_alive: int = 60
    
    # Detection parameters
    buffer_size: int = 1000
    batch_size: int = 10
    detection_interval: float = 1.0
    anomaly_threshold: float = 0.8
    
    # Data processing
    json_path: Optional[str] = None  # JSONPath for extracting values
    timestamp_field: str = "timestamp"
    value_fields: List[str] = None

@dataclass
class MQTTMessage:
    """MQTT message wrapper."""
    topic: str
    payload: Any
    timestamp: datetime
    qos: int
    retain: bool
    message_id: Optional[int] = None

class MQTTDetector:
    """MQTT-based anomaly detector for IoT devices."""
    
    def __init__(self, config: MQTTConfig = None):
        """Initialize MQTT detector.
        
        Args:
            config: MQTT configuration
        """
        self.config = config or MQTTConfig()
        if self.config.input_topics is None:
            self.config.input_topics = ["sensors/+/data"]
        if self.config.value_fields is None:
            self.config.value_fields = ["value", "temperature", "humidity", "pressure"]
        
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt is required for MQTT functionality")
        
        self.core_service = CoreDetectionService()
        
        # MQTT client
        self.client = None
        self.is_connected = False
        self.is_running = False
        
        # Data management
        self.message_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.processed_data: deque = deque(maxlen=self.config.buffer_size)
        self.anomaly_history: deque = deque(maxlen=1000)
        
        # Topic-specific buffers
        self.topic_buffers: Dict[str, deque] = {}
        
        # Threading
        self.processing_thread = None
        self.lock = threading.RLock()
        
        # Callbacks
        self.message_callbacks: List[Callable[[MQTTMessage], None]] = []
        self.anomaly_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'anomalies_detected': 0,
            'connection_errors': 0,
            'processing_errors': 0,
            'avg_processing_time': 0,
            'active_topics': set(),
            'last_message_time': None
        }
        
        logger.info(f"MQTT Detector initialized for broker {self.config.broker_host}:{self.config.broker_port}")
    
    def connect(self) -> bool:
        """Connect to MQTT broker.
        
        Returns:
            Connection success status
        """
        try:
            if self.is_connected:
                logger.warning("MQTT client already connected")
                return True
            
            # Create client
            self.client = mqtt.Client(
                client_id=self.config.client_id,
                clean_session=self.config.clean_session
            )
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.on_publish = self._on_publish
            self.client.on_subscribe = self._on_subscribe
            
            # Set credentials if provided
            if self.config.username and self.config.password:
                self.client.username_pw_set(self.config.username, self.config.password)
            
            # Connect to broker
            self.client.connect(
                self.config.broker_host,
                self.config.broker_port,
                self.config.keep_alive
            )
            
            # Start network loop
            self.client.loop_start()
            
            # Wait for connection
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < 10:
                time.sleep(0.1)
            
            if self.is_connected:
                logger.info("MQTT client connected successfully")
                return True
            else:
                logger.error("MQTT connection timeout")
                return False
                
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            self.stats['connection_errors'] += 1
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        try:
            if self.client and self.is_connected:
                self.client.loop_stop()
                self.client.disconnect()
            self.is_connected = False
            logger.info("MQTT client disconnected")
            
        except Exception as e:
            logger.error(f"MQTT disconnection error: {e}")
    
    def start_detection(self):
        """Start anomaly detection."""
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("Cannot start detection without MQTT connection")
        
        if self.is_running:
            logger.warning("Detection already running")
            return
        
        try:
            # Subscribe to input topics
            for topic in self.config.input_topics:
                self.client.subscribe(topic, qos=self.config.qos)
                logger.info(f"Subscribed to topic: {topic}")
            
            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Publish status
            self._publish_status("started")
            
            logger.info("MQTT anomaly detection started")
            
        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
            self.is_running = False
            raise
    
    def stop_detection(self):
        """Stop anomaly detection."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Unsubscribe from topics
        if self.client:
            for topic in self.config.input_topics:
                self.client.unsubscribe(topic)
        
        # Publish status
        self._publish_status("stopped")
        
        logger.info("MQTT anomaly detection stopped")
    
    def publish_anomaly(self, anomaly_data: Dict[str, Any]):
        """Publish anomaly detection result.
        
        Args:
            anomaly_data: Anomaly information to publish
        """
        try:
            if not self.is_connected:
                logger.warning("Cannot publish - MQTT not connected")
                return
            
            payload = json.dumps(anomaly_data)
            result = self.client.publish(
                self.config.output_topic,
                payload,
                qos=self.config.qos,
                retain=self.config.retain
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Anomaly published to {self.config.output_topic}")
            else:
                logger.error(f"Failed to publish anomaly: {result.rc}")
                
        except Exception as e:
            logger.error(f"Anomaly publishing error: {e}")
    
    def add_message_callback(self, callback: Callable[[MQTTMessage], None]):
        """Add callback for message processing.
        
        Args:
            callback: Function to call with each message
        """
        self.message_callbacks.append(callback)
    
    def add_anomaly_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for anomaly detection.
        
        Args:
            callback: Function to call with anomaly data
        """
        self.anomaly_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'is_connected': self.is_connected,
                'is_running': self.is_running,
                'buffer_size': len(self.message_buffer),
                'processed_data_size': len(self.processed_data),
                'active_topics': list(self.stats['active_topics']),
                'anomaly_history_size': len(self.anomaly_history)
            })
            return stats
    
    def get_recent_anomalies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent anomaly detections.
        
        Args:
            limit: Maximum number of anomalies to return
            
        Returns:
            List of recent anomalies
        """
        with self.lock:
            return list(self.anomaly_history)[-limit:]
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.is_connected = True
            logger.info("MQTT connected successfully")
        else:
            self.is_connected = False
            logger.error(f"MQTT connection failed with code {rc}")
            self.stats['connection_errors'] += 1
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self.is_connected = False
        if rc != 0:
            logger.warning(f"MQTT unexpected disconnection: {rc}")
        else:
            logger.info("MQTT disconnected cleanly")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            # Create message wrapper
            mqtt_msg = MQTTMessage(
                topic=msg.topic,
                payload=msg.payload.decode('utf-8'),
                timestamp=datetime.now(),
                qos=msg.qos,
                retain=msg.retain
            )
            
            # Add to buffer
            with self.lock:
                self.message_buffer.append(mqtt_msg)
                self.stats['messages_received'] += 1
                self.stats['active_topics'].add(msg.topic)
                self.stats['last_message_time'] = mqtt_msg.timestamp
                
                # Add to topic-specific buffer
                if msg.topic not in self.topic_buffers:
                    self.topic_buffers[msg.topic] = deque(maxlen=self.config.buffer_size)
                self.topic_buffers[msg.topic].append(mqtt_msg)
            
            # Trigger message callbacks
            for callback in self.message_callbacks:
                try:
                    callback(mqtt_msg)
                except Exception as e:
                    logger.error(f"Message callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.stats['processing_errors'] += 1
    
    def _on_publish(self, client, userdata, mid):
        """MQTT publish callback."""
        logger.debug(f"Message published: {mid}")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """MQTT subscribe callback."""
        logger.debug(f"Subscribed with QoS: {granted_qos}")
    
    def _processing_loop(self):
        """Main processing loop for anomaly detection."""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Process buffered messages
                if len(self.message_buffer) >= self.config.batch_size:
                    self._process_message_batch()
                
                # Sleep based on detection interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.detection_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(1.0)
    
    def _process_message_batch(self):
        """Process a batch of messages for anomaly detection."""
        try:
            with self.lock:
                if len(self.message_buffer) == 0:
                    return
                
                # Get batch of messages
                batch_size = min(self.config.batch_size, len(self.message_buffer))
                messages = [self.message_buffer.popleft() for _ in range(batch_size)]
            
            # Extract features from messages
            features = []
            message_metadata = []
            
            for msg in messages:
                try:
                    # Parse message payload
                    if isinstance(msg.payload, str):
                        data = json.loads(msg.payload)
                    else:
                        data = msg.payload
                    
                    # Extract feature values
                    values = self._extract_features(data)
                    if values:
                        features.append(values)
                        message_metadata.append({
                            'topic': msg.topic,
                            'timestamp': msg.timestamp,
                            'original_data': data
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features from message: {e}")
                    continue
            
            if not features:
                return
            
            # Convert to numpy array
            features_array = np.array(features)
            
            # Detect anomalies
            result = self.core_service.detect_anomalies(
                features_array,
                algorithm='isolation_forest',
                contamination=0.1
            )
            
            predictions = result['predictions']
            scores = result.get('scores', np.zeros(len(predictions)))
            
            # Process results
            for i, (prediction, score) in enumerate(zip(predictions, scores)):
                metadata = message_metadata[i]
                
                # Store processed data
                processed_point = {
                    'timestamp': metadata['timestamp'],
                    'topic': metadata['topic'],
                    'features': features[i],
                    'is_anomaly': prediction == -1,
                    'anomaly_score': abs(score),
                    'original_data': metadata['original_data']
                }
                
                self.processed_data.append(processed_point)
                
                # Handle anomalies
                if prediction == -1 and abs(score) > self.config.anomaly_threshold:
                    self._handle_anomaly(processed_point)
            
            # Update statistics
            self.stats['messages_processed'] += len(features)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.stats['processing_errors'] += 1
    
    def _extract_features(self, data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numerical features from message data.
        
        Args:
            data: Parsed message data
            
        Returns:
            List of feature values or None
        """
        try:
            features = []
            
            # Extract specified value fields
            for field in self.config.value_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (int, float)):
                        features.append(float(value))
                    elif isinstance(value, str):
                        try:
                            features.append(float(value))
                        except ValueError:
                            continue
            
            # If no specific fields found, try to extract all numeric values
            if not features:
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        features.append(float(value))
            
            return features if features else None
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def _handle_anomaly(self, anomaly_data: Dict[str, Any]):
        """Handle detected anomaly.
        
        Args:
            anomaly_data: Anomaly information
        """
        try:
            # Add to history
            self.anomaly_history.append(anomaly_data)
            self.stats['anomalies_detected'] += 1
            
            # Create anomaly report
            anomaly_report = {
                'timestamp': anomaly_data['timestamp'].isoformat(),
                'topic': anomaly_data['topic'],
                'anomaly_score': anomaly_data['anomaly_score'],
                'features': anomaly_data['features'],
                'original_data': anomaly_data['original_data'],
                'detector': 'mqtt_detector',
                'severity': self._determine_severity(anomaly_data['anomaly_score'])
            }
            
            # Publish anomaly
            self.publish_anomaly(anomaly_report)
            
            # Trigger anomaly callbacks
            for callback in self.anomaly_callbacks:
                try:
                    callback(anomaly_report)
                except Exception as e:
                    logger.error(f"Anomaly callback failed: {e}")
            
            logger.warning(f"Anomaly detected on topic {anomaly_data['topic']}: score={anomaly_data['anomaly_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Anomaly handling failed: {e}")
    
    def _determine_severity(self, score: float) -> str:
        """Determine anomaly severity based on score.
        
        Args:
            score: Anomaly score
            
        Returns:
            Severity level
        """
        if score > 0.9:
            return 'critical'
        elif score > 0.7:
            return 'high'
        elif score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _publish_status(self, status: str):
        """Publish detector status.
        
        Args:
            status: Status message
        """
        try:
            if not self.is_connected:
                return
            
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'detector': 'mqtt_detector',
                'status': status,
                'statistics': self.get_statistics()
            }
            
            payload = json.dumps(status_data)
            self.client.publish(
                self.config.status_topic,
                payload,
                qos=1,
                retain=True
            )
            
        except Exception as e:
            logger.error(f"Status publishing failed: {e}")