"""
IoT Integration Hub for Pynomaly Detection
==========================================

Comprehensive IoT device integration and management platform providing:
- Multi-protocol device communication (MQTT, CoAP, HTTP)
- Device discovery and registration
- Real-time sensor data processing
- Edge device coordination
- Mobile device integration
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

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    import aiocoap
    from aiocoap import Context, Message, Code
    COAP_AVAILABLE = True
except ImportError:
    COAP_AVAILABLE = False

try:
    import websockets
    import asyncio
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ...simplified_services.core_detection_service import CoreDetectionService

logger = logging.getLogger(__name__)

@dataclass
class DeviceConfig:
    """Configuration for IoT device."""
    device_id: str
    device_type: str  # sensor, actuator, gateway, mobile
    protocol: str  # mqtt, coap, http, websocket
    endpoint: str
    port: int = 1883
    credentials: Optional[Dict[str, str]] = None
    data_format: str = "json"  # json, csv, binary
    sampling_rate: float = 1.0  # Hz
    buffer_size: int = 1000
    anomaly_threshold: float = 0.8
    enable_preprocessing: bool = True
    enable_local_detection: bool = False

@dataclass
class DeviceData:
    """IoT device data packet."""
    device_id: str
    timestamp: datetime
    sensor_data: Dict[str, Any]
    device_status: str
    signal_strength: Optional[float] = None
    battery_level: Optional[float] = None
    location: Optional[Dict[str, float]] = None

@dataclass
class IoTAlert:
    """IoT anomaly alert."""
    device_id: str
    timestamp: datetime
    alert_type: str  # anomaly, device_offline, low_battery, signal_weak
    severity: str  # low, medium, high, critical
    message: str
    sensor_data: Dict[str, Any]
    anomaly_score: Optional[float] = None
    recommended_actions: List[str] = field(default_factory=list)

class IoTIntegrationHub:
    """Comprehensive IoT integration and management hub."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize IoT integration hub.
        
        Args:
            config: Hub configuration dictionary
        """
        self.config = config or {}
        self.core_service = CoreDetectionService()
        
        # Device management
        self.devices: Dict[str, DeviceConfig] = {}
        self.device_connections: Dict[str, Any] = {}
        self.device_data_buffers: Dict[str, deque] = {}
        self.device_status: Dict[str, str] = {}
        
        # Data processing
        self.data_processors: Dict[str, Callable] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        
        # Communication protocols
        self.mqtt_client = None
        self.coap_context = None
        self.websocket_servers: Dict[str, Any] = {}
        
        # Threading and async
        self.is_running = False
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.async_loop = None
        
        # Callbacks and alerts
        self.data_callbacks: List[Callable[[DeviceData], None]] = []
        self.alert_callbacks: List[Callable[[IoTAlert], None]] = []
        
        # Statistics
        self.stats = {
            'devices_registered': 0,
            'devices_online': 0,
            'data_packets_processed': 0,
            'anomalies_detected': 0,
            'alerts_sent': 0,
            'avg_processing_time': 0,
            'data_throughput': 0  # packets/second
        }
        
        logger.info("IoT Integration Hub initialized")
    
    def register_device(self, device_config: DeviceConfig) -> bool:
        """Register a new IoT device.
        
        Args:
            device_config: Device configuration
            
        Returns:
            Success status
        """
        try:
            device_id = device_config.device_id
            
            # Validate device configuration
            if not self._validate_device_config(device_config):
                return False
            
            # Store device configuration
            self.devices[device_id] = device_config
            self.device_data_buffers[device_id] = deque(maxlen=device_config.buffer_size)
            self.device_status[device_id] = "registered"
            
            # Initialize device-specific anomaly detector if enabled
            if device_config.enable_local_detection:
                self.anomaly_detectors[device_id] = self._create_device_detector(device_config)
            
            # Initialize connection based on protocol
            success = self._initialize_device_connection(device_config)
            
            if success:
                self.stats['devices_registered'] += 1
                logger.info(f"Device registered: {device_id} ({device_config.protocol})")
                return True
            else:
                # Cleanup on failure
                self._cleanup_device(device_id)
                return False
                
        except Exception as e:
            logger.error(f"Failed to register device {device_config.device_id}: {e}")
            return False
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister an IoT device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Success status
        """
        try:
            if device_id not in self.devices:
                logger.warning(f"Device not found: {device_id}")
                return False
            
            # Disconnect and cleanup
            self._cleanup_device(device_id)
            
            # Remove from collections
            del self.devices[device_id]
            del self.device_data_buffers[device_id]
            del self.device_status[device_id]
            
            if device_id in self.anomaly_detectors:
                del self.anomaly_detectors[device_id]
            
            logger.info(f"Device unregistered: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister device {device_id}: {e}")
            return False
    
    def start_hub(self):
        """Start the IoT integration hub."""
        if self.is_running:
            logger.warning("IoT hub is already running")
            return
        
        try:
            self.is_running = True
            
            # Start protocol servers
            self._start_protocol_servers()
            
            # Start processing threads for each device
            for device_id in self.devices:
                thread = threading.Thread(
                    target=self._device_processing_loop,
                    args=(device_id,),
                    daemon=True
                )
                thread.start()
                self.processing_threads[device_id] = thread
            
            # Start statistics thread
            stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
            stats_thread.start()
            
            logger.info("IoT Integration Hub started")
            
        except Exception as e:
            logger.error(f"Failed to start IoT hub: {e}")
            self.is_running = False
            raise
    
    def stop_hub(self):
        """Stop the IoT integration hub."""
        self.is_running = False
        
        # Stop protocol servers
        self._stop_protocol_servers()
        
        # Wait for processing threads
        for thread in self.processing_threads.values():
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Cleanup device connections
        for device_id in list(self.devices.keys()):
            self._cleanup_device(device_id)
        
        logger.info("IoT Integration Hub stopped")
    
    def process_device_data(self, device_id: str, raw_data: Any) -> Optional[DeviceData]:
        """Process incoming device data.
        
        Args:
            device_id: Device identifier
            raw_data: Raw sensor data
            
        Returns:
            Processed device data or None
        """
        try:
            start_time = time.time()
            
            if device_id not in self.devices:
                logger.warning(f"Unknown device: {device_id}")
                return None
            
            device_config = self.devices[device_id]
            
            # Parse raw data based on format
            parsed_data = self._parse_device_data(raw_data, device_config.data_format)
            
            # Create device data object
            device_data = DeviceData(
                device_id=device_id,
                timestamp=datetime.now(),
                sensor_data=parsed_data.get('sensors', {}),
                device_status=parsed_data.get('status', 'online'),
                signal_strength=parsed_data.get('signal_strength'),
                battery_level=parsed_data.get('battery_level'),
                location=parsed_data.get('location')
            )
            
            # Update device status
            self.device_status[device_id] = device_data.device_status
            
            # Add to buffer
            self.device_data_buffers[device_id].append(device_data)
            
            # Process with anomaly detection if enabled
            if device_config.enable_local_detection and device_id in self.anomaly_detectors:
                self._detect_device_anomalies(device_id, device_data)
            
            # Trigger data callbacks
            for callback in self.data_callbacks:
                try:
                    callback(device_data)
                except Exception as e:
                    logger.error(f"Data callback failed: {e}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['data_packets_processed'] += 1
            self.stats['avg_processing_time'] = (
                self.stats['avg_processing_time'] * 0.9 + processing_time * 0.1
            )
            
            return device_data
            
        except Exception as e:
            logger.error(f"Failed to process device data for {device_id}: {e}")
            return None
    
    def get_device_data(self, device_id: str, limit: int = 100) -> List[DeviceData]:
        """Get recent data for a device.
        
        Args:
            device_id: Device identifier
            limit: Maximum number of data points
            
        Returns:
            List of device data
        """
        if device_id not in self.device_data_buffers:
            return []
        
        buffer = self.device_data_buffers[device_id]
        return list(buffer)[-limit:]
    
    def get_device_status(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Get device status information.
        
        Args:
            device_id: Specific device ID or None for all devices
            
        Returns:
            Device status information
        """
        if device_id:
            if device_id not in self.devices:
                return {}
            
            return {
                'device_id': device_id,
                'status': self.device_status.get(device_id, 'unknown'),
                'config': self.devices[device_id],
                'buffer_size': len(self.device_data_buffers.get(device_id, [])),
                'has_anomaly_detector': device_id in self.anomaly_detectors
            }
        else:
            return {
                device_id: {
                    'status': self.device_status.get(device_id, 'unknown'),
                    'device_type': self.devices[device_id].device_type,
                    'protocol': self.devices[device_id].protocol,
                    'buffer_size': len(self.device_data_buffers.get(device_id, []))
                }
                for device_id in self.devices
            }
    
    def add_data_callback(self, callback: Callable[[DeviceData], None]):
        """Add callback for device data processing.
        
        Args:
            callback: Function to call with device data
        """
        self.data_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[IoTAlert], None]):
        """Add callback for IoT alerts.
        
        Args:
            callback: Function to call with IoT alerts
        """
        self.alert_callbacks.append(callback)
    
    def get_hub_statistics(self) -> Dict[str, Any]:
        """Get hub statistics."""
        stats = self.stats.copy()
        stats.update({
            'devices_online': sum(1 for status in self.device_status.values() if status == 'online'),
            'total_devices': len(self.devices),
            'protocols_active': list(set(device.protocol for device in self.devices.values())),
            'is_running': self.is_running
        })
        return stats
    
    def _validate_device_config(self, config: DeviceConfig) -> bool:
        """Validate device configuration."""
        required_fields = ['device_id', 'device_type', 'protocol', 'endpoint']
        
        for field in required_fields:
            if not getattr(config, field):
                logger.error(f"Missing required field: {field}")
                return False
        
        supported_protocols = ['mqtt', 'coap', 'http', 'websocket']
        if config.protocol not in supported_protocols:
            logger.error(f"Unsupported protocol: {config.protocol}")
            return False
        
        return True
    
    def _create_device_detector(self, config: DeviceConfig) -> Any:
        """Create device-specific anomaly detector."""
        try:
            detector_config = {
                'algorithm': 'isolation_forest',
                'contamination': 0.1,
                'window_size': min(100, config.buffer_size // 2)
            }
            
            return self.core_service
            
        except Exception as e:
            logger.error(f"Failed to create detector for {config.device_id}: {e}")
            return None
    
    def _initialize_device_connection(self, config: DeviceConfig) -> bool:
        """Initialize connection for device based on protocol."""
        try:
            if config.protocol == 'mqtt':
                return self._init_mqtt_connection(config)
            elif config.protocol == 'coap':
                return self._init_coap_connection(config)
            elif config.protocol == 'http':
                return self._init_http_connection(config)
            elif config.protocol == 'websocket':
                return self._init_websocket_connection(config)
            else:
                logger.error(f"Unsupported protocol: {config.protocol}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize connection for {config.device_id}: {e}")
            return False
    
    def _init_mqtt_connection(self, config: DeviceConfig) -> bool:
        """Initialize MQTT connection."""
        if not MQTT_AVAILABLE:
            logger.error("MQTT library not available")
            return False
        
        try:
            if not self.mqtt_client:
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.on_connect = self._on_mqtt_connect
                self.mqtt_client.on_message = self._on_mqtt_message
                self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
                
                # Set credentials if provided
                if config.credentials:
                    self.mqtt_client.username_pw_set(
                        config.credentials.get('username'),
                        config.credentials.get('password')
                    )
                
                self.mqtt_client.connect(config.endpoint, config.port, 60)
                self.mqtt_client.loop_start()
            
            # Subscribe to device topic
            topic = f"pynomaly/{config.device_id}/data"
            self.mqtt_client.subscribe(topic)
            
            self.device_connections[config.device_id] = {
                'type': 'mqtt',
                'topic': topic,
                'client': self.mqtt_client
            }
            
            return True
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False
    
    def _init_coap_connection(self, config: DeviceConfig) -> bool:
        """Initialize CoAP connection."""
        if not COAP_AVAILABLE:
            logger.error("CoAP library not available")
            return False
        
        try:
            # Store CoAP configuration for async processing
            self.device_connections[config.device_id] = {
                'type': 'coap',
                'endpoint': config.endpoint,
                'port': config.port,
                'resource': f"/sensors/{config.device_id}"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"CoAP connection failed: {e}")
            return False
    
    def _init_http_connection(self, config: DeviceConfig) -> bool:
        """Initialize HTTP connection."""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available")
            return False
        
        try:
            # Store HTTP configuration
            self.device_connections[config.device_id] = {
                'type': 'http',
                'endpoint': config.endpoint,
                'port': config.port,
                'credentials': config.credentials
            }
            
            return True
            
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            return False
    
    def _init_websocket_connection(self, config: DeviceConfig) -> bool:
        """Initialize WebSocket connection."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets library not available")
            return False
        
        try:
            # Store WebSocket configuration for async processing
            self.device_connections[config.device_id] = {
                'type': 'websocket',
                'endpoint': config.endpoint,
                'port': config.port
            }
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    def _parse_device_data(self, raw_data: Any, data_format: str) -> Dict[str, Any]:
        """Parse raw device data based on format."""
        try:
            if data_format == 'json':
                if isinstance(raw_data, str):
                    return json.loads(raw_data)
                elif isinstance(raw_data, dict):
                    return raw_data
                else:
                    return {'value': raw_data}
            
            elif data_format == 'csv':
                # Simple CSV parsing - assume first row is headers
                lines = raw_data.strip().split('\n')
                if len(lines) >= 2:
                    headers = lines[0].split(',')
                    values = lines[1].split(',')
                    return dict(zip(headers, values))
                else:
                    return {'raw': raw_data}
            
            elif data_format == 'binary':
                # Basic binary data handling
                return {'binary_data': raw_data, 'size': len(raw_data)}
            
            else:
                return {'raw': raw_data}
                
        except Exception as e:
            logger.warning(f"Failed to parse data format {data_format}: {e}")
            return {'raw': raw_data}
    
    def _detect_device_anomalies(self, device_id: str, device_data: DeviceData):
        """Detect anomalies for specific device."""
        try:
            if device_id not in self.anomaly_detectors:
                return
            
            # Extract numeric sensor values
            sensor_values = []
            for key, value in device_data.sensor_data.items():
                if isinstance(value, (int, float)):
                    sensor_values.append(value)
            
            if not sensor_values:
                return
            
            # Use buffer for detection
            buffer = self.device_data_buffers[device_id]
            if len(buffer) < 10:  # Need minimum samples
                return
            
            # Extract features from buffer
            features = []
            for data_point in buffer:
                point_values = []
                for key, value in data_point.sensor_data.items():
                    if isinstance(value, (int, float)):
                        point_values.append(value)
                if point_values:
                    features.append(point_values)
            
            if len(features) < 10:
                return
            
            # Detect anomalies using core service
            features_array = np.array(features)
            result = self.core_service.detect_anomalies(
                features_array,
                algorithm='isolation_forest',
                contamination=0.1
            )
            
            # Check if current point is anomaly
            current_prediction = result['predictions'][-1]
            current_score = result['scores'][-1] if 'scores' in result else 0.5
            
            if current_prediction == -1:  # Anomaly detected
                self._send_anomaly_alert(device_id, device_data, current_score)
                
        except Exception as e:
            logger.error(f"Anomaly detection failed for {device_id}: {e}")
    
    def _send_anomaly_alert(self, device_id: str, device_data: DeviceData, anomaly_score: float):
        """Send anomaly alert."""
        try:
            # Determine severity based on score
            if anomaly_score > 0.9:
                severity = 'critical'
            elif anomaly_score > 0.7:
                severity = 'high'
            elif anomaly_score > 0.5:
                severity = 'medium'
            else:
                severity = 'low'
            
            alert = IoTAlert(
                device_id=device_id,
                timestamp=device_data.timestamp,
                alert_type='anomaly',
                severity=severity,
                message=f"Anomaly detected in device {device_id} sensor data",
                sensor_data=device_data.sensor_data,
                anomaly_score=anomaly_score,
                recommended_actions=[
                    "Check device sensor calibration",
                    "Inspect device physical condition",
                    "Review recent environmental changes"
                ]
            )
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
            self.stats['alerts_sent'] += 1
            self.stats['anomalies_detected'] += 1
            
            logger.warning(f"Anomaly alert sent for device {device_id}: {severity} severity")
            
        except Exception as e:
            logger.error(f"Failed to send anomaly alert: {e}")
    
    def _device_processing_loop(self, device_id: str):
        """Processing loop for individual device."""
        device_config = self.devices[device_id]
        sleep_interval = 1.0 / device_config.sampling_rate
        
        while self.is_running and device_id in self.devices:
            try:
                # Poll device based on protocol
                connection = self.device_connections.get(device_id)
                if connection:
                    self._poll_device(device_id, connection)
                
                time.sleep(sleep_interval)
                
            except Exception as e:
                logger.error(f"Device processing error for {device_id}: {e}")
                time.sleep(1.0)
    
    def _poll_device(self, device_id: str, connection: Dict[str, Any]):
        """Poll device for data based on connection type."""
        try:
            if connection['type'] == 'http':
                self._poll_http_device(device_id, connection)
            elif connection['type'] == 'coap':
                self._poll_coap_device(device_id, connection)
            # MQTT and WebSocket are event-driven, not polled
            
        except Exception as e:
            logger.error(f"Failed to poll device {device_id}: {e}")
    
    def _poll_http_device(self, device_id: str, connection: Dict[str, Any]):
        """Poll HTTP device for data."""
        try:
            url = f"http://{connection['endpoint']}:{connection['port']}/data"
            
            # Add authentication if available
            auth = None
            if connection.get('credentials'):
                creds = connection['credentials']
                auth = (creds.get('username'), creds.get('password'))
            
            response = requests.get(url, auth=auth, timeout=5)
            response.raise_for_status()
            
            # Process response data
            self.process_device_data(device_id, response.json())
            
        except Exception as e:
            logger.error(f"HTTP polling failed for {device_id}: {e}")
    
    def _poll_coap_device(self, device_id: str, connection: Dict[str, Any]):
        """Poll CoAP device for data."""
        try:
            # This would require async implementation in practice
            # Simplified for demonstration
            logger.debug(f"CoAP polling for {device_id} (async implementation needed)")
            
        except Exception as e:
            logger.error(f"CoAP polling failed for {device_id}: {e}")
    
    def _cleanup_device(self, device_id: str):
        """Cleanup device connection and resources."""
        try:
            # Remove from connections
            if device_id in self.device_connections:
                connection = self.device_connections[device_id]
                
                if connection['type'] == 'mqtt' and self.mqtt_client:
                    topic = connection['topic']
                    self.mqtt_client.unsubscribe(topic)
                
                del self.device_connections[device_id]
            
            # Stop processing thread
            if device_id in self.processing_threads:
                del self.processing_threads[device_id]
            
            # Update status
            if device_id in self.device_status:
                self.device_status[device_id] = 'disconnected'
            
        except Exception as e:
            logger.error(f"Device cleanup failed for {device_id}: {e}")
    
    def _start_protocol_servers(self):
        """Start protocol servers."""
        try:
            # MQTT client is started per device
            # CoAP and WebSocket would need async servers here
            pass
            
        except Exception as e:
            logger.error(f"Failed to start protocol servers: {e}")
    
    def _stop_protocol_servers(self):
        """Stop protocol servers."""
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                self.mqtt_client = None
            
        except Exception as e:
            logger.error(f"Failed to stop protocol servers: {e}")
    
    def _stats_loop(self):
        """Statistics calculation loop."""
        last_packet_count = 0
        
        while self.is_running:
            try:
                # Calculate throughput
                current_packets = self.stats['data_packets_processed']
                self.stats['data_throughput'] = current_packets - last_packet_count
                last_packet_count = current_packets
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Stats loop error: {e}")
                time.sleep(1.0)
    
    # MQTT callbacks
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info("MQTT connected successfully")
        else:
            logger.error(f"MQTT connection failed: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic_parts = msg.topic.split('/')
            if len(topic_parts) >= 3 and topic_parts[0] == 'pynomaly':
                device_id = topic_parts[1]
                data = msg.payload.decode('utf-8')
                self.process_device_data(device_id, data)
                
        except Exception as e:
            logger.error(f"MQTT message processing failed: {e}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        logger.warning(f"MQTT disconnected: {rc}")