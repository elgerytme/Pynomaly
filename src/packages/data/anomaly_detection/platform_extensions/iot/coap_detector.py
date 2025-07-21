"""
CoAP Anomaly Detection for Pynomaly Detection
==============================================

Specialized CoAP integration for IoT anomaly detection with:
- CoAP client and server functionality
- Observe pattern for real-time updates
- Resource discovery and management
- Constrained device optimization
"""

import logging
import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import numpy as np

try:
    import aiocoap
    from aiocoap import Context, Message, Code
    from aiocoap.resource import Site, Resource
    COAP_AVAILABLE = True
except ImportError:
    COAP_AVAILABLE = False

from ...simplified_services.core_detection_service import CoreDetectionService

logger = logging.getLogger(__name__)

@dataclass
class CoAPConfig:
    """CoAP detector configuration."""
    server_host: str = "localhost"
    server_port: int = 5683
    client_timeout: float = 30.0
    
    # Resource configuration
    observation_resources: List[str] = None
    polling_resources: List[str] = None
    polling_interval: float = 10.0
    
    # Detection parameters
    buffer_size: int = 1000
    batch_size: int = 10
    detection_interval: float = 5.0
    anomaly_threshold: float = 0.8
    
    # Data processing
    value_fields: List[str] = None
    timestamp_field: str = "timestamp"
    device_id_field: str = "device_id"
    
    # Constrained device optimization
    enable_caching: bool = True
    max_age: int = 60
    block_size: int = 1024

@dataclass
class CoAPMessage:
    """CoAP message wrapper."""
    resource_uri: str
    payload: Any
    timestamp: datetime
    message_code: int
    device_id: Optional[str] = None
    observe_option: Optional[int] = None

class CoAPResource(Resource):
    """Custom CoAP resource for anomaly detection."""
    
    def __init__(self, detector_instance, resource_name: str):
        super().__init__()
        self.detector = detector_instance
        self.resource_name = resource_name
        self.data_buffer = deque(maxlen=100)
    
    async def render_get(self, request):
        """Handle GET requests."""
        try:
            # Return recent data or statistics
            response_data = {
                'resource': self.resource_name,
                'timestamp': datetime.now().isoformat(),
                'buffer_size': len(self.data_buffer),
                'statistics': self.detector.get_statistics()
            }
            
            payload = json.dumps(response_data).encode('utf-8')
            return Message(payload=payload, code=Code.CONTENT)
            
        except Exception as e:
            logger.error(f"CoAP GET error: {e}")
            return Message(code=Code.INTERNAL_SERVER_ERROR)
    
    async def render_post(self, request):
        """Handle POST requests with sensor data."""
        try:
            # Parse incoming data
            payload = request.payload.decode('utf-8')
            data = json.loads(payload)
            
            # Create CoAP message
            coap_msg = CoAPMessage(
                resource_uri=f"/{self.resource_name}",
                payload=data,
                timestamp=datetime.now(),
                message_code=request.code,
                device_id=data.get(self.detector.config.device_id_field)
            )
            
            # Process data
            self.detector._process_coap_message(coap_msg)
            
            return Message(code=Code.CREATED)
            
        except Exception as e:
            logger.error(f"CoAP POST error: {e}")
            return Message(code=Code.BAD_REQUEST)

class CoAPDetector:
    """CoAP-based anomaly detector for constrained IoT devices."""
    
    def __init__(self, config: CoAPConfig = None):
        """Initialize CoAP detector.
        
        Args:
            config: CoAP configuration
        """
        self.config = config or CoAPConfig()
        if self.config.observation_resources is None:
            self.config.observation_resources = ["/sensors/temperature", "/sensors/humidity"]
        if self.config.polling_resources is None:
            self.config.polling_resources = ["/devices/status"]
        if self.config.value_fields is None:
            self.config.value_fields = ["value", "temperature", "humidity", "pressure"]
        
        if not COAP_AVAILABLE:
            raise ImportError("aiocoap is required for CoAP functionality")
        
        self.core_service = CoreDetectionService()
        
        # CoAP context and server
        self.context = None
        self.server_site = None
        self.server_task = None
        
        # Client management
        self.observed_resources: Dict[str, Any] = {}
        self.polling_tasks: Dict[str, asyncio.Task] = {}
        
        # Data management
        self.message_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.processed_data: deque = deque(maxlen=self.config.buffer_size)
        self.anomaly_history: deque = deque(maxlen=1000)
        
        # Resource-specific buffers
        self.resource_buffers: Dict[str, deque] = {}
        
        # Threading and async
        self.is_running = False
        self.async_loop = None
        self.processing_thread = None
        self.lock = threading.RLock()
        
        # Callbacks
        self.message_callbacks: List[Callable[[CoAPMessage], None]] = []
        self.anomaly_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'anomalies_detected': 0,
            'observation_errors': 0,
            'polling_errors': 0,
            'avg_processing_time': 0,
            'active_resources': set(),
            'last_message_time': None,
            'observed_devices': set()
        }
        
        logger.info(f"CoAP Detector initialized on {self.config.server_host}:{self.config.server_port}")
    
    def start_detection(self):
        """Start CoAP anomaly detection."""
        if self.is_running:
            logger.warning("CoAP detection already running")
            return
        
        try:
            self.is_running = True
            
            # Start async event loop in separate thread
            self.processing_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.processing_thread.start()
            
            # Wait for initialization
            time.sleep(1.0)
            
            logger.info("CoAP anomaly detection started")
            
        except Exception as e:
            logger.error(f"Failed to start CoAP detection: {e}")
            self.is_running = False
            raise
    
    def stop_detection(self):
        """Stop CoAP anomaly detection."""
        self.is_running = False
        
        # Cancel async tasks
        if self.async_loop and not self.async_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._cleanup_async(), self.async_loop)
        
        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        logger.info("CoAP anomaly detection stopped")
    
    def add_observation_resource(self, uri: str, device_endpoint: str):
        """Add resource for observation.
        
        Args:
            uri: Resource URI to observe
            device_endpoint: Device endpoint (coap://host:port)
        """
        if self.async_loop:
            asyncio.run_coroutine_threadsafe(
                self._start_observation(uri, device_endpoint),
                self.async_loop
            )
        else:
            logger.warning("Cannot add observation - async loop not running")
    
    def add_polling_resource(self, uri: str, device_endpoint: str):
        """Add resource for polling.
        
        Args:
            uri: Resource URI to poll
            device_endpoint: Device endpoint (coap://host:port)
        """
        if self.async_loop:
            asyncio.run_coroutine_threadsafe(
                self._start_polling(uri, device_endpoint),
                self.async_loop
            )
        else:
            logger.warning("Cannot add polling - async loop not running")
    
    def add_message_callback(self, callback: Callable[[CoAPMessage], None]):
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
                'is_running': self.is_running,
                'buffer_size': len(self.message_buffer),
                'processed_data_size': len(self.processed_data),
                'active_resources': list(self.stats['active_resources']),
                'observed_devices': list(self.stats['observed_devices']),
                'anomaly_history_size': len(self.anomaly_history),
                'observation_count': len(self.observed_resources),
                'polling_count': len(self.polling_tasks)
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
    
    def _run_async_loop(self):
        """Run async event loop in separate thread."""
        try:
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
            
            # Start CoAP context and server
            self.async_loop.run_until_complete(self._initialize_coap())
            
            # Start processing loop
            processing_task = self.async_loop.create_task(self._async_processing_loop())
            
            # Run event loop
            self.async_loop.run_until_complete(processing_task)
            
        except Exception as e:
            logger.error(f"Async loop error: {e}")
        finally:
            if self.async_loop and not self.async_loop.is_closed():
                self.async_loop.close()
    
    async def _initialize_coap(self):
        """Initialize CoAP context and server."""
        try:
            # Create CoAP context
            self.context = await Context.create_client_context()
            
            # Create server site with resources
            self.server_site = Site()
            
            # Add anomaly detection resource
            self.server_site.add_resource(['.well-known', 'core'], self._discovery_resource())
            self.server_site.add_resource(['anomaly', 'detect'], CoAPResource(self, 'detect'))
            self.server_site.add_resource(['anomaly', 'status'], CoAPResource(self, 'status'))
            
            # Start server
            server_endpoint = f"coap://{self.config.server_host}:{self.config.server_port}"
            self.server_task = self.async_loop.create_task(
                Context.create_server_context(self.server_site, bind=(self.config.server_host, self.config.server_port))
            )
            
            logger.info(f"CoAP server started on {server_endpoint}")
            
            # Start observations and polling
            await self._start_default_observations()
            await self._start_default_polling()
            
        except Exception as e:
            logger.error(f"CoAP initialization failed: {e}")
            raise
    
    async def _start_observation(self, uri: str, device_endpoint: str):
        """Start observing a CoAP resource.
        
        Args:
            uri: Resource URI
            device_endpoint: Device endpoint
        """
        try:
            resource_url = f"{device_endpoint}{uri}"
            
            # Create observation request
            request = Message(code=Code.GET, uri=resource_url)
            request.opt.observe = 0
            
            # Start observation
            observation = self.context.request(request)
            
            # Store observation
            observation_key = f"{device_endpoint}{uri}"
            self.observed_resources[observation_key] = observation
            
            # Process observation responses
            async for response in observation.observation:
                await self._handle_observation_response(uri, device_endpoint, response)
            
        except Exception as e:
            logger.error(f"Observation failed for {uri}: {e}")
            self.stats['observation_errors'] += 1
    
    async def _start_polling(self, uri: str, device_endpoint: str):
        """Start polling a CoAP resource.
        
        Args:
            uri: Resource URI
            device_endpoint: Device endpoint
        """
        try:
            async def poll_resource():
                while self.is_running:
                    try:
                        resource_url = f"{device_endpoint}{uri}"
                        request = Message(code=Code.GET, uri=resource_url)
                        
                        response = await asyncio.wait_for(
                            self.context.request(request).response,
                            timeout=self.config.client_timeout
                        )
                        
                        await self._handle_polling_response(uri, device_endpoint, response)
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Polling timeout for {uri}")
                        self.stats['polling_errors'] += 1
                    except Exception as e:
                        logger.error(f"Polling error for {uri}: {e}")
                        self.stats['polling_errors'] += 1
                    
                    await asyncio.sleep(self.config.polling_interval)
            
            # Start polling task
            polling_key = f"{device_endpoint}{uri}"
            self.polling_tasks[polling_key] = self.async_loop.create_task(poll_resource())
            
        except Exception as e:
            logger.error(f"Polling setup failed for {uri}: {e}")
    
    async def _handle_observation_response(self, uri: str, device_endpoint: str, response):
        """Handle observation response.
        
        Args:
            uri: Resource URI
            device_endpoint: Device endpoint
            response: CoAP response
        """
        try:
            if response.payload:
                payload_str = response.payload.decode('utf-8')
                data = json.loads(payload_str)
                
                coap_msg = CoAPMessage(
                    resource_uri=uri,
                    payload=data,
                    timestamp=datetime.now(),
                    message_code=response.code,
                    device_id=data.get(self.config.device_id_field),
                    observe_option=response.opt.observe
                )
                
                self._process_coap_message(coap_msg)
                
        except Exception as e:
            logger.error(f"Observation response handling failed: {e}")
    
    async def _handle_polling_response(self, uri: str, device_endpoint: str, response):
        """Handle polling response.
        
        Args:
            uri: Resource URI
            device_endpoint: Device endpoint
            response: CoAP response
        """
        try:
            if response.payload:
                payload_str = response.payload.decode('utf-8')
                data = json.loads(payload_str)
                
                coap_msg = CoAPMessage(
                    resource_uri=uri,
                    payload=data,
                    timestamp=datetime.now(),
                    message_code=response.code,
                    device_id=data.get(self.config.device_id_field)
                )
                
                self._process_coap_message(coap_msg)
                
        except Exception as e:
            logger.error(f"Polling response handling failed: {e}")
    
    def _process_coap_message(self, message: CoAPMessage):
        """Process incoming CoAP message.
        
        Args:
            message: CoAP message to process
        """
        try:
            # Add to buffer
            with self.lock:
                self.message_buffer.append(message)
                self.stats['messages_received'] += 1
                self.stats['active_resources'].add(message.resource_uri)
                self.stats['last_message_time'] = message.timestamp
                
                if message.device_id:
                    self.stats['observed_devices'].add(message.device_id)
                
                # Add to resource-specific buffer
                if message.resource_uri not in self.resource_buffers:
                    self.resource_buffers[message.resource_uri] = deque(maxlen=self.config.buffer_size)
                self.resource_buffers[message.resource_uri].append(message)
            
            # Trigger message callbacks
            for callback in self.message_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Message callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"CoAP message processing error: {e}")
    
    async def _async_processing_loop(self):
        """Async processing loop for anomaly detection."""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Process buffered messages
                if len(self.message_buffer) >= self.config.batch_size:
                    await self._process_message_batch()
                
                # Sleep based on detection interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.detection_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Async processing loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_message_batch(self):
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
                    # Extract feature values
                    values = self._extract_features(msg.payload)
                    if values:
                        features.append(values)
                        message_metadata.append({
                            'resource_uri': msg.resource_uri,
                            'timestamp': msg.timestamp,
                            'device_id': msg.device_id,
                            'original_data': msg.payload
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features from CoAP message: {e}")
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
                    'resource_uri': metadata['resource_uri'],
                    'device_id': metadata['device_id'],
                    'features': features[i],
                    'is_anomaly': prediction == -1,
                    'anomaly_score': abs(score),
                    'original_data': metadata['original_data']
                }
                
                self.processed_data.append(processed_point)
                
                # Handle anomalies
                if prediction == -1 and abs(score) > self.config.anomaly_threshold:
                    await self._handle_anomaly(processed_point)
            
            # Update statistics
            self.stats['messages_processed'] += len(features)
            
        except Exception as e:
            logger.error(f"CoAP batch processing failed: {e}")
    
    def _extract_features(self, data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numerical features from CoAP message data.
        
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
            logger.warning(f"CoAP feature extraction failed: {e}")
            return None
    
    async def _handle_anomaly(self, anomaly_data: Dict[str, Any]):
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
                'resource_uri': anomaly_data['resource_uri'],
                'device_id': anomaly_data['device_id'],
                'anomaly_score': anomaly_data['anomaly_score'],
                'features': anomaly_data['features'],
                'original_data': anomaly_data['original_data'],
                'detector': 'coap_detector',
                'severity': self._determine_severity(anomaly_data['anomaly_score'])
            }
            
            # Trigger anomaly callbacks
            for callback in self.anomaly_callbacks:
                try:
                    callback(anomaly_report)
                except Exception as e:
                    logger.error(f"CoAP anomaly callback failed: {e}")
            
            logger.warning(f"CoAP anomaly detected on {anomaly_data['resource_uri']}: score={anomaly_data['anomaly_score']:.3f}")
            
        except Exception as e:
            logger.error(f"CoAP anomaly handling failed: {e}")
    
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
    
    async def _start_default_observations(self):
        """Start default observations."""
        for resource_uri in self.config.observation_resources:
            # This would need actual device endpoints
            # await self._start_observation(resource_uri, "coap://device-endpoint")
            pass
    
    async def _start_default_polling(self):
        """Start default polling."""
        for resource_uri in self.config.polling_resources:
            # This would need actual device endpoints
            # await self._start_polling(resource_uri, "coap://device-endpoint")
            pass
    
    async def _cleanup_async(self):
        """Cleanup async resources."""
        try:
            # Cancel observation tasks
            for observation in self.observed_resources.values():
                if hasattr(observation, 'cancel'):
                    observation.cancel()
            
            # Cancel polling tasks
            for task in self.polling_tasks.values():
                task.cancel()
            
            # Close server
            if self.server_task:
                self.server_task.cancel()
            
            # Close context
            if self.context:
                await self.context.shutdown()
            
        except Exception as e:
            logger.error(f"Async cleanup error: {e}")
    
    def _discovery_resource(self):
        """Create resource discovery resource."""
        class DiscoveryResource(Resource):
            async def render_get(self, request):
                """Return available resources."""
                discovery_data = {
                    'resources': [
                        {'uri': '/anomaly/detect', 'ct': 0, 'title': 'Anomaly Detection'},
                        {'uri': '/anomaly/status', 'ct': 0, 'title': 'Detector Status'}
                    ]
                }
                payload = json.dumps(discovery_data).encode('utf-8')
                return Message(payload=payload, code=Code.CONTENT)
        
        return DiscoveryResource()