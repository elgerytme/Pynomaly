"""
Platform Extensions for Pynomaly Detection
===========================================

This module provides comprehensive platform extensions including:
- Time series forecasting and anomaly detection
- Real-time streaming analytics
- Edge computing deployment capabilities
- Mobile and IoT integration
- Data pipeline orchestration
"""

from .time_series.time_series_detector import TimeSeriesDetector
from .streaming.streaming_detector import StreamingDetector, KafkaStreaming, PulsarStreaming
from .edge_computing.edge_deployment_manager import EdgeDeploymentManager, LiteModelOptimizer
from .iot.iot_integration_hub import IoTIntegrationHub
from .iot.mqtt_detector import MQTTDetector
from .iot.coap_detector import CoAPDetector
from .data_pipeline.data_pipeline_orchestrator import DataPipelineOrchestrator
from .data_pipeline.airflow_integration import AirflowIntegration

__all__ = [
    # Time Series
    'TimeSeriesDetector',
    
    # Streaming
    'StreamingDetector',
    'KafkaStreaming',
    'PulsarStreaming',
    
    # Edge Computing
    'EdgeDeploymentManager',
    'LiteModelOptimizer',
    
    # IoT Integration
    'IoTIntegrationHub',
    'MQTTDetector',
    'CoAPDetector',
    
    # Data Pipeline
    'DataPipelineOrchestrator',
    'AirflowIntegration'
]

# Platform Extensions Version
__version__ = "1.0.0"

def get_platform_extensions_info():
    """Get platform extensions information."""
    return {
        "version": __version__,
        "capabilities": {
            "time_series": "Time series anomaly detection and forecasting",
            "streaming": "Real-time streaming analytics with Kafka/Pulsar",
            "edge_computing": "Lightweight models for edge deployment",
            "iot": "IoT device integration with MQTT/CoAP protocols",
            "data_pipeline": "Pipeline orchestration with Airflow integration"
        },
        "supported_protocols": [
            "HTTP/HTTPS", "WebSocket", "gRPC", "MQTT", "CoAP",
            "Kafka", "Pulsar", "Redis Streams", "Apache Arrow"
        ]
    }