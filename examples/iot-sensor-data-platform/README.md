# IoT Sensor Data Platform Example

A scalable IoT platform for collecting, processing, and analyzing sensor data from thousands of connected devices, built with event-driven architecture for real-time processing.

## Features

- **Device Management**: Registration and lifecycle management of IoT devices
- **Real-time Data Ingestion**: High-throughput sensor data collection
- **Stream Processing**: Real-time analytics and anomaly detection
- **Time-Series Storage**: Optimized storage for sensor data
- **Rule Engine**: Configurable alerting and automation rules
- **Edge Computing**: Local processing capabilities
- **Device Provisioning**: Secure device onboarding
- **Firmware Updates**: Over-the-air (OTA) update management
- **Data Visualization**: Real-time dashboards and historical analysis

## Architecture

Event-driven microservices with CQRS pattern:

```
src/iot_platform/
├── domain/                    # Core IoT concepts
│   ├── entities/             # Device, Sensor, Reading
│   ├── aggregates/           # DeviceAggregate
│   ├── events/               # Domain events
│   └── value_objects/        # SensorReading, Timestamp
├── application/              # Use cases
│   ├── commands/             # Register device, process reading
│   ├── queries/              # Device status, sensor data
│   ├── handlers/             # Command and event handlers
│   └── services/             # Application services
├── infrastructure/          # External integrations
│   ├── mqtt/                # MQTT broker integration
│   ├── timeseries/          # InfluxDB adapter
│   ├── streaming/           # Apache Kafka processing
│   ├── edge/                # Edge computing gateway
│   └── protocols/           # CoAP, LoRaWAN, etc.
└── presentation/            # API and protocols
    ├── rest/                # Device management API
    ├── mqtt/                # MQTT endpoints
    ├── websocket/           # Real-time data streaming
    └── graphql/             # Flexible data queries
```

## Quick Start

```bash
# Start the IoT platform
cd examples/iot-sensor-data-platform
docker-compose up -d

# Register a device
curl -X POST http://localhost:8000/api/devices \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor-001",
    "device_type": "temperature_humidity",
    "location": {
      "latitude": 37.7749,
      "longitude": -122.4194,
      "address": "San Francisco, CA"
    },
    "metadata": {
      "manufacturer": "ACME Sensors",
      "model": "TH-2000",
      "firmware_version": "1.2.3"
    }
  }'

# Send sensor data via MQTT
mosquitto_pub -h localhost -p 1883 \
  -t "devices/sensor-001/telemetry" \
  -m '{"temperature": 23.5, "humidity": 45.2, "timestamp": "2024-01-01T12:00:00Z"}'
```

## Supported Protocols

### MQTT
```python
# Device telemetry publishing
topic = f"devices/{device_id}/telemetry"
payload = {
    "temperature": 23.5,
    "humidity": 45.2,
    "battery_level": 87,
    "timestamp": "2024-01-01T12:00:00Z"
}
client.publish(topic, json.dumps(payload))

# Device commands
command_topic = f"devices/{device_id}/commands"
command = {
    "command": "set_sampling_rate",
    "parameters": {"rate_seconds": 60}
}
```

### CoAP (Constrained Application Protocol)
```python
# CoAP endpoint for resource-constrained devices
@coap_server.resource("sensors")
class SensorResource(Resource):
    async def post(self, request):
        # Process sensor data from CoAP client
        sensor_data = cbor.loads(request.payload)
        await self.process_sensor_reading(sensor_data)
        return Message(code=CREATED)
```

### LoRaWAN Integration
```python
# LoRaWAN device communication
class LoRaWANGateway:
    async def handle_uplink(self, message: LoRaWANMessage):
        """Process LoRaWAN uplink message"""
        device_eui = message.dev_eui
        payload = message.frm_payload
        
        # Decode payload based on device type
        sensor_data = await self.decode_payload(device_eui, payload)
        
        # Process the sensor reading
        await self.sensor_service.process_reading(
            device_id=device_eui,
            data=sensor_data,
            metadata={
                "rssi": message.rssi,
                "snr": message.snr,
                "gateway_id": message.gateway_id
            }
        )
```

## Real-time Data Processing

### Stream Processing with Apache Kafka
```python
# Kafka Streams for real-time analytics
class SensorDataProcessor:
    async def process_temperature_stream(self):
        """Process temperature sensor data stream"""
        async for record in self.kafka_consumer:
            sensor_data = json.loads(record.value)
            
            # Calculate rolling average
            avg_temp = await self.calculate_rolling_average(
                device_id=sensor_data["device_id"],
                value=sensor_data["temperature"],
                window_minutes=15
            )
            
            # Check for anomalies
            if abs(sensor_data["temperature"] - avg_temp) > 5.0:
                await self.trigger_alert(
                    device_id=sensor_data["device_id"],
                    alert_type="temperature_anomaly",
                    current_value=sensor_data["temperature"],
                    expected_value=avg_temp
                )
            
            # Store aggregated data
            await self.store_aggregated_reading(sensor_data, avg_temp)
```

### Rule Engine
```python
# Configurable alerting rules
class RuleEngine:
    async def evaluate_rules(self, sensor_reading: SensorReading):
        """Evaluate all rules for incoming sensor data"""
        device_rules = await self.rule_repository.get_rules_for_device(
            sensor_reading.device_id
        )
        
        for rule in device_rules:
            if await self.evaluate_condition(rule.condition, sensor_reading):
                await self.execute_action(rule.action, sensor_reading)
    
    async def evaluate_condition(self, condition: RuleCondition, reading: SensorReading) -> bool:
        """Evaluate rule condition"""
        if condition.type == "threshold":
            return getattr(reading, condition.field) > condition.threshold
        elif condition.type == "range":
            value = getattr(reading, condition.field)
            return not (condition.min_value <= value <= condition.max_value)
        elif condition.type == "rate_of_change":
            previous = await self.get_previous_reading(reading.device_id)
            if previous:
                change_rate = abs(reading.value - previous.value) / (reading.timestamp - previous.timestamp).seconds
                return change_rate > condition.max_rate
        
        return False
```

## Time-Series Data Management

### InfluxDB Integration
```python
# Optimized time-series storage
class TimeSeriesRepository:
    async def store_sensor_reading(self, reading: SensorReading):
        """Store sensor reading in InfluxDB"""
        point = Point("sensor_data") \
            .tag("device_id", reading.device_id) \
            .tag("sensor_type", reading.sensor_type) \
            .tag("location", reading.location) \
            .field("value", reading.value) \
            .field("quality", reading.quality) \
            .time(reading.timestamp)
        
        await self.influx_client.write(point)
    
    async def query_historical_data(
        self, 
        device_id: str, 
        start_time: datetime, 
        end_time: datetime,
        aggregation: str = "mean",
        interval: str = "1h"
    ) -> List[SensorReading]:
        """Query historical sensor data with aggregation"""
        query = f'''
        from(bucket: "sensor_data")
        |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
        |> filter(fn: (r) => r["device_id"] == "{device_id}")
        |> aggregateWindow(every: {interval}, fn: {aggregation}, createEmpty: false)
        '''
        
        result = await self.influx_client.query(query)
        return self.convert_to_sensor_readings(result)
```

## Edge Computing

### Edge Gateway
```python
# Local processing at the edge
class EdgeGateway:
    def __init__(self):
        self.local_storage = SQLiteRepository()
        self.ml_models = ModelRegistry()
        self.sync_manager = CloudSyncManager()
    
    async def process_local_data(self, sensor_data: dict):
        """Process data locally when cloud connection is unavailable"""
        # Store locally
        await self.local_storage.store(sensor_data)
        
        # Run local ML inference
        if sensor_data["sensor_type"] == "vibration":
            prediction = await self.ml_models.predict_equipment_failure(sensor_data)
            if prediction.probability > 0.8:
                await self.trigger_local_alert(prediction)
        
        # Sync with cloud when connection is restored
        if await self.sync_manager.is_connected():
            await self.sync_manager.upload_pending_data()
```

### Local Analytics
```python
# Edge analytics for reduced latency
class EdgeAnalytics:
    async def detect_anomalies_locally(self, readings: List[SensorReading]) -> List[Anomaly]:
        """Local anomaly detection using lightweight algorithms"""
        anomalies = []
        
        # Statistical anomaly detection
        for reading in readings:
            z_score = await self.calculate_z_score(reading)
            if abs(z_score) > 3.0:
                anomalies.append(Anomaly(
                    device_id=reading.device_id,
                    timestamp=reading.timestamp,
                    type="statistical_outlier",
                    severity="medium",
                    z_score=z_score
                ))
        
        return anomalies
```

## Device Management

### Device Lifecycle
```python
# Complete device lifecycle management
class DeviceManager:
    async def register_device(self, device_info: DeviceRegistration) -> Device:
        """Register new IoT device"""
        # Generate device credentials
        credentials = await self.generate_device_credentials(device_info.device_id)
        
        # Create device entity
        device = Device(
            device_id=device_info.device_id,
            device_type=device_info.device_type,
            location=device_info.location,
            credentials=credentials,
            status=DeviceStatus.REGISTERED,
            registered_at=datetime.utcnow()
        )
        
        # Store in repository
        await self.device_repository.save(device)
        
        # Publish device registered event
        await self.event_publisher.publish(DeviceRegisteredEvent(device))
        
        return device
    
    async def provision_device(self, device_id: str) -> ProvisioningResult:
        """Provision device with certificates and configuration"""
        device = await self.device_repository.get(device_id)
        
        # Generate X.509 certificates
        cert_bundle = await self.certificate_authority.issue_certificate(device_id)
        
        # Create device configuration
        config = DeviceConfiguration(
            mqtt_broker=self.config.mqtt_broker_url,
            reporting_interval=self.config.default_reporting_interval,
            encryption_enabled=True,
            certificates=cert_bundle
        )
        
        # Update device status
        device.status = DeviceStatus.PROVISIONED
        device.provisioned_at = datetime.utcnow()
        await self.device_repository.save(device)
        
        return ProvisioningResult(
            device_id=device_id,
            configuration=config,
            status="success"
        )
```

### Firmware Updates (OTA)
```python
# Over-the-air firmware updates
class FirmwareUpdateManager:
    async def deploy_firmware_update(
        self, 
        device_ids: List[str], 
        firmware_version: str
    ) -> UpdateCampaign:
        """Deploy firmware update to devices"""
        campaign = UpdateCampaign(
            campaign_id=str(uuid.uuid4()),
            firmware_version=firmware_version,
            target_devices=device_ids,
            status=CampaignStatus.SCHEDULED,
            created_at=datetime.utcnow()
        )
        
        # Schedule updates in batches to avoid network congestion
        for batch in self.create_update_batches(device_ids, batch_size=100):
            await self.schedule_batch_update(batch, firmware_version, delay_minutes=10)
        
        await self.campaign_repository.save(campaign)
        return campaign
    
    async def handle_update_progress(self, device_id: str, progress: UpdateProgress):
        """Handle firmware update progress from device"""
        campaign = await self.get_active_campaign_for_device(device_id)
        
        # Update progress
        campaign.update_device_progress(device_id, progress)
        
        if progress.status == UpdateStatus.COMPLETED:
            # Verify firmware version
            await self.verify_firmware_version(device_id, campaign.firmware_version)
        elif progress.status == UpdateStatus.FAILED:
            # Schedule retry if within retry limit
            await self.schedule_update_retry(device_id, campaign)
        
        await self.campaign_repository.save(campaign)
```

## API Endpoints

### Device Management
- `POST /api/devices` - Register device
- `GET /api/devices/{device_id}` - Get device details
- `PUT /api/devices/{device_id}/provision` - Provision device
- `DELETE /api/devices/{device_id}` - Decommission device

### Sensor Data
- `GET /api/devices/{device_id}/data` - Get sensor data
- `GET /api/devices/{device_id}/data/latest` - Get latest readings
- `POST /api/devices/{device_id}/commands` - Send device command

### Analytics
- `GET /api/analytics/devices/{device_id}/anomalies` - Get anomalies
- `GET /api/analytics/aggregated` - Get aggregated data
- `POST /api/analytics/rules` - Create alerting rule

### Firmware Management
- `POST /api/firmware/upload` - Upload firmware
- `POST /api/firmware/deploy` - Deploy firmware update
- `GET /api/firmware/campaigns/{campaign_id}` - Get update campaign status

## Configuration

```yaml
# MQTT broker settings
mqtt:
  broker_host: "mqtt.iot-platform.com"
  broker_port: 8883
  use_tls: true
  qos_level: 1
  keep_alive: 60

# Time-series database
timeseries:
  influxdb:
    url: "http://influxdb:8086"
    bucket: "sensor_data"
    retention_policy: "30d"
    
# Stream processing
streaming:
  kafka:
    brokers: ["kafka:9092"]
    topics:
      sensor_data: "sensor-readings"
      alerts: "iot-alerts"
      commands: "device-commands"

# Edge computing
edge:
  gateway_enabled: true
  local_storage_days: 7
  sync_interval_minutes: 5
  offline_processing: true

# Device security
security:
  certificate_authority: "internal"
  device_cert_validity_days: 365
  mutual_tls_required: true
  encryption_algorithm: "AES-256"

# Analytics
analytics:
  anomaly_detection: true
  ml_models:
    - equipment_failure_prediction
    - energy_optimization
    - predictive_maintenance
  real_time_processing: true
```

## Monitoring & Alerts

### Device Health Monitoring
- Connection status and uptime
- Battery levels and power consumption
- Signal strength and network quality
- Data transmission rates

### Platform Metrics
- Message throughput (messages/second)
- Processing latency
- Storage utilization
- Alert response times

### Business Metrics
- Device activation rates
- Data quality scores
- Predictive maintenance savings
- Energy efficiency improvements

## Testing

### Device Simulation
```bash
# Simulate temperature sensors
python tools/device-simulator.py \
  --device-count 1000 \
  --sensor-types temperature,humidity \
  --interval 30 \
  --duration 3600

# Load testing
locust -f tests/load/iot_load_test.py \
  --users 10000 \
  --spawn-rate 100
```

## Deployment

### Environment Variables
```bash
# Database
TIMESERIES_DB_URL=http://influxdb:8086
DEVICE_DB_URL=postgresql://user:pass@postgres:5432/devices

# Message brokers
MQTT_BROKER_URL=mqtt://mosquitto:1883
KAFKA_BROKERS=kafka:9092

# Security
DEVICE_CA_CERT_PATH=/certs/ca.crt
DEVICE_CA_KEY_PATH=/certs/ca.key
ENCRYPTION_KEY=your-32-byte-key

# Analytics
ML_MODEL_REGISTRY_URL=http://mlflow:5000
ANOMALY_DETECTION_ENABLED=true
```

## Extensions

This IoT platform can be extended with:
- **Digital Twins**: Virtual representations of physical devices
- **Blockchain Integration**: Secure device identity and data provenance
- **5G/LTE-M**: Cellular connectivity for mobile IoT devices
- **Computer Vision**: Image processing from camera sensors
- **Voice Control**: Integration with voice assistants
- **Augmented Reality**: AR interfaces for device maintenance