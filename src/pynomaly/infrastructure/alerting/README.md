# Pynomaly Real-time Alerting System

A comprehensive real-time alerting and notification system for the Pynomaly anomaly detection platform.

## Features

### 🚨 Alert Management

- **Rule-based alerting** with flexible conditions and thresholds
- **Multiple severity levels** (Low, Medium, High, Critical)
- **Configurable alert conditions** (>, <, >=, <=, ==, !=)
- **Alert cooldown periods** to prevent spam
- **Alert acknowledgment and resolution** workflow

### 📡 Real-time Monitoring

- **Metric collection** from system and application sources
- **Prometheus integration** for metrics export
- **WebSocket support** for real-time alert notifications
- **Background processing** for high-performance metric ingestion

### 🔔 Multi-channel Notifications

- **Email notifications** with HTML templates
- **Slack integration** with rich message formatting
- **Discord webhook** support
- **Microsoft Teams** notifications
- **PagerDuty integration** for critical alerts
- **Webhook notifications** for custom integrations

### 📊 System Integration

- **SQLAlchemy ORM** for database persistence
- **FastAPI REST API** for management and monitoring
- **Pydantic validation** for type safety
- **Structured logging** with JSON output

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Alerting Service (FastAPI)                  │
├─────────────────────────────────────────────────────────────────┤
│  REST API Endpoints  │  WebSocket Endpoints  │  Demo Endpoints  │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Alert Manager                              │
├─────────────────────────────────────────────────────────────────┤
│  Rule Management  │  Alert Processing  │  Notification Sending  │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Metric Collector                           │
├─────────────────────────────────────────────────────────────────┤
│  System Metrics  │  Application Metrics  │  Prometheus Export  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Alert Manager (`alert_manager.py`)

Core component responsible for:

- Managing alert rules (CRUD operations)
- Processing incoming metrics against rules
- Triggering alerts when conditions are met
- Sending notifications through multiple channels
- Managing alert lifecycle (acknowledgment, resolution)

### 2. Metric Collector (`metric_collector.py`)

Handles metric collection and processing:

- Collects system metrics (CPU, memory, disk, network)
- Accepts custom application metrics
- Exports metrics to Prometheus
- Provides real-time metric streaming

### 3. Alerting Service (`alerting_service.py`)

FastAPI-based REST API providing:

- Alert rule management endpoints
- Metric submission endpoints
- Real-time WebSocket connections
- System status and health checks
- Demo endpoints for testing

## Quick Start

### 1. Import the Components

```python
from pynomaly.infrastructure.alerting import (
    AlertManager,
    MetricCollector,
    alerting_router,
    AlertSeverity,
    NotificationChannel,
    get_alert_manager,
    get_metric_collector,
)
```

### 2. Set Up the Database

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Create database engine
engine = create_engine("sqlite:///alerting.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
from pynomaly.infrastructure.alerting.alert_manager import Base
Base.metadata.create_all(bind=engine)
```

### 3. Create Alert Rules

```python
import asyncio
from pynomaly.infrastructure.alerting import AlertRuleCreate, AlertSeverity, NotificationChannel

async def create_sample_rules():
    alert_manager = get_alert_manager()
    
    # High CPU usage rule
    rule = AlertRuleCreate(
        name="High CPU Usage",
        description="Alert when CPU usage exceeds 80%",
        metric_name="system.cpu.usage",
        condition=">",
        threshold="80.0",
        duration=60,
        severity=AlertSeverity.HIGH,
        enabled=True,
        notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
        cooldown_period=300,
    )
    
    rule_id = await alert_manager.create_alert_rule(rule)
    print(f"Created rule: {rule_id}")

# Run the example
asyncio.run(create_sample_rules())
```

### 4. Submit Metrics

```python
async def submit_metrics():
    alert_manager = get_alert_manager()
    
    # Submit a high CPU usage metric
    await alert_manager.process_metric(
        metric_name="system.cpu.usage",
        value=85.5,
        metadata={"host": "web-server-01"}
    )
    
    print("Metric submitted and processed")

asyncio.run(submit_metrics())
```

### 5. Add to FastAPI Application

```python
from fastapi import FastAPI
from pynomaly.infrastructure.alerting import alerting_router

app = FastAPI()

# Include the alerting router
app.include_router(alerting_router, prefix="/api/v1")

# Run with: uvicorn main:app --reload
```

## API Endpoints

### Alert Rules Management

- `POST /alerting/rules` - Create alert rule
- `GET /alerting/rules` - List all rules
- `PUT /alerting/rules/{rule_id}` - Update rule
- `DELETE /alerting/rules/{rule_id}` - Delete rule

### Alert Management

- `GET /alerting/alerts` - Get active alerts
- `POST /alerting/alerts/{alert_id}/acknowledge` - Acknowledge alert
- `POST /alerting/alerts/{alert_id}/resolve` - Resolve alert

### Metrics Submission

- `POST /alerting/metrics` - Submit single metric
- `POST /alerting/metrics/batch` - Submit multiple metrics

### System Monitoring

- `GET /alerting/status` - Get system status
- `GET /alerting/health` - Health check

### WebSocket

- `WS /alerting/ws/{client_id}` - Real-time alert notifications

### Demo Endpoints

- `POST /alerting/demo/trigger-alert` - Trigger demo alert
- `POST /alerting/demo/create-rule` - Create demo rule

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite:///alerting.db

# Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...
PAGERDUTY_API_KEY=your-pagerduty-key
```

### Notification Templates

```python
# Email template
email_template = """
Alert: {rule_name}
Severity: {severity}
Message: {message}
Value: {value}
Timestamp: {timestamp}
"""

# Slack template
slack_template = {
    "text": "🚨 *{rule_name}*",
    "attachments": [
        {
            "color": "danger",
            "fields": [
                {"title": "Severity", "value": "{severity}", "short": True},
                {"title": "Value", "value": "{value}", "short": True},
                {"title": "Message", "value": "{message}", "short": False}
            ]
        }
    ]
}
```

## Running the Demo

A comprehensive demo script is available to test the alerting system:

```bash
# Make sure your Pynomaly server is running
python -m uvicorn main:app --reload

# In another terminal, run the demo
cd /path/to/pynomaly
python scripts/demo_alerting.py
```

The demo will:

1. Check system health
2. Create sample alert rules
3. Submit test metrics
4. Trigger alerts
5. Show real-time WebSocket notifications

## Testing

### Unit Tests

```bash
pytest tests/unit/infrastructure/alerting/
```

### Integration Tests

```bash
pytest tests/integration/alerting/
```

### Load Testing

```bash
python scripts/load_test_alerting.py
```

## Monitoring

### Prometheus Metrics

The system exports metrics to Prometheus:

- `alerting_rules_total` - Total number of alert rules
- `alerting_alerts_active` - Number of active alerts
- `alerting_notifications_sent_total` - Total notifications sent
- `alerting_metric_processing_duration_seconds` - Metric processing time

### Health Checks

- Database connectivity
- Notification service availability
- Metric collection status
- WebSocket connection count

## Security

### Input Validation

- All API inputs are validated using Pydantic
- SQL injection prevention through SQLAlchemy ORM
- XSS protection in notification templates

### Authentication

- API key authentication for REST endpoints
- WebSocket connection validation
- Rate limiting for metric submission

### Data Protection

- Sensitive data encryption in database
- Secure webhook URL validation
- Audit logging for all operations

## Performance

### Optimizations

- Background metric processing
- Database connection pooling
- Efficient alert rule evaluation
- Batch notification sending

### Scalability

- Horizontal scaling support
- Redis caching for frequently accessed data
- Asynchronous processing throughout
- Load balancing for WebSocket connections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
