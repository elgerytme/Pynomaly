# Enterprise Governance & Compliance Package

This package provides comprehensive governance, audit, compliance, and SLA management capabilities for anomaly_detection, supporting multiple regulatory frameworks and enterprise governance requirements.

## Features

### 🔍 **Comprehensive Audit Logging**
- **Universal Audit Trail**: Complete audit logging for all system activities
- **Compliance-Ready Events**: Pre-configured event types for major compliance frameworks
- **Integrity Protection**: Cryptographic checksums and tamper detection
- **Real-Time Monitoring**: Live audit event streaming and alerting
- **Advanced Search**: Full-text search with filtering and aggregation capabilities

### 📋 **Multi-Framework Compliance**
- **SOC 2 Type II**: Complete control framework with automated evidence collection
- **GDPR Compliance**: Data protection and privacy rights management
- **ISO 27001**: Information security management system controls
- **HIPAA**: Healthcare data protection and privacy safeguards
- **PCI-DSS**: Payment card industry security standards
- **CCPA**: California consumer privacy act compliance
- **FedRAMP**: Federal risk and authorization management program
- **NIST CSF**: Cybersecurity framework implementation

### 📊 **Service Level Agreement Management**
- **Multi-Metric SLAs**: Support for availability, performance, and custom metrics
- **Real-Time Monitoring**: Continuous SLA compliance tracking
- **Violation Management**: Automated violation detection and escalation
- **Service Credits**: Automatic calculation of penalties and credits
- **Performance Analytics**: Historical trends and predictive analytics

### 🔒 **Data Privacy & Protection**
- **Data Subject Rights**: GDPR/CCPA rights management (access, rectification, erasure)
- **Consent Management**: Granular consent tracking and withdrawal
- **Data Retention**: Automated retention policy enforcement
- **Cross-Border Transfers**: International transfer safeguards and documentation
- **Breach Management**: Data breach incident tracking and notification

## Quick Start

### Installation

```bash
pip install anomaly_detection-enterprise-governance
```

### Basic Usage

```python
from enterprise_governance import GovernanceService, AuditLog, ComplianceFramework

# Initialize governance service
governance_service = GovernanceService(
    audit_repository=audit_repo,
    compliance_repository=compliance_repo,
    sla_repository=sla_repo,
    notification_service=notification_service,
    report_generator=report_generator
)

# Create audit log entry
await governance_service.create_audit_log(
    tenant_id=tenant_id,
    event_type="data.accessed",
    user_id=user_id,
    resource_type="user_data",
    details={"dataset": "customer_records", "rows_accessed": 1500}
)

# Start compliance assessment
assessment = await governance_service.create_compliance_assessment(
    tenant_id=tenant_id,
    framework=ComplianceFramework.SOC2,
    assessment_name="Q1 2024 SOC2 Assessment",
    scope="All production systems and data processing",
    lead_assessor="security-team@company.com"
)
```

### API Integration

```python
from fastapi import FastAPI, Depends
from enterprise_governance import GovernanceService
from enterprise_governance.application.dto import AuditLogRequest, SLARequest

app = FastAPI()

@app.post("/audit/logs")
async def create_audit_log(
    request: AuditLogRequest,
    governance: GovernanceService = Depends()
):
    audit_log = await governance.create_audit_log(
        tenant_id=get_current_tenant_id(),
        event_type=request.event_type,
        user_id=request.user_id,
        details=request.details
    )
    return {"audit_log_id": audit_log.id}

@app.get("/compliance/assessments/{assessment_id}/report")
async def get_compliance_report(
    assessment_id: UUID,
    governance: GovernanceService = Depends()
):
    report = await governance.generate_compliance_report(assessment_id)
    return report.dict()
```

### CLI Usage

```bash
# Generate audit report
anomaly_detection-enterprise-governance audit report \
    --tenant-id <tenant-id> \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --format pdf \
    --compliance-framework soc2

# Create SLA
anomaly_detection-enterprise-governance sla create \
    --name "API Availability SLA" \
    --type availability \
    --target 99.9 \
    --service "anomaly_detection-api" \
    --provider "Internal IT"

# Check compliance status
anomaly_detection-enterprise-governance compliance status \
    --framework soc2 \
    --tenant-id <tenant-id>
```

## Configuration

### Environment Variables

```bash
# Database Configuration
ANOMALY_DETECTION_GOVERNANCE_DATABASE_URL=postgresql://user:pass@localhost/governance
ANOMALY_DETECTION_GOVERNANCE_REDIS_URL=redis://localhost:6379/2

# Audit Settings
ANOMALY_DETECTION_AUDIT_RETENTION_DAYS=2557  # 7 years default
ANOMALY_DETECTION_AUDIT_ENCRYPTION_ENABLED=true
ANOMALY_DETECTION_AUDIT_REAL_TIME_ALERTS=true

# Compliance Settings
ANOMALY_DETECTION_COMPLIANCE_AUTO_ASSESSMENT=true
ANOMALY_DETECTION_COMPLIANCE_NOTIFICATION_THRESHOLD=90  # Alert below 90% compliance

# SLA Settings
ANOMALY_DETECTION_SLA_CHECK_INTERVAL_MINUTES=5
ANOMALY_DETECTION_SLA_VIOLATION_ESCALATION=true
ANOMALY_DETECTION_SLA_CREDITS_AUTO_APPLY=true

# Privacy Settings
ANOMALY_DETECTION_PRIVACY_DATA_RETENTION_CHECK_DAILY=true
ANOMALY_DETECTION_PRIVACY_CONSENT_RENEWAL_DAYS=365
ANOMALY_DETECTION_PRIVACY_BREACH_NOTIFICATION_HOURS=72

# Notification Settings
ANOMALY_DETECTION_NOTIFICATIONS_SLACK_WEBHOOK=https://hooks.slack.com/...
ANOMALY_DETECTION_NOTIFICATIONS_EMAIL_SENDER=governance@company.com
ANOMALY_DETECTION_NOTIFICATIONS_PAGERDUTY_API_KEY=<api-key>

# Report Generation
ANOMALY_DETECTION_REPORTS_STORAGE_PATH=/var/lib/anomaly_detection/reports
ANOMALY_DETECTION_REPORTS_RETENTION_DAYS=2557
ANOMALY_DETECTION_REPORTS_ENCRYPTION_ENABLED=true
```

### Configuration File

Create `governance_config.yaml`:

```yaml
governance:
  audit:
    retention_policy:
      default_days: 2557  # 7 years
      critical_events_days: 3653  # 10 years
      security_events_days: 2557  # 7 years
    
    encryption:
      enabled: true
      key_rotation_days: 90
    
    real_time:
      enabled: true
      batch_size: 1000
      flush_interval_seconds: 30

  compliance:
    frameworks:
      soc2:
        enabled: true
        auto_assessment: true
        control_review_frequency: "quarterly"
      
      gdpr:
        enabled: true
        data_retention_monitoring: true
        consent_renewal_days: 365
        breach_notification_hours: 72
      
      iso27001:
        enabled: true
        risk_assessment_frequency: "annual"
        control_testing_frequency: "quarterly"

  sla:
    monitoring:
      check_interval_minutes: 5
      violation_threshold_minutes: 15
    
    notifications:
      immediate_severity: ["critical", "high"]
      escalation_delays: [5, 15, 60]  # minutes
    
    credits:
      auto_calculate: true
      auto_apply: false  # Require approval
      minimum_credit_amount: 10.0

  privacy:
    data_retention:
      check_frequency: "daily"
      deletion_grace_period_days: 30
    
    consent:
      renewal_notification_days: 30
      withdrawal_confirmation_required: true
    
    rights_requests:
      response_time_days: 30
      auto_acknowledgment: true

# Notification channels
notifications:
  channels:
    security_alerts:
      - type: "slack"
        webhook: "${SLACK_SECURITY_WEBHOOK}"
      - type: "email"
        recipients: ["security-team@company.com"]
      - type: "pagerduty"
        service_key: "${PAGERDUTY_SERVICE_KEY}"
    
    compliance_reports:
      - type: "email"
        recipients: ["compliance@company.com", "audit@company.com"]
    
    sla_violations:
      - type: "slack"
        webhook: "${SLACK_SLA_WEBHOOK}"
      - type: "email"
        recipients: ["operations@company.com"]

# Report generation
reports:
  storage:
    type: "s3"  # or "local", "azure_blob", "gcs"
    bucket: "company-governance-reports"
    encryption: true
    retention_days: 2557
  
  templates:
    audit_report: "templates/audit_report.html"
    compliance_report: "templates/compliance_report.html"
    sla_report: "templates/sla_report.html"
  
  generation:
    batch_processing: true
    concurrent_reports: 3
    timeout_minutes: 30
```

## Architecture

### Domain-Driven Design

The package follows Domain-Driven Design (DDD) principles with clear bounded contexts:

```
enterprise_governance/
├── domain/                    # Pure business logic
│   ├── entities/             # Core business entities
│   │   ├── audit_log.py      # Audit trail management
│   │   ├── compliance.py     # Compliance frameworks & controls
│   │   └── sla.py           # Service level agreements
│   ├── services/            # Domain services
│   └── repositories/        # Repository interfaces
├── application/             # Use cases and orchestration
│   ├── services/           # Application services
│   │   └── governance_service.py  # Main orchestrator
│   ├── use_cases/          # Specific use cases
│   └── dto/                # Data transfer objects
│       └── governance_dto.py    # API communication models
├── infrastructure/         # External integrations
│   ├── persistence/        # Database implementations
│   ├── external/           # External service integrations
│   ├── notifications/      # Notification providers
│   └── reports/           # Report generators
└── presentation/          # User interfaces
    ├── api/               # REST API endpoints
    └── cli/               # Command-line interface
```

### Key Components

#### Domain Entities

- **AuditLog**: Comprehensive audit trail with integrity protection
- **ComplianceControl**: Individual compliance controls with evidence tracking
- **ComplianceAssessment**: Full compliance assessments with findings
- **ServiceLevelAgreement**: SLA definitions with metrics and penalties
- **SLAMetric**: Individual performance metrics with thresholds
- **DataPrivacyRecord**: GDPR/CCPA privacy rights and consent tracking

#### Application Services

- **GovernanceService**: Main orchestration service for all governance operations
- **ComplianceService**: Specialized compliance assessment and reporting
- **SLAMonitoringService**: Real-time SLA monitoring and violation detection
- **AuditService**: Audit log management and search capabilities

#### Infrastructure Adapters

- **AuditRepository**: Persistent audit log storage with search capabilities
- **ComplianceRepository**: Compliance data storage and retrieval
- **NotificationService**: Multi-channel notification delivery
- **ReportGenerator**: Automated report generation in multiple formats

## API Reference

### Audit Endpoints

```http
POST   /audit/logs                    # Create audit log entry
GET    /audit/logs                    # Search audit logs
GET    /audit/logs/{id}               # Get specific audit log
POST   /audit/reports                 # Generate audit report
GET    /audit/statistics              # Get audit statistics
```

### Compliance Endpoints

```http
POST   /compliance/assessments        # Create compliance assessment
GET    /compliance/assessments        # List assessments
GET    /compliance/assessments/{id}   # Get assessment details
PUT    /compliance/controls/{id}      # Update control status
GET    /compliance/controls           # List controls
POST   /compliance/reports            # Generate compliance report
GET    /compliance/frameworks         # List supported frameworks
```

### SLA Endpoints

```http
POST   /slas                         # Create SLA
GET    /slas                         # List SLAs
GET    /slas/{id}                    # Get SLA details
PUT    /slas/{id}                    # Update SLA
POST   /slas/{id}/metrics            # Add SLA metric
POST   /metrics/{id}/measurements    # Record metric measurement
GET    /slas/{id}/violations         # Get SLA violations
GET    /slas/compliance              # Check SLA compliance
```

### Data Privacy Endpoints

```http
POST   /privacy/records              # Create privacy record
GET    /privacy/records/{subject_id} # Get subject's privacy records
POST   /privacy/rights-requests      # Submit data rights request
GET    /privacy/consent/{subject_id} # Get consent status
POST   /privacy/consent/withdraw     # Withdraw consent
GET    /privacy/retention-due        # Get records due for deletion
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# E2E tests
pytest tests/e2e/

# Compliance tests
pytest tests/compliance/

# With coverage
pytest --cov=enterprise_governance --cov-report=html
```

### Test Categories

- **Unit Tests**: Domain logic and service tests
- **Integration Tests**: Database and external service integration
- **E2E Tests**: Complete governance workflows
- **Compliance Tests**: Compliance framework validation
- **Performance Tests**: Load testing for audit logging and SLA monitoring

### Test Configuration

```python
# conftest.py
import pytest
from enterprise_governance import GovernanceService
from enterprise_governance.infrastructure.persistence import (
    InMemoryAuditRepository, InMemoryComplianceRepository
)

@pytest.fixture
def governance_service():
    return GovernanceService(
        audit_repository=InMemoryAuditRepository(),
        compliance_repository=InMemoryComplianceRepository(),
        sla_repository=InMemorySLARepository(),
        notification_service=MockNotificationService(),
        report_generator=MockReportGenerator()
    )

@pytest.fixture
def sample_audit_log():
    return AuditLog(
        event_type="data.accessed",
        category="data_access",
        severity="medium",
        message="User accessed customer data"
    )
```

## Security Considerations

### Audit Security
- Cryptographic integrity protection for audit logs
- Tamper-evident audit trails with checksums
- Encrypted storage of sensitive audit data
- Access controls for audit log viewing and export

### Compliance Security
- Evidence encryption and secure storage
- Role-based access to compliance data
- Audit trails for all compliance activities
- Secure report generation and distribution

### Data Privacy Security
- Encryption of personal data records
- Secure consent management
- Access controls for privacy data
- Audit trails for all privacy operations

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "enterprise_governance.presentation.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly_detection-enterprise-governance
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly_detection-enterprise-governance
  template:
    metadata:
      labels:
        app: anomaly_detection-enterprise-governance
    spec:
      containers:
      - name: governance-service
        image: anomaly_detection/enterprise-governance:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: governance-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: governance-secrets
              key: redis-url
```

## Performance Optimization

### Audit Log Performance
- Asynchronous audit log writing with batching
- Partitioned database tables by date
- Indexed search fields for fast queries
- Configurable retention policies for data lifecycle

### Compliance Monitoring
- Cached compliance status calculations
- Incremental compliance updates
- Background compliance checking
- Optimized control status queries

### SLA Monitoring
- Real-time metric collection with buffering
- Efficient violation detection algorithms
- Cached SLA compliance calculations
- Optimized time-series data storage

## Monitoring and Observability

### Metrics
- Audit log ingestion rates and volumes
- Compliance assessment completion times
- SLA violation detection latency
- Report generation performance

### Alerts
- Critical security event notifications
- Compliance threshold breach alerts
- SLA violation escalations
- System health and performance alerts

### Dashboards
- Real-time governance overview
- Compliance status across frameworks
- SLA performance trending
- Audit activity monitoring

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/compliance-enhancement`
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/anomaly_detection.git
cd anomaly_detection/src/packages/enterprise/enterprise_governance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev,test,lint,all]"

# Run tests
pytest

# Run linting
ruff check .
mypy enterprise_governance/

# Generate documentation
mkdocs serve
```

## License

This package is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [https://docs.anomaly_detection.org/enterprise/governance](https://docs.anomaly_detection.org/enterprise/governance)
- **Issues**: [GitHub Issues](https://github.com/yourusername/anomaly_detection/issues)
- **Enterprise Support**: enterprise-support@anomaly_detection.org
- **Security Issues**: security@anomaly_detection.org

---

**Enterprise Governance & Compliance Package** - Part of the anomaly_detection Enterprise Suite