"""
Pytest configuration for Enterprise Governance package testing.
Provides fixtures for governance, audit, compliance, and SLA testing.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import time
from uuid import uuid4, UUID
from datetime import datetime, timedelta, date
from unittest.mock import Mock


@pytest.fixture
def sample_tenant_id() -> UUID:
    """Sample tenant ID for testing."""
    return uuid4()


@pytest.fixture
def sample_user_id() -> UUID:
    """Sample user ID for testing."""
    return uuid4()


@pytest.fixture
def audit_event_data() -> Dict[str, Any]:
    """Sample audit event data for testing."""
    return {
        'event_type': 'data.access',
        'resource_type': 'anomaly_model',
        'resource_id': str(uuid4()),
        'details': {
            'model_name': 'fraud_detection_v1',
            'action': 'inference',
            'data_size': 1000,
            'confidence_threshold': 0.85
        },
        'ip_address': '192.168.1.100'
    }


@pytest.fixture
def compliance_framework_data() -> Dict[str, Any]:
    """Sample compliance framework data."""
    return {
        'framework': 'gdpr',
        'assessment_name': 'GDPR Compliance Assessment 2024',
        'scope': 'Data processing for anomaly detection models',
        'lead_assessor': 'compliance_officer',
        'controls': [
            {
                'control_id': 'GDPR-7.1',
                'title': 'Data Subject Rights',
                'description': 'Implement data subject access rights',
                'risk_level': 'high',
                'implementation_status': 'not_implemented'
            },
            {
                'control_id': 'GDPR-7.2', 
                'title': 'Data Retention',
                'description': 'Implement data retention policies',
                'risk_level': 'medium',
                'implementation_status': 'partially_implemented'
            }
        ]
    }


@pytest.fixture
def sla_configuration() -> Dict[str, Any]:
    """Sample SLA configuration for testing."""
    return {
        'name': 'Anomaly Detection Service SLA',
        'sla_type': 'availability',
        'service_provider': 'ML Platform Team',
        'service_consumer': 'Business Applications',
        'services_covered': [
            'model_inference',
            'batch_processing',
            'real_time_detection'
        ],
        'overall_target': 99.5,
        'effective_date': datetime.utcnow(),
        'expiry_date': datetime.utcnow() + timedelta(days=365),
        'metrics': [
            {
                'name': 'availability',
                'metric_type': 'percentage',
                'target_value': 99.5,
                'minimum_acceptable': 99.0,
                'measurement_unit': 'percentage'
            },
            {
                'name': 'response_time',
                'metric_type': 'latency',
                'target_value': 100.0,
                'minimum_acceptable': 200.0,
                'measurement_unit': 'milliseconds'
            },
            {
                'name': 'throughput',
                'metric_type': 'rate',
                'target_value': 1000.0,
                'minimum_acceptable': 800.0,
                'measurement_unit': 'requests_per_second'
            }
        ]
    }


@pytest.fixture
def security_event_scenarios() -> List[Dict[str, Any]]:
    """Security event scenarios for testing."""
    return [
        {
            'event_type': 'security.unauthorized_access',
            'severity': 'critical',
            'details': {
                'attempted_resource': 'production_model',
                'source_ip': '192.168.1.255',
                'failure_reason': 'invalid_token'
            },
            'expected_alert': True
        },
        {
            'event_type': 'security.data_breach_attempt',
            'severity': 'critical',
            'details': {
                'data_type': 'personal_data',
                'volume_attempted': 10000,
                'blocked': True
            },
            'expected_alert': True
        },
        {
            'event_type': 'user.login_failed',
            'severity': 'high',
            'details': {
                'username': 'admin',
                'consecutive_failures': 5,
                'account_locked': True
            },
            'expected_alert': True
        },
        {
            'event_type': 'data.access',
            'severity': 'medium',
            'details': {
                'resource': 'model_metrics',
                'authorized': True
            },
            'expected_alert': False
        }
    ]


@pytest.fixture
def compliance_violation_scenarios() -> List[Dict[str, Any]]:
    """Compliance violation scenarios for testing."""
    return [
        {
            'framework': 'gdpr',
            'violation_type': 'data_retention',
            'severity': 'high',
            'description': 'Data retained beyond policy limits',
            'affected_records': 1500,
            'remediation_required': True
        },
        {
            'framework': 'hipaa',
            'violation_type': 'unauthorized_access',
            'severity': 'critical',
            'description': 'PHI accessed without authorization',
            'affected_records': 50,
            'remediation_required': True
        },
        {
            'framework': 'sox',
            'violation_type': 'audit_trail',
            'severity': 'medium',
            'description': 'Incomplete audit trail for financial data',
            'affected_records': 250,
            'remediation_required': True
        }
    ]


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return Timer()


@pytest.fixture
def mock_audit_repository():
    """Mock audit repository for testing."""
    class MockAuditRepository:
        def __init__(self):
            self.audit_logs = {}
            
        def create(self, audit_log) -> Dict[str, Any]:
            audit_log.id = uuid4()
            audit_log.created_at = datetime.utcnow()
            self.audit_logs[audit_log.id] = audit_log
            return audit_log
            
        def search(self, query) -> List[Dict[str, Any]]:
            results = []
            for log in self.audit_logs.values():
                if self._matches_query(log, query):
                    results.append(log)
            return results[:query.page_size] if query.page_size else results
            
        def _matches_query(self, log, query) -> bool:
            if query.tenant_id and log.tenant_id != query.tenant_id:
                return False
            if query.event_types and log.event_type not in query.event_types:
                return False
            if query.start_time and log.created_at < query.start_time:
                return False
            if query.end_time and log.created_at > query.end_time:
                return False
            return True
    
    return MockAuditRepository()


@pytest.fixture
def mock_compliance_repository():
    """Mock compliance repository for testing."""
    class MockComplianceRepository:
        def __init__(self):
            self.assessments = {}
            self.controls = {}
            self.reports = {}
            
        def create_assessment(self, assessment):
            assessment.id = uuid4()
            self.assessments[assessment.id] = assessment
            return assessment
            
        def get_assessment(self, assessment_id: UUID):
            return self.assessments.get(assessment_id)
            
        def create_control(self, control):
            control.id = uuid4()
            self.controls[control.id] = control
            return control
            
        def get_control(self, control_id: UUID):
            return self.controls.get(control_id)
            
        def update_control(self, control):
            if control.id in self.controls:
                self.controls[control.id] = control
            return control
            
        def get_controls_by_framework(self, tenant_id: UUID, framework):
            return [c for c in self.controls.values() 
                   if c.tenant_id == tenant_id and c.framework == framework]
            
        def create_report(self, report):
            report.id = uuid4()
            self.reports[report.id] = report
            return report
    
    return MockComplianceRepository()


@pytest.fixture
def mock_sla_repository():
    """Mock SLA repository for testing."""
    class MockSLARepository:
        def __init__(self):
            self.slas = {}
            self.metrics = {}
            self.violations = {}
            
        def create_sla(self, sla):
            sla.id = uuid4()
            sla.created_at = datetime.utcnow()
            self.slas[sla.id] = sla
            return sla
            
        def get_sla(self, sla_id: UUID):
            return self.slas.get(sla_id)
            
        def update_sla(self, sla):
            if sla.id in self.slas:
                self.slas[sla.id] = sla
            return sla
            
        def create_metric(self, metric):
            metric.id = uuid4()
            self.metrics[metric.id] = metric
            return metric
            
        def get_metric(self, metric_id: UUID):
            return self.metrics.get(metric_id)
            
        def update_metric(self, metric):
            if metric.id in self.metrics:
                self.metrics[metric.id] = metric
            return metric
            
        def get_slas_by_tenant(self, tenant_id: UUID):
            return [sla for sla in self.slas.values() if sla.tenant_id == tenant_id]
            
        def get_metrics_by_sla(self, sla_id: UUID):
            sla = self.slas.get(sla_id)
            if sla and sla.metrics:
                return [self.metrics[mid] for mid in sla.metrics if mid in self.metrics]
            return []
            
        def create_violation(self, violation):
            violation.id = uuid4()
            self.violations[violation.id] = violation
            return violation
            
        def get_violations_by_date(self, tenant_id: UUID, date_filter: date):
            return [
                v for v in self.violations.values()
                if v.tenant_id == tenant_id and v.start_time.date() == date_filter
            ]
    
    return MockSLARepository()


@pytest.fixture
def mock_notification_service():
    """Mock notification service for testing."""
    class MockNotificationService:
        def __init__(self):
            self.sent_alerts = []
            
        def send_security_alert(self, tenant_id: UUID, event, severity: str):
            alert = {
                'type': 'security_alert',
                'tenant_id': tenant_id,
                'event': event,
                'severity': severity,
                'timestamp': datetime.utcnow()
            }
            self.sent_alerts.append(alert)
            return {'success': True}
            
        def send_sla_violation_alert(self, violation):
            alert = {
                'type': 'sla_violation',
                'violation': violation,
                'timestamp': datetime.utcnow()
            }
            self.sent_alerts.append(alert)
            return {'success': True}
            
        def send_compliance_violation_alert(self, violation):
            alert = {
                'type': 'compliance_violation', 
                'violation': violation,
                'timestamp': datetime.utcnow()
            }
            self.sent_alerts.append(alert)
            return {'success': True}
    
    return MockNotificationService()


@pytest.fixture
def large_audit_dataset() -> List[Dict[str, Any]]:
    """Generate large audit dataset for performance testing."""
    np.random.seed(42)
    
    events = []
    event_types = [
        'user.login', 'user.logout', 'data.access', 'model.inference',
        'security.scan', 'system.startup', 'compliance.check'
    ]
    
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i in range(10000):
        event_time = base_time + timedelta(seconds=i * 30)
        
        event = {
            'event_type': np.random.choice(event_types),
            'tenant_id': str(uuid4()) if i % 100 == 0 else 'common_tenant',
            'user_id': str(uuid4()) if i % 50 == 0 else 'common_user',
            'timestamp': event_time,
            'details': {
                'session_id': f'session_{i % 1000}',
                'request_id': f'req_{i}',
                'duration_ms': np.random.randint(10, 1000)
            }
        }
        events.append(event)
    
    return events


def pytest_configure(config):
    """Configure pytest markers for governance testing."""
    markers = [
        "governance: Enterprise governance tests",
        "audit: Audit logging and trail tests",
        "compliance: Compliance framework tests",
        "sla: Service Level Agreement tests",
        "security: Security and access control tests",
        "performance: Governance performance tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)
