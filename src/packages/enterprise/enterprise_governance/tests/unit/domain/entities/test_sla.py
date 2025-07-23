"""
Unit tests for SLA domain entities.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from enterprise_governance.domain.entities.sla import (
    SLAMetric, ServiceLevelAgreement, SLAViolation,
    SLAStatus, SLAType, SLAMetricType, SLAViolationSeverity
)


class TestSLAMetric:
    """Test cases for SLAMetric entity."""
    
    def test_sla_metric_creation_basic(self):
        """Test basic SLA metric creation."""
        metric = SLAMetric(
            name="Response Time",
            description="Average API response time",
            metric_type=SLAMetricType.TIME_DURATION,
            target_value=500.0,
            minimum_acceptable=1000.0,
            measurement_unit="milliseconds",
            measurement_frequency="5 minutes",
            calculation_method="Average over measurement period",
            data_source="API Gateway metrics",
            aggregation_period="5 minutes"
        )
        
        assert isinstance(metric.id, UUID)
        assert metric.name == "Response Time"
        assert metric.description == "Average API response time"
        assert metric.metric_type == SLAMetricType.TIME_DURATION
        assert metric.target_value == 500.0
        assert metric.minimum_acceptable == 1000.0
        assert metric.measurement_unit == "milliseconds"
        assert metric.measurement_frequency == "5 minutes"
        assert metric.calculation_method == "Average over measurement period"
        assert metric.data_source == "API Gateway metrics"
        assert metric.aggregation_period == "5 minutes"
        assert metric.warning_threshold is None
        assert metric.critical_threshold is None
        assert metric.current_value is None
        assert metric.last_measured_at is None
        assert metric.measurement_history == []
        assert metric.is_meeting_target is True
        assert metric.consecutive_failures == 0
        assert metric.tags == []
        
    def test_sla_metric_creation_comprehensive(self):
        """Test comprehensive SLA metric creation."""
        tags = ["performance", "critical"]
        
        metric = SLAMetric(
            name="System Availability",
            description="System uptime percentage",
            metric_type=SLAMetricType.PERCENTAGE,
            target_value=99.9,
            minimum_acceptable=99.5,
            measurement_unit="percent",
            measurement_frequency="1 minute",
            warning_threshold=99.8,
            critical_threshold=99.5,
            calculation_method="Uptime / Total time * 100",
            data_source="Health check monitor",
            aggregation_period="1 hour",
            current_value=99.95,
            last_measured_at=datetime.utcnow(),
            is_meeting_target=True,
            consecutive_failures=0,
            tags=tags
        )
        
        assert metric.name == "System Availability"
        assert metric.metric_type == SLAMetricType.PERCENTAGE
        assert metric.target_value == 99.9
        assert metric.warning_threshold == 99.8
        assert metric.critical_threshold == 99.5
        assert metric.current_value == 99.95
        assert metric.is_meeting_target is True
        assert metric.tags == tags
        
    def test_record_measurement_percentage_metric(self):
        """Test recording measurement for percentage metric."""
        metric = SLAMetric(
            name="System Availability",
            description="System uptime percentage",
            metric_type=SLAMetricType.PERCENTAGE,
            target_value=99.5,
            minimum_acceptable=99.0,
            measurement_unit="percent",
            measurement_frequency="1 minute",
            calculation_method="Uptime calculation",
            data_source="Monitor",
            aggregation_period="1 hour"
        )
        
        timestamp = datetime.utcnow()
        
        # Record measurement that meets target
        metric.record_measurement(99.8, timestamp)
        
        assert metric.current_value == 99.8
        assert metric.last_measured_at == timestamp
        assert metric.is_meeting_target is True
        assert metric.consecutive_failures == 0
        assert len(metric.measurement_history) == 1
        
        history_entry = metric.measurement_history[0]
        assert history_entry["value"] == 99.8
        assert history_entry["timestamp"] == timestamp.isoformat()
        assert history_entry["meets_target"] is True
        
    def test_record_measurement_time_duration_metric(self):
        """Test recording measurement for time duration metric."""
        metric = SLAMetric(
            name="Response Time",
            description="API response time",
            metric_type=SLAMetricType.TIME_DURATION,
            target_value=500.0,
            minimum_acceptable=1000.0,
            measurement_unit="milliseconds",
            measurement_frequency="1 minute",
            calculation_method="Average response time",
            data_source="API Gateway",
            aggregation_period="5 minutes"
        )
        
        timestamp = datetime.utcnow()
        
        # Record measurement that meets target (lower is better for time)
        metric.record_measurement(300.0, timestamp)
        
        assert metric.current_value == 300.0
        assert metric.is_meeting_target is True
        assert metric.consecutive_failures == 0
        
        # Record measurement that doesn't meet target
        metric.record_measurement(800.0, timestamp + timedelta(minutes=1))
        
        assert metric.current_value == 800.0
        assert metric.is_meeting_target is False
        assert metric.consecutive_failures == 1
        
    def test_record_measurement_consecutive_failures(self):
        """Test consecutive failures tracking."""
        metric = SLAMetric(
            name="Test Metric",
            description="Test metric",
            metric_type=SLAMetricType.PERCENTAGE,
            target_value=95.0,
            minimum_acceptable=90.0,
            measurement_unit="percent",
            measurement_frequency="1 minute",
            calculation_method="Test calculation",
            data_source="Test source",
            aggregation_period="1 hour"
        )
        
        # Record multiple failures
        metric.record_measurement(80.0)  # Failure 1
        assert metric.consecutive_failures == 1
        
        metric.record_measurement(85.0)  # Failure 2
        assert metric.consecutive_failures == 2
        
        metric.record_measurement(98.0)  # Success - should reset
        assert metric.consecutive_failures == 0
        assert metric.is_meeting_target is True
        
    def test_get_compliance_percentage(self):
        """Test compliance percentage calculation."""
        metric = SLAMetric(
            name="Test Metric",
            description="Test metric",
            metric_type=SLAMetricType.PERCENTAGE,
            target_value=95.0,
            minimum_acceptable=90.0,
            measurement_unit="percent",
            measurement_frequency="1 minute",
            calculation_method="Test calculation",
            data_source="Test source",
            aggregation_period="1 hour"
        )
        
        # No measurements should return 100%
        compliance = metric.get_compliance_percentage(30)
        assert compliance == 100.0
        
        # Add measurements within the time window
        now = datetime.utcnow()
        
        # 3 measurements: 2 compliant, 1 non-compliant
        metric.record_measurement(98.0, now - timedelta(days=5))      # Compliant
        metric.record_measurement(92.0, now - timedelta(days=10))     # Non-compliant  
        metric.record_measurement(97.0, now - timedelta(days=15))     # Compliant
        
        compliance = metric.get_compliance_percentage(30)
        assert compliance == (2/3) * 100  # 66.67%
        
        # Test with measurements outside time window
        metric.record_measurement(80.0, now - timedelta(days=35))     # Outside window
        
        compliance = metric.get_compliance_percentage(30)
        assert compliance == (2/3) * 100  # Should still be 66.67%
        
    def test_get_compliance_percentage_no_recent_measurements(self):
        """Test compliance percentage with no recent measurements."""
        metric = SLAMetric(
            name="Test Metric",
            description="Test metric",
            metric_type=SLAMetricType.PERCENTAGE,
            target_value=95.0,
            minimum_acceptable=90.0,
            measurement_unit="percent",
            measurement_frequency="1 minute",
            calculation_method="Test calculation",
            data_source="Test source",
            aggregation_period="1 hour"
        )
        
        # Add old measurements
        old_time = datetime.utcnow() - timedelta(days=60)
        metric.record_measurement(98.0, old_time)
        
        # Should return 100% since no recent measurements
        compliance = metric.get_compliance_percentage(30)
        assert compliance == 100.0
        
    def test_is_in_violation(self):
        """Test violation detection."""
        metric = SLAMetric(
            name="Test Metric",
            description="Test metric",
            metric_type=SLAMetricType.PERCENTAGE,
            target_value=95.0,
            minimum_acceptable=90.0,
            measurement_unit="percent",
            measurement_frequency="1 minute",
            calculation_method="Test calculation",
            data_source="Test source",
            aggregation_period="1 hour"
        )
        
        # No current value should not be in violation
        assert metric.is_in_violation() is False
        
        # Record compliant measurement
        metric.record_measurement(97.0)
        assert metric.is_in_violation() is False
        
        # Record non-compliant measurement
        metric.record_measurement(80.0)
        assert metric.is_in_violation() is True
        
    def test_get_violation_severity(self):
        """Test violation severity assessment."""
        metric = SLAMetric(
            name="Test Metric",
            description="Test metric",
            metric_type=SLAMetricType.PERCENTAGE,
            target_value=95.0,
            minimum_acceptable=90.0,
            measurement_unit="percent",
            measurement_frequency="1 minute",
            warning_threshold=92.0,
            critical_threshold=88.0,
            calculation_method="Test calculation",
            data_source="Test source",
            aggregation_period="1 hour"
        )
        
        # Compliant measurement should have no severity
        metric.record_measurement(97.0)
        assert metric.get_violation_severity() is None
        
        # Below critical threshold
        metric.record_measurement(85.0)
        assert metric.get_violation_severity() == SLAViolationSeverity.CRITICAL
        
        # Below warning threshold but above critical
        metric.record_measurement(90.0)
        assert metric.get_violation_severity() == SLAViolationSeverity.HIGH
        
        # Below target but above warning threshold
        metric.record_measurement(93.5)
        assert metric.get_violation_severity() == SLAViolationSeverity.MEDIUM


class TestServiceLevelAgreement:
    """Test cases for ServiceLevelAgreement entity."""
    
    def test_sla_creation_basic(self):
        """Test basic SLA creation."""
        tenant_id = uuid4()
        effective_date = datetime.utcnow()
        metric_ids = [uuid4(), uuid4()]
        services_covered = ["API Gateway", "Database", "Cache"]
        
        sla = ServiceLevelAgreement(
            name="Production Services SLA",
            description="SLA for production services",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Platform Team",
            service_consumer="Business Units",
            tenant_id=tenant_id,
            services_covered=services_covered,
            service_hours="24x7",
            metrics=metric_ids,
            overall_target=99.5,
            measurement_period="Monthly",
            effective_date=effective_date,
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        
        assert isinstance(sla.id, UUID)
        assert sla.name == "Production Services SLA"
        assert sla.version == "1.0"
        assert sla.description == "SLA for production services"
        assert sla.sla_type == SLAType.AVAILABILITY
        assert sla.service_provider == "Platform Team"
        assert sla.service_consumer == "Business Units"
        assert sla.tenant_id == tenant_id
        assert sla.services_covered == services_covered
        assert sla.service_hours == "24x7"
        assert sla.metrics == metric_ids
        assert sla.overall_target == 99.5
        assert sla.measurement_period == "Monthly"
        assert sla.effective_date == effective_date
        assert sla.expiry_date is None
        assert sla.auto_renewal is False
        assert sla.status == SLAStatus.ACTIVE
        assert sla.current_compliance == 100.0
        assert sla.penalties_enabled is False
        assert sla.credits_earned == 0.0
        assert sla.exclusions == []
        assert sla.penalty_structure == []
        assert sla.notification_contacts == []
        
    def test_sla_creation_comprehensive(self):
        """Test comprehensive SLA creation."""
        tenant_id = uuid4()
        effective_date = datetime.utcnow()
        expiry_date = effective_date + timedelta(days=365)
        next_review = effective_date + timedelta(days=90)
        
        exclusions = ["Planned maintenance", "Force majeure events"]
        notification_contacts = ["ops@company.com", "management@company.com"]
        escalation_matrix = [
            {"level": 1, "contact": "ops@company.com", "time_threshold": "15 minutes"},
            {"level": 2, "contact": "manager@company.com", "time_threshold": "1 hour"}
        ]
        stakeholders = ["Operations Team", "Business Stakeholders", "Executive Team"]
        tags = ["critical", "production", "customer-facing"]
        attachments = ["sla_document.pdf", "escalation_procedures.pdf"]
        external_refs = ["Contract #SLA-2024-001"]
        
        sla = ServiceLevelAgreement(
            name="Enterprise API SLA",
            version="2.1",
            description="Comprehensive SLA for enterprise API services",
            sla_type=SLAType.PERFORMANCE,
            service_provider="API Platform Team",
            service_consumer="Enterprise Customers",
            tenant_id=tenant_id,
            services_covered=["Authentication API", "Data API", "Analytics API"],
            service_hours="24x7x365",
            exclusions=exclusions,
            metrics=[uuid4(), uuid4(), uuid4()],
            overall_target=99.9,
            measurement_period="Monthly",
            effective_date=effective_date,
            expiry_date=expiry_date,
            auto_renewal=True,
            renewal_period="12 months",
            status=SLAStatus.ACTIVE,
            current_compliance=99.85,
            penalties_enabled=True,
            credits_earned=150.0,
            reporting_frequency="Daily",
            notification_contacts=notification_contacts,
            escalation_matrix=escalation_matrix,
            review_schedule="Monthly",
            next_review_date=next_review,
            stakeholders=stakeholders,
            tags=tags,
            attachments=attachments,
            external_references=external_refs
        )
        
        assert sla.version == "2.1"
        assert sla.sla_type == SLAType.PERFORMANCE
        assert sla.service_hours == "24x7x365"
        assert sla.exclusions == exclusions
        assert sla.overall_target == 99.9
        assert sla.expiry_date == expiry_date
        assert sla.auto_renewal is True
        assert sla.renewal_period == "12 months"
        assert sla.current_compliance == 99.85
        assert sla.penalties_enabled is True
        assert sla.credits_earned == 150.0
        assert sla.notification_contacts == notification_contacts
        assert sla.escalation_matrix == escalation_matrix
        assert sla.next_review_date == next_review
        assert sla.stakeholders == stakeholders
        assert sla.tags == tags
        assert sla.attachments == attachments
        assert sla.external_references == external_refs
        
    def test_expiry_date_validation(self):
        """Test expiry date validation."""
        tenant_id = uuid4()
        effective_date = datetime.utcnow()
        invalid_expiry = effective_date - timedelta(days=1)  # Before effective date
        
        with pytest.raises(ValueError, match="Expiry date must be after effective date"):
            ServiceLevelAgreement(
                name="Invalid SLA",
                description="SLA with invalid expiry",
                sla_type=SLAType.AVAILABILITY,
                service_provider="Provider",
                service_consumer="Consumer",
                tenant_id=tenant_id,
                services_covered=["Service"],
                service_hours="24x7",
                metrics=[uuid4()],
                overall_target=99.0,
                measurement_period="Monthly",
                effective_date=effective_date,
                expiry_date=invalid_expiry,
                reporting_frequency="Weekly",
                review_schedule="Quarterly"
            )
            
    def test_is_active(self):
        """Test SLA active status check."""
        tenant_id = uuid4()
        now = datetime.utcnow()
        
        # Active SLA within date range
        active_sla = ServiceLevelAgreement(
            name="Active SLA",
            description="Active SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=tenant_id,
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=now - timedelta(days=30),
            expiry_date=now + timedelta(days=30),
            status=SLAStatus.ACTIVE,
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        assert active_sla.is_active() is True
        
        # Inactive status
        inactive_sla = ServiceLevelAgreement(
            name="Inactive SLA",
            description="Inactive SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=tenant_id,
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=now - timedelta(days=30),
            expiry_date=now + timedelta(days=30),
            status=SLAStatus.INACTIVE,
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        assert inactive_sla.is_active() is False
        
        # Not yet effective
        future_sla = ServiceLevelAgreement(
            name="Future SLA",
            description="Future SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=tenant_id,
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=now + timedelta(days=30),
            status=SLAStatus.ACTIVE,
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        assert future_sla.is_active() is False
        
        # Expired
        expired_sla = ServiceLevelAgreement(
            name="Expired SLA",
            description="Expired SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=tenant_id,
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=now - timedelta(days=60),
            expiry_date=now - timedelta(days=1),
            status=SLAStatus.ACTIVE,
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        assert expired_sla.is_active() is False
        
    def test_is_expired(self):
        """Test SLA expiry check."""
        tenant_id = uuid4()
        now = datetime.utcnow()
        
        # Not expired
        active_sla = ServiceLevelAgreement(
            name="Active SLA",
            description="Active SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=tenant_id,
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=now - timedelta(days=30),
            expiry_date=now + timedelta(days=30),
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        assert active_sla.is_expired() is False
        
        # Expired
        expired_sla = ServiceLevelAgreement(
            name="Expired SLA",
            description="Expired SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=tenant_id,
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=now - timedelta(days=60),
            expiry_date=now - timedelta(days=1),
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        assert expired_sla.is_expired() is True
        
        # No expiry date
        no_expiry_sla = ServiceLevelAgreement(
            name="No Expiry SLA",
            description="SLA without expiry",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=tenant_id,
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=now - timedelta(days=30),
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        assert no_expiry_sla.is_expired() is False
        
    def test_update_compliance(self):
        """Test compliance update."""
        sla = ServiceLevelAgreement(
            name="Test SLA",
            description="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=uuid4(),
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=datetime.utcnow(),
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        
        original_updated_at = sla.updated_at
        
        sla.update_compliance(97.5)
        
        assert sla.current_compliance == 97.5
        assert sla.last_compliance_check is not None
        assert sla.updated_at > original_updated_at
        
        # Test boundary values
        sla.update_compliance(-5.0)  # Should be clamped to 0
        assert sla.current_compliance == 0.0
        
        sla.update_compliance(105.0)  # Should be clamped to 100
        assert sla.current_compliance == 100.0
        
    def test_add_penalty(self):
        """Test adding penalty structure."""
        sla = ServiceLevelAgreement(
            name="Test SLA",
            description="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=uuid4(),
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=datetime.utcnow(),
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        
        original_updated_at = sla.updated_at
        
        sla.add_penalty("availability_violation", 100.0, "Service credit for availability violations")
        
        assert len(sla.penalty_structure) == 1
        penalty = sla.penalty_structure[0]
        assert penalty["violation_type"] == "availability_violation"
        assert penalty["penalty_amount"] == 100.0
        assert penalty["description"] == "Service credit for availability violations"
        assert "id" in penalty
        assert "created_at" in penalty
        assert sla.updated_at > original_updated_at
        
    def test_calculate_service_credits(self):
        """Test service credit calculation."""
        sla = ServiceLevelAgreement(
            name="Test SLA",
            description="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=uuid4(),
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=datetime.utcnow(),
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        
        # Add penalty structure
        sla.add_penalty("downtime", 50.0, "Credit per hour of downtime")
        
        original_credits = sla.credits_earned
        original_updated_at = sla.updated_at
        
        # Calculate credits for 2 hours of downtime
        credits = sla.calculate_service_credits(2.0, "downtime")
        
        assert credits == 100.0  # 50.0 * 2.0
        assert sla.credits_earned == original_credits + 100.0
        assert sla.updated_at > original_updated_at
        
        # Test with non-matching violation type
        credits = sla.calculate_service_credits(1.0, "performance")
        assert credits == 0.0
        
    def test_schedule_next_review(self):
        """Test scheduling next review."""
        sla = ServiceLevelAgreement(
            name="Test SLA",
            description="Test SLA",
            sla_type=SLAType.AVAILABILITY,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=uuid4(),
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.0,
            measurement_period="Monthly",
            effective_date=datetime.utcnow(),
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        
        original_updated_at = sla.updated_at
        
        sla.schedule_next_review(6)  # 6 months ahead
        
        assert sla.next_review_date is not None
        assert sla.next_review_date > datetime.utcnow()
        assert sla.updated_at > original_updated_at
        
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        sla = ServiceLevelAgreement(
            name="Performance SLA",
            description="Test SLA",
            sla_type=SLAType.PERFORMANCE,
            service_provider="Provider",
            service_consumer="Consumer",
            tenant_id=uuid4(),
            services_covered=["Service"],
            service_hours="24x7",
            metrics=[uuid4()],
            overall_target=99.5,
            measurement_period="Monthly",
            effective_date=datetime.utcnow() - timedelta(days=30),
            expiry_date=datetime.utcnow() + timedelta(days=30),
            status=SLAStatus.ACTIVE,
            current_compliance=97.8,
            credits_earned=250.0,
            reporting_frequency="Weekly",
            review_schedule="Quarterly"
        )
        
        summary = sla.get_performance_summary()
        
        assert summary["sla_name"] == "Performance SLA"
        assert summary["status"] == SLAStatus.ACTIVE
        assert summary["current_compliance"] == 97.8
        assert summary["overall_target"] == 99.5
        assert summary["is_meeting_target"] is False  # 97.8 < 99.5
        assert summary["credits_earned"] == 250.0
        assert summary["is_active"] is True
        assert summary["is_expired"] is False
        assert summary["days_until_expiry"] is not None


class TestSLAViolation:
    """Test cases for SLAViolation entity."""
    
    def test_sla_violation_creation_basic(self):
        """Test basic SLA violation creation."""
        sla_id = uuid4()
        metric_id = uuid4()
        tenant_id = uuid4()
        start_time = datetime.utcnow()
        
        violation = SLAViolation(
            sla_id=sla_id,
            metric_id=metric_id,
            tenant_id=tenant_id,
            violation_type="availability_breach",
            severity=SLAViolationSeverity.HIGH,
            description="System availability dropped below threshold",
            start_time=start_time,
            target_value=99.5,
            actual_value=98.2,
            deviation_percentage=1.3
        )
        
        assert isinstance(violation.id, UUID)
        assert violation.sla_id == sla_id
        assert violation.metric_id == metric_id
        assert violation.tenant_id == tenant_id
        assert violation.violation_type == "availability_breach"
        assert violation.severity == SLAViolationSeverity.HIGH
        assert violation.description == "System availability dropped below threshold"
        assert violation.start_time == start_time
        assert violation.end_time is None
        assert violation.duration_minutes is None
        assert violation.target_value == 99.5
        assert violation.actual_value == 98.2
        assert violation.deviation_percentage == 1.3
        assert violation.affected_services == []
        assert violation.affected_users is None
        assert violation.business_impact == ""
        assert violation.financial_impact is None
        assert violation.root_cause == ""
        assert violation.resolution_actions == []
        assert violation.preventive_measures == []
        assert violation.notifications_sent == []
        assert violation.escalated_to == []
        assert violation.escalation_level == 0
        assert violation.is_resolved is False
        assert violation.requires_follow_up is False
        assert violation.service_credits_due == 0.0
        assert violation.penalty_applied == 0.0
        assert violation.credit_applied is False
        
    def test_sla_violation_creation_comprehensive(self):
        """Test comprehensive SLA violation creation."""
        sla_id = uuid4()
        metric_id = uuid4()
        tenant_id = uuid4()
        start_time = datetime.utcnow() - timedelta(hours=2)
        end_time = datetime.utcnow() - timedelta(minutes=30)
        affected_services = ["API Gateway", "Database", "Cache"]
        resolution_actions = ["Restarted services", "Scaled up infrastructure"]
        preventive_measures = ["Add more monitoring", "Implement auto-scaling"]
        escalated_to = ["ops-manager@company.com", "vp-engineering@company.com"]
        tags = ["critical", "customer-impact", "infrastructure"]
        attachments = ["incident_report.pdf", "performance_graphs.png"]
        
        violation = SLAViolation(
            sla_id=sla_id,
            metric_id=metric_id,
            tenant_id=tenant_id,
            violation_type="performance_degradation",
            severity=SLAViolationSeverity.CRITICAL,
            description="Severe performance degradation affecting all services",
            start_time=start_time,
            end_time=end_time,
            duration_minutes=90.0,
            affected_services=affected_services,
            affected_users=5000,
            business_impact="Customer complaints increased by 300%",
            financial_impact=25000.0,
            target_value=500.0,
            actual_value=2500.0,
            deviation_percentage=400.0,
            root_cause="Database connection pool exhaustion",
            resolution_actions=resolution_actions,
            preventive_measures=preventive_measures,
            escalation_level=2,
            escalated_to=escalated_to,
            is_resolved=True,
            requires_follow_up=True,
            follow_up_date=datetime.utcnow() + timedelta(days=7),
            service_credits_due=500.0,
            penalty_applied=1000.0,
            credit_applied=True,
            tags=tags,
            attachments=attachments,
            acknowledged_at=start_time + timedelta(minutes=5),
            resolved_at=end_time
        )
        
        assert violation.violation_type == "performance_degradation"
        assert violation.severity == SLAViolationSeverity.CRITICAL
        assert violation.end_time == end_time
        assert violation.duration_minutes == 90.0
        assert violation.affected_services == affected_services
        assert violation.affected_users == 5000
        assert violation.business_impact == "Customer complaints increased by 300%"
        assert violation.financial_impact == 25000.0
        assert violation.deviation_percentage == 400.0
        assert violation.root_cause == "Database connection pool exhaustion"
        assert violation.resolution_actions == resolution_actions
        assert violation.preventive_measures == preventive_measures
        assert violation.escalation_level == 2
        assert violation.escalated_to == escalated_to
        assert violation.is_resolved is True
        assert violation.requires_follow_up is True
        assert violation.service_credits_due == 500.0
        assert violation.penalty_applied == 1000.0
        assert violation.credit_applied is True
        assert violation.tags == tags
        assert violation.attachments == attachments
        
    def test_deviation_percentage_calculation(self):
        """Test automatic deviation percentage calculation."""
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="test_violation",
            severity=SLAViolationSeverity.MEDIUM,
            description="Test violation",
            start_time=datetime.utcnow(),
            target_value=100.0,
            actual_value=80.0
        )
        
        # Should calculate: abs((80 - 100) / 100) * 100 = 20%
        assert violation.deviation_percentage == 20.0
        
    def test_deviation_percentage_zero_target(self):
        """Test deviation calculation with zero target."""
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="test_violation",
            severity=SLAViolationSeverity.MEDIUM,
            description="Test violation",
            start_time=datetime.utcnow(),
            target_value=0.0,
            actual_value=10.0
        )
        
        # Should default to 0 when target is 0
        assert violation.deviation_percentage == 0.0
        
    def test_acknowledge(self):
        """Test violation acknowledgment."""
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="test_violation",
            severity=SLAViolationSeverity.MEDIUM,
            description="Test violation",
            start_time=datetime.utcnow(),
            target_value=100.0,
            actual_value=80.0
        )
        
        acknowledged_by = "ops-team@company.com"
        original_updated_at = violation.updated_at
        
        violation.acknowledge(acknowledged_by)
        
        assert violation.acknowledged_at is not None
        assert len(violation.notifications_sent) == 1
        notification = violation.notifications_sent[0]
        assert notification["type"] == "acknowledged"
        assert notification["recipient"] == acknowledged_by
        assert notification["message"] == "Violation acknowledged"
        assert violation.updated_at > original_updated_at
        
    def test_resolve(self):
        """Test violation resolution."""
        start_time = datetime.utcnow() - timedelta(hours=2)
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="test_violation",
            severity=SLAViolationSeverity.MEDIUM,
            description="Test violation",
            start_time=start_time,
            target_value=100.0,
            actual_value=80.0
        )
        
        resolved_by = "engineering-team@company.com"
        resolution_notes = "Fixed by restarting the database connection pool"
        original_updated_at = violation.updated_at
        
        violation.resolve(resolved_by, resolution_notes)
        
        assert violation.is_resolved is True
        assert violation.resolved_at is not None
        assert violation.end_time is not None
        assert violation.duration_minutes is not None
        assert violation.duration_minutes > 0
        assert resolution_notes in violation.resolution_actions[0]
        assert len(violation.notifications_sent) == 1
        notification = violation.notifications_sent[0]
        assert notification["type"] == "resolved"
        assert notification["recipient"] == resolved_by
        assert violation.updated_at > original_updated_at
        
    def test_escalate(self):
        """Test violation escalation."""
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="test_violation",
            severity=SLAViolationSeverity.HIGH,
            description="Test violation",
            start_time=datetime.utcnow(),
            target_value=100.0,
            actual_value=70.0
        )
        
        escalated_to = "engineering-manager@company.com"
        escalation_reason = "Violation not resolved within SLA timeframe"
        original_updated_at = violation.updated_at
        
        violation.escalate(escalated_to, escalation_reason)
        
        assert violation.escalation_level == 1
        assert escalated_to in violation.escalated_to
        assert len(violation.notifications_sent) == 1
        notification = violation.notifications_sent[0]
        assert notification["type"] == "escalated"
        assert notification["recipient"] == escalated_to
        assert "Escalated to level 1" in notification["message"]
        assert escalation_reason in notification["message"]
        assert violation.updated_at > original_updated_at
        
        # Test multiple escalations
        violation.escalate("vp-engineering@company.com", "Still not resolved")
        assert violation.escalation_level == 2
        assert len(violation.escalated_to) == 2
        
    def test_add_notification(self):
        """Test adding notification records."""
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="test_violation",
            severity=SLAViolationSeverity.MEDIUM,
            description="Test violation",
            start_time=datetime.utcnow(),
            target_value=100.0,
            actual_value=80.0
        )
        
        violation.add_notification("alert", "ops@company.com", "SLA violation detected")
        
        assert len(violation.notifications_sent) == 1
        notification = violation.notifications_sent[0]
        assert notification["type"] == "alert"
        assert notification["recipient"] == "ops@company.com"
        assert notification["message"] == "SLA violation detected"
        assert "id" in notification
        assert "sent_at" in notification
        
    def test_calculate_service_credits(self):
        """Test service credit calculation."""
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="availability_violation",
            severity=SLAViolationSeverity.HIGH,
            description="Availability violation",
            start_time=datetime.utcnow() - timedelta(hours=2),
            target_value=99.5,
            actual_value=97.0,
            duration_minutes=120.0
        )
        
        credit_rate = 1.5  # $1.50 per minute
        original_updated_at = violation.updated_at
        
        credits = violation.calculate_service_credits(credit_rate)
        
        # Expected: 120 minutes * $1.50 * 2.0 (HIGH severity multiplier) = $360
        expected_credits = 120.0 * 1.5 * 2.0
        assert credits == expected_credits
        assert violation.service_credits_due == expected_credits
        assert violation.updated_at > original_updated_at
        
        # Test with no duration
        no_duration_violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="test_violation",
            severity=SLAViolationSeverity.LOW,
            description="Test violation",
            start_time=datetime.utcnow(),
            target_value=100.0,
            actual_value=90.0
        )
        
        credits = no_duration_violation.calculate_service_credits(1.0)
        assert credits == 0.0
        
    def test_severity_multiplier(self):
        """Test severity multiplier calculation."""
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="test_violation",
            severity=SLAViolationSeverity.CRITICAL,
            description="Test violation",
            start_time=datetime.utcnow(),
            target_value=100.0,
            actual_value=80.0
        )
        
        assert violation.severity_multiplier() == 3.0
        
        violation.severity = SLAViolationSeverity.HIGH
        assert violation.severity_multiplier() == 2.0
        
        violation.severity = SLAViolationSeverity.MEDIUM
        assert violation.severity_multiplier() == 1.5
        
        violation.severity = SLAViolationSeverity.LOW
        assert violation.severity_multiplier() == 1.0
        
        violation.severity = SLAViolationSeverity.INFO
        assert violation.severity_multiplier() == 0.5
        
    def test_is_ongoing(self):
        """Test ongoing violation check."""
        ongoing_violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="ongoing_violation",
            severity=SLAViolationSeverity.MEDIUM,
            description="Ongoing violation",
            start_time=datetime.utcnow() - timedelta(hours=1),
            target_value=100.0,
            actual_value=80.0,
            is_resolved=False
        )
        assert ongoing_violation.is_ongoing() is True
        
        resolved_violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="resolved_violation",
            severity=SLAViolationSeverity.MEDIUM,
            description="Resolved violation",
            start_time=datetime.utcnow() - timedelta(hours=2),
            target_value=100.0,
            actual_value=80.0,
            is_resolved=True,
            end_time=datetime.utcnow() - timedelta(hours=1)
        )
        assert resolved_violation.is_ongoing() is False
        
        ended_violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="ended_violation",
            severity=SLAViolationSeverity.MEDIUM,
            description="Ended violation",
            start_time=datetime.utcnow() - timedelta(hours=2),
            target_value=100.0,
            actual_value=80.0,
            end_time=datetime.utcnow() - timedelta(hours=1)
        )
        assert ended_violation.is_ongoing() is False
        
    def test_get_violation_summary(self):
        """Test violation summary generation."""
        start_time = datetime.utcnow() - timedelta(hours=1)
        violation = SLAViolation(
            sla_id=uuid4(),
            metric_id=uuid4(),
            tenant_id=uuid4(),
            violation_type="performance_issue",
            severity=SLAViolationSeverity.HIGH,
            description="Performance degradation",
            start_time=start_time,
            duration_minutes=60.0,
            target_value=500.0,
            actual_value=1200.0,
            affected_services=["API", "Database"],
            business_impact="Customer complaints increased",
            service_credits_due=300.0,
            escalation_level=1,
            is_resolved=True
        )
        
        summary = violation.get_violation_summary()
        
        assert summary["id"] == str(violation.id)
        assert summary["severity"] == SLAViolationSeverity.HIGH
        assert summary["violation_type"] == "performance_issue"
        assert summary["start_time"] == start_time.isoformat()
        assert summary["duration_minutes"] == 60.0
        assert summary["is_resolved"] is True
        assert summary["is_ongoing"] is False
        assert summary["deviation_percentage"] == violation.deviation_percentage
        assert summary["service_credits_due"] == 300.0
        assert summary["escalation_level"] == 1
        assert summary["affected_services"] == ["API", "Database"]
        assert summary["business_impact"] == "Customer complaints increased"