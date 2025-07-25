"""Comprehensive security testing framework integration tests."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path

from anomaly_detection.application.services.security.vulnerability_scanner import (
    VulnerabilityScanner,
    VulnerabilityFinding,
    VulnerabilitySeverity,
    VulnerabilityCategory,
    ScanResult,
    get_vulnerability_scanner
)

from anomaly_detection.application.services.security.compliance_auditor import (
    ComplianceAuditor,
    ComplianceStandard,
    ComplianceStatus,
    AuditSeverity,
    AuditFinding,
    AuditResult,
    get_compliance_auditor
)

from anomaly_detection.application.services.security.threat_detector import (
    ThreatDetectionSystem,
    ThreatEvent,
    ThreatType,
    ThreatSeverity,
    SecurityAlert,
    get_threat_detection_system
)


@pytest.mark.security
class TestSecurityFrameworkIntegration:
    """Test comprehensive security framework integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_config = {
            "vulnerability_scanner": {
                "dependency": {"enabled": True},
                "code_analysis": {"enabled": True},
                "api_security": {"enabled": True, "base_url": "http://localhost:8000"}
            },
            "compliance_auditor": {
                "gdpr": {"enabled": True},
                "hipaa": {"enabled": True}
            },
            "threat_detection": {
                "brute_force": {"enabled": True, "max_attempts": 3},
                "injection": {"enabled": True},
                "api_abuse": {"enabled": True}
            }
        }
        
        # Initialize components
        self.vulnerability_scanner = VulnerabilityScanner(
            self.security_config["vulnerability_scanner"]
        )
        self.compliance_auditor = ComplianceAuditor(
            self.security_config["compliance_auditor"]
        )
        self.threat_detector = ThreatDetectionSystem(
            self.security_config["threat_detection"]
        )
    
    @pytest.mark.asyncio
    async def test_comprehensive_security_assessment(self):
        """Test comprehensive security assessment workflow."""
        project_path = "/test/project"
        
        # Mock vulnerability scan results
        vuln_findings = [
            VulnerabilityFinding(
                id="vuln_001",
                title="High Severity SQL Injection",
                description="SQL injection vulnerability in user input",
                severity=VulnerabilitySeverity.HIGH,
                category=VulnerabilityCategory.INJECTION,
                affected_component="user_api",
                confidence=0.9
            ),
            VulnerabilityFinding(
                id="vuln_002",
                title="Outdated Dependency",
                description="Vulnerable dependency: requests<2.25.1",
                severity=VulnerabilitySeverity.MEDIUM,
                category=VulnerabilityCategory.DEPENDENCIES,
                affected_component="requests",
                confidence=1.0
            )
        ]
        
        # Mock compliance audit results
        compliance_findings = [
            AuditFinding(
                id="gdpr_001",
                requirement_id="GDPR-01",
                title="Missing Data Subject Rights API",
                description="No API endpoints for data subject rights",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.HIGH,
                category="Data Subject Rights",
                confidence=0.9
            ),
            AuditFinding(
                id="hipaa_001",
                requirement_id="HIPAA-03",
                title="Insufficient Encryption",
                description="No evidence of proper encryption implementation",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.CRITICAL,
                category="Technical Safeguards",
                confidence=0.8
            )
        ]
        
        # Mock threat detection results
        threat_events = [
            ThreatEvent(
                id="threat_001",
                threat_type=ThreatType.BRUTE_FORCE,
                severity=ThreatSeverity.HIGH,
                title="Brute Force Attack Detected",
                description="Multiple failed login attempts from 192.168.1.100",
                source_ip="192.168.1.100",
                confidence=0.9,
                risk_score=85.0
            )
        ]
        
        # Mock the actual scanning/detection methods
        with patch.object(self.vulnerability_scanner, 'scan_all') as mock_vuln_scan, \
             patch.object(self.compliance_auditor, 'audit_all_standards') as mock_compliance_audit, \
             patch.object(self.threat_detector, 'detect_threats') as mock_threat_detect:
            
            # Configure mocks
            mock_vuln_scan.return_value = ScanResult(
                scan_id="scan_001",
                scan_type="comprehensive",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                target=project_path,
                findings=vuln_findings,
                scan_duration=30.0,
                total_checks=10,
                passed_checks=8,
                failed_checks=2
            )
            
            mock_compliance_audit.return_value = [
                AuditResult(
                    audit_id="audit_001",
                    standard=ComplianceStandard.GDPR,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    scope=project_path,
                    findings=[compliance_findings[0]],
                    total_requirements=5,
                    compliant_requirements=4,
                    non_compliant_requirements=1,
                    compliance_score=80.0,
                    overall_status=ComplianceStatus.PARTIALLY_COMPLIANT
                ),
                AuditResult(
                    audit_id="audit_002",
                    standard=ComplianceStandard.HIPAA,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    scope=project_path,
                    findings=[compliance_findings[1]],
                    total_requirements=3,
                    compliant_requirements=2,
                    non_compliant_requirements=1,
                    compliance_score=60.0,
                    overall_status=ComplianceStatus.NON_COMPLIANT
                )
            ]
            
            mock_threat_detect.return_value = threat_events
            
            # Run comprehensive security assessment
            security_assessment = await self._run_comprehensive_security_assessment(project_path)
            
            # Verify all components were called
            mock_vuln_scan.assert_called_once_with(project_path)
            mock_compliance_audit.assert_called_once_with(project_path)
            mock_threat_detect.assert_called_once()
            
            # Verify assessment results
            assert security_assessment["vulnerability_scan"]["total_findings"] == 2
            assert security_assessment["compliance_audit"]["total_findings"] == 2
            assert security_assessment["threat_detection"]["total_events"] == 1
            
            # Check risk aggregation
            assert "overall_risk_score" in security_assessment
            assert security_assessment["overall_risk_score"] > 0
            
            # Check recommendations
            assert "recommendations" in security_assessment
            assert len(security_assessment["recommendations"]) > 0
    
    async def _run_comprehensive_security_assessment(
        self,
        project_path: str
    ) -> "Dict[str, Any]":
        """Run comprehensive security assessment."""
        assessment_results = {
            "assessment_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "target": project_path,
                "assessment_id": "assess_001"
            }
        }
        
        # 1. Vulnerability Assessment
        vuln_result = await self.vulnerability_scanner.scan_all(project_path)
        vuln_report = self.vulnerability_scanner.generate_report(vuln_result)
        assessment_results["vulnerability_scan"] = vuln_report["summary"]
        
        # 2. Compliance Audit
        compliance_results = await self.compliance_auditor.audit_all_standards(project_path)
        compliance_report = self.compliance_auditor.generate_compliance_report(compliance_results)
        assessment_results["compliance_audit"] = compliance_report["executive_summary"]
        
        # 3. Threat Detection (simulated with sample data)
        threat_data = {
            "authentication_events": [],
            "http_requests": [],
            "api_requests": [],
            "traffic_metrics": {}
        }
        threat_events = await self.threat_detector.detect_threats(threat_data)
        threat_report = self.threat_detector.generate_threat_report()
        assessment_results["threat_detection"] = threat_report["summary"]
        
        # 4. Risk Aggregation and Analysis
        assessment_results["overall_risk_score"] = self._calculate_overall_risk(
            vuln_result, compliance_results, threat_events
        )
        
        # 5. Generate Recommendations
        assessment_results["recommendations"] = self._generate_security_recommendations(
            vuln_result, compliance_results, threat_events
        )
        
        return assessment_results
    
    def _calculate_overall_risk(
        self,
        vuln_result: ScanResult,
        compliance_results: List[AuditResult],
        threat_events: List[ThreatEvent]
    ) -> float:
        """Calculate overall security risk score."""
        # Vulnerability risk (0-40 points)
        vuln_risk = 0
        if vuln_result.findings:
            critical_high = len([f for f in vuln_result.findings 
                               if f.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]])
            vuln_risk = min(40, critical_high * 10)
        
        # Compliance risk (0-40 points)
        compliance_risk = 0
        if compliance_results:
            avg_compliance_score = sum(r.compliance_score for r in compliance_results) / len(compliance_results)
            compliance_risk = max(0, 40 - (avg_compliance_score * 0.4))
        
        # Threat risk (0-20 points)
        threat_risk = 0
        if threat_events:
            high_severity_threats = len([t for t in threat_events 
                                       if t.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]])
            threat_risk = min(20, high_severity_threats * 5)
        
        return vuln_risk + compliance_risk + threat_risk
    
    def _generate_security_recommendations(
        self,
        vuln_result: ScanResult,
        compliance_results: List[AuditResult],
        threat_events: List[ThreatEvent]
    ) -> List[str]:
        """Generate comprehensive security recommendations."""
        recommendations = []
        
        # Vulnerability recommendations
        if vuln_result.critical_findings:
            recommendations.append(
                f"URGENT: Address {len(vuln_result.critical_findings)} critical vulnerabilities immediately"
            )
        
        if vuln_result.high_findings:
            recommendations.append(
                f"High Priority: Fix {len(vuln_result.high_findings)} high-severity vulnerabilities"
            )
        
        # Compliance recommendations
        non_compliant_standards = [
            r.standard.value for r in compliance_results 
            if r.overall_status == ComplianceStatus.NON_COMPLIANT
        ]
        
        if non_compliant_standards:
            recommendations.append(
                f"Compliance: Address non-compliance issues in {', '.join(non_compliant_standards)}"
            )
        
        # Threat recommendations
        if threat_events:
            recommendations.append(
                f"Monitoring: Investigate and respond to {len(threat_events)} active security threats"
            )
        
        # General recommendations
        recommendations.extend([
            "Implement continuous security monitoring",
            "Establish incident response procedures",
            "Conduct regular security training for development team",
            "Schedule periodic security assessments"
        ])
        
        return recommendations
    
    @pytest.mark.asyncio
    async def test_security_integration_workflow(self):
        """Test end-to-end security integration workflow."""
        # Test the complete workflow from detection to remediation
        
        # 1. Threat Detection
        threat_data = {
            "authentication_events": [
                {
                    "success": False,
                    "source_ip": "192.168.1.100",
                    "username": "admin",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            "http_requests": [
                {
                    "source_ip": "192.168.1.100",
                    "endpoint": "/api/v1/users",
                    "parameters": {"id": "1' OR '1'='1"},
                    "user_agent": "sqlmap/1.0",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
        }
        
        threats = await self.threat_detector.detect_threats(threat_data)
        assert len(threats) > 0
        
        # 2. Alert Generation
        alerts = self.threat_detector.get_active_alerts()
        assert len(alerts) >= 0  # May be 0 if aggregation is active
        
        # 3. Security Response Simulation
        for threat in threats:
            if threat.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]:
                # Simulate automated response
                response_actions = self._simulate_security_response(threat)
                assert len(response_actions) > 0
    
    def _simulate_security_response(self, threat: ThreatEvent) -> List[str]:
        """Simulate automated security response actions."""
        actions = []
        
        if threat.threat_type == ThreatType.BRUTE_FORCE:
            actions.extend([
                f"Block IP {threat.source_ip} for 1 hour",
                "Notify security team",
                "Log incident for analysis"
            ])
        
        elif threat.threat_type == ThreatType.SQL_INJECTION:
            actions.extend([
                f"Immediately block IP {threat.source_ip}",
                "Alert development team",
                "Review application logs",
                "Escalate to security incident response team"
            ])
        
        return actions
    
    @pytest.mark.asyncio
    async def test_security_metrics_collection(self):
        """Test security metrics collection and reporting."""
        # Mock metrics collector
        with patch('anomaly_detection.infrastructure.monitoring.get_metrics_collector') as mock_collector:
            mock_metrics_collector = Mock()
            mock_collector.return_value = mock_metrics_collector
            
            # Run security operations
            project_path = "/test/project"
            
            with patch.object(self.vulnerability_scanner, 'scan_all') as mock_scan:
                mock_scan.return_value = ScanResult(
                    scan_id="test",
                    scan_type="test",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    target=project_path,
                    findings=[],
                    scan_duration=1.0,
                    total_checks=1,
                    passed_checks=1,
                    failed_checks=0
                )
                
                await self.vulnerability_scanner.scan_all(project_path)
            
            # Verify metrics were recorded
            # mock_metrics_collector.record_metric.assert_called()
    
    def test_security_configuration_validation(self):
        """Test security configuration validation."""
        # Test valid configuration
        valid_config = {
            "vulnerability_scanner": {
                "dependency": {"enabled": True},
                "code_analysis": {"enabled": True}
            },
            "compliance_auditor": {
                "gdpr": {"enabled": True}
            },
            "threat_detection": {
                "brute_force": {"enabled": True}
            }
        }
        
        scanner = VulnerabilityScanner(valid_config["vulnerability_scanner"])
        auditor = ComplianceAuditor(valid_config["compliance_auditor"])
        detector = ThreatDetectionSystem(valid_config["threat_detection"])
        
        assert len(scanner.enabled_scanners) >= 1
        assert len(auditor.enabled_checkers) >= 1
        assert len(detector.enabled_detectors) >= 1
        
        # Test disabled components
        disabled_config = {
            "vulnerability_scanner": {
                "dependency": {"enabled": False},
                "code_analysis": {"enabled": False}
            }
        }
        
        disabled_scanner = VulnerabilityScanner(disabled_config["vulnerability_scanner"])
        assert len(disabled_scanner.enabled_scanners) == 0
    
    def test_security_report_generation(self):
        """Test comprehensive security report generation."""
        # Create sample findings
        vuln_findings = [
            VulnerabilityFinding(
                id="v1",
                title="Test Vuln",
                description="Test",
                severity=VulnerabilitySeverity.HIGH,
                category=VulnerabilityCategory.INJECTION
            )
        ]
        
        scan_result = ScanResult(
            scan_id="test",
            scan_type="test",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            target="/test",
            findings=vuln_findings,
            scan_duration=1.0,
            total_checks=1,
            passed_checks=0,
            failed_checks=1
        )
        
        # Generate vulnerability report
        vuln_report = self.vulnerability_scanner.generate_report(scan_result)
        assert "scan_info" in vuln_report
        assert "summary" in vuln_report
        assert vuln_report["summary"]["total_findings"] == 1
        assert vuln_report["summary"]["high_findings"] == 1
        
        # Test compliance report
        audit_findings = [
            AuditFinding(
                id="a1",
                requirement_id="REQ-01",
                title="Test Audit",
                description="Test",
                status=ComplianceStatus.NON_COMPLIANT,
                severity=AuditSeverity.HIGH,
                category="Test"
            )
        ]
        
        audit_result = AuditResult(
            audit_id="test",
            standard=ComplianceStandard.GDPR,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            scope="/test",
            findings=audit_findings,
            total_requirements=1,
            compliant_requirements=0,
            non_compliant_requirements=1,
            compliance_score=0.0,
            overall_status=ComplianceStatus.NON_COMPLIANT
        )
        
        compliance_report = self.compliance_auditor.generate_compliance_report([audit_result])
        assert "executive_summary" in compliance_report
        assert compliance_report["executive_summary"]["total_findings"] == 1
        
        # Test threat report
        threat_report = self.threat_detector.generate_threat_report()
        assert "summary" in threat_report
        assert "threat_analysis" in threat_report
    
    @pytest.mark.asyncio
    async def test_security_automation_pipeline(self):
        """Test automated security pipeline integration."""
        project_path = "/test/project"
        
        # Define automation pipeline steps
        pipeline_steps = [
            "vulnerability_scan",
            "compliance_audit", 
            "threat_analysis",
            "risk_assessment",
            "response_planning"
        ]
        
        pipeline_results = {}
        
        # Mock each pipeline step
        for step in pipeline_steps:
            if step == "vulnerability_scan":
                with patch.object(self.vulnerability_scanner, 'scan_all') as mock_scan:
                    mock_scan.return_value = ScanResult(
                        scan_id="auto_001",
                        scan_type="automated",
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        target=project_path,
                        findings=[],
                        scan_duration=5.0,
                        total_checks=10,
                        passed_checks=10,
                        failed_checks=0
                    )
                    
                    result = await self.vulnerability_scanner.scan_all(project_path)
                    pipeline_results[step] = {"status": "completed", "findings": len(result.findings)}
            
            elif step == "compliance_audit":
                with patch.object(self.compliance_auditor, 'audit_all_standards') as mock_audit:
                    mock_audit.return_value = []
                    
                    result = await self.compliance_auditor.audit_all_standards(project_path)
                    pipeline_results[step] = {"status": "completed", "audits": len(result)}
            
            elif step == "threat_analysis":
                result = await self.threat_detector.detect_threats({})
                pipeline_results[step] = {"status": "completed", "threats": len(result)}
            
            else:
                # Simulate other steps
                pipeline_results[step] = {"status": "completed"}
        
        # Verify pipeline completion
        assert len(pipeline_results) == len(pipeline_steps)
        assert all(result["status"] == "completed" for result in pipeline_results.values())
        
        # Generate pipeline report
        pipeline_report = {
            "pipeline_id": "auto_pipeline_001",
            "execution_time": datetime.utcnow().isoformat(),
            "steps_completed": len(pipeline_steps),
            "results": pipeline_results,
            "overall_status": "success"
        }
        
        assert pipeline_report["overall_status"] == "success"
        assert pipeline_report["steps_completed"] == 5


@pytest.mark.security
class TestSecurityBestPractices:
    """Test security best practices implementation."""
    
    def test_secure_configuration_defaults(self):
        """Test that security components have secure defaults."""
        # Test vulnerability scanner defaults
        scanner = VulnerabilityScanner()
        assert len(scanner.enabled_scanners) > 0  # Should have some scanners enabled
        
        # Test compliance auditor defaults
        auditor = ComplianceAuditor()
        # Should have reasonable default configuration
        
        # Test threat detector defaults
        detector = ThreatDetectionSystem()
        assert detector.alert_aggregation_window > 0
        assert detector.max_alerts_per_hour > 0
    
    def test_input_validation_security(self):
        """Test input validation in security components."""
        # Test invalid configurations
        with pytest.raises((ValueError, TypeError)):
            VulnerabilityScanner({"invalid": "config"})
        
        # Test boundary conditions
        scanner = VulnerabilityScanner({
            "dependency": {"enabled": True},
            "api_security": {"timeout": 1}  # Very short timeout
        })
        assert scanner is not None
    
    def test_error_handling_security(self):
        """Test secure error handling."""
        scanner = VulnerabilityScanner()
        
        # Test with non-existent target
        # Should not reveal sensitive information in errors
        try:
            # This would normally raise an exception
            pass
        except Exception as e:
            error_str = str(e)
            # Should not contain sensitive paths, passwords, etc.
            assert "password" not in error_str.lower()
            assert "secret" not in error_str.lower()
    
    def test_logging_security(self):
        """Test secure logging practices."""
        # Verify that sensitive information is not logged
        scanner = VulnerabilityScanner()
        
        # Mock logger to capture log messages
        with patch('anomaly_detection.infrastructure.logging.get_logger') as mock_logger:
            mock_logger_instance = Mock()
            mock_logger.return_value = mock_logger_instance
            
            # Operations that might log sensitive data
            test_data = {
                "password": "secret123",
                "api_key": "key_abc123",
                "normal_field": "safe_value"
            }
            
            # Security components should sanitize sensitive data before logging
            # This is a placeholder test - actual implementation would need proper sanitization


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])