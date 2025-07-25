#!/usr/bin/env python3

"""
Comprehensive Security Audit and Penetration Testing Framework
This script performs automated security testing for the MLOps platform
"""

import asyncio
import json
import subprocess
import yaml
import requests
import ssl
import socket
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import urllib3
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

# Disable SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityFinding:
    """Represents a security finding"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str
    title: str
    description: str
    recommendation: str
    affected_component: str
    cve_id: Optional[str] = None
    confidence: str = "HIGH"
    evidence: Optional[Dict] = None

@dataclass
class SecurityAuditReport:
    """Complete security audit report"""
    timestamp: str
    environment: str
    target_domain: str
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    info_count: int
    findings: List[SecurityFinding]
    scan_duration: float
    tools_used: List[str]
    compliance_status: Dict[str, str]

class SecurityAuditor:
    """Main security auditing class"""
    
    def __init__(self, target_domain: str, environment: str = "staging"):
        self.target_domain = target_domain
        self.environment = environment
        self.findings: List[SecurityFinding] = []
        self.tools_used: List[str] = []
        self.start_time = datetime.now()
        
        # Define target endpoints
        self.endpoints = {
            'api': f'https://api.{target_domain}',
            'web': f'https://app.{target_domain}',
            'monitoring': f'https://monitoring.{target_domain}'
        }
    
    def add_finding(self, finding: SecurityFinding):
        """Add a security finding"""
        self.findings.append(finding)
        logger.warning(f"Security Finding [{finding.severity}]: {finding.title}")
    
    async def run_comprehensive_audit(self) -> SecurityAuditReport:
        """Run comprehensive security audit"""
        logger.info(f"Starting comprehensive security audit for {self.target_domain}")
        
        # Run all security tests
        await asyncio.gather(
            self.test_ssl_tls_configuration(),
            self.test_authentication_security(),
            self.test_api_security(),
            self.test_infrastructure_security(),
            self.test_container_security(),
            self.test_network_security(),
            self.test_data_protection(),
            self.test_compliance_controls()
        )
        
        # Generate final report
        return self._generate_report()
    
    async def test_ssl_tls_configuration(self):
        """Test SSL/TLS configuration and certificate security"""
        logger.info("Testing SSL/TLS configuration...")
        self.tools_used.append("SSL/TLS Scanner")
        
        for name, endpoint in self.endpoints.items():
            try:
                # Test SSL certificate
                hostname = endpoint.replace('https://', '').replace('http://', '')
                context = ssl.create_default_context()
                
                with socket.create_connection((hostname, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Check certificate expiry
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.now()).days
                        
                        if days_until_expiry < 30:
                            self.add_finding(SecurityFinding(
                                severity="HIGH",
                                category="SSL/TLS",
                                title=f"SSL Certificate Expiring Soon - {name}",
                                description=f"SSL certificate for {hostname} expires in {days_until_expiry} days",
                                recommendation="Renew SSL certificate before expiry",
                                affected_component=f"{name} endpoint",
                                evidence={"expiry_date": cert['notAfter'], "days_remaining": days_until_expiry}
                            ))
                        
                        # Check for weak cipher suites
                        cipher = ssock.cipher()
                        if cipher and len(cipher) >= 3:
                            cipher_suite = cipher[0]
                            if any(weak in cipher_suite.lower() for weak in ['rc4', 'des', 'md5']):
                                self.add_finding(SecurityFinding(
                                    severity="MEDIUM",
                                    category="SSL/TLS",
                                    title=f"Weak Cipher Suite - {name}",
                                    description=f"Weak cipher suite detected: {cipher_suite}",
                                    recommendation="Disable weak cipher suites and use modern TLS configuration",
                                    affected_component=f"{name} endpoint",
                                    evidence={"cipher_suite": cipher_suite}
                                ))
                
                # Test for HSTS header
                response = requests.get(endpoint, timeout=10, verify=False)
                if 'Strict-Transport-Security' not in response.headers:
                    self.add_finding(SecurityFinding(
                        severity="MEDIUM",
                        category="SSL/TLS",
                        title=f"Missing HSTS Header - {name}",
                        description="HTTP Strict Transport Security header is missing",
                        recommendation="Add HSTS header to prevent protocol downgrade attacks",
                        affected_component=f"{name} endpoint"
                    ))
                
                # Test for secure headers
                security_headers = {
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'DENY',
                    'X-XSS-Protection': '1; mode=block',
                    'Content-Security-Policy': 'default-src'
                }
                
                for header, expected in security_headers.items():
                    if header not in response.headers:
                        self.add_finding(SecurityFinding(
                            severity="MEDIUM",
                            category="HTTP Security",
                            title=f"Missing Security Header: {header} - {name}",
                            description=f"Missing security header: {header}",
                            recommendation=f"Add {header} header with appropriate value",
                            affected_component=f"{name} endpoint"
                        ))
                
            except Exception as e:
                logger.error(f"SSL/TLS test failed for {name}: {e}")
                self.add_finding(SecurityFinding(
                    severity="HIGH",
                    category="SSL/TLS",
                    title=f"SSL/TLS Connection Failed - {name}",
                    description=f"Failed to establish secure connection: {str(e)}",
                    recommendation="Investigate SSL/TLS configuration issues",
                    affected_component=f"{name} endpoint"
                ))
    
    async def test_authentication_security(self):
        """Test authentication and authorization security"""
        logger.info("Testing authentication security...")
        self.tools_used.append("Authentication Scanner")
        
        api_endpoint = self.endpoints['api']
        
        # Test for default credentials
        default_creds = [
            ('admin', 'admin'),
            ('admin', 'password'),
            ('root', 'root'),
            ('admin', '123456'),
            ('test', 'test')
        ]
        
        for username, password in default_creds:
            try:
                response = requests.post(
                    f"{api_endpoint}/api/v1/auth/login",
                    json={"username": username, "password": password},
                    timeout=5,
                    verify=False
                )
                
                if response.status_code == 200:
                    self.add_finding(SecurityFinding(
                        severity="CRITICAL",
                        category="Authentication",
                        title="Default Credentials Found",
                        description=f"Default credentials work: {username}:{password}",
                        recommendation="Change all default passwords immediately",
                        affected_component="Authentication system",
                        evidence={"username": username, "status_code": response.status_code}
                    ))
            except requests.exceptions.RequestException:
                pass  # Expected for secure systems
        
        # Test for weak password policy
        weak_passwords = ['123456', 'password', 'admin', '12345678']
        for weak_pass in weak_passwords:
            try:
                response = requests.post(
                    f"{api_endpoint}/api/v1/auth/register",
                    json={"username": "testuser", "password": weak_pass},
                    timeout=5,
                    verify=False
                )
                
                if response.status_code == 201:
                    self.add_finding(SecurityFinding(
                        severity="HIGH",
                        category="Authentication",
                        title="Weak Password Policy",
                        description="System accepts weak passwords",
                        recommendation="Implement strong password policy with complexity requirements",
                        affected_component="User registration",
                        evidence={"weak_password": weak_pass}
                    ))
            except requests.exceptions.RequestException:
                pass
        
        # Test for JWT security
        try:
            # Try to access protected endpoint without token
            response = requests.get(
                f"{api_endpoint}/api/v1/admin/users",
                timeout=5,
                verify=False
            )
            
            if response.status_code != 401:
                self.add_finding(SecurityFinding(
                    severity="HIGH",
                    category="Authorization",
                    title="Missing Authentication Check",
                    description="Protected endpoint accessible without authentication",
                    recommendation="Implement proper authentication checks for protected endpoints",
                    affected_component="API authorization",
                    evidence={"endpoint": "/api/v1/admin/users", "status_code": response.status_code}
                ))
        except requests.exceptions.RequestException:
            pass
    
    async def test_api_security(self):
        """Test API security vulnerabilities"""
        logger.info("Testing API security...")
        self.tools_used.append("API Security Scanner")
        
        api_endpoint = self.endpoints['api']
        
        # Test for SQL injection
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT NULL--",
            "1' OR 1=1#"
        ]
        
        for payload in sql_payloads:
            try:
                response = requests.get(
                    f"{api_endpoint}/api/v1/users",
                    params={"search": payload},
                    timeout=5,
                    verify=False
                )
                
                # Look for database error messages
                if any(error in response.text.lower() for error in ['sql', 'mysql', 'postgres', 'sqlite', 'oracle']):
                    self.add_finding(SecurityFinding(
                        severity="CRITICAL",
                        category="Injection",
                        title="SQL Injection Vulnerability",
                        description="SQL injection vulnerability detected in search parameter",
                        recommendation="Use parameterized queries and input validation",
                        affected_component="User search API",
                        cve_id="CWE-89",
                        evidence={"payload": payload, "response_snippet": response.text[:200]}
                    ))
            except requests.exceptions.RequestException:
                pass
        
        # Test for XSS
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            try:
                response = requests.post(
                    f"{api_endpoint}/api/v1/feedback",
                    json={"message": payload},
                    timeout=5,
                    verify=False
                )
                
                if payload in response.text:
                    self.add_finding(SecurityFinding(
                        severity="HIGH",
                        category="Injection",
                        title="Cross-Site Scripting (XSS) Vulnerability",
                        description="XSS vulnerability detected in feedback form",
                        recommendation="Implement proper output encoding and input validation",
                        affected_component="Feedback API",
                        cve_id="CWE-79",
                        evidence={"payload": payload}
                    ))
            except requests.exceptions.RequestException:
                pass
        
        # Test for IDOR (Insecure Direct Object References)
        try:
            for user_id in [1, 2, 9999, -1]:
                response = requests.get(
                    f"{api_endpoint}/api/v1/users/{user_id}",
                    timeout=5,
                    verify=False
                )
                
                if response.status_code == 200 and 'user' in response.text.lower():
                    self.add_finding(SecurityFinding(
                        severity="HIGH",
                        category="Authorization",
                        title="Insecure Direct Object Reference",
                        description=f"User data accessible without proper authorization (user_id: {user_id})",
                        recommendation="Implement proper access controls and user authorization checks",
                        affected_component="User API",
                        cve_id="CWE-639",
                        evidence={"user_id": user_id, "status_code": response.status_code}
                    ))
        except requests.exceptions.RequestException:
            pass
    
    async def test_infrastructure_security(self):
        """Test infrastructure security"""
        logger.info("Testing infrastructure security...")
        self.tools_used.append("Infrastructure Scanner")
        
        # Test for exposed admin interfaces
        admin_paths = [
            '/admin',
            '/admin/',
            '/administrator',
            '/wp-admin',
            '/phpmyadmin',
            '/adminer',
            '/grafana',
            '/prometheus',
            '/kibana',
            '/kubernetes',
            '/k8s'
        ]
        
        for endpoint_name, base_url in self.endpoints.items():
            for admin_path in admin_paths:
                try:
                    response = requests.get(
                        f"{base_url}{admin_path}",
                        timeout=5,
                        verify=False,
                        allow_redirects=False
                    )
                    
                    if response.status_code in [200, 301, 302]:
                        self.add_finding(SecurityFinding(
                            severity="MEDIUM",
                            category="Information Disclosure",
                            title=f"Exposed Admin Interface - {endpoint_name}",
                            description=f"Admin interface accessible at {admin_path}",
                            recommendation="Restrict access to admin interfaces or implement additional authentication",
                            affected_component=f"{endpoint_name} - {admin_path}",
                            evidence={"path": admin_path, "status_code": response.status_code}
                        ))
                except requests.exceptions.RequestException:
                    pass
        
        # Test for information disclosure
        info_paths = [
            '/.env',
            '/config.json',
            '/app.config',
            '/.git/config',
            '/swagger.json',
            '/api/docs',
            '/health/detailed',
            '/metrics',
            '/debug'
        ]
        
        for endpoint_name, base_url in self.endpoints.items():
            for info_path in info_paths:
                try:
                    response = requests.get(
                        f"{base_url}{info_path}",
                        timeout=5,
                        verify=False
                    )
                    
                    if response.status_code == 200:
                        # Check for sensitive information
                        sensitive_patterns = ['password', 'secret', 'key', 'token', 'api_key', 'database']
                        if any(pattern in response.text.lower() for pattern in sensitive_patterns):
                            self.add_finding(SecurityFinding(
                                severity="HIGH",
                                category="Information Disclosure",
                                title=f"Sensitive Information Exposed - {endpoint_name}",
                                description=f"Sensitive information exposed at {info_path}",
                                recommendation="Remove or restrict access to sensitive configuration files",
                                affected_component=f"{endpoint_name} - {info_path}",
                                evidence={"path": info_path, "contains_sensitive": True}
                            ))
                except requests.exceptions.RequestException:
                    pass
    
    async def test_container_security(self):
        """Test container security using Docker and Kubernetes scanners"""
        logger.info("Testing container security...")
        self.tools_used.append("Container Security Scanner")
        
        try:
            # Check if we can access Kubernetes
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", f"mlops-{self.environment}"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Check for privileged containers
                pods_output = result.stdout
                if "privileged" in pods_output.lower():
                    self.add_finding(SecurityFinding(
                        severity="HIGH",
                        category="Container Security",
                        title="Privileged Containers Detected",
                        description="Privileged containers found in the cluster",
                        recommendation="Remove privileged mode unless absolutely necessary",
                        affected_component="Kubernetes pods"
                    ))
                
                # Check pod security contexts
                result = subprocess.run(
                    ["kubectl", "get", "pods", "-n", f"mlops-{self.environment}", "-o", "yaml"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    try:
                        pods_yaml = yaml.safe_load(result.stdout)
                        for item in pods_yaml.get('items', []):
                            pod_name = item.get('metadata', {}).get('name', 'unknown')
                            spec = item.get('spec', {})
                            
                            # Check for missing security context
                            security_context = spec.get('securityContext')
                            if not security_context:
                                self.add_finding(SecurityFinding(
                                    severity="MEDIUM",
                                    category="Container Security",
                                    title=f"Missing Security Context - {pod_name}",
                                    description="Pod is missing security context configuration",
                                    recommendation="Add appropriate security context with non-root user",
                                    affected_component=f"Pod: {pod_name}"
                                ))
                            
                            # Check containers for security issues
                            for container in spec.get('containers', []):
                                container_name = container.get('name', 'unknown')
                                container_security = container.get('securityContext', {})
                                
                                # Check if running as root
                                if container_security.get('runAsUser') == 0:
                                    self.add_finding(SecurityFinding(
                                        severity="HIGH",
                                        category="Container Security",
                                        title=f"Container Running as Root - {container_name}",
                                        description="Container is configured to run as root user",
                                        recommendation="Configure container to run as non-root user",
                                        affected_component=f"Container: {container_name} in {pod_name}"
                                    ))
                                
                                # Check for privileged escalation
                                if container_security.get('allowPrivilegeEscalation', True):
                                    self.add_finding(SecurityFinding(
                                        severity="MEDIUM",
                                        category="Container Security",
                                        title=f"Privilege Escalation Allowed - {container_name}",
                                        description="Container allows privilege escalation",
                                        recommendation="Set allowPrivilegeEscalation to false",
                                        affected_component=f"Container: {container_name} in {pod_name}"
                                    ))
                    except yaml.YAMLError:
                        pass
            
        except subprocess.TimeoutExpired:
            logger.warning("Kubernetes security check timed out")
        except FileNotFoundError:
            logger.warning("kubectl not found, skipping Kubernetes security checks")
        except Exception as e:
            logger.error(f"Container security test failed: {e}")
    
    async def test_network_security(self):
        """Test network security configuration"""
        logger.info("Testing network security...")
        self.tools_used.append("Network Security Scanner")
        
        # Test for open ports
        common_ports = [22, 23, 21, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 3306, 6379, 9200, 9300]
        
        hostname = self.target_domain
        open_ports = []
        
        def scan_port(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((hostname, port))
                sock.close()
                return port if result == 0 else None
            except Exception:
                return None
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(scan_port, common_ports))
            open_ports = [port for port in results if port is not None]
        
        # Check for unnecessary open ports
        expected_ports = [80, 443]  # Only HTTP and HTTPS should be open
        unexpected_ports = [port for port in open_ports if port not in expected_ports]
        
        for port in unexpected_ports:
            severity = "HIGH" if port in [22, 23, 21, 3389] else "MEDIUM"
            self.add_finding(SecurityFinding(
                severity=severity,
                category="Network Security",
                title=f"Unexpected Open Port: {port}",
                description=f"Port {port} is open and may not be necessary",
                recommendation="Close unnecessary ports or restrict access",
                affected_component="Network configuration",
                evidence={"port": port, "all_open_ports": open_ports}
            ))
    
    async def test_data_protection(self):
        """Test data protection and privacy controls"""
        logger.info("Testing data protection...")
        self.tools_used.append("Data Protection Scanner")
        
        api_endpoint = self.endpoints['api']
        
        # Test for data encryption in transit
        try:
            # Try to access API over HTTP (should fail or redirect)
            http_endpoint = api_endpoint.replace('https://', 'http://')
            response = requests.get(http_endpoint, timeout=5, allow_redirects=False)
            
            if response.status_code == 200:
                self.add_finding(SecurityFinding(
                    severity="HIGH",
                    category="Data Protection",
                    title="Unencrypted Data Transmission",
                    description="API is accessible over unencrypted HTTP",
                    recommendation="Enforce HTTPS for all communications",
                    affected_component="API server"
                ))
        except requests.exceptions.RequestException:
            pass  # Expected for secure configurations
        
        # Test for sensitive data in responses
        try:
            response = requests.get(f"{api_endpoint}/health/detailed", timeout=5, verify=False)
            if response.status_code == 200:
                sensitive_patterns = [
                    'password', 'secret', 'key', 'token', 'api_key',
                    'database_url', 'connection_string', 'private_key'
                ]
                
                response_lower = response.text.lower()
                for pattern in sensitive_patterns:
                    if pattern in response_lower:
                        self.add_finding(SecurityFinding(
                            severity="HIGH",
                            category="Data Protection",
                            title="Sensitive Data Exposed in API Response",
                            description=f"Sensitive data pattern '{pattern}' found in API response",
                            recommendation="Remove sensitive data from API responses",
                            affected_component="Health check endpoint",
                            evidence={"pattern": pattern}
                        ))
        except requests.exceptions.RequestException:
            pass
    
    async def test_compliance_controls(self):
        """Test compliance controls (GDPR, SOC2, etc.)"""
        logger.info("Testing compliance controls...")
        self.tools_used.append("Compliance Scanner")
        
        api_endpoint = self.endpoints['api']
        
        # Test for privacy policy endpoint
        try:
            response = requests.get(f"{api_endpoint}/privacy-policy", timeout=5, verify=False)
            if response.status_code != 200:
                self.add_finding(SecurityFinding(
                    severity="MEDIUM",
                    category="Compliance",
                    title="Missing Privacy Policy",
                    description="Privacy policy endpoint not accessible",
                    recommendation="Implement privacy policy endpoint for GDPR compliance",
                    affected_component="Privacy compliance"
                ))
        except requests.exceptions.RequestException:
            pass
        
        # Test for data deletion endpoint (GDPR right to be forgotten)
        try:
            response = requests.delete(f"{api_endpoint}/api/v1/users/me", timeout=5, verify=False)
            if response.status_code == 404:
                self.add_finding(SecurityFinding(
                    severity="MEDIUM",
                    category="Compliance",
                    title="Missing Data Deletion Capability",
                    description="User data deletion endpoint not implemented",
                    recommendation="Implement data deletion capabilities for GDPR compliance",
                    affected_component="User data management"
                ))
        except requests.exceptions.RequestException:
            pass
        
        # Test for audit logging
        try:
            response = requests.get(f"{api_endpoint}/api/v1/audit/logs", timeout=5, verify=False)
            if response.status_code == 404:
                self.add_finding(SecurityFinding(
                    severity="MEDIUM",
                    category="Compliance",
                    title="Missing Audit Logging",
                    description="Audit logging endpoint not available",
                    recommendation="Implement comprehensive audit logging for compliance",
                    affected_component="Audit system"
                ))
        except requests.exceptions.RequestException:
            pass
    
    def _generate_report(self) -> SecurityAuditReport:
        """Generate final security audit report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Count findings by severity
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'INFO': 0
        }
        
        for finding in self.findings:
            severity_counts[finding.severity] += 1
        
        # Determine compliance status
        compliance_status = {
            'GDPR': 'PARTIAL' if severity_counts['CRITICAL'] == 0 else 'NON_COMPLIANT',
            'SOC2': 'PARTIAL' if severity_counts['CRITICAL'] == 0 else 'NON_COMPLIANT',
            'ISO27001': 'PARTIAL' if severity_counts['CRITICAL'] == 0 else 'NON_COMPLIANT',
            'PCI_DSS': 'PARTIAL' if severity_counts['CRITICAL'] == 0 else 'NON_COMPLIANT'
        }
        
        return SecurityAuditReport(
            timestamp=self.start_time.isoformat(),
            environment=self.environment,
            target_domain=self.target_domain,
            total_findings=len(self.findings),
            critical_count=severity_counts['CRITICAL'],
            high_count=severity_counts['HIGH'],
            medium_count=severity_counts['MEDIUM'],
            low_count=severity_counts['LOW'],
            info_count=severity_counts['INFO'],
            findings=self.findings,
            scan_duration=duration,
            tools_used=self.tools_used,
            compliance_status=compliance_status
        )

class SecurityReportGenerator:
    """Generate detailed security reports"""
    
    @staticmethod
    def generate_json_report(report: SecurityAuditReport, output_file: str):
        """Generate JSON report"""
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
    
    @staticmethod
    def generate_html_report(report: SecurityAuditReport, output_file: str):
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Security Audit Report - {report.target_domain}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .critical {{ color: #d32f2f; }}
        .high {{ color: #f57c00; }}
        .medium {{ color: #fbc02d; }}
        .low {{ color: #388e3c; }}
        .finding {{ margin: 20px 0; padding: 15px; border-left: 4px solid #ccc; }}
        .finding.critical {{ border-left-color: #d32f2f; }}
        .finding.high {{ border-left-color: #f57c00; }}
        .finding.medium {{ border-left-color: #fbc02d; }}
        .finding.low {{ border-left-color: #388e3c; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Audit Report</h1>
        <p><strong>Target:</strong> {report.target_domain}</p>
        <p><strong>Environment:</strong> {report.environment}</p>
        <p><strong>Scan Date:</strong> {report.timestamp}</p>
        <p><strong>Duration:</strong> {report.scan_duration:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3 class="critical">{report.critical_count}</h3>
            <p>Critical</p>
        </div>
        <div class="metric">
            <h3 class="high">{report.high_count}</h3>
            <p>High</p>
        </div>
        <div class="metric">
            <h3 class="medium">{report.medium_count}</h3>
            <p>Medium</p>
        </div>
        <div class="metric">
            <h3 class="low">{report.low_count}</h3>
            <p>Low</p>
        </div>
    </div>
    
    <h2>Findings</h2>
"""
        
        for finding in sorted(report.findings, key=lambda x: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].index(x.severity)):
            html_content += f"""
    <div class="finding {finding.severity.lower()}">
        <h3>{finding.title} <span class="{finding.severity.lower()}">[{finding.severity}]</span></h3>
        <p><strong>Category:</strong> {finding.category}</p>
        <p><strong>Component:</strong> {finding.affected_component}</p>
        <p><strong>Description:</strong> {finding.description}</p>
        <p><strong>Recommendation:</strong> {finding.recommendation}</p>
        {f'<p><strong>CVE:</strong> {finding.cve_id}</p>' if finding.cve_id else ''}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MLOps Platform Security Auditor')
    parser.add_argument('--domain', required=True, help='Target domain (e.g., staging.mlops-platform.com)')
    parser.add_argument('--environment', default='staging', help='Environment name')
    parser.add_argument('--output-dir', default='./security-reports', help='Output directory for reports')
    parser.add_argument('--format', choices=['json', 'html', 'both'], default='both', help='Report format')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run security audit
    auditor = SecurityAuditor(args.domain, args.environment)
    report = await auditor.run_comprehensive_audit()
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.format in ['json', 'both']:
        json_file = output_dir / f'security-audit-{args.environment}-{timestamp}.json'
        SecurityReportGenerator.generate_json_report(report, str(json_file))
        logger.info(f"JSON report generated: {json_file}")
    
    if args.format in ['html', 'both']:
        html_file = output_dir / f'security-audit-{args.environment}-{timestamp}.html'
        SecurityReportGenerator.generate_html_report(report, str(html_file))
        logger.info(f"HTML report generated: {html_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SECURITY AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Target: {report.target_domain}")
    print(f"Environment: {report.environment}")
    print(f"Total Findings: {report.total_findings}")
    print(f"Critical: {report.critical_count}")
    print(f"High: {report.high_count}")
    print(f"Medium: {report.medium_count}")
    print(f"Low: {report.low_count}")
    print(f"Scan Duration: {report.scan_duration:.2f} seconds")
    print(f"{'='*60}")
    
    # Exit with error code if critical issues found
    if report.critical_count > 0:
        logger.error("Critical security issues found!")
        return 1
    elif report.high_count > 0:
        logger.warning("High severity security issues found!")
        return 2
    else:
        logger.info("No critical security issues found!")
        return 0

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)