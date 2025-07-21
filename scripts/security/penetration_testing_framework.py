#!/usr/bin/env python3
"""
Penetration Testing Framework for anomaly_detection

This framework provides automated penetration testing capabilities for defensive
security purposes. It helps identify vulnerabilities that attackers might exploit
so they can be fixed before malicious actors find them.

DEFENSIVE USE ONLY: This tool is designed for authorized security testing of your
own systems or systems you have explicit permission to test.
"""

import asyncio
import json
import logging
import requests
import time
import random
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import tempfile
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PenetrationTestResult:
    """Result of a penetration test."""
    test_name: str
    vulnerability: str
    severity: str  # critical, high, medium, low
    description: str
    evidence: str
    remediation: str
    cvss_score: Optional[float] = None
    exploitable: bool = False


@dataclass
class PenetrationTestReport:
    """Complete penetration test report."""
    timestamp: str
    target: str
    test_duration: float
    tests_executed: int
    vulnerabilities_found: int
    critical_vulns: int
    high_vulns: int
    medium_vulns: int
    low_vulns: int
    results: List[PenetrationTestResult]
    recommendations: List[str]


class PenetrationTester:
    """Automated penetration testing framework for defensive security."""
    
    def __init__(self, target_url: str, config_path: Optional[str] = None):
        self.target_url = target_url.rstrip('/')
        self.config = self._load_config(config_path)
        self.results: List[PenetrationTestResult] = []
        self.session = requests.Session()
        self.start_time = time.time()
        
        # Set reasonable timeouts and headers
        self.session.timeout = 10
        self.session.headers.update({
            'User-Agent': 'anomaly_detection-Security-Scanner/1.0 (Defensive Security Testing)'
        })
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load penetration testing configuration."""
        default_config = {
            'authentication': {
                'enabled': True,
                'test_endpoints': ['/auth/login', '/api/auth/login'],
                'test_credentials': [
                    {'username': 'admin', 'password': 'admin'},
                    {'username': 'admin', 'password': 'password'},
                    {'username': 'admin', 'password': '123456'},
                    {'username': 'test', 'password': 'test'},
                ]
            },
            'authorization': {
                'enabled': True,
                'test_privilege_escalation': True,
                'test_idor': True,
            },
            'injection': {
                'enabled': True,
                'sql_injection': True,
                'command_injection': True,
                'xss': True,
                'ldap_injection': True,
            },
            'broken_access_control': {
                'enabled': True,
                'test_directory_traversal': True,
                'test_file_inclusion': True,
                'test_unrestricted_file_upload': True,
            },
            'security_misconfiguration': {
                'enabled': True,
                'test_default_credentials': True,
                'test_information_disclosure': True,
                'test_security_headers': True,
            },
            'sensitive_data_exposure': {
                'enabled': True,
                'test_data_leakage': True,
                'test_crypto_issues': True,
            },
            'rate_limiting': {
                'enabled': True,
                'max_requests_per_second': 5,
                'test_dos': True,
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return {**default_config, **user_config}
        
        return default_config
    
    async def run_penetration_tests(self) -> PenetrationTestReport:
        """Execute comprehensive penetration testing."""
        logger.info(f"🎯 Starting Penetration Testing for {self.target_url}")
        logger.info("🛡️  DEFENSIVE SECURITY TESTING - AUTHORIZED USE ONLY")
        
        # Pre-flight checks
        if not await self._verify_target_authorization():
            raise ValueError("Target authorization check failed - ensure you have permission to test this system")
        
        test_tasks = []
        
        # Authentication Testing
        if self.config['authentication']['enabled']:
            test_tasks.append(self._test_authentication_vulnerabilities())
        
        # Authorization Testing
        if self.config['authorization']['enabled']:
            test_tasks.append(self._test_authorization_vulnerabilities())
        
        # Injection Testing
        if self.config['injection']['enabled']:
            test_tasks.append(self._test_injection_vulnerabilities())
        
        # Broken Access Control Testing
        if self.config['broken_access_control']['enabled']:
            test_tasks.append(self._test_broken_access_control())
        
        # Security Misconfiguration Testing
        if self.config['security_misconfiguration']['enabled']:
            test_tasks.append(self._test_security_misconfiguration())
        
        # Sensitive Data Exposure Testing
        if self.config['sensitive_data_exposure']['enabled']:
            test_tasks.append(self._test_sensitive_data_exposure())
        
        # Rate Limiting and DoS Testing
        if self.config['rate_limiting']['enabled']:
            test_tasks.append(self._test_rate_limiting())
        
        # Execute all tests
        await asyncio.gather(*test_tasks, return_exceptions=True)
        
        return self._generate_report()
    
    async def _verify_target_authorization(self) -> bool:
        """Verify authorization to test the target system."""
        try:
            # Check if target is localhost or a development environment
            if any(host in self.target_url for host in ['localhost', '127.0.0.1', '0.0.0.0', 'dev.', 'test.']):
                return True
            
            # Look for security.txt or robots.txt that might indicate testing policy
            try:
                response = await self._safe_request('GET', '/.well-known/security.txt')
                if response and 'security testing' in response.text.lower():
                    return True
            except:
                pass
            
            # For production systems, require explicit authorization
            logger.warning("⚠️  Production system detected. Ensure you have explicit authorization to test this system.")
            return False
            
        except Exception as e:
            logger.error(f"Authorization check failed: {e}")
            return False
    
    async def _safe_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make a safe HTTP request with error handling and rate limiting."""
        try:
            # Rate limiting
            if hasattr(self, '_last_request_time'):
                time_since_last = time.time() - self._last_request_time
                min_interval = 1.0 / self.config['rate_limiting']['max_requests_per_second']
                if time_since_last < min_interval:
                    await asyncio.sleep(min_interval - time_since_last)
            
            self._last_request_time = time.time()
            
            # Make request
            full_url = url if url.startswith('http') else self.target_url + url
            response = self.session.request(method, full_url, **kwargs)
            
            return response
            
        except requests.RequestException as e:
            logger.debug(f"Request failed for {url}: {e}")
            return None
    
    async def _test_authentication_vulnerabilities(self) -> None:
        """Test for authentication vulnerabilities."""
        logger.info("🔐 Testing authentication vulnerabilities...")
        
        # Test for default credentials
        await self._test_default_credentials()
        
        # Test for brute force protection
        await self._test_brute_force_protection()
        
        # Test for session management issues
        await self._test_session_management()
        
        # Test for password policy
        await self._test_password_policy()
    
    async def _test_default_credentials(self) -> None:
        """Test for default credentials vulnerability."""
        endpoints = self.config['authentication']['test_endpoints']
        credentials = self.config['authentication']['test_credentials']
        
        for endpoint in endpoints:
            for cred in credentials:
                try:
                    response = await self._safe_request('POST', endpoint, json=cred)
                    
                    if response and response.status_code == 200:
                        # Check for successful authentication indicators
                        response_text = response.text.lower()
                        if any(indicator in response_text for indicator in ['token', 'success', 'dashboard', 'welcome']):
                            result = PenetrationTestResult(
                                test_name="Default Credentials",
                                vulnerability="Weak Default Credentials",
                                severity="critical",
                                description=f"Default credentials accepted: {cred['username']}:{cred['password']}",
                                evidence=f"HTTP {response.status_code} response with authentication success indicators",
                                remediation="Change default credentials and enforce strong password policy",
                                cvss_score=9.8,
                                exploitable=True
                            )
                            self.results.append(result)
                            
                except Exception as e:
                    logger.debug(f"Error testing credentials {cred}: {e}")
    
    async def _test_brute_force_protection(self) -> None:
        """Test for brute force protection."""
        endpoints = self.config['authentication']['test_endpoints']
        
        for endpoint in endpoints:
            # Attempt multiple failed logins
            failed_attempts = 0
            
            for i in range(10):  # Try 10 failed attempts
                try:
                    response = await self._safe_request('POST', endpoint, json={
                        'username': 'testuser',
                        'password': f'wrongpassword{i}'
                    })
                    
                    if response:
                        if response.status_code == 429:  # Rate limited
                            break
                        elif response.status_code in [401, 403]:  # Failed login
                            failed_attempts += 1
                        
                except Exception:
                    continue
            
            # If we made many failed attempts without rate limiting
            if failed_attempts >= 8:
                result = PenetrationTestResult(
                    test_name="Brute Force Protection",
                    vulnerability="Missing Brute Force Protection",
                    severity="high",
                    description="No rate limiting or account lockout after multiple failed login attempts",
                    evidence=f"Successfully made {failed_attempts} failed login attempts without rate limiting",
                    remediation="Implement account lockout and rate limiting for authentication endpoints",
                    cvss_score=7.5
                )
                self.results.append(result)
    
    async def _test_session_management(self) -> None:
        """Test session management vulnerabilities."""
        # Test for session fixation
        try:
            # Get initial session
            response1 = await self._safe_request('GET', '/login')
            if response1:
                session_id_before = response1.cookies.get('sessionid') or response1.cookies.get('JSESSIONID')
                
                if session_id_before:
                    # Attempt login
                    login_response = await self._safe_request('POST', '/auth/login', json={
                        'username': 'admin',
                        'password': 'admin'
                    })
                    
                    if login_response:
                        session_id_after = login_response.cookies.get('sessionid') or login_response.cookies.get('JSESSIONID')
                        
                        # Session should change after authentication
                        if session_id_before == session_id_after:
                            result = PenetrationTestResult(
                                test_name="Session Management",
                                vulnerability="Session Fixation",
                                severity="medium",
                                description="Session ID does not change after authentication",
                                evidence=f"Session ID remained {session_id_before} after login",
                                remediation="Regenerate session ID after successful authentication",
                                cvss_score=5.4
                            )
                            self.results.append(result)
                            
        except Exception as e:
            logger.debug(f"Session management test failed: {e}")
    
    async def _test_password_policy(self) -> None:
        """Test password policy enforcement."""
        # This would test registration endpoints with weak passwords
        registration_endpoints = ['/register', '/api/register', '/auth/register']
        weak_passwords = ['123', 'password', 'abc']
        
        for endpoint in registration_endpoints:
            for weak_password in weak_passwords:
                try:
                    response = await self._safe_request('POST', endpoint, json={
                        'username': f'testuser_{random.randint(1000, 9999)}',
                        'email': 'test@example.com',
                        'password': weak_password
                    })
                    
                    if response and response.status_code == 201:  # User created
                        result = PenetrationTestResult(
                            test_name="Password Policy",
                            vulnerability="Weak Password Policy",
                            severity="medium",
                            description=f"Weak password '{weak_password}' accepted during registration",
                            evidence=f"HTTP {response.status_code} - User created with weak password",
                            remediation="Implement strong password policy with minimum length, complexity requirements",
                            cvss_score=4.3
                        )
                        self.results.append(result)
                        break
                        
                except Exception:
                    continue
    
    async def _test_authorization_vulnerabilities(self) -> None:
        """Test for authorization vulnerabilities."""
        logger.info("🔒 Testing authorization vulnerabilities...")
        
        # Test for Insecure Direct Object Reference (IDOR)
        await self._test_idor()
        
        # Test for privilege escalation
        await self._test_privilege_escalation()
        
        # Test for missing function level access control
        await self._test_function_level_access_control()
    
    async def _test_idor(self) -> None:
        """Test for Insecure Direct Object Reference vulnerabilities."""
        # Common IDOR patterns
        idor_patterns = [
            '/api/users/{id}',
            '/api/documents/{id}',
            '/api/orders/{id}',
            '/user/{id}',
            '/profile/{id}'
        ]
        
        for pattern in idor_patterns:
            # Test with different user IDs
            for user_id in [1, 2, 100, 999]:
                endpoint = pattern.format(id=user_id)
                
                try:
                    response = await self._safe_request('GET', endpoint)
                    
                    if response and response.status_code == 200:
                        response_text = response.text.lower()
                        
                        # Look for sensitive information that might indicate IDOR
                        sensitive_indicators = ['email', 'phone', 'address', 'ssn', 'credit']
                        if any(indicator in response_text for indicator in sensitive_indicators):
                            result = PenetrationTestResult(
                                test_name="IDOR Vulnerability",
                                vulnerability="Insecure Direct Object Reference",
                                severity="high",
                                description=f"Direct access to user data via {endpoint}",
                                evidence=f"HTTP {response.status_code} - Sensitive data exposed",
                                remediation="Implement proper authorization checks for object access",
                                cvss_score=8.1
                            )
                            self.results.append(result)
                            break
                            
                except Exception:
                    continue
    
    async def _test_privilege_escalation(self) -> None:
        """Test for privilege escalation vulnerabilities."""
        admin_endpoints = [
            '/admin',
            '/api/admin',
            '/admin/users',
            '/admin/settings',
            '/dashboard/admin'
        ]
        
        for endpoint in admin_endpoints:
            try:
                # Test access without authentication
                response = await self._safe_request('GET', endpoint)
                
                if response and response.status_code == 200:
                    result = PenetrationTestResult(
                        test_name="Privilege Escalation",
                        vulnerability="Missing Authorization",
                        severity="critical",
                        description=f"Administrative endpoint {endpoint} accessible without authentication",
                        evidence=f"HTTP {response.status_code} - Admin endpoint accessible",
                        remediation="Implement proper authentication and authorization for administrative functions",
                        cvss_score=9.1
                    )
                    self.results.append(result)
                    
            except Exception:
                continue
    
    async def _test_function_level_access_control(self) -> None:
        """Test function level access control."""
        # Test HTTP method override
        sensitive_endpoints = ['/api/users', '/api/config', '/api/settings']
        
        for endpoint in sensitive_endpoints:
            # Test if GET allows access when it should require authentication
            try:
                response = await self._safe_request('GET', endpoint)
                
                if response and response.status_code == 200 and len(response.text) > 100:
                    result = PenetrationTestResult(
                        test_name="Function Level Access Control",
                        vulnerability="Missing Function Level Authorization",
                        severity="medium",
                        description=f"Sensitive endpoint {endpoint} accessible without proper authorization",
                        evidence=f"HTTP {response.status_code} - Data returned from sensitive endpoint",
                        remediation="Implement function-level access control for all sensitive operations",
                        cvss_score=6.5
                    )
                    self.results.append(result)
                    
            except Exception:
                continue
    
    async def _test_injection_vulnerabilities(self) -> None:
        """Test for injection vulnerabilities."""
        logger.info("💉 Testing injection vulnerabilities...")
        
        if self.config['injection']['sql_injection']:
            await self._test_sql_injection()
        
        if self.config['injection']['command_injection']:
            await self._test_command_injection()
        
        if self.config['injection']['xss']:
            await self._test_xss()
    
    async def _test_sql_injection(self) -> None:
        """Test for SQL injection vulnerabilities."""
        # Common SQL injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "' OR 1=1--",
            "'; DROP TABLE users--",
            "' UNION SELECT NULL--",
            "admin'--",
            "' OR 1=1#"
        ]
        
        # Common vulnerable endpoints
        test_endpoints = [
            '/api/users?id={payload}',
            '/search?q={payload}',
            '/login',  # Test in POST body
            '/api/products?category={payload}'
        ]
        
        for endpoint_template in test_endpoints:
            for payload in sql_payloads:
                try:
                    if '{payload}' in endpoint_template:
                        # GET request with payload in URL
                        endpoint = endpoint_template.format(payload=payload)
                        response = await self._safe_request('GET', endpoint)
                    else:
                        # POST request with payload in body
                        response = await self._safe_request('POST', endpoint_template, json={
                            'username': payload,
                            'password': 'test'
                        })
                    
                    if response:
                        response_text = response.text.lower()
                        
                        # Look for SQL error messages
                        sql_error_indicators = [
                            'sql syntax', 'mysql', 'postgresql', 'sqlite', 'oracle',
                            'syntax error', 'database error', 'sql statement',
                            'unknown column', 'table doesn\'t exist'
                        ]
                        
                        if any(indicator in response_text for indicator in sql_error_indicators):
                            result = PenetrationTestResult(
                                test_name="SQL Injection",
                                vulnerability="SQL Injection",
                                severity="critical",
                                description=f"SQL injection vulnerability detected at {endpoint_template}",
                                evidence=f"SQL error message in response to payload: {payload}",
                                remediation="Use parameterized queries and input validation",
                                cvss_score=9.8,
                                exploitable=True
                            )
                            self.results.append(result)
                            break
                            
                except Exception:
                    continue
    
    async def _test_command_injection(self) -> None:
        """Test for command injection vulnerabilities."""
        # Command injection payloads
        command_payloads = [
            '; ls -la',
            '| whoami',
            '& ping -c 1 127.0.0.1',
            '; cat /etc/passwd',
            '`id`',
            '$(whoami)'
        ]
        
        # Common vulnerable endpoints
        endpoints = ['/api/exec', '/api/system', '/api/ping', '/api/file']
        
        for endpoint in endpoints:
            for payload in command_payloads:
                try:
                    response = await self._safe_request('POST', endpoint, json={
                        'command': payload,
                        'input': payload,
                        'filename': payload
                    })
                    
                    if response:
                        response_text = response.text
                        
                        # Look for command execution indicators
                        command_indicators = [
                            'root:', 'usr/bin', '/home/', 'uid=', 'gid=',
                            'total ', 'drwx', 'PING', 'packets transmitted'
                        ]
                        
                        if any(indicator in response_text for indicator in command_indicators):
                            result = PenetrationTestResult(
                                test_name="Command Injection",
                                vulnerability="OS Command Injection",
                                severity="critical",
                                description=f"Command injection vulnerability at {endpoint}",
                                evidence=f"Command execution output in response: {response_text[:200]}...",
                                remediation="Sanitize user input and avoid system command execution",
                                cvss_score=9.8,
                                exploitable=True
                            )
                            self.results.append(result)
                            break
                            
                except Exception:
                    continue
    
    async def _test_xss(self) -> None:
        """Test for Cross-Site Scripting vulnerabilities."""
        # XSS payloads
        xss_payloads = [
            '<script>alert("XSS")</script>',
            '<img src=x onerror=alert("XSS")>',
            '"><script>alert("XSS")</script>',
            "javascript:alert('XSS')",
            '<svg onload=alert("XSS")>'
        ]
        
        # Common vulnerable endpoints
        endpoints = ['/search', '/api/comments', '/api/posts', '/feedback']
        
        for endpoint in endpoints:
            for payload in xss_payloads:
                try:
                    # Test GET parameter
                    get_response = await self._safe_request('GET', f'{endpoint}?q={payload}')
                    
                    # Test POST body
                    post_response = await self._safe_request('POST', endpoint, json={
                        'content': payload,
                        'message': payload,
                        'comment': payload
                    })
                    
                    for response in [get_response, post_response]:
                        if response and payload in response.text:
                            # Check if payload is reflected without encoding
                            if '<script>' in response.text or 'onerror=' in response.text:
                                result = PenetrationTestResult(
                                    test_name="Cross-Site Scripting",
                                    vulnerability="Reflected XSS",
                                    severity="high",
                                    description=f"XSS vulnerability at {endpoint}",
                                    evidence=f"Unescaped payload reflected in response: {payload}",
                                    remediation="Implement proper input validation and output encoding",
                                    cvss_score=8.8
                                )
                                self.results.append(result)
                                break
                                
                except Exception:
                    continue
    
    async def _test_broken_access_control(self) -> None:
        """Test for broken access control vulnerabilities."""
        logger.info("🚪 Testing broken access control...")
        
        await self._test_directory_traversal()
        await self._test_file_inclusion()
    
    async def _test_directory_traversal(self) -> None:
        """Test for directory traversal vulnerabilities."""
        traversal_payloads = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\drivers\\etc\\hosts',
            '....//....//....//etc/passwd',
            '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd'
        ]
        
        endpoints = ['/api/file', '/download', '/api/download', '/files']
        
        for endpoint in endpoints:
            for payload in traversal_payloads:
                try:
                    response = await self._safe_request('GET', f'{endpoint}?file={payload}')
                    
                    if response:
                        response_text = response.text
                        
                        # Look for system file content
                        if 'root:' in response_text or 'localhost' in response_text:
                            result = PenetrationTestResult(
                                test_name="Directory Traversal",
                                vulnerability="Path Traversal",
                                severity="high",
                                description=f"Directory traversal vulnerability at {endpoint}",
                                evidence=f"System file content retrieved: {response_text[:100]}...",
                                remediation="Validate and sanitize file paths, use allowlists",
                                cvss_score=7.5,
                                exploitable=True
                            )
                            self.results.append(result)
                            break
                            
                except Exception:
                    continue
    
    async def _test_file_inclusion(self) -> None:
        """Test for file inclusion vulnerabilities."""
        # Local File Inclusion payloads
        lfi_payloads = [
            '/etc/passwd',
            '/proc/version',
            '/var/log/apache2/access.log',
            'file:///etc/passwd'
        ]
        
        endpoints = ['/api/include', '/api/template', '/view']
        
        for endpoint in endpoints:
            for payload in lfi_payloads:
                try:
                    response = await self._safe_request('GET', f'{endpoint}?page={payload}')
                    
                    if response and ('root:' in response.text or 'Linux version' in response.text):
                        result = PenetrationTestResult(
                            test_name="Local File Inclusion",
                            vulnerability="Local File Inclusion (LFI)",
                            severity="high",
                            description=f"LFI vulnerability at {endpoint}",
                            evidence=f"Local file content retrieved via include parameter",
                            remediation="Use allowlists for file inclusion, validate file paths",
                            cvss_score=8.6
                        )
                        self.results.append(result)
                        break
                        
                except Exception:
                    continue
    
    async def _test_security_misconfiguration(self) -> None:
        """Test for security misconfigurations."""
        logger.info("⚙️ Testing security misconfigurations...")
        
        await self._test_information_disclosure()
        await self._test_security_headers()
        await self._test_default_pages()
    
    async def _test_information_disclosure(self) -> None:
        """Test for information disclosure vulnerabilities."""
        info_endpoints = [
            '/api/info',
            '/debug',
            '/api/debug',
            '/.env',
            '/config',
            '/server-status',
            '/server-info',
            '/.git/config',
            '/robots.txt'
        ]
        
        for endpoint in info_endpoints:
            try:
                response = await self._safe_request('GET', endpoint)
                
                if response and response.status_code == 200:
                    response_text = response.text.lower()
                    
                    # Look for sensitive information
                    sensitive_keywords = [
                        'database', 'password', 'secret', 'key', 'token',
                        'config', 'debug', 'error', 'stack trace', 'version'
                    ]
                    
                    if any(keyword in response_text for keyword in sensitive_keywords):
                        result = PenetrationTestResult(
                            test_name="Information Disclosure",
                            vulnerability="Sensitive Information Exposure",
                            severity="medium",
                            description=f"Sensitive information exposed at {endpoint}",
                            evidence=f"HTTP {response.status_code} - Sensitive data in response",
                            remediation="Remove debug endpoints and sensitive information from public access",
                            cvss_score=5.3
                        )
                        self.results.append(result)
                        
            except Exception:
                continue
    
    async def _test_security_headers(self) -> None:
        """Test for missing security headers."""
        try:
            response = await self._safe_request('GET', '/')
            
            if response:
                headers = response.headers
                
                # Check for missing security headers
                security_headers = {
                    'X-Frame-Options': 'Clickjacking protection missing',
                    'X-Content-Type-Options': 'MIME type sniffing protection missing',
                    'X-XSS-Protection': 'XSS protection missing',
                    'Strict-Transport-Security': 'HTTPS enforcement missing',
                    'Content-Security-Policy': 'Content Security Policy missing'
                }
                
                for header, description in security_headers.items():
                    if header not in headers:
                        result = PenetrationTestResult(
                            test_name="Security Headers",
                            vulnerability=f"Missing Security Header: {header}",
                            severity="low",
                            description=description,
                            evidence=f"Response headers do not include {header}",
                            remediation=f"Add {header} header to HTTP responses",
                            cvss_score=3.1
                        )
                        self.results.append(result)
                        
        except Exception:
            pass
    
    async def _test_default_pages(self) -> None:
        """Test for default pages and directories."""
        default_paths = [
            '/admin/admin',
            '/administrator',
            '/phpmyadmin',
            '/wp-admin',
            '/admin/login',
            '/management',
            '/test',
            '/demo'
        ]
        
        for path in default_paths:
            try:
                response = await self._safe_request('GET', path)
                
                if response and response.status_code == 200:
                    result = PenetrationTestResult(
                        test_name="Default Pages",
                        vulnerability="Default Administrative Interface",
                        severity="medium",
                        description=f"Default administrative interface accessible at {path}",
                        evidence=f"HTTP {response.status_code} - Default page accessible",
                        remediation="Remove or secure default administrative interfaces",
                        cvss_score=4.3
                    )
                    self.results.append(result)
                    
            except Exception:
                continue
    
    async def _test_sensitive_data_exposure(self) -> None:
        """Test for sensitive data exposure."""
        logger.info("🔓 Testing sensitive data exposure...")
        
        # Test for backup files
        backup_extensions = ['.bak', '.backup', '.old', '.orig', '.save', '.tmp']
        common_files = ['config', 'database', 'app', 'settings', 'env']
        
        for file_base in common_files:
            for ext in backup_extensions:
                endpoint = f'/{file_base}{ext}'
                try:
                    response = await self._safe_request('GET', endpoint)
                    
                    if response and response.status_code == 200 and len(response.text) > 50:
                        result = PenetrationTestResult(
                            test_name="Sensitive Data Exposure",
                            vulnerability="Backup File Exposure",
                            severity="medium",
                            description=f"Backup file exposed at {endpoint}",
                            evidence=f"HTTP {response.status_code} - Backup file accessible",
                            remediation="Remove backup files from web root",
                            cvss_score=5.3
                        )
                        self.results.append(result)
                        
                except Exception:
                    continue
    
    async def _test_rate_limiting(self) -> None:
        """Test rate limiting and DoS protection."""
        logger.info("⚡ Testing rate limiting...")
        
        # Test rate limiting on authentication endpoints
        auth_endpoints = ['/login', '/api/auth/login', '/auth/login']
        
        for endpoint in auth_endpoints:
            request_count = 0
            rate_limited = False
            
            # Send rapid requests
            for i in range(20):
                try:
                    response = await self._safe_request('POST', endpoint, json={
                        'username': f'test{i}',
                        'password': 'wrongpassword'
                    })
                    
                    if response:
                        if response.status_code == 429:  # Too Many Requests
                            rate_limited = True
                            break
                        request_count += 1
                    
                    # Small delay to avoid overwhelming the server
                    await asyncio.sleep(0.1)
                    
                except Exception:
                    continue
            
            # If we made many requests without rate limiting
            if request_count > 15 and not rate_limited:
                result = PenetrationTestResult(
                    test_name="Rate Limiting",
                    vulnerability="Missing Rate Limiting",
                    severity="medium",
                    description=f"No rate limiting on {endpoint}",
                    evidence=f"Successfully made {request_count} requests without rate limiting",
                    remediation="Implement rate limiting to prevent abuse and DoS attacks",
                    cvss_score=5.3
                )
                self.results.append(result)
    
    def _generate_report(self) -> PenetrationTestReport:
        """Generate comprehensive penetration test report."""
        end_time = time.time()
        test_duration = end_time - self.start_time
        
        # Count vulnerabilities by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for result in self.results:
            severity_counts[result.severity] += 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        report = PenetrationTestReport(
            timestamp=datetime.now().isoformat(),
            target=self.target_url,
            test_duration=test_duration,
            tests_executed=len(set(result.test_name for result in self.results)) + 10,  # Approximate
            vulnerabilities_found=len(self.results),
            critical_vulns=severity_counts['critical'],
            high_vulns=severity_counts['high'],
            medium_vulns=severity_counts['medium'],
            low_vulns=severity_counts['low'],
            results=self.results,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Analysis findings by category
        categories = set(result.vulnerability for result in self.results)
        
        if any('SQL Injection' in cat for cat in categories):
            recommendations.append("Implement parameterized queries and input validation to prevent SQL injection")
        
        if any('XSS' in cat for cat in categories):
            recommendations.append("Implement proper output encoding and Content Security Policy to prevent XSS")
        
        if any('Default Credentials' in cat for cat in categories):
            recommendations.append("Change all default credentials and implement strong password policies")
        
        if any('Missing' in cat for cat in categories):
            recommendations.append("Implement proper authentication and authorization controls")
        
        if any('Rate Limiting' in cat for cat in categories):
            recommendations.append("Implement rate limiting and DoS protection mechanisms")
        
        # General recommendations
        recommendations.extend([
            "Conduct regular security assessments and penetration testing",
            "Keep all software components up to date with security patches",
            "Implement security logging and monitoring",
            "Follow secure coding practices and security frameworks",
            "Provide security training for development team"
        ])
        
        return recommendations
    
    def save_report(self, report: PenetrationTestReport, output_dir: str = "pentest_reports") -> None:
        """Save penetration test report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = output_path / f"pentest_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # HTML report
        html_file = output_path / f"pentest_report_{timestamp}.html"
        self._generate_html_report(report, html_file)
        
        logger.info(f"Penetration test reports saved to {output_path}")
        logger.info(f"JSON: {json_file}")
        logger.info(f"HTML: {html_file}")
    
    def _generate_html_report(self, report: PenetrationTestReport, output_file: Path) -> None:
        """Generate HTML penetration test report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Penetration Test Report - {report.target}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #dc3545; color: white; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
                .critical {{ background: #dc3545; color: white; }}
                .high {{ background: #fd7e14; color: white; }}
                .medium {{ background: #ffc107; color: black; }}
                .low {{ background: #28a745; color: white; }}
                .vulnerability {{ border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; background: #f8f9fa; }}
                .vulnerability.critical {{ border-color: #dc3545; }}
                .vulnerability.high {{ border-color: #fd7e14; }}
                .vulnerability.medium {{ border-color: #ffc107; }}
                .vulnerability.low {{ border-color: #28a745; }}
                .exploitable {{ background: #ffebee; border: 1px solid #f44336; }}
                .recommendations {{ background: #e8f5e8; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Penetration Test Report</h1>
                <p><strong>Target:</strong> {report.target}</p>
                <p><strong>Generated:</strong> {report.timestamp}</p>
                <p><strong>Test Duration:</strong> {report.test_duration:.2f} seconds</p>
                <p><strong>Tests Executed:</strong> {report.tests_executed}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Total Vulnerabilities</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.vulnerabilities_found}</div>
                </div>
                <div class="metric critical">
                    <h3>Critical</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.critical_vulns}</div>
                </div>
                <div class="metric high">
                    <h3>High</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.high_vulns}</div>
                </div>
                <div class="metric medium">
                    <h3>Medium</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.medium_vulns}</div>
                </div>
                <div class="metric low">
                    <h3>Low</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.low_vulns}</div>
                </div>
            </div>
            
            <h2>🚨 Vulnerabilities Found</h2>
        """
        
        for vuln in sorted(report.results, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x.severity]):
            exploitable_class = " exploitable" if vuln.exploitable else ""
            cvss_info = f"<br><strong>CVSS Score:</strong> {vuln.cvss_score}" if vuln.cvss_score else ""
            
            html_content += f"""
            <div class="vulnerability {vuln.severity}{exploitable_class}">
                <h4>{vuln.vulnerability} <span class="{vuln.severity}">({vuln.severity.upper()})</span>
                {' ⚠️ EXPLOITABLE' if vuln.exploitable else ''}</h4>
                <p><strong>Test:</strong> {vuln.test_name}</p>
                <p><strong>Description:</strong> {vuln.description}</p>
                <p><strong>Evidence:</strong> {vuln.evidence}</p>
                <p><strong>Remediation:</strong> {vuln.remediation}</p>
                {cvss_info}
            </div>
            """
        
        html_content += f"""
            <div class="recommendations">
                <h2>📋 Security Recommendations</h2>
                <ul>
        """
        
        for recommendation in report.recommendations:
            html_content += f"<li>{recommendation}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div style="margin-top: 40px; padding: 20px; background: #fff3cd; border-radius: 5px;">
                <h3>⚠️ Important Notice</h3>
                <p><strong>This penetration test was conducted for defensive security purposes only.</strong></p>
                <p>All vulnerabilities identified should be remediated immediately. Consider engaging a professional security firm for comprehensive security assessment.</p>
                <p>Regular penetration testing should be conducted as part of a comprehensive security program.</p>
            </div>
            
            </body>
            </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)


async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Penetration Testing Framework for Defensive Security")
    parser.add_argument("target", help="Target URL to test (must have authorization)")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-dir", default="pentest_reports", help="Output directory for reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Warning about authorized use
    print("🛡️  DEFENSIVE SECURITY PENETRATION TESTING FRAMEWORK")
    print("⚠️  WARNING: Only use this tool on systems you own or have explicit permission to test.")
    print("📝 This tool is designed for defensive security purposes only.")
    print()
    
    # Confirm authorization
    if not args.target.startswith(('http://localhost', 'http://127.0.0.1', 'http://0.0.0.0')):
        confirm = input(f"Do you have explicit authorization to test {args.target}? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Exiting - Authorization required for security testing.")
            return 1
    
    # Initialize penetration tester
    try:
        tester = PenetrationTester(target_url=args.target, config_path=args.config)
        
        # Run penetration tests
        report = await tester.run_penetration_tests()
        
        # Save reports
        tester.save_report(report, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 PENETRATION TEST SUMMARY")
        print("="*60)
        print(f"Target: {report.target}")
        print(f"Vulnerabilities Found: {report.vulnerabilities_found}")
        print(f"Critical: {report.critical_vulns}")
        print(f"High: {report.high_vulns}")
        print(f"Medium: {report.medium_vulns}")
        print(f"Low: {report.low_vulns}")
        print(f"Test Duration: {report.test_duration:.2f} seconds")
        
        if report.critical_vulns > 0:
            print(f"\n🚨 CRITICAL VULNERABILITIES FOUND!")
            print("Immediate remediation required.")
            return 1
        elif report.high_vulns > 0:
            print(f"\n⚠️  High severity vulnerabilities found. Urgent remediation recommended.")
            return 0
        else:
            print(f"\n✅ No critical or high severity vulnerabilities found.")
            return 0
            
    except Exception as e:
        logger.error(f"Penetration testing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))