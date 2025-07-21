#!/usr/bin/env python3
"""
Automated Security Scanner for anomaly_detection Web Application
Performs comprehensive security testing including OWASP Top 10 vulnerabilities
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import socket
import ssl
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VulnerabilityType(Enum):
    """Types of vulnerabilities to scan for"""
    XSS_REFLECTED = "xss_reflected"
    XSS_STORED = "xss_stored"
    SQL_INJECTION = "sql_injection"
    CSRF = "csrf"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    BROKEN_AUTHENTICATION = "broken_authentication"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    XML_EXTERNAL_ENTITIES = "xml_external_entities"
    BROKEN_ACCESS_CONTROL = "broken_access_control"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    KNOWN_VULNERABILITIES = "known_vulnerabilities"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    SSL_TLS_ISSUES = "ssl_tls_issues"
    INFORMATION_DISCLOSURE = "information_disclosure"
    CLICKJACKING = "clickjacking"
    WEAK_AUTHENTICATION = "weak_authentication"

class SeverityLevel(Enum):
    """Vulnerability severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Vulnerability:
    """Vulnerability finding"""
    vuln_type: VulnerabilityType
    severity: SeverityLevel
    url: str
    parameter: Optional[str]
    payload: Optional[str]
    description: str
    evidence: str
    remediation: str
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None

@dataclass
class ScanResult:
    """Complete scan results"""
    target_url: str
    scan_start: datetime
    scan_end: Optional[datetime]
    vulnerabilities: List[Vulnerability]
    scan_coverage: Dict[str, int]
    total_requests: int
    scan_duration: float

class SecurityScanner:
    """Main security scanner class"""

    def __init__(self, target_url: str, options: Dict[str, Any] = None):
        self.target_url = target_url.rstrip('/')
        self.parsed_url = urlparse(self.target_url)
        self.base_domain = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}"
        self.session = None
        self.vulnerabilities = []
        self.total_requests = 0
        self.scan_start = datetime.now()

        # Configuration
        self.options = options or {}
        self.max_concurrent = self.options.get('max_concurrent', 10)
        self.request_delay = self.options.get('request_delay', 0.1)
        self.timeout = self.options.get('timeout', 30)
        self.user_agent = self.options.get('user_agent', 'anomaly_detection-SecurityScanner/1.0')
        self.include_tests = self.options.get('include_tests', [])
        self.exclude_tests = self.options.get('exclude_tests', [])

        # Attack payloads
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "\"><script>alert('XSS')</script>",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<body onload=alert('XSS')>",
            "<input type=image src=x onerror=alert('XSS')>",
            "<details open ontoggle=alert('XSS')>"
        ]

        self.sql_payloads = [
            "' OR '1'='1",
            "' OR 1=1--",
            "\" OR \"1\"=\"1",
            "' UNION SELECT NULL--",
            "'; DROP TABLE users;--",
            "' OR 1=1#",
            "admin'--",
            "' OR 'a'='a",
            "1' OR '1'='1' --",
            "' OR 1=1 LIMIT 1--"
        ]

        self.command_injection_payloads = [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "; cat /etc/passwd",
            "| dir",
            "&& dir",
            "; dir",
            "`whoami`",
            "$(whoami)",
            "| id"
        ]

        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..//..//..//etc/passwd",
            "../../../../../../etc/passwd%00",
            "../../../../../../../../etc/passwd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]

    async def init_session(self):
        """Initialize HTTP session"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            ssl=False  # Allow insecure connections for testing
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'User-Agent': self.user_agent}
        )

    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def scan(self) -> ScanResult:
        """Perform comprehensive security scan"""
        logger.info(f"Starting security scan of {self.target_url}")

        await self.init_session()

        try:
            # Discover endpoints
            endpoints = await self.discover_endpoints()
            logger.info(f"Discovered {len(endpoints)} endpoints")

            # Run vulnerability tests
            scan_tasks = []

            if self._should_run_test('xss'):
                scan_tasks.append(self.test_xss_vulnerabilities(endpoints))

            if self._should_run_test('sql'):
                scan_tasks.append(self.test_sql_injection(endpoints))

            if self._should_run_test('csrf'):
                scan_tasks.append(self.test_csrf_protection())

            if self._should_run_test('command_injection'):
                scan_tasks.append(self.test_command_injection(endpoints))

            if self._should_run_test('path_traversal'):
                scan_tasks.append(self.test_path_traversal())

            if self._should_run_test('authentication'):
                scan_tasks.append(self.test_authentication_weaknesses())

            if self._should_run_test('ssl'):
                scan_tasks.append(self.test_ssl_configuration())

            if self._should_run_test('headers'):
                scan_tasks.append(self.test_security_headers())

            if self._should_run_test('information_disclosure'):
                scan_tasks.append(self.test_information_disclosure())

            if self._should_run_test('clickjacking'):
                scan_tasks.append(self.test_clickjacking_protection())

            # Execute all tests concurrently
            await asyncio.gather(*scan_tasks, return_exceptions=True)

        finally:
            await self.close_session()

        scan_end = datetime.now()
        scan_duration = (scan_end - self.scan_start).total_seconds()

        # Generate scan coverage statistics
        scan_coverage = {
            'endpoints_tested': len(endpoints) if 'endpoints' in locals() else 0,
            'vulnerabilities_found': len(self.vulnerabilities),
            'critical_vulns': len([v for v in self.vulnerabilities if v.severity == SeverityLevel.CRITICAL]),
            'high_vulns': len([v for v in self.vulnerabilities if v.severity == SeverityLevel.HIGH]),
            'medium_vulns': len([v for v in self.vulnerabilities if v.severity == SeverityLevel.MEDIUM]),
            'low_vulns': len([v for v in self.vulnerabilities if v.severity == SeverityLevel.LOW])
        }

        return ScanResult(
            target_url=self.target_url,
            scan_start=self.scan_start,
            scan_end=scan_end,
            vulnerabilities=self.vulnerabilities,
            scan_coverage=scan_coverage,
            total_requests=self.total_requests,
            scan_duration=scan_duration
        )

    def _should_run_test(self, test_name: str) -> bool:
        """Check if a test should be run based on include/exclude lists"""
        if self.include_tests and test_name not in self.include_tests:
            return False
        if test_name in self.exclude_tests:
            return False
        return True

    async def discover_endpoints(self) -> List[str]:
        """Discover application endpoints"""
        endpoints = [
            '/',
            '/login',
            '/dashboard',
            '/api/v1/health',
            '/api/v1/datasets',
            '/api/v1/detectors',
            '/api/v1/detection',
            '/api/v1/auth/login',
            '/api/v1/auth/register',
            '/admin',
            '/config',
            '/debug',
            '/test',
            '/backup',
            '/static',
            '/docs',
            '/api',
            '/web/upload',
            '/web/dashboard',
            '/web/settings'
        ]

        # Add sitemap and robots.txt discovery
        try:
            sitemap_urls = await self.discover_from_sitemap()
            endpoints.extend(sitemap_urls)
        except:
            pass

        try:
            robots_urls = await self.discover_from_robots()
            endpoints.extend(robots_urls)
        except:
            pass

        # Filter and validate endpoints
        valid_endpoints = []
        for endpoint in set(endpoints):
            try:
                url = urljoin(self.target_url, endpoint)
                async with self.session.get(url) as response:
                    self.total_requests += 1
                    if response.status < 500:  # Include client errors but not server errors
                        valid_endpoints.append(endpoint)
                await asyncio.sleep(self.request_delay)
            except:
                pass

        return valid_endpoints

    async def discover_from_sitemap(self) -> List[str]:
        """Discover URLs from sitemap.xml"""
        sitemap_url = urljoin(self.target_url, '/sitemap.xml')
        async with self.session.get(sitemap_url) as response:
            self.total_requests += 1
            if response.status == 200:
                content = await response.text()
                # Extract URLs from sitemap
                urls = re.findall(r'<loc>(.*?)</loc>', content)
                return [urlparse(url).path for url in urls if url.startswith(self.base_domain)]
        return []

    async def discover_from_robots(self) -> List[str]:
        """Discover URLs from robots.txt"""
        robots_url = urljoin(self.target_url, '/robots.txt')
        async with self.session.get(robots_url) as response:
            self.total_requests += 1
            if response.status == 200:
                content = await response.text()
                # Extract disallowed paths
                paths = re.findall(r'Disallow:\s*([^\n\r]+)', content)
                return [path.strip() for path in paths if path.strip()]
        return []

    async def test_xss_vulnerabilities(self, endpoints: List[str]):
        """Test for XSS vulnerabilities"""
        logger.info("Testing for XSS vulnerabilities")

        for endpoint in endpoints:
            for payload in self.xss_payloads:
                await self._test_xss_parameter(endpoint, payload)
                await asyncio.sleep(self.request_delay)

    async def _test_xss_parameter(self, endpoint: str, payload: str):
        """Test XSS in URL parameters and form inputs"""
        url = urljoin(self.target_url, endpoint)

        # Test GET parameters
        test_params = ['q', 'search', 'query', 'name', 'value', 'data', 'input', 'term']
        for param in test_params:
            try:
                params = {param: payload}
                async with self.session.get(url, params=params) as response:
                    self.total_requests += 1
                    content = await response.text()

                    if payload in content and 'text/html' in response.headers.get('content-type', ''):
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type=VulnerabilityType.XSS_REFLECTED,
                            severity=SeverityLevel.HIGH,
                            url=str(response.url),
                            parameter=param,
                            payload=payload,
                            description=f"Reflected XSS vulnerability found in parameter '{param}'",
                            evidence=f"Payload '{payload}' was reflected in response without proper encoding",
                            remediation="Implement proper input validation and output encoding. Use Content Security Policy (CSP).",
                            cwe_id="CWE-79",
                            owasp_category="A7: Cross-Site Scripting (XSS)"
                        ))
            except Exception as e:
                logger.debug(f"Error testing XSS on {url}: {e}")

    async def test_sql_injection(self, endpoints: List[str]):
        """Test for SQL injection vulnerabilities"""
        logger.info("Testing for SQL injection vulnerabilities")

        for endpoint in endpoints:
            for payload in self.sql_payloads:
                await self._test_sql_parameter(endpoint, payload)
                await asyncio.sleep(self.request_delay)

    async def _test_sql_parameter(self, endpoint: str, payload: str):
        """Test SQL injection in parameters"""
        url = urljoin(self.target_url, endpoint)
        test_params = ['id', 'user_id', 'search', 'query', 'filter', 'sort', 'order']

        for param in test_params:
            try:
                params = {param: payload}
                async with self.session.get(url, params=params) as response:
                    self.total_requests += 1
                    content = await response.text()

                    # Check for SQL error indicators
                    sql_errors = [
                        'SQL syntax error', 'mysql_fetch', 'ORA-', 'PostgreSQL',
                        'Warning: mysql', 'MySQLSyntaxErrorException', 'valid MySQL result',
                        'SQLServer JDBC Driver', 'Oracle error', 'sqlite3.OperationalError',
                        'Unclosed quotation mark', 'Microsoft OLE DB Provider',
                        'SQL Server', 'sqlite3', 'psycopg2'
                    ]

                    for error in sql_errors:
                        if error.lower() in content.lower():
                            self.vulnerabilities.append(Vulnerability(
                                vuln_type=VulnerabilityType.SQL_INJECTION,
                                severity=SeverityLevel.CRITICAL,
                                url=str(response.url),
                                parameter=param,
                                payload=payload,
                                description=f"SQL injection vulnerability found in parameter '{param}'",
                                evidence=f"SQL error message detected: {error}",
                                remediation="Use parameterized queries and input validation. Implement proper error handling.",
                                cwe_id="CWE-89",
                                owasp_category="A1: Injection"
                            ))
                            break
            except Exception as e:
                logger.debug(f"Error testing SQL injection on {url}: {e}")

    async def test_csrf_protection(self):
        """Test for CSRF protection"""
        logger.info("Testing CSRF protection")

        # Test login form for CSRF token
        login_url = urljoin(self.target_url, '/login')
        try:
            async with self.session.get(login_url) as response:
                self.total_requests += 1
                if response.status == 200:
                    content = await response.text()

                    # Check for CSRF token patterns
                    csrf_patterns = [
                        r'name=["\']csrf[_-]?token["\']',
                        r'name=["\']_token["\']',
                        r'name=["\']authenticity_token["\']',
                        r'meta.*csrf.*content'
                    ]

                    has_csrf_token = any(re.search(pattern, content, re.IGNORECASE) for pattern in csrf_patterns)

                    if not has_csrf_token:
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type=VulnerabilityType.CSRF,
                            severity=SeverityLevel.HIGH,
                            url=login_url,
                            parameter=None,
                            payload=None,
                            description="Missing CSRF protection on login form",
                            evidence="No CSRF token found in login form",
                            remediation="Implement CSRF tokens for all state-changing operations",
                            cwe_id="CWE-352",
                            owasp_category="A8: Cross-Site Request Forgery (CSRF)"
                        ))
        except Exception as e:
            logger.debug(f"Error testing CSRF protection: {e}")

    async def test_command_injection(self, endpoints: List[str]):
        """Test for command injection vulnerabilities"""
        logger.info("Testing for command injection vulnerabilities")

        for endpoint in endpoints:
            for payload in self.command_injection_payloads:
                await self._test_command_injection_parameter(endpoint, payload)
                await asyncio.sleep(self.request_delay)

    async def _test_command_injection_parameter(self, endpoint: str, payload: str):
        """Test command injection in parameters"""
        url = urljoin(self.target_url, endpoint)
        test_params = ['cmd', 'command', 'exec', 'system', 'shell', 'run', 'file', 'path']

        for param in test_params:
            try:
                params = {param: payload}
                async with self.session.get(url, params=params) as response:
                    self.total_requests += 1
                    content = await response.text()

                    # Check for command execution indicators
                    command_indicators = [
                        'uid=', 'gid=', 'groups=', 'root:', '/bin/', '/usr/',
                        'Volume in drive', 'Directory of', 'total ',
                        'drwx', '-rw-', 'Permission denied'
                    ]

                    for indicator in command_indicators:
                        if indicator in content:
                            self.vulnerabilities.append(Vulnerability(
                                vuln_type=VulnerabilityType.COMMAND_INJECTION,
                                severity=SeverityLevel.CRITICAL,
                                url=str(response.url),
                                parameter=param,
                                payload=payload,
                                description=f"Command injection vulnerability found in parameter '{param}'",
                                evidence=f"Command execution indicator detected: {indicator}",
                                remediation="Avoid system calls with user input. Use safe APIs and input validation.",
                                cwe_id="CWE-78",
                                owasp_category="A1: Injection"
                            ))
                            break
            except Exception as e:
                logger.debug(f"Error testing command injection on {url}: {e}")

    async def test_path_traversal(self):
        """Test for path traversal vulnerabilities"""
        logger.info("Testing for path traversal vulnerabilities")

        vulnerable_params = ['file', 'path', 'dir', 'folder', 'document', 'page', 'include']

        for payload in self.path_traversal_payloads:
            for param in vulnerable_params:
                try:
                    url = self.target_url
                    params = {param: payload}

                    async with self.session.get(url, params=params) as response:
                        self.total_requests += 1
                        content = await response.text()

                        # Check for file content indicators
                        file_indicators = [
                            'root:x:0:0:', 'daemon:', 'bin:', 'sys:', 'nobody:',
                            '# localhost', '127.0.0.1', '[boot loader]',
                            'Windows Registry', 'HKEY_'
                        ]

                        for indicator in file_indicators:
                            if indicator in content:
                                self.vulnerabilities.append(Vulnerability(
                                    vuln_type=VulnerabilityType.PATH_TRAVERSAL,
                                    severity=SeverityLevel.HIGH,
                                    url=str(response.url),
                                    parameter=param,
                                    payload=payload,
                                    description=f"Path traversal vulnerability found in parameter '{param}'",
                                    evidence=f"System file content detected: {indicator}",
                                    remediation="Implement proper file path validation and use safe file access APIs.",
                                    cwe_id="CWE-22",
                                    owasp_category="A5: Broken Access Control"
                                ))
                                break

                        await asyncio.sleep(self.request_delay)

                except Exception as e:
                    logger.debug(f"Error testing path traversal: {e}")

    async def test_authentication_weaknesses(self):
        """Test for authentication weaknesses"""
        logger.info("Testing authentication weaknesses")

        # Test weak credentials
        weak_credentials = [
            ('admin', 'admin'),
            ('admin', 'password'),
            ('admin', '123456'),
            ('admin', 'admin123'),
            ('user', 'user'),
            ('test', 'test'),
            ('guest', 'guest'),
            ('admin', ''),
            ('', ''),
            ('administrator', 'administrator')
        ]

        login_url = urljoin(self.target_url, '/login')
        api_login_url = urljoin(self.target_url, '/api/v1/auth/login')

        for username, password in weak_credentials:
            await self._test_weak_credentials(login_url, username, password)
            await self._test_weak_credentials(api_login_url, username, password)
            await asyncio.sleep(self.request_delay)

    async def _test_weak_credentials(self, url: str, username: str, password: str):
        """Test specific username/password combination"""
        try:
            data = {
                'username': username,
                'password': password,
                'email': username,  # Some forms use email
                'login': username
            }

            async with self.session.post(url, data=data) as response:
                self.total_requests += 1

                # Check for successful login indicators
                if response.status in [200, 302]:
                    content = await response.text()

                    success_indicators = [
                        'dashboard', 'welcome', 'logout', 'profile',
                        'success', 'authenticated', 'session'
                    ]

                    failure_indicators = [
                        'invalid', 'incorrect', 'failed', 'error',
                        'wrong', 'denied', 'unauthorized'
                    ]

                    has_success = any(indicator in content.lower() for indicator in success_indicators)
                    has_failure = any(indicator in content.lower() for indicator in failure_indicators)

                    # Also check for redirect to dashboard/home
                    is_redirect = response.status == 302 and any(
                        redirect in str(response.headers.get('location', '')).lower()
                        for redirect in ['dashboard', 'home', 'main', 'index']
                    )

                    if (has_success and not has_failure) or is_redirect:
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type=VulnerabilityType.WEAK_AUTHENTICATION,
                            severity=SeverityLevel.CRITICAL,
                            url=url,
                            parameter=f"{username}:{password}",
                            payload=None,
                            description=f"Weak credentials allow authentication: {username}/{password}",
                            evidence=f"Successful login with credentials {username}:{password}",
                            remediation="Enforce strong password policies and remove default credentials.",
                            cwe_id="CWE-521",
                            owasp_category="A2: Broken Authentication"
                        ))

        except Exception as e:
            logger.debug(f"Error testing credentials {username}:{password}: {e}")

    async def test_ssl_configuration(self):
        """Test SSL/TLS configuration"""
        logger.info("Testing SSL/TLS configuration")

        if not self.target_url.startswith('https://'):
            self.vulnerabilities.append(Vulnerability(
                vuln_type=VulnerabilityType.SSL_TLS_ISSUES,
                severity=SeverityLevel.HIGH,
                url=self.target_url,
                parameter=None,
                payload=None,
                description="Application not using HTTPS",
                evidence="URL uses HTTP instead of HTTPS",
                remediation="Configure HTTPS with valid SSL certificate and redirect HTTP to HTTPS.",
                cwe_id="CWE-319",
                owasp_category="A3: Sensitive Data Exposure"
            ))
            return

        # Test SSL certificate
        try:
            hostname = self.parsed_url.hostname
            port = self.parsed_url.port or 443

            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()

                    # Check certificate validity
                    import datetime
                    not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')

                    if not_after < datetime.datetime.now():
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type=VulnerabilityType.SSL_TLS_ISSUES,
                            severity=SeverityLevel.HIGH,
                            url=self.target_url,
                            parameter=None,
                            payload=None,
                            description="SSL certificate has expired",
                            evidence=f"Certificate expired on {not_after}",
                            remediation="Renew SSL certificate and implement certificate monitoring.",
                            cwe_id="CWE-295",
                            owasp_category="A3: Sensitive Data Exposure"
                        ))

        except Exception as e:
            logger.debug(f"Error testing SSL configuration: {e}")

    async def test_security_headers(self):
        """Test for security headers"""
        logger.info("Testing security headers")

        try:
            async with self.session.get(self.target_url) as response:
                self.total_requests += 1
                headers = response.headers

                # Required security headers
                required_headers = {
                    'X-Frame-Options': 'Missing X-Frame-Options header allows clickjacking attacks',
                    'X-Content-Type-Options': 'Missing X-Content-Type-Options allows MIME type sniffing',
                    'X-XSS-Protection': 'Missing X-XSS-Protection removes browser XSS filtering',
                    'Strict-Transport-Security': 'Missing HSTS header allows protocol downgrade attacks',
                    'Content-Security-Policy': 'Missing CSP header allows code injection attacks'
                }

                for header, description in required_headers.items():
                    if header.lower() not in [h.lower() for h in headers.keys()]:
                        severity = SeverityLevel.HIGH if header in ['Content-Security-Policy', 'X-Frame-Options'] else SeverityLevel.MEDIUM

                        self.vulnerabilities.append(Vulnerability(
                            vuln_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                            severity=severity,
                            url=self.target_url,
                            parameter=header,
                            payload=None,
                            description=f"Missing security header: {header}",
                            evidence=description,
                            remediation=f"Add {header} header with appropriate value.",
                            cwe_id="CWE-16",
                            owasp_category="A6: Security Misconfiguration"
                        ))

        except Exception as e:
            logger.debug(f"Error testing security headers: {e}")

    async def test_information_disclosure(self):
        """Test for information disclosure"""
        logger.info("Testing for information disclosure")

        # Test for sensitive files
        sensitive_files = [
            '/.env',
            '/config.php',
            '/config.json',
            '/backup.sql',
            '/dump.sql',
            '/admin.php',
            '/test.php',
            '/phpinfo.php',
            '/info.php',
            '/debug.log',
            '/error.log',
            '/access.log',
            '/.git/config',
            '/.svn/entries',
            '/web.config',
            '/.htaccess',
            '/composer.json',
            '/package.json'
        ]

        for file_path in sensitive_files:
            try:
                url = urljoin(self.target_url, file_path)
                async with self.session.get(url) as response:
                    self.total_requests += 1

                    if response.status == 200:
                        content = await response.text()

                        # Check for sensitive content patterns
                        sensitive_patterns = [
                            'password', 'secret', 'key', 'token', 'api_key',
                            'database', 'connection', 'config', 'admin',
                            'debug', 'error', 'exception', 'stack trace'
                        ]

                        if any(pattern in content.lower() for pattern in sensitive_patterns):
                            self.vulnerabilities.append(Vulnerability(
                                vuln_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                                severity=SeverityLevel.MEDIUM,
                                url=url,
                                parameter=None,
                                payload=None,
                                description=f"Sensitive file accessible: {file_path}",
                                evidence=f"File contains sensitive information and is publicly accessible",
                                remediation="Remove or protect sensitive files from public access.",
                                cwe_id="CWE-200",
                                owasp_category="A3: Sensitive Data Exposure"
                            ))

                await asyncio.sleep(self.request_delay)

        except Exception as e:
            logger.debug(f"Error testing information disclosure: {e}")

    async def test_clickjacking_protection(self):
        """Test for clickjacking protection"""
        logger.info("Testing clickjacking protection")

        try:
            async with self.session.get(self.target_url) as response:
                self.total_requests += 1
                headers = response.headers

                # Check for frame protection headers
                frame_headers = ['X-Frame-Options', 'Content-Security-Policy']
                has_protection = False

                for header in frame_headers:
                    if header.lower() in [h.lower() for h in headers.keys()]:
                        header_value = headers.get(header, '').lower()

                        if header.lower() == 'x-frame-options':
                            if 'deny' in header_value or 'sameorigin' in header_value:
                                has_protection = True
                        elif header.lower() == 'content-security-policy':
                            if 'frame-ancestors' in header_value:
                                has_protection = True

                if not has_protection:
                    self.vulnerabilities.append(Vulnerability(
                        vuln_type=VulnerabilityType.CLICKJACKING,
                        severity=SeverityLevel.MEDIUM,
                        url=self.target_url,
                        parameter=None,
                        payload=None,
                        description="Missing clickjacking protection",
                        evidence="No X-Frame-Options or CSP frame-ancestors directive found",
                        remediation="Add X-Frame-Options: DENY or CSP frame-ancestors directive.",
                        cwe_id="CWE-1021",
                        owasp_category="A6: Security Misconfiguration"
                    ))

        except Exception as e:
            logger.debug(f"Error testing clickjacking protection: {e}")

def generate_report(scan_result: ScanResult, output_format: str = 'json') -> str:
    """Generate scan report in specified format"""

    if output_format == 'json':
        return json.dumps(asdict(scan_result), indent=2, default=str)

    elif output_format == 'html':
        return generate_html_report(scan_result)

    elif output_format == 'text':
        return generate_text_report(scan_result)

    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def generate_html_report(scan_result: ScanResult) -> str:
    """Generate HTML security report"""

    vulns_by_severity = {
        'critical': [v for v in scan_result.vulnerabilities if v.severity == SeverityLevel.CRITICAL],
        'high': [v for v in scan_result.vulnerabilities if v.severity == SeverityLevel.HIGH],
        'medium': [v for v in scan_result.vulnerabilities if v.severity == SeverityLevel.MEDIUM],
        'low': [v for v in scan_result.vulnerabilities if v.severity == SeverityLevel.LOW]
    }

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>anomaly_detection Security Scan Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #2563eb; color: white; padding: 20px; border-radius: 8px; }}
            .summary {{ background: #f8fafc; padding: 15px; margin: 20px 0; border-radius: 8px; }}
            .vulnerability {{ border: 1px solid #e2e8f0; margin: 10px 0; padding: 15px; border-radius: 8px; }}
            .critical {{ border-left: 5px solid #dc2626; }}
            .high {{ border-left: 5px solid #ea580c; }}
            .medium {{ border-left: 5px solid #d97706; }}
            .low {{ border-left: 5px solid #65a30d; }}
            .vuln-title {{ font-weight: bold; color: #1f2937; }}
            .vuln-details {{ margin: 10px 0; }}
            .stats {{ display: flex; gap: 20px; }}
            .stat {{ text-align: center; padding: 10px; background: white; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ anomaly_detection Security Scan Report</h1>
            <p>Target: {scan_result.target_url}</p>
            <p>Scan Date: {scan_result.scan_start.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Duration: {scan_result.scan_duration:.2f} seconds</p>
        </div>

        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="stats">
                <div class="stat">
                    <h3>{len(scan_result.vulnerabilities)}</h3>
                    <p>Total Vulnerabilities</p>
                </div>
                <div class="stat">
                    <h3>{len(vulns_by_severity['critical'])}</h3>
                    <p>Critical</p>
                </div>
                <div class="stat">
                    <h3>{len(vulns_by_severity['high'])}</h3>
                    <p>High</p>
                </div>
                <div class="stat">
                    <h3>{len(vulns_by_severity['medium'])}</h3>
                    <p>Medium</p>
                </div>
                <div class="stat">
                    <h3>{len(vulns_by_severity['low'])}</h3>
                    <p>Low</p>
                </div>
            </div>
        </div>

        <h2>Vulnerability Details</h2>
    """

    for severity, vulns in vulns_by_severity.items():
        if vulns:
            html += f"<h3>{severity.title()} Severity ({len(vulns)} findings)</h3>"

            for vuln in vulns:
                html += f"""
                <div class="vulnerability {severity}">
                    <div class="vuln-title">{vuln.vuln_type.value.replace('_', ' ').title()}</div>
                    <div class="vuln-details">
                        <p><strong>URL:</strong> {vuln.url}</p>
                        {f'<p><strong>Parameter:</strong> {vuln.parameter}</p>' if vuln.parameter else ''}
                        <p><strong>Description:</strong> {vuln.description}</p>
                        <p><strong>Evidence:</strong> {vuln.evidence}</p>
                        <p><strong>Remediation:</strong> {vuln.remediation}</p>
                        {f'<p><strong>CWE:</strong> {vuln.cwe_id}</p>' if vuln.cwe_id else ''}
                        {f'<p><strong>OWASP:</strong> {vuln.owasp_category}</p>' if vuln.owasp_category else ''}
                    </div>
                </div>
                """

    html += """
        </body>
        </html>
    """

    return html

def generate_text_report(scan_result: ScanResult) -> str:
    """Generate text security report"""

    report = f"""
anomaly_detection SECURITY SCAN REPORT
=============================

Target: {scan_result.target_url}
Scan Date: {scan_result.scan_start.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {scan_result.scan_duration:.2f} seconds
Total Requests: {scan_result.total_requests}

SUMMARY
-------
Total Vulnerabilities: {len(scan_result.vulnerabilities)}
Critical: {scan_result.scan_coverage.get('critical_vulns', 0)}
High: {scan_result.scan_coverage.get('high_vulns', 0)}
Medium: {scan_result.scan_coverage.get('medium_vulns', 0)}
Low: {scan_result.scan_coverage.get('low_vulns', 0)}

VULNERABILITIES
---------------
"""

    for i, vuln in enumerate(scan_result.vulnerabilities, 1):
        report += f"""
{i}. {vuln.vuln_type.value.replace('_', ' ').title()} [{vuln.severity.value.upper()}]
   URL: {vuln.url}
   {f'Parameter: {vuln.parameter}' if vuln.parameter else ''}
   Description: {vuln.description}
   Evidence: {vuln.evidence}
   Remediation: {vuln.remediation}
   {f'CWE: {vuln.cwe_id}' if vuln.cwe_id else ''}
   {f'OWASP: {vuln.owasp_category}' if vuln.owasp_category else ''}
"""

    return report

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='anomaly_detection Security Scanner')
    parser.add_argument('target', help='Target URL to scan')
    parser.add_argument('--output', '-o', choices=['json', 'html', 'text'], default='json', help='Output format')
    parser.add_argument('--output-file', '-f', help='Output file path')
    parser.add_argument('--include-tests', nargs='+', help='Tests to include')
    parser.add_argument('--exclude-tests', nargs='+', help='Tests to exclude')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Max concurrent requests')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between requests')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Scanner options
    options = {
        'max_concurrent': args.max_concurrent,
        'request_delay': args.delay,
        'timeout': args.timeout,
        'include_tests': args.include_tests or [],
        'exclude_tests': args.exclude_tests or []
    }

    # Run scan
    scanner = SecurityScanner(args.target, options)
    scan_result = await scanner.scan()

    # Generate report
    report = generate_report(scan_result, args.output)

    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output_file}")
    else:
        print(report)

    # Exit with error code if critical vulnerabilities found
    critical_count = scan_result.scan_coverage.get('critical_vulns', 0)
    if critical_count > 0:
        print(f"\n⚠️  {critical_count} critical vulnerabilities found!")
        exit(1)

    print(f"\n✅ Scan completed. {len(scan_result.vulnerabilities)} vulnerabilities found.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Scan interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Scan failed: {e}")
        exit(1)
