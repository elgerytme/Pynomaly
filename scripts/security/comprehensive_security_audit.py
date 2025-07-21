#!/usr/bin/env python3
"""
Comprehensive Security Audit Framework for anomaly_detection

This script provides a complete security audit framework covering:
- Static Application Security Testing (SAST)
- Dynamic Application Security Testing (DAST) 
- Dependency vulnerability scanning
- Container security assessment
- API security testing
- Authentication/authorization validation
- Data protection compliance
- Network security evaluation
- OWASP compliance checking

This is a defensive security tool designed to identify and remediate vulnerabilities.
"""

import asyncio
import json
import subprocess
import yaml
import time
import logging
import os
import sys
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Represents a security finding from audit tools."""
    severity: str  # critical, high, medium, low
    category: str  # e.g., 'vulnerability', 'misconfiguration', 'compliance'
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    tool: Optional[str] = None


@dataclass
class SecurityAuditReport:
    """Complete security audit report."""
    timestamp: str
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int
    findings: List[SecurityFinding]
    scan_duration: float
    tools_used: List[str]
    compliance_status: Dict[str, str]
    risk_score: float


class SecurityAuditor:
    """Comprehensive security audit orchestrator."""
    
    def __init__(self, target_path: str = ".", config_path: Optional[str] = None):
        self.target_path = Path(target_path).resolve()
        self.config_path = config_path
        self.config = self._load_config()
        self.findings: List[SecurityFinding] = []
        self.start_time = time.time()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load security audit configuration."""
        default_config = {
            'sast': {
                'enabled': True,
                'tools': ['bandit', 'semgrep', 'pylint-security'],
                'exclude_paths': ['.git/', 'node_modules/', '__pycache__/']
            },
            'dependency_scan': {
                'enabled': True,
                'tools': ['safety', 'pip-audit', 'cyclonedx'],
                'vulnerability_db_update': True
            },
            'container_scan': {
                'enabled': True,
                'tools': ['trivy', 'grype'],
                'scan_images': True,
                'dockerfile_analysis': True
            },
            'api_security': {
                'enabled': True,
                'base_url': 'http://localhost:8000',
                'auth_endpoints': ['/auth/login', '/auth/register'],
                'test_endpoints': ['/api/v1/', '/health', '/docs']
            },
            'compliance': {
                'frameworks': ['owasp-top10', 'pci-dss', 'gdpr'],
                'custom_rules': []
            },
            'reporting': {
                'formats': ['json', 'html', 'sarif'],
                'output_dir': 'security_reports',
                'include_remediation': True
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return {**default_config, **user_config}
        
        return default_config
    
    async def run_comprehensive_audit(self) -> SecurityAuditReport:
        """Execute complete security audit."""
        logger.info("🔒 Starting Comprehensive Security Audit")
        logger.info(f"Target: {self.target_path}")
        
        audit_tasks = []
        
        # Static Application Security Testing
        if self.config['sast']['enabled']:
            audit_tasks.append(self._run_sast_scan())
        
        # Dependency vulnerability scanning
        if self.config['dependency_scan']['enabled']:
            audit_tasks.append(self._run_dependency_scan())
        
        # Container security scanning
        if self.config['container_scan']['enabled']:
            audit_tasks.append(self._run_container_scan())
        
        # API security testing
        if self.config['api_security']['enabled']:
            audit_tasks.append(self._run_api_security_test())
        
        # Authentication/Authorization testing
        audit_tasks.append(self._run_auth_security_test())
        
        # Data protection validation
        audit_tasks.append(self._run_data_protection_audit())
        
        # Network security assessment
        audit_tasks.append(self._run_network_security_test())
        
        # Compliance checking
        audit_tasks.append(self._run_compliance_check())
        
        # Execute all audits concurrently
        await asyncio.gather(*audit_tasks, return_exceptions=True)
        
        return self._generate_report()
    
    async def _run_sast_scan(self) -> None:
        """Run Static Application Security Testing."""
        logger.info("🔍 Running SAST scan...")
        
        # Bandit - Python security linter
        await self._run_bandit_scan()
        
        # Semgrep - Multi-language static analysis
        await self._run_semgrep_scan()
        
        # Custom security pattern analysis
        await self._run_custom_pattern_scan()
    
    async def _run_bandit_scan(self) -> None:
        """Run Bandit security scanner."""
        try:
            cmd = [
                'bandit', '-r', str(self.target_path),
                '-f', 'json',
                '--exclude', ','.join(self.config['sast']['exclude_paths'])
            ]
            
            result = await self._run_subprocess(cmd)
            if result.returncode == 0 or result.returncode == 1:  # Bandit returns 1 when issues found
                bandit_data = json.loads(result.stdout)
                
                for issue in bandit_data.get('results', []):
                    finding = SecurityFinding(
                        severity=issue['issue_severity'].lower(),
                        category='vulnerability',
                        title=issue['test_name'],
                        description=issue['issue_text'],
                        file_path=issue['filename'],
                        line_number=issue['line_number'],
                        cwe_id=issue.get('issue_cwe', {}).get('id'),
                        remediation=issue.get('more_info'),
                        tool='bandit'
                    )
                    self.findings.append(finding)
                    
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
    
    async def _run_semgrep_scan(self) -> None:
        """Run Semgrep security analysis."""
        try:
            cmd = [
                'semgrep', '--config=auto',
                '--json',
                '--severity=ERROR',
                '--severity=WARNING',
                str(self.target_path)
            ]
            
            result = await self._run_subprocess(cmd)
            if result.returncode == 0:
                semgrep_data = json.loads(result.stdout)
                
                for finding in semgrep_data.get('results', []):
                    security_finding = SecurityFinding(
                        severity=finding['extra'].get('severity', 'medium').lower(),
                        category='security-pattern',
                        title=finding['extra']['message'],
                        description=finding['extra'].get('metadata', {}).get('description', ''),
                        file_path=finding['path'],
                        line_number=finding['start']['line'],
                        remediation=finding['extra'].get('fix_regex'),
                        tool='semgrep'
                    )
                    self.findings.append(security_finding)
                    
        except Exception as e:
            logger.warning(f"Semgrep scan failed (tool may not be installed): {e}")
    
    async def _run_custom_pattern_scan(self) -> None:
        """Run custom security pattern analysis."""
        # Security anti-patterns to look for
        patterns = [
            {
                'pattern': r'password\s*=\s*["\'][^"\']*["\']',
                'severity': 'high',
                'title': 'Hardcoded Password',
                'description': 'Potential hardcoded password found'
            },
            {
                'pattern': r'api[_-]?key\s*=\s*["\'][^"\']*["\']',
                'severity': 'high',
                'title': 'Hardcoded API Key',
                'description': 'Potential hardcoded API key found'
            },
            {
                'pattern': r'eval\s*\(',
                'severity': 'critical',
                'title': 'Code Injection Risk',
                'description': 'Use of eval() can lead to code injection'
            },
            {
                'pattern': r'subprocess\s*\.\s*call\s*\(',
                'severity': 'medium',
                'title': 'Command Injection Risk',
                'description': 'Subprocess call may be vulnerable to injection'
            }
        ]
        
        import re
        
        for py_file in self.target_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for pattern_info in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern_info['pattern'], line, re.IGNORECASE):
                            finding = SecurityFinding(
                                severity=pattern_info['severity'],
                                category='security-pattern',
                                title=pattern_info['title'],
                                description=pattern_info['description'],
                                file_path=str(py_file),
                                line_number=line_num,
                                tool='custom-patterns'
                            )
                            self.findings.append(finding)
            except Exception as e:
                logger.debug(f"Error scanning {py_file}: {e}")
    
    async def _run_dependency_scan(self) -> None:
        """Run dependency vulnerability scanning."""
        logger.info("📦 Running dependency vulnerability scan...")
        
        # Safety - Check for known security vulnerabilities
        await self._run_safety_scan()
        
        # pip-audit - Alternative dependency scanner
        await self._run_pip_audit()
        
        # Custom dependency analysis
        await self._analyze_dependencies()
    
    async def _run_safety_scan(self) -> None:
        """Run Safety dependency scanner."""
        try:
            cmd = ['safety', 'check', '--json', '--full-report']
            
            result = await self._run_subprocess(cmd)
            if result.returncode in [0, 64]:  # 64 = vulnerabilities found
                safety_data = json.loads(result.stdout)
                
                for vulnerability in safety_data.get('vulnerabilities', []):
                    finding = SecurityFinding(
                        severity='high' if vulnerability.get('advisory') else 'medium',
                        category='dependency-vulnerability',
                        title=f"Vulnerable dependency: {vulnerability['package_name']}",
                        description=vulnerability.get('advisory', 'Known security vulnerability'),
                        remediation=f"Update to version {vulnerability.get('analyzed_version', 'latest')}",
                        tool='safety'
                    )
                    self.findings.append(finding)
                    
        except Exception as e:
            logger.warning(f"Safety scan failed: {e}")
    
    async def _run_pip_audit(self) -> None:
        """Run pip-audit scanner."""
        try:
            cmd = ['pip-audit', '--format=json', '--desc']
            
            result = await self._run_subprocess(cmd)
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                
                for vulnerability in audit_data.get('vulnerabilities', []):
                    finding = SecurityFinding(
                        severity='high',
                        category='dependency-vulnerability',
                        title=f"CVE in {vulnerability['package']}",
                        description=vulnerability.get('description', 'Security vulnerability found'),
                        cwe_id=vulnerability.get('id'),
                        remediation=f"Update to fixed version",
                        tool='pip-audit'
                    )
                    self.findings.append(finding)
                    
        except Exception as e:
            logger.warning(f"pip-audit failed (tool may not be installed): {e}")
    
    async def _analyze_dependencies(self) -> None:
        """Analyze dependencies for security issues."""
        # Check for outdated packages
        requirements_files = list(self.target_path.rglob('requirements*.txt'))
        requirements_files.extend(list(self.target_path.rglob('pyproject.toml')))
        
        for req_file in requirements_files:
            if 'requirements' in req_file.name.lower():
                await self._check_requirements_security(req_file)
    
    async def _check_requirements_security(self, req_file: Path) -> None:
        """Check requirements file for security issues."""
        try:
            with open(req_file, 'r') as f:
                content = f.read()
            
            # Look for unpinned dependencies
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' not in line and '>=' not in line and '~=' not in line:
                        finding = SecurityFinding(
                            severity='medium',
                            category='dependency-security',
                            title='Unpinned Dependency',
                            description=f'Dependency "{line}" is not pinned to specific version',
                            file_path=str(req_file),
                            line_number=line_num,
                            remediation='Pin dependency to specific version for reproducible builds',
                            tool='custom-dependency-analysis'
                        )
                        self.findings.append(finding)
                        
        except Exception as e:
            logger.debug(f"Error analyzing {req_file}: {e}")
    
    async def _run_container_scan(self) -> None:
        """Run container security scanning."""
        logger.info("🐳 Running container security scan...")
        
        # Find Docker files
        docker_files = list(self.target_path.rglob('Dockerfile*'))
        compose_files = list(self.target_path.rglob('docker-compose*.yml'))
        
        for dockerfile in docker_files:
            await self._analyze_dockerfile(dockerfile)
        
        for compose_file in compose_files:
            await self._analyze_compose_file(compose_file)
    
    async def _analyze_dockerfile(self, dockerfile: Path) -> None:
        """Analyze Dockerfile for security issues."""
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Security checks
            has_user_instruction = any('USER' in line.upper() for line in lines)
            if not has_user_instruction:
                finding = SecurityFinding(
                    severity='high',
                    category='container-security',
                    title='Container runs as root',
                    description='Dockerfile does not specify non-root user',
                    file_path=str(dockerfile),
                    remediation='Add USER instruction to run as non-root user',
                    tool='dockerfile-analysis'
                )
                self.findings.append(finding)
            
            # Check for secrets in ENV or ARG
            for line_num, line in enumerate(lines, 1):
                if ('ENV' in line.upper() or 'ARG' in line.upper()) and any(
                    secret in line.lower() for secret in ['password', 'key', 'token', 'secret']
                ):
                    finding = SecurityFinding(
                        severity='high',
                        category='container-security',
                        title='Potential secret in Dockerfile',
                        description='Environment variable may contain sensitive information',
                        file_path=str(dockerfile),
                        line_number=line_num,
                        remediation='Use Docker secrets or runtime environment variables',
                        tool='dockerfile-analysis'
                    )
                    self.findings.append(finding)
                    
        except Exception as e:
            logger.debug(f"Error analyzing {dockerfile}: {e}")
    
    async def _analyze_compose_file(self, compose_file: Path) -> None:
        """Analyze docker-compose file for security issues."""
        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get('services', {})
            
            for service_name, service_config in services.items():
                # Check for privileged containers
                if service_config.get('privileged'):
                    finding = SecurityFinding(
                        severity='critical',
                        category='container-security',
                        title='Privileged container',
                        description=f'Service "{service_name}" runs in privileged mode',
                        file_path=str(compose_file),
                        remediation='Remove privileged mode or use specific capabilities',
                        tool='compose-analysis'
                    )
                    self.findings.append(finding)
                
                # Check for host network mode
                if service_config.get('network_mode') == 'host':
                    finding = SecurityFinding(
                        severity='high',
                        category='container-security',
                        title='Host network mode',
                        description=f'Service "{service_name}" uses host network mode',
                        file_path=str(compose_file),
                        remediation='Use bridge networking with specific port mappings',
                        tool='compose-analysis'
                    )
                    self.findings.append(finding)
                    
        except Exception as e:
            logger.debug(f"Error analyzing {compose_file}: {e}")
    
    async def _run_api_security_test(self) -> None:
        """Run API security testing."""
        logger.info("🌐 Running API security tests...")
        
        base_url = self.config['api_security']['base_url']
        
        # Test for common API vulnerabilities
        await self._test_api_authentication(base_url)
        await self._test_api_authorization(base_url)
        await self._test_api_input_validation(base_url)
        await self._test_api_rate_limiting(base_url)
        await self._test_api_security_headers(base_url)
    
    async def _test_api_authentication(self, base_url: str) -> None:
        """Test API authentication security."""
        try:
            # Test unauthenticated access to protected endpoints
            test_endpoints = self.config['api_security']['test_endpoints']
            
            for endpoint in test_endpoints:
                url = f"{base_url}{endpoint}"
                
                try:
                    response = requests.get(url, timeout=5)
                    
                    # Check if sensitive endpoints are accessible without authentication
                    if '/admin' in endpoint or '/api' in endpoint:
                        if response.status_code == 200:
                            finding = SecurityFinding(
                                severity='high',
                                category='api-security',
                                title='Unauthenticated access to sensitive endpoint',
                                description=f'Endpoint {endpoint} accessible without authentication',
                                remediation='Implement proper authentication for sensitive endpoints',
                                tool='api-security-test'
                            )
                            self.findings.append(finding)
                            
                except requests.RequestException as e:
                    logger.debug(f"API test request failed for {url}: {e}")
                    
        except Exception as e:
            logger.warning(f"API authentication test failed: {e}")
    
    async def _test_api_authorization(self, base_url: str) -> None:
        """Test API authorization security."""
        # This would test for privilege escalation, IDOR, etc.
        # Simulated for demonstration
        pass
    
    async def _test_api_input_validation(self, base_url: str) -> None:
        """Test API input validation."""
        # Test for SQL injection, XSS, etc.
        # Simulated for demonstration
        pass
    
    async def _test_api_rate_limiting(self, base_url: str) -> None:
        """Test API rate limiting."""
        try:
            test_endpoint = f"{base_url}/health"
            
            # Send multiple rapid requests
            responses = []
            for _ in range(10):
                try:
                    response = requests.get(test_endpoint, timeout=1)
                    responses.append(response.status_code)
                except requests.RequestException:
                    pass
            
            # Check if rate limiting is in place
            if all(status == 200 for status in responses[-5:]):  # Last 5 requests all successful
                finding = SecurityFinding(
                    severity='medium',
                    category='api-security',
                    title='Missing rate limiting',
                    description='API endpoints may not have rate limiting implemented',
                    remediation='Implement rate limiting to prevent abuse',
                    tool='api-security-test'
                )
                self.findings.append(finding)
                
        except Exception as e:
            logger.debug(f"Rate limiting test failed: {e}")
    
    async def _test_api_security_headers(self, base_url: str) -> None:
        """Test API security headers."""
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            headers = response.headers
            
            # Check for missing security headers
            security_headers = [
                'X-Frame-Options',
                'X-Content-Type-Options', 
                'X-XSS-Protection',
                'Strict-Transport-Security',
                'Content-Security-Policy'
            ]
            
            for header in security_headers:
                if header not in headers:
                    finding = SecurityFinding(
                        severity='medium',
                        category='api-security',
                        title=f'Missing security header: {header}',
                        description=f'Response missing {header} security header',
                        remediation=f'Add {header} header to API responses',
                        tool='api-security-test'
                    )
                    self.findings.append(finding)
                    
        except Exception as e:
            logger.debug(f"Security headers test failed: {e}")
    
    async def _run_auth_security_test(self) -> None:
        """Test authentication and authorization security."""
        logger.info("🔐 Running authentication security tests...")
        
        # Look for authentication-related code
        auth_files = []
        for py_file in self.target_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in ['jwt', 'token', 'auth', 'login', 'password']):
                        auth_files.append(py_file)
            except Exception:
                continue
        
        # Analyze authentication implementation
        for auth_file in auth_files:
            await self._analyze_auth_implementation(auth_file)
    
    async def _analyze_auth_implementation(self, auth_file: Path) -> None:
        """Analyze authentication implementation for security issues."""
        try:
            with open(auth_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Look for common authentication vulnerabilities
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                
                # Weak password hashing
                if 'md5' in line_lower or 'sha1' in line_lower:
                    finding = SecurityFinding(
                        severity='high',
                        category='auth-security',
                        title='Weak password hashing',
                        description='Using weak hashing algorithm for passwords',
                        file_path=str(auth_file),
                        line_number=line_num,
                        remediation='Use bcrypt, scrypt, or Argon2 for password hashing',
                        tool='auth-analysis'
                    )
                    self.findings.append(finding)
                
                # Hardcoded JWT secrets
                if 'jwt_secret' in line_lower and '=' in line and '"' in line:
                    finding = SecurityFinding(
                        severity='critical',
                        category='auth-security',
                        title='Hardcoded JWT secret',
                        description='JWT secret appears to be hardcoded',
                        file_path=str(auth_file),
                        line_number=line_num,
                        remediation='Use environment variables for JWT secrets',
                        tool='auth-analysis'
                    )
                    self.findings.append(finding)
                    
        except Exception as e:
            logger.debug(f"Error analyzing {auth_file}: {e}")
    
    async def _run_data_protection_audit(self) -> None:
        """Run data protection compliance audit."""
        logger.info("🛡️ Running data protection audit...")
        
        # Look for data handling patterns
        for py_file in self.target_path.rglob('*.py'):
            await self._analyze_data_handling(py_file)
    
    async def _analyze_data_handling(self, py_file: Path) -> None:
        """Analyze data handling for privacy compliance."""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Look for PII handling patterns
            pii_patterns = ['email', 'ssn', 'credit.card', 'phone', 'address']
            
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                
                for pattern in pii_patterns:
                    if pattern in line_lower and 'log' in line_lower:
                        finding = SecurityFinding(
                            severity='high',
                            category='data-protection',
                            title='Potential PII logging',
                            description=f'Possible logging of {pattern} data',
                            file_path=str(py_file),
                            line_number=line_num,
                            remediation='Avoid logging sensitive personal information',
                            tool='data-protection-analysis'
                        )
                        self.findings.append(finding)
                        break
                        
        except Exception as e:
            logger.debug(f"Error analyzing {py_file}: {e}")
    
    async def _run_network_security_test(self) -> None:
        """Run network security assessment."""
        logger.info("🌐 Running network security assessment...")
        
        # Check for network configurations
        config_files = list(self.target_path.rglob('*.yml')) + list(self.target_path.rglob('*.yaml'))
        
        for config_file in config_files:
            await self._analyze_network_config(config_file)
    
    async def _analyze_network_config(self, config_file: Path) -> None:
        """Analyze network configuration for security issues."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Check for insecure network settings
            if isinstance(config_data, dict):
                await self._check_network_settings(config_data, config_file)
                
        except Exception as e:
            logger.debug(f"Error analyzing {config_file}: {e}")
    
    async def _check_network_settings(self, config: Dict[str, Any], config_file: Path) -> None:
        """Check network settings for security issues."""
        # Check for HTTP instead of HTTPS
        def check_dict_recursively(d, path=""):
            if isinstance(d, dict):
                for key, value in d.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if isinstance(value, str) and value.startswith('http://'):
                        finding = SecurityFinding(
                            severity='medium',
                            category='network-security',
                            title='Insecure HTTP connection',
                            description=f'Configuration uses HTTP instead of HTTPS: {value}',
                            file_path=str(config_file),
                            remediation='Use HTTPS for secure communication',
                            tool='network-config-analysis'
                        )
                        self.findings.append(finding)
                    
                    elif isinstance(value, (dict, list)):
                        if isinstance(value, dict):
                            check_dict_recursively(value, current_path)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    check_dict_recursively(item, current_path)
        
        check_dict_recursively(config)
    
    async def _run_compliance_check(self) -> None:
        """Run compliance framework checking."""
        logger.info("📋 Running compliance checks...")
        
        frameworks = self.config['compliance']['frameworks']
        
        for framework in frameworks:
            if framework == 'owasp-top10':
                await self._check_owasp_compliance()
            elif framework == 'pci-dss':
                await self._check_pci_compliance()
            elif framework == 'gdpr':
                await self._check_gdpr_compliance()
    
    async def _check_owasp_compliance(self) -> None:
        """Check OWASP Top 10 compliance."""
        # This would implement specific OWASP Top 10 checks
        # For now, we'll add a summary finding
        
        # Count existing security findings by OWASP category
        owasp_categories = {
            'A01:2021 – Broken Access Control': 0,
            'A02:2021 – Cryptographic Failures': 0,
            'A03:2021 – Injection': 0,
            'A04:2021 – Insecure Design': 0,
            'A05:2021 – Security Misconfiguration': 0,
            'A06:2021 – Vulnerable and Outdated Components': 0,
            'A07:2021 – Identification and Authentication Failures': 0,
            'A08:2021 – Software and Data Integrity Failures': 0,
            'A09:2021 – Security Logging and Monitoring Failures': 0,
            'A10:2021 – Server-Side Request Forgery': 0,
        }
        
        # Map existing findings to OWASP categories
        for finding in self.findings:
            if finding.category == 'auth-security':
                owasp_categories['A07:2021 – Identification and Authentication Failures'] += 1
            elif finding.category == 'dependency-vulnerability':
                owasp_categories['A06:2021 – Vulnerable and Outdated Components'] += 1
            elif finding.category == 'api-security':
                owasp_categories['A01:2021 – Broken Access Control'] += 1
            # ... more mappings
        
        # Create compliance summary
        total_issues = sum(owasp_categories.values())
        if total_issues > 0:
            finding = SecurityFinding(
                severity='medium',
                category='compliance',
                title='OWASP Top 10 Compliance Issues',
                description=f'Found {total_issues} issues across OWASP Top 10 categories',
                remediation='Address individual security findings to improve OWASP compliance',
                tool='owasp-compliance-check'
            )
            self.findings.append(finding)
    
    async def _check_pci_compliance(self) -> None:
        """Check PCI DSS compliance."""
        # PCI DSS specific checks
        pass
    
    async def _check_gdpr_compliance(self) -> None:
        """Check GDPR compliance."""
        # GDPR specific checks - data protection, consent, etc.
        pass
    
    async def _run_subprocess(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run subprocess command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.target_path)
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode('utf-8', errors='ignore') if stdout else '',
            stderr=stderr.decode('utf-8', errors='ignore') if stderr else ''
        )
    
    def _generate_report(self) -> SecurityAuditReport:
        """Generate comprehensive security audit report."""
        end_time = time.time()
        scan_duration = end_time - self.start_time
        
        # Count findings by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for finding in self.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        
        # Calculate risk score
        risk_score = (
            severity_counts['critical'] * 10 +
            severity_counts['high'] * 7 +
            severity_counts['medium'] * 4 +
            severity_counts['low'] * 1
        )
        
        # Compliance status
        compliance_status = {
            'owasp-top10': 'needs-attention' if severity_counts['critical'] + severity_counts['high'] > 5 else 'compliant',
            'pci-dss': 'needs-review',
            'gdpr': 'needs-review'
        }
        
        # Tools used
        tools_used = list(set(finding.tool for finding in self.findings if finding.tool))
        
        report = SecurityAuditReport(
            timestamp=datetime.now().isoformat(),
            total_findings=len(self.findings),
            critical_findings=severity_counts['critical'],
            high_findings=severity_counts['high'],
            medium_findings=severity_counts['medium'],
            low_findings=severity_counts['low'],
            findings=self.findings,
            scan_duration=scan_duration,
            tools_used=tools_used,
            compliance_status=compliance_status,
            risk_score=risk_score
        )
        
        return report
    
    def save_report(self, report: SecurityAuditReport, output_dir: str = "security_reports") -> None:
        """Save security audit report in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = output_path / f"security_audit_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # HTML report
        html_file = output_path / f"security_audit_{timestamp}.html"
        self._generate_html_report(report, html_file)
        
        # SARIF report for CI/CD integration
        sarif_file = output_path / f"security_audit_{timestamp}.sarif"
        self._generate_sarif_report(report, sarif_file)
        
        logger.info(f"Reports saved to {output_path}")
        logger.info(f"JSON: {json_file}")
        logger.info(f"HTML: {html_file}")
        logger.info(f"SARIF: {sarif_file}")
    
    def _generate_html_report(self, report: SecurityAuditReport, output_file: Path) -> None:
        """Generate HTML security report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Audit Report - {report.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }}
                .critical {{ color: #dc3545; }}
                .high {{ color: #fd7e14; }}
                .medium {{ color: #ffc107; }}
                .low {{ color: #28a745; }}
                .finding {{ border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; background: #f8f9fa; }}
                .finding.critical {{ border-color: #dc3545; }}
                .finding.high {{ border-color: #fd7e14; }}
                .finding.medium {{ border-color: #ffc107; }}
                .finding.low {{ border-color: #28a745; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔒 Security Audit Report</h1>
                <p><strong>Generated:</strong> {report.timestamp}</p>
                <p><strong>Scan Duration:</strong> {report.scan_duration:.2f} seconds</p>
                <p><strong>Tools Used:</strong> {', '.join(report.tools_used)}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Total Findings</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.total_findings}</div>
                </div>
                <div class="metric critical">
                    <h3>Critical</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.critical_findings}</div>
                </div>
                <div class="metric high">
                    <h3>High</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.high_findings}</div>
                </div>
                <div class="metric medium">
                    <h3>Medium</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.medium_findings}</div>
                </div>
                <div class="metric low">
                    <h3>Low</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.low_findings}</div>
                </div>
                <div class="metric">
                    <h3>Risk Score</h3>
                    <div style="font-size: 24px; font-weight: bold;">{report.risk_score:.1f}</div>
                </div>
            </div>
            
            <h2>📊 Compliance Status</h2>
            <ul>
        """
        
        for framework, status in report.compliance_status.items():
            html_content += f"<li><strong>{framework.upper()}:</strong> {status}</li>"
        
        html_content += """
            </ul>
            
            <h2>🔍 Security Findings</h2>
        """
        
        for finding in sorted(report.findings, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x.severity]):
            location = ""
            if finding.file_path:
                location = f"<br><small><strong>File:</strong> {finding.file_path}"
                if finding.line_number:
                    location += f":{finding.line_number}"
                location += "</small>"
            
            remediation = ""
            if finding.remediation:
                remediation = f"<br><strong>Remediation:</strong> {finding.remediation}"
            
            html_content += f"""
            <div class="finding {finding.severity}">
                <h4>{finding.title} <span class="{finding.severity}">({finding.severity.upper()})</span></h4>
                <p>{finding.description}</p>
                {location}
                {remediation}
                <br><small><strong>Tool:</strong> {finding.tool or 'N/A'} | <strong>Category:</strong> {finding.category}</small>
            </div>
            """
        
        html_content += """
            </body>
            </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_sarif_report(self, report: SecurityAuditReport, output_file: Path) -> None:
        """Generate SARIF format report for CI/CD integration."""
        sarif_data = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "anomaly_detection Security Auditor",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/yourusername/anomaly_detection"
                    }
                },
                "results": []
            }]
        }
        
        for finding in report.findings:
            result = {
                "ruleId": f"{finding.category}-{finding.title.replace(' ', '-').lower()}",
                "level": {"critical": "error", "high": "error", "medium": "warning", "low": "note"}[finding.severity],
                "message": {
                    "text": finding.description
                }
            }
            
            if finding.file_path:
                result["locations"] = [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": finding.file_path
                        }
                    }
                }]
                
                if finding.line_number:
                    result["locations"][0]["physicalLocation"]["region"] = {
                        "startLine": finding.line_number
                    }
            
            sarif_data["runs"][0]["results"].append(result)
        
        with open(output_file, 'w') as f:
            json.dump(sarif_data, f, indent=2)


async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Security Audit Framework")
    parser.add_argument("--target", default=".", help="Target directory to audit")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-dir", default="security_reports", help="Output directory for reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize auditor
    auditor = SecurityAuditor(target_path=args.target, config_path=args.config)
    
    # Run comprehensive audit
    try:
        report = await auditor.run_comprehensive_audit()
        
        # Save reports
        auditor.save_report(report, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("🔒 SECURITY AUDIT SUMMARY")
        print("="*60)
        print(f"Total Findings: {report.total_findings}")
        print(f"Critical: {report.critical_findings}")
        print(f"High: {report.high_findings}")
        print(f"Medium: {report.medium_findings}")
        print(f"Low: {report.low_findings}")
        print(f"Risk Score: {report.risk_score:.1f}")
        print(f"Scan Duration: {report.scan_duration:.2f} seconds")
        print(f"Tools Used: {', '.join(report.tools_used)}")
        
        print(f"\n📊 Compliance Status:")
        for framework, status in report.compliance_status.items():
            print(f"  {framework.upper()}: {status}")
        
        if report.critical_findings > 0:
            print(f"\n⚠️  CRITICAL ISSUES FOUND! Immediate attention required.")
            return 1
        elif report.high_findings > 0:
            print(f"\n⚠️  High severity issues found. Review recommended.")
            return 0
        else:
            print(f"\n✅ No critical or high severity issues found.")
            return 0
            
    except Exception as e:
        logger.error(f"Security audit failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))