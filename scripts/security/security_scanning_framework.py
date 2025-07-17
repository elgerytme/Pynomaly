#!/usr/bin/env python3
"""
Security Scanning and Vulnerability Assessment Framework

Implements automated security scanning and vulnerability assessment to ensure 
all packages meet security standards.

Issue: #822 - Implement Security Scanning and Vulnerability Assessment
"""

import os
import sys
import json
import time
import subprocess
import logging
import re
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import yaml
from packaging import version
import requests


@dataclass
class SecurityVulnerability:
    """Security vulnerability data structure"""
    id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    title: str
    description: str
    affected_package: str
    affected_version: str
    fixed_version: Optional[str] = None
    cve_id: Optional[str] = None
    references: List[str] = None
    
    def __post_init__(self):
        if self.references is None:
            self.references = []


@dataclass
class SecurityScanResult:
    """Security scan result"""
    scanner: str
    package: str
    vulnerabilities: List[SecurityVulnerability]
    scan_time: float
    passed: bool
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    max_critical_vulnerabilities: int = 0
    max_high_vulnerabilities: int = 0
    max_medium_vulnerabilities: int = 5
    max_low_vulnerabilities: int = 10
    allowed_licenses: List[str] = None
    blocked_packages: List[str] = None
    
    def __post_init__(self):
        if self.allowed_licenses is None:
            self.allowed_licenses = ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause"]
        if self.blocked_packages is None:
            self.blocked_packages = []


class SecurityScanner:
    """Base class for security scanners"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"security.{name}")
    
    def scan(self, package_path: str) -> SecurityScanResult:
        """Perform security scan on package"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if scanner is available"""
        raise NotImplementedError


class DependencyScanner(SecurityScanner):
    """Scanner for dependency vulnerabilities"""
    
    def __init__(self):
        super().__init__("dependency")
        self.vulnerability_db = self._load_vulnerability_database()
    
    def _load_vulnerability_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load vulnerability database (mock implementation)"""
        # In real implementation, this would load from OSV, NVD, or other sources
        return {
            "urllib3": [
                {
                    "id": "GHSA-v845-jxx5-vc9f",
                    "severity": "MEDIUM",
                    "title": "urllib3 vulnerable to proxy-authorization header leakage",
                    "affected_versions": "<1.26.5",
                    "fixed_version": "1.26.5",
                    "cve_id": "CVE-2021-28363"
                }
            ],
            "requests": [
                {
                    "id": "GHSA-j8r2-6x86-q33q",
                    "severity": "HIGH",
                    "title": "Requests vulnerable to unintended proxy authentication leakage",
                    "affected_versions": "<2.27.1",
                    "fixed_version": "2.27.1",
                    "cve_id": "CVE-2022-23607"
                }
            ]
        }
    
    def scan(self, package_path: str) -> SecurityScanResult:
        """Scan package dependencies for vulnerabilities"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            # Find requirements files
            requirements_files = [
                Path(package_path) / "requirements.txt",
                Path(package_path) / "pyproject.toml"
            ]
            
            dependencies = self._extract_dependencies(requirements_files)
            
            # Check each dependency
            for dep_name, dep_version in dependencies.items():
                if dep_name in self.vulnerability_db:
                    for vuln_info in self.vulnerability_db[dep_name]:
                        if self._is_vulnerable(dep_version, vuln_info["affected_versions"]):
                            vulnerability = SecurityVulnerability(
                                id=vuln_info["id"],
                                severity=vuln_info["severity"],
                                title=vuln_info["title"],
                                description=vuln_info.get("description", vuln_info["title"]),
                                affected_package=dep_name,
                                affected_version=dep_version,
                                fixed_version=vuln_info.get("fixed_version"),
                                cve_id=vuln_info.get("cve_id"),
                                references=vuln_info.get("references", [])
                            )
                            vulnerabilities.append(vulnerability)
            
            duration = time.time() - start_time
            
            return SecurityScanResult(
                scanner=self.name,
                package=package_path,
                vulnerabilities=vulnerabilities,
                scan_time=duration,
                passed=len(vulnerabilities) == 0
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return SecurityScanResult(
                scanner=self.name,
                package=package_path,
                vulnerabilities=[],
                scan_time=duration,
                passed=False,
                error=str(e)
            )
    
    def _extract_dependencies(self, requirements_files: List[Path]) -> Dict[str, str]:
        """Extract dependencies from requirements files"""
        dependencies = {}
        
        for req_file in requirements_files:
            if req_file.exists():
                if req_file.name == "requirements.txt":
                    dependencies.update(self._parse_requirements_txt(req_file))
                elif req_file.name == "pyproject.toml":
                    dependencies.update(self._parse_pyproject_toml(req_file))
        
        return dependencies
    
    def _parse_requirements_txt(self, file_path: Path) -> Dict[str, str]:
        """Parse requirements.txt file"""
        dependencies = {}
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse dependency line (e.g., "requests>=2.25.0")
                        match = re.match(r'^([a-zA-Z0-9_-]+)([>=<~!]+)(.+)$', line)
                        if match:
                            name, operator, version_str = match.groups()
                            dependencies[name] = version_str
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {e}")
        
        return dependencies
    
    def _parse_pyproject_toml(self, file_path: Path) -> Dict[str, str]:
        """Parse pyproject.toml file"""
        dependencies = {}
        
        try:
            import toml
            with open(file_path, 'r') as f:
                pyproject = toml.load(f)
            
            # Extract dependencies from project.dependencies
            project_deps = pyproject.get("project", {}).get("dependencies", [])
            for dep in project_deps:
                match = re.match(r'^([a-zA-Z0-9_-]+)([>=<~!]+)(.+)$', dep)
                if match:
                    name, operator, version_str = match.groups()
                    dependencies[name] = version_str
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {e}")
        
        return dependencies
    
    def _is_vulnerable(self, current_version: str, affected_versions: str) -> bool:
        """Check if current version is vulnerable"""
        try:
            # Simple version comparison (in real implementation, use packaging.version)
            if "<" in affected_versions:
                fix_version = affected_versions.replace("<", "").strip()
                return version.parse(current_version) < version.parse(fix_version)
            elif ">" in affected_versions:
                min_version = affected_versions.replace(">", "").strip()
                return version.parse(current_version) > version.parse(min_version)
            elif "==" in affected_versions:
                exact_version = affected_versions.replace("==", "").strip()
                return version.parse(current_version) == version.parse(exact_version)
        except Exception:
            pass
        
        return False
    
    def is_available(self) -> bool:
        """Check if scanner is available"""
        return True


class StaticCodeAnalyzer(SecurityScanner):
    """Static Application Security Testing (SAST) scanner"""
    
    def __init__(self):
        super().__init__("sast")
        self.security_patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security patterns for static analysis"""
        return {
            "hardcoded_secrets": [
                {
                    "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
                    "severity": "HIGH",
                    "description": "Hardcoded password found"
                },
                {
                    "pattern": r"api_key\s*=\s*['\"][^'\"]+['\"]",
                    "severity": "HIGH",
                    "description": "Hardcoded API key found"
                },
                {
                    "pattern": r"secret\s*=\s*['\"][^'\"]+['\"]",
                    "severity": "HIGH",
                    "description": "Hardcoded secret found"
                }
            ],
            "sql_injection": [
                {
                    "pattern": r"execute\s*\(\s*['\"].*%s.*['\"]",
                    "severity": "HIGH",
                    "description": "Potential SQL injection vulnerability"
                },
                {
                    "pattern": r"query\s*\(\s*f['\"].*{.*}.*['\"]",
                    "severity": "MEDIUM",
                    "description": "Potential SQL injection with f-string"
                }
            ],
            "command_injection": [
                {
                    "pattern": r"os\.system\s*\(\s*.*\+.*\)",
                    "severity": "HIGH",
                    "description": "Potential command injection vulnerability"
                },
                {
                    "pattern": r"subprocess\.call\s*\(\s*.*\+.*\)",
                    "severity": "HIGH",
                    "description": "Potential command injection vulnerability"
                }
            ],
            "crypto_issues": [
                {
                    "pattern": r"hashlib\.md5\s*\(",
                    "severity": "MEDIUM",
                    "description": "MD5 is cryptographically broken"
                },
                {
                    "pattern": r"hashlib\.sha1\s*\(",
                    "severity": "MEDIUM",
                    "description": "SHA1 is cryptographically weak"
                }
            ]
        }
    
    def scan(self, package_path: str) -> SecurityScanResult:
        """Perform static code analysis"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            # Find Python files
            python_files = list(Path(package_path).rglob("*.py"))
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for security patterns
                    for category, patterns in self.security_patterns.items():
                        for pattern_info in patterns:
                            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE)
                            for match in matches:
                                line_number = content[:match.start()].count('\n') + 1
                                
                                vulnerability = SecurityVulnerability(
                                    id=f"SAST-{category}-{hashlib.md5(str(match.group()).encode()).hexdigest()[:8]}",
                                    severity=pattern_info["severity"],
                                    title=pattern_info["description"],
                                    description=f"{pattern_info['description']} in {file_path}:{line_number}",
                                    affected_package=str(file_path),
                                    affected_version="current",
                                    references=[f"Line {line_number}: {match.group()}"]
                                )
                                vulnerabilities.append(vulnerability)
                
                except Exception as e:
                    self.logger.warning(f"Failed to scan {file_path}: {e}")
            
            duration = time.time() - start_time
            
            return SecurityScanResult(
                scanner=self.name,
                package=package_path,
                vulnerabilities=vulnerabilities,
                scan_time=duration,
                passed=len(vulnerabilities) == 0
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return SecurityScanResult(
                scanner=self.name,
                package=package_path,
                vulnerabilities=[],
                scan_time=duration,
                passed=False,
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if scanner is available"""
        return True


class LicenseScanner(SecurityScanner):
    """Scanner for license compliance"""
    
    def __init__(self):
        super().__init__("license")
        self.license_db = self._load_license_database()
    
    def _load_license_database(self) -> Dict[str, Dict[str, Any]]:
        """Load license database"""
        return {
            "MIT": {"compatible": True, "type": "permissive"},
            "Apache-2.0": {"compatible": True, "type": "permissive"},
            "BSD-3-Clause": {"compatible": True, "type": "permissive"},
            "BSD-2-Clause": {"compatible": True, "type": "permissive"},
            "GPL-3.0": {"compatible": False, "type": "copyleft"},
            "AGPL-3.0": {"compatible": False, "type": "copyleft"},
            "Unknown": {"compatible": False, "type": "unknown"}
        }
    
    def scan(self, package_path: str) -> SecurityScanResult:
        """Scan package licenses"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            # Find license files
            license_files = [
                Path(package_path) / "LICENSE",
                Path(package_path) / "LICENSE.txt",
                Path(package_path) / "LICENSE.md",
                Path(package_path) / "COPYING"
            ]
            
            detected_licenses = []
            
            for license_file in license_files:
                if license_file.exists():
                    license_type = self._detect_license_type(license_file)
                    detected_licenses.append(license_type)
            
            # Check pyproject.toml for license info
            pyproject_file = Path(package_path) / "pyproject.toml"
            if pyproject_file.exists():
                try:
                    import toml
                    with open(pyproject_file, 'r') as f:
                        pyproject = toml.load(f)
                    
                    license_info = pyproject.get("project", {}).get("license", {})
                    if "text" in license_info:
                        detected_licenses.append(license_info["text"])
                except Exception:
                    pass
            
            # Check for incompatible licenses
            for license_name in detected_licenses:
                license_info = self.license_db.get(license_name, {"compatible": False, "type": "unknown"})
                if not license_info["compatible"]:
                    vulnerability = SecurityVulnerability(
                        id=f"LICENSE-{license_name}",
                        severity="MEDIUM",
                        title=f"Incompatible license: {license_name}",
                        description=f"Package uses {license_name} license which may not be compatible",
                        affected_package=package_path,
                        affected_version="current"
                    )
                    vulnerabilities.append(vulnerability)
            
            duration = time.time() - start_time
            
            return SecurityScanResult(
                scanner=self.name,
                package=package_path,
                vulnerabilities=vulnerabilities,
                scan_time=duration,
                passed=len(vulnerabilities) == 0
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return SecurityScanResult(
                scanner=self.name,
                package=package_path,
                vulnerabilities=[],
                scan_time=duration,
                passed=False,
                error=str(e)
            )
    
    def _detect_license_type(self, license_file: Path) -> str:
        """Detect license type from file content"""
        try:
            with open(license_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            # Simple license detection
            if "mit license" in content or "mit" in content:
                return "MIT"
            elif "apache license" in content or "apache" in content:
                return "Apache-2.0"
            elif "bsd" in content:
                if "3-clause" in content:
                    return "BSD-3-Clause"
                else:
                    return "BSD-2-Clause"
            elif "gpl" in content:
                return "GPL-3.0"
            elif "agpl" in content:
                return "AGPL-3.0"
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
    def is_available(self) -> bool:
        """Check if scanner is available"""
        return True


class SecurityAssessmentFramework:
    """Main security assessment framework"""
    
    def __init__(self, policy_file: str = "security_policy.yaml"):
        self.policy = self._load_security_policy(policy_file)
        self.scanners = [
            DependencyScanner(),
            StaticCodeAnalyzer(),
            LicenseScanner()
        ]
        self.results: List[SecurityScanResult] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_security_policy(self, policy_file: str) -> SecurityPolicy:
        """Load security policy from file"""
        policy_path = Path(policy_file)
        
        if policy_path.exists():
            try:
                with open(policy_path, 'r') as f:
                    policy_data = yaml.safe_load(f)
                return SecurityPolicy(**policy_data)
            except Exception as e:
                self.logger.warning(f"Failed to load policy file: {e}")
        
        return SecurityPolicy()
    
    def run_security_assessment(self, package_paths: List[str]) -> List[SecurityScanResult]:
        """Run comprehensive security assessment"""
        
        self.logger.info(f"Running security assessment on {len(package_paths)} packages")
        
        # Run scanners in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for package_path in package_paths:
                for scanner in self.scanners:
                    if scanner.is_available():
                        future = executor.submit(scanner.scan, package_path)
                        futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    if result.passed:
                        self.logger.info(f"✓ {result.scanner} scan passed for {result.package}")
                    else:
                        self.logger.warning(f"✗ {result.scanner} scan failed for {result.package}")
                        if result.error:
                            self.logger.error(f"Error: {result.error}")
                        
                        for vuln in result.vulnerabilities:
                            self.logger.warning(f"  {vuln.severity}: {vuln.title}")
                
                except Exception as e:
                    self.logger.error(f"Scanner execution failed: {e}")
        
        return self.results
    
    def enforce_security_policy(self) -> bool:
        """Enforce security policy on scan results"""
        
        # Aggregate vulnerabilities by severity
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for result in self.results:
            for vuln in result.vulnerabilities:
                severity_counts[vuln.severity] += 1
        
        # Check against policy
        policy_violations = []
        
        if severity_counts["CRITICAL"] > self.policy.max_critical_vulnerabilities:
            policy_violations.append(f"Critical vulnerabilities: {severity_counts['CRITICAL']} > {self.policy.max_critical_vulnerabilities}")
        
        if severity_counts["HIGH"] > self.policy.max_high_vulnerabilities:
            policy_violations.append(f"High vulnerabilities: {severity_counts['HIGH']} > {self.policy.max_high_vulnerabilities}")
        
        if severity_counts["MEDIUM"] > self.policy.max_medium_vulnerabilities:
            policy_violations.append(f"Medium vulnerabilities: {severity_counts['MEDIUM']} > {self.policy.max_medium_vulnerabilities}")
        
        if severity_counts["LOW"] > self.policy.max_low_vulnerabilities:
            policy_violations.append(f"Low vulnerabilities: {severity_counts['LOW']} > {self.policy.max_low_vulnerabilities}")
        
        if policy_violations:
            self.logger.error("Security policy violations:")
            for violation in policy_violations:
                self.logger.error(f"  - {violation}")
            return False
        
        self.logger.info("Security policy compliance: PASSED")
        return True
    
    def generate_security_report(self, output_file: str = "security_report.html"):
        """Generate security assessment report"""
        
        # Calculate summary statistics
        total_vulnerabilities = sum(len(result.vulnerabilities) for result in self.results)
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for result in self.results:
            for vuln in result.vulnerabilities:
                severity_counts[vuln.severity] += 1
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .summary-item {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 5px; }}
                .critical {{ background: #dc3545; color: white; }}
                .high {{ background: #fd7e14; color: white; }}
                .medium {{ background: #ffc107; color: black; }}
                .low {{ background: #28a745; color: white; }}
                .vulnerability {{ margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid; }}
                .vulnerability.CRITICAL {{ background: #f8d7da; border-left-color: #dc3545; }}
                .vulnerability.HIGH {{ background: #fff3cd; border-left-color: #fd7e14; }}
                .vulnerability.MEDIUM {{ background: #fff3cd; border-left-color: #ffc107; }}
                .vulnerability.LOW {{ background: #d1ecf1; border-left-color: #28a745; }}
                .scanner-results {{ margin: 20px 0; }}
                .scanner-result {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Security Assessment Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="summary-item">
                    <h3>{total_vulnerabilities}</h3>
                    <p>Total Vulnerabilities</p>
                </div>
                <div class="summary-item critical">
                    <h3>{severity_counts['CRITICAL']}</h3>
                    <p>Critical</p>
                </div>
                <div class="summary-item high">
                    <h3>{severity_counts['HIGH']}</h3>
                    <p>High</p>
                </div>
                <div class="summary-item medium">
                    <h3>{severity_counts['MEDIUM']}</h3>
                    <p>Medium</p>
                </div>
                <div class="summary-item low">
                    <h3>{severity_counts['LOW']}</h3>
                    <p>Low</p>
                </div>
            </div>
            
            <h2>Security Policy Compliance</h2>
            <div class="scanner-result">
                <p><strong>Policy Status:</strong> {'PASSED' if self.enforce_security_policy() else 'FAILED'}</p>
                <p><strong>Max Critical:</strong> {self.policy.max_critical_vulnerabilities}</p>
                <p><strong>Max High:</strong> {self.policy.max_high_vulnerabilities}</p>
                <p><strong>Max Medium:</strong> {self.policy.max_medium_vulnerabilities}</p>
                <p><strong>Max Low:</strong> {self.policy.max_low_vulnerabilities}</p>
            </div>
            
            <h2>Scanner Results</h2>
            <div class="scanner-results">
        """
        
        for result in self.results:
            status = "PASSED" if result.passed else "FAILED"
            html_content += f"""
            <div class="scanner-result">
                <h3>{result.scanner.upper()} Scanner - {status}</h3>
                <p><strong>Package:</strong> {result.package}</p>
                <p><strong>Scan Time:</strong> {result.scan_time:.2f}s</p>
                <p><strong>Vulnerabilities:</strong> {len(result.vulnerabilities)}</p>
                
                {f'<p><strong>Error:</strong> {result.error}</p>' if result.error else ''}
                
                <div class="vulnerabilities">
            """
            
            for vuln in result.vulnerabilities:
                html_content += f"""
                <div class="vulnerability {vuln.severity}">
                    <h4>{vuln.title} ({vuln.severity})</h4>
                    <p>{vuln.description}</p>
                    <p><strong>Affected:</strong> {vuln.affected_package} v{vuln.affected_version}</p>
                    {f'<p><strong>Fixed in:</strong> {vuln.fixed_version}</p>' if vuln.fixed_version else ''}
                    {f'<p><strong>CVE:</strong> {vuln.cve_id}</p>' if vuln.cve_id else ''}
                </div>
                """
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Security report generated: {output_file}")
    
    def run_ci_security_scan(self, package_paths: List[str]) -> bool:
        """Run security scan in CI/CD environment"""
        
        self.logger.info("Running CI security scan")
        
        # Run security assessment
        results = self.run_security_assessment(package_paths)
        
        # Generate report
        self.generate_security_report()
        
        # Enforce policy
        policy_passed = self.enforce_security_policy()
        
        # Check for critical issues
        critical_issues = []
        for result in results:
            if result.error:
                critical_issues.append(f"Scanner {result.scanner} failed: {result.error}")
            
            for vuln in result.vulnerabilities:
                if vuln.severity in ["CRITICAL", "HIGH"]:
                    critical_issues.append(f"{vuln.severity}: {vuln.title}")
        
        if critical_issues:
            self.logger.error("Critical security issues found:")
            for issue in critical_issues:
                self.logger.error(f"  - {issue}")
            return False
        
        if not policy_passed:
            self.logger.error("Security policy compliance failed")
            return False
        
        self.logger.info("Security scan passed")
        return True


def main():
    """Main entry point for security scanning"""
    if len(sys.argv) > 1 and sys.argv[1] == "ci":
        # Run CI security scan
        framework = SecurityAssessmentFramework()
        
        # Default package paths
        package_paths = [
            "src/packages",
            ".",
        ]
        
        # Override with command line arguments if provided
        if len(sys.argv) > 2:
            package_paths = sys.argv[2:]
        
        success = framework.run_ci_security_scan(package_paths)
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        print("Security Scanning and Vulnerability Assessment Framework")
        print("Usage: python security_scanning_framework.py [ci] [package_paths...]")
        print("  ci: Run CI security scan")
        print("  package_paths: Paths to scan (default: src/packages .)")


if __name__ == "__main__":
    main()