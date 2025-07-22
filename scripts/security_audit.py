#!/usr/bin/env python3
"""
Comprehensive Security Audit

This script performs security audits across all packages, checking for
common vulnerabilities, security best practices, and compliance requirements.
"""

import os
import sys
import subprocess
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class SecurityAuditor:
    """
    Comprehensive security auditor for the entire codebase.
    """
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.audit_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scope": "comprehensive_security_audit",
            "findings": [],
            "summary": {},
            "compliance_status": "unknown"
        }
        
    def scan_sql_injection_vulnerabilities(self) -> List[Dict]:
        """Scan for potential SQL injection vulnerabilities."""
        print("🔍 Scanning for SQL injection vulnerabilities...")
        
        findings = []
        sql_patterns = [
            r'execute\s*\(\s*[\'"].*%.*[\'"]',
            r'cursor\.execute\s*\(\s*[\'"].*\+.*[\'"]',
            r'query\s*=\s*[\'"].*%.*[\'"]',
            r'sql\s*=\s*[\'"].*\+.*[\'"]',
            r'SELECT\s+.*\+.*FROM',
            r'INSERT\s+.*\+.*VALUES'
        ]
        
        python_files = list(self.base_dir.rglob("*.py"))
        for file_path in python_files:
            if any(excluded in str(file_path) for excluded in ["__pycache__", ".venv", "venv", "test_"]):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in sql_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append({
                                "type": "sql_injection_risk",
                                "severity": "high",
                                "file": str(file_path.relative_to(self.base_dir)),
                                "line": line_num,
                                "pattern": pattern,
                                "description": "Potential SQL injection vulnerability detected",
                                "recommendation": "Use parameterized queries or ORM methods"
                            })
            except Exception:
                continue
        
        return findings
    
    def scan_hardcoded_secrets(self) -> List[Dict]:
        """Scan for hardcoded secrets and credentials."""
        print("🔍 Scanning for hardcoded secrets...")
        
        findings = []
        secret_patterns = [
            (r'password\s*=\s*[\'"][^\'"\s]+[\'"]', "hardcoded_password"),
            (r'api[_-]?key\s*=\s*[\'"][^\'"\s]+[\'"]', "api_key"),
            (r'secret[_-]?key\s*=\s*[\'"][^\'"\s]+[\'"]', "secret_key"),
            (r'token\s*=\s*[\'"][^\'"\s]{10,}[\'"]', "access_token"),
            (r'aws[_-]?secret\s*=\s*[\'"][^\'"\s]+[\'"]', "aws_secret"),
            (r'database[_-]?url\s*=\s*[\'"].*://.*:[^@]+@.*[\'"]', "database_credentials"),
        ]
        
        python_files = list(self.base_dir.rglob("*.py"))
        config_files = list(self.base_dir.rglob("*.json")) + list(self.base_dir.rglob("*.yaml")) + list(self.base_dir.rglob("*.yml"))
        
        for file_path in python_files + config_files:
            if any(excluded in str(file_path) for excluded in ["__pycache__", ".venv", "venv", "test_"]):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip obvious test/example values
                            if any(test_val in line.lower() for test_val in ["test", "example", "dummy", "fake", "mock"]):
                                continue
                                
                            findings.append({
                                "type": "hardcoded_secret",
                                "severity": "critical",
                                "file": str(file_path.relative_to(self.base_dir)),
                                "line": line_num,
                                "secret_type": secret_type,
                                "description": f"Potential hardcoded {secret_type.replace('_', ' ')} detected",
                                "recommendation": "Use environment variables or secure vaults for secrets"
                            })
            except Exception:
                continue
        
        return findings
    
    def scan_authentication_vulnerabilities(self) -> List[Dict]:
        """Scan for authentication and authorization vulnerabilities."""
        print("🔍 Scanning for authentication vulnerabilities...")
        
        findings = []
        auth_patterns = [
            (r'md5\s*\(', "weak_hashing", "MD5 is cryptographically broken"),
            (r'sha1\s*\(', "weak_hashing", "SHA1 is deprecated for passwords"),
            (r'password\s*==\s*[\'"][^\'"]', "plaintext_comparison", "Plain text password comparison"),
            (r'admin\s*==\s*user', "privilege_escalation", "Potential privilege escalation"),
            (r'if\s+not\s+auth', "auth_bypass", "Potential authentication bypass"),
            (r'decode\s*\(\s*[\'"]base64[\'"]', "base64_credentials", "Base64 is not encryption"),
        ]
        
        python_files = list(self.base_dir.rglob("*.py"))
        for file_path in python_files:
            if any(excluded in str(file_path) for excluded in ["__pycache__", ".venv", "venv"]):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern, vuln_type, description in auth_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            severity = "critical" if vuln_type in ["plaintext_comparison", "privilege_escalation"] else "high"
                            
                            findings.append({
                                "type": "authentication_vulnerability",
                                "severity": severity,
                                "file": str(file_path.relative_to(self.base_dir)),
                                "line": line_num,
                                "vulnerability_type": vuln_type,
                                "description": description,
                                "recommendation": "Implement secure authentication practices"
                            })
            except Exception:
                continue
        
        return findings
    
    def scan_input_validation_issues(self) -> List[Dict]:
        """Scan for input validation vulnerabilities."""
        print("🔍 Scanning for input validation issues...")
        
        findings = []
        validation_patterns = [
            (r'eval\s*\(', "code_injection", "eval() can execute arbitrary code"),
            (r'exec\s*\(', "code_injection", "exec() can execute arbitrary code"),
            (r'os\.system\s*\(', "command_injection", "os.system() vulnerable to command injection"),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "shell_injection", "shell=True enables command injection"),
            (r'pickle\.loads\s*\(', "deserialization", "pickle.loads() can execute arbitrary code"),
            (r'yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader', "yaml_injection", "yaml.Loader enables arbitrary code execution"),
        ]
        
        python_files = list(self.base_dir.rglob("*.py"))
        for file_path in python_files:
            if any(excluded in str(file_path) for excluded in ["__pycache__", ".venv", "venv"]):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern, vuln_type, description in validation_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            severity = "critical" if vuln_type in ["code_injection", "deserialization"] else "high"
                            
                            findings.append({
                                "type": "input_validation_issue",
                                "severity": severity,
                                "file": str(file_path.relative_to(self.base_dir)),
                                "line": line_num,
                                "vulnerability_type": vuln_type,
                                "description": description,
                                "recommendation": "Validate and sanitize all user inputs"
                            })
            except Exception:
                continue
        
        return findings
    
    def check_dependency_vulnerabilities(self) -> List[Dict]:
        """Check for known vulnerabilities in dependencies."""
        print("🔍 Checking dependency vulnerabilities...")
        
        findings = []
        
        # Check for requirements.txt files
        req_files = list(self.base_dir.rglob("requirements*.txt")) + list(self.base_dir.rglob("pyproject.toml"))
        
        vulnerable_packages = {
            "django": "4.0",
            "flask": "2.0", 
            "requests": "2.25.0",
            "pyyaml": "5.4.0",
            "pillow": "8.2.0",
            "cryptography": "3.4.0"
        }
        
        for req_file in req_files:
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for package, min_version in vulnerable_packages.items():
                    pattern = rf'{package}\s*[<>=!]*\s*([0-9.]+)'
                    match = re.search(pattern, content, re.IGNORECASE)
                    
                    if match:
                        version = match.group(1)
                        # Simplified version comparison
                        if version < min_version:
                            findings.append({
                                "type": "vulnerable_dependency",
                                "severity": "high",
                                "file": str(req_file.relative_to(self.base_dir)),
                                "package": package,
                                "current_version": version,
                                "min_safe_version": min_version,
                                "description": f"Vulnerable version of {package} detected",
                                "recommendation": f"Update {package} to version {min_version} or later"
                            })
            except Exception:
                continue
        
        return findings
    
    def check_file_permissions(self) -> List[Dict]:
        """Check for insecure file permissions."""
        print("🔍 Checking file permissions...")
        
        findings = []
        
        # Check for overly permissive files
        sensitive_files = list(self.base_dir.rglob("*.key")) + list(self.base_dir.rglob("*.pem")) + \
                         list(self.base_dir.rglob("config.py")) + list(self.base_dir.rglob(".env*"))
        
        for file_path in sensitive_files:
            try:
                # Get file permissions (Unix-like systems)
                stat_info = file_path.stat()
                permissions = oct(stat_info.st_mode)[-3:]
                
                # Check for world-readable files
                if permissions[2] in ['4', '5', '6', '7']:
                    findings.append({
                        "type": "insecure_file_permissions",
                        "severity": "medium",
                        "file": str(file_path.relative_to(self.base_dir)),
                        "permissions": permissions,
                        "description": "Sensitive file is world-readable",
                        "recommendation": "Restrict file permissions to owner only (600 or 640)"
                    })
            except Exception:
                continue
        
        return findings
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security audit report."""
        findings_by_severity = {}
        findings_by_type = {}
        
        for finding in self.audit_results["findings"]:
            severity = finding["severity"]
            finding_type = finding["type"]
            
            if severity not in findings_by_severity:
                findings_by_severity[severity] = []
            findings_by_severity[severity].append(finding)
            
            if finding_type not in findings_by_type:
                findings_by_type[finding_type] = []
            findings_by_type[finding_type].append(finding)
        
        # Generate report
        report_lines = []
        
        report_lines.append("🔒 COMPREHENSIVE SECURITY AUDIT REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {self.audit_results['timestamp']}")
        report_lines.append(f"Scope: {self.audit_results['scope']}")
        report_lines.append("")
        
        # Executive summary
        total_findings = len(self.audit_results["findings"])
        critical_count = len(findings_by_severity.get("critical", []))
        high_count = len(findings_by_severity.get("high", []))
        medium_count = len(findings_by_severity.get("medium", []))
        
        report_lines.append("📊 EXECUTIVE SUMMARY")
        report_lines.append("-" * 30)
        report_lines.append(f"Total Security Findings: {total_findings}")
        report_lines.append(f"Critical Issues: {critical_count}")
        report_lines.append(f"High Risk Issues: {high_count}")
        report_lines.append(f"Medium Risk Issues: {medium_count}")
        
        # Security risk level
        if critical_count > 0:
            risk_level = "🚨 CRITICAL RISK"
            self.audit_results["compliance_status"] = "non_compliant"
        elif high_count > 5:
            risk_level = "⚠️  HIGH RISK"
            self.audit_results["compliance_status"] = "partial_compliance"
        elif high_count > 0 or medium_count > 10:
            risk_level = "⚠️  MEDIUM RISK"
            self.audit_results["compliance_status"] = "partial_compliance"
        else:
            risk_level = "✅ LOW RISK"
            self.audit_results["compliance_status"] = "compliant"
        
        report_lines.append(f"Overall Risk Level: {risk_level}")
        report_lines.append("")
        
        # Detailed findings by severity
        for severity in ["critical", "high", "medium", "low"]:
            if severity in findings_by_severity:
                severity_findings = findings_by_severity[severity]
                severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "📝"}[severity]
                
                report_lines.append(f"{severity_emoji} {severity.upper()} SEVERITY FINDINGS")
                report_lines.append("-" * 40)
                
                for finding in severity_findings[:10]:  # Limit to first 10 per severity
                    report_lines.append(f"  • {finding['description']}")
                    report_lines.append(f"    File: {finding['file']}")
                    if 'line' in finding:
                        report_lines.append(f"    Line: {finding['line']}")
                    report_lines.append(f"    Recommendation: {finding['recommendation']}")
                    report_lines.append("")
                
                if len(severity_findings) > 10:
                    report_lines.append(f"  ... and {len(severity_findings) - 10} more {severity} findings")
                    report_lines.append("")
        
        # Security categories summary
        report_lines.append("🛡️  SECURITY CATEGORIES")
        report_lines.append("-" * 30)
        
        for category, category_findings in findings_by_type.items():
            category_name = category.replace("_", " ").title()
            report_lines.append(f"  • {category_name}: {len(category_findings)} issues")
        
        # Compliance recommendations
        report_lines.append("")
        report_lines.append("📋 COMPLIANCE RECOMMENDATIONS")
        report_lines.append("-" * 35)
        
        if critical_count > 0:
            report_lines.append("1. 🚨 IMMEDIATE ACTION REQUIRED:")
            report_lines.append("   - Address all critical security vulnerabilities")
            report_lines.append("   - Block production deployment until resolved")
        
        if high_count > 0:
            report_lines.append("2. ⚠️  HIGH PRIORITY:")
            report_lines.append("   - Implement secure coding practices")
            report_lines.append("   - Add security testing to CI/CD pipeline")
        
        if medium_count > 0:
            report_lines.append("3. ℹ️  MEDIUM PRIORITY:")
            report_lines.append("   - Review and improve security controls")
            report_lines.append("   - Schedule security training for development team")
        
        report_lines.append("")
        report_lines.append("4. 🔄 ONGOING SECURITY PRACTICES:")
        report_lines.append("   - Regular security audits and penetration testing")
        report_lines.append("   - Dependency vulnerability scanning")
        report_lines.append("   - Security code reviews")
        report_lines.append("   - Incident response planning")
        
        return "\\n".join(report_lines)
    
    def run_comprehensive_audit(self) -> Dict:
        """Run comprehensive security audit."""
        print("🚀 Starting Comprehensive Security Audit")
        print("=" * 70)
        
        # Run all security scans
        all_findings = []
        
        all_findings.extend(self.scan_sql_injection_vulnerabilities())
        all_findings.extend(self.scan_hardcoded_secrets())
        all_findings.extend(self.scan_authentication_vulnerabilities())
        all_findings.extend(self.scan_input_validation_issues())
        all_findings.extend(self.check_dependency_vulnerabilities())
        all_findings.extend(self.check_file_permissions())
        
        self.audit_results["findings"] = all_findings
        
        # Generate report
        report = self.generate_security_report()
        print(report)
        
        # Save audit results
        with open("security-audit-report.json", "w") as f:
            json.dump(self.audit_results, f, indent=2)
        
        with open("security-audit-report.txt", "w") as f:
            f.write(report)
        
        print(f"\\n📄 Security audit reports saved:")
        print(f"   • JSON: security-audit-report.json")
        print(f"   • Text: security-audit-report.txt")
        
        return {
            "total_findings": len(all_findings),
            "compliance_status": self.audit_results["compliance_status"],
            "risk_level": "critical" if any(f["severity"] == "critical" for f in all_findings) else "acceptable",
            "report": report
        }


def main():
    """Main entry point for security audit."""
    parser = argparse.ArgumentParser(description="Comprehensive Security Audit")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (default: current working directory)"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text", "both"],
        default="both",
        help="Output format (default: both)"
    )
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor(base_dir=args.base_dir)
    results = auditor.run_comprehensive_audit()
    
    # Exit with appropriate code for CI/CD integration
    if results["risk_level"] == "critical":
        print("\\n🚨 CRITICAL: Security vulnerabilities detected - deployment blocked!")
        sys.exit(2)
    elif results["compliance_status"] == "partial_compliance":
        print("\\n⚠️  WARNING: Security issues detected - review required!")
        sys.exit(1)
    else:
        print("\\n✅ SUCCESS: No critical security issues detected!")
        sys.exit(0)


if __name__ == "__main__":
    main()