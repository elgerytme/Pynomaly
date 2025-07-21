#!/usr/bin/env python3
"""
Comprehensive Security Scanner for anomaly_detection

This script provides:
- Automated vulnerability scanning using existing infrastructure
- Dependency vulnerability checks with bandit and safety
- Configuration security assessment
- SAST (Static Application Security Testing)
- Docker security scanning
- Infrastructure security checks
- Compliance validation
- Security metrics reporting
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("security_scan.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveSecurityScanner:
    """Comprehensive security scanner integrating multiple tools."""
    
    def __init__(self, project_root: str = None):
        """Initialize the security scanner."""
        self.project_root = Path(project_root or os.getcwd())
        self.scan_results = {}
        self.scan_timestamp = datetime.now().isoformat()
        self.scan_id = f"scan_{int(datetime.now().timestamp())}"
        
        # Initialize security tools
        self.security_tools = {
            "bandit": self._check_bandit_installed(),
            "safety": self._check_safety_installed(),
            "semgrep": self._check_semgrep_installed(),
            "hadolint": self._check_hadolint_installed(),
            "trivy": self._check_trivy_installed(),
        }
        
        logger.info(f"Security scanner initialized for {self.project_root}")
        logger.info(f"Available security tools: {list(k for k, v in self.security_tools.items() if v)}")
    
    def _check_bandit_installed(self) -> bool:
        """Check if bandit is installed."""
        try:
            subprocess.run(["bandit", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_safety_installed(self) -> bool:
        """Check if safety is installed."""
        try:
            subprocess.run(["safety", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_semgrep_installed(self) -> bool:
        """Check if semgrep is installed."""
        try:
            subprocess.run(["semgrep", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_hadolint_installed(self) -> bool:
        """Check if hadolint is installed."""
        try:
            subprocess.run(["hadolint", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_trivy_installed(self) -> bool:
        """Check if trivy is installed."""
        try:
            subprocess.run(["trivy", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    async def run_comprehensive_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        logger.info("Starting comprehensive security scan...")
        
        # Initialize scan results
        self.scan_results = {
            "scan_id": self.scan_id,
            "timestamp": self.scan_timestamp,
            "project_root": str(self.project_root),
            "tools_used": [],
            "scans": {},
            "summary": {},
            "recommendations": [],
            "compliance_status": {}
        }
        
        # Run all security scans
        tasks = []
        
        # Static Application Security Testing (SAST)
        if self.security_tools["bandit"]:
            tasks.append(self._run_bandit_scan())
        
        if self.security_tools["semgrep"]:
            tasks.append(self._run_semgrep_scan())
        
        # Dependency vulnerability scanning
        if self.security_tools["safety"]:
            tasks.append(self._run_safety_scan())
        
        # Docker security scanning
        if self.security_tools["hadolint"]:
            tasks.append(self._run_hadolint_scan())
        
        if self.security_tools["trivy"]:
            tasks.append(self._run_trivy_scan())
        
        # Configuration security
        tasks.append(self._run_configuration_scan())
        
        # Infrastructure security
        tasks.append(self._run_infrastructure_scan())
        
        # Secret scanning
        tasks.append(self._run_secret_scan())
        
        # Custom vulnerability scanning using existing infrastructure
        tasks.append(self._run_anomaly_detection_vulnerability_scan())
        
        # Run all scans concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Generate summary and recommendations
        self._generate_scan_summary()
        self._generate_recommendations()
        self._assess_compliance()
        
        # Save results
        self._save_scan_results()
        
        logger.info("Comprehensive security scan completed")
        return self.scan_results
    
    async def _run_bandit_scan(self) -> None:
        """Run bandit static security analysis."""
        logger.info("Running bandit security scan...")
        
        try:
            cmd = [
                "bandit",
                "-r", str(self.project_root / "src"),
                "-f", "json",
                "--skip", "B101,B601",  # Skip assert and shell usage warnings
                "--exclude", "*/tests/*,*/venv/*,*/node_modules/*"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                self.scan_results["scans"]["bandit"] = {
                    "tool": "bandit",
                    "status": "completed",
                    "results": bandit_results,
                    "issues_found": len(bandit_results.get("results", [])),
                    "severity_breakdown": self._analyze_bandit_results(bandit_results)
                }
                self.scan_results["tools_used"].append("bandit")
            
            logger.info(f"Bandit scan completed: {len(bandit_results.get('results', []))} issues found")
            
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            self.scan_results["scans"]["bandit"] = {
                "tool": "bandit",
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_semgrep_scan(self) -> None:
        """Run semgrep security analysis."""
        logger.info("Running semgrep security scan...")
        
        try:
            cmd = [
                "semgrep",
                "--config=auto",
                "--json",
                "--exclude=tests/",
                "--exclude=venv/",
                "--exclude=node_modules/",
                str(self.project_root / "src")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                semgrep_results = json.loads(result.stdout)
                self.scan_results["scans"]["semgrep"] = {
                    "tool": "semgrep",
                    "status": "completed",
                    "results": semgrep_results,
                    "issues_found": len(semgrep_results.get("results", [])),
                    "severity_breakdown": self._analyze_semgrep_results(semgrep_results)
                }
                self.scan_results["tools_used"].append("semgrep")
            
            logger.info(f"Semgrep scan completed: {len(semgrep_results.get('results', []))} issues found")
            
        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
            self.scan_results["scans"]["semgrep"] = {
                "tool": "semgrep",
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_safety_scan(self) -> None:
        """Run safety dependency vulnerability scan."""
        logger.info("Running safety dependency scan...")
        
        try:
            # Check for requirements.txt files
            requirements_files = list(self.project_root.rglob("requirements*.txt"))
            
            if not requirements_files:
                logger.warning("No requirements.txt files found")
                return
            
            safety_results = []
            for req_file in requirements_files:
                cmd = ["safety", "check", "-r", str(req_file), "--json"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.stdout:
                    try:
                        file_results = json.loads(result.stdout)
                        safety_results.extend(file_results)
                    except json.JSONDecodeError:
                        # Safety might return non-JSON output
                        safety_results.append({
                            "file": str(req_file),
                            "output": result.stdout,
                            "error": result.stderr
                        })
            
            self.scan_results["scans"]["safety"] = {
                "tool": "safety",
                "status": "completed",
                "results": safety_results,
                "issues_found": len(safety_results),
                "files_scanned": [str(f) for f in requirements_files]
            }
            self.scan_results["tools_used"].append("safety")
            
            logger.info(f"Safety scan completed: {len(safety_results)} vulnerabilities found")
            
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            self.scan_results["scans"]["safety"] = {
                "tool": "safety",
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_hadolint_scan(self) -> None:
        """Run hadolint Docker security scan."""
        logger.info("Running hadolint Docker scan...")
        
        try:
            # Find Dockerfile
            dockerfiles = list(self.project_root.rglob("Dockerfile*"))
            
            if not dockerfiles:
                logger.warning("No Dockerfiles found")
                return
            
            hadolint_results = []
            for dockerfile in dockerfiles:
                cmd = ["hadolint", "--format", "json", str(dockerfile)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.stdout:
                    try:
                        file_results = json.loads(result.stdout)
                        hadolint_results.extend(file_results)
                    except json.JSONDecodeError:
                        hadolint_results.append({
                            "file": str(dockerfile),
                            "output": result.stdout,
                            "error": result.stderr
                        })
            
            self.scan_results["scans"]["hadolint"] = {
                "tool": "hadolint",
                "status": "completed",
                "results": hadolint_results,
                "issues_found": len(hadolint_results),
                "files_scanned": [str(f) for f in dockerfiles]
            }
            self.scan_results["tools_used"].append("hadolint")
            
            logger.info(f"Hadolint scan completed: {len(hadolint_results)} issues found")
            
        except Exception as e:
            logger.error(f"Hadolint scan failed: {e}")
            self.scan_results["scans"]["hadolint"] = {
                "tool": "hadolint",
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_trivy_scan(self) -> None:
        """Run trivy vulnerability scan."""
        logger.info("Running trivy vulnerability scan...")
        
        try:
            # Scan filesystem
            cmd = [
                "trivy", "fs", "--format", "json",
                "--skip-dirs", "venv,node_modules,.git",
                str(self.project_root)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                trivy_results = json.loads(result.stdout)
                self.scan_results["scans"]["trivy"] = {
                    "tool": "trivy",
                    "status": "completed",
                    "results": trivy_results,
                    "issues_found": sum(
                        len(target.get("Vulnerabilities", []))
                        for target in trivy_results.get("Results", [])
                    )
                }
                self.scan_results["tools_used"].append("trivy")
            
            logger.info("Trivy scan completed")
            
        except Exception as e:
            logger.error(f"Trivy scan failed: {e}")
            self.scan_results["scans"]["trivy"] = {
                "tool": "trivy",
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_configuration_scan(self) -> None:
        """Run configuration security scan."""
        logger.info("Running configuration security scan...")
        
        try:
            config_issues = []
            
            # Check for common configuration issues
            sensitive_files = [
                ".env", ".env.local", ".env.production",
                "config.py", "settings.py", "secrets.json"
            ]
            
            for file_pattern in sensitive_files:
                files = list(self.project_root.rglob(file_pattern))
                for file_path in files:
                    if file_path.exists():
                        # Check file permissions
                        stat = file_path.stat()
                        permissions = oct(stat.st_mode)[-3:]
                        
                        if permissions[2] != '0':  # World-readable
                            config_issues.append({
                                "type": "file_permissions",
                                "file": str(file_path),
                                "issue": f"World-readable permissions: {permissions}",
                                "severity": "medium",
                                "recommendation": "Change permissions to 600"
                            })
                        
                        # Check for hardcoded secrets
                        if file_path.suffix == '.py':
                            content = file_path.read_text()
                            if 'password' in content.lower() or 'secret' in content.lower():
                                config_issues.append({
                                    "type": "hardcoded_secrets",
                                    "file": str(file_path),
                                    "issue": "Potential hardcoded secrets",
                                    "severity": "high",
                                    "recommendation": "Use environment variables"
                                })
            
            self.scan_results["scans"]["configuration"] = {
                "tool": "configuration_scanner",
                "status": "completed",
                "results": config_issues,
                "issues_found": len(config_issues)
            }
            self.scan_results["tools_used"].append("configuration_scanner")
            
            logger.info(f"Configuration scan completed: {len(config_issues)} issues found")
            
        except Exception as e:
            logger.error(f"Configuration scan failed: {e}")
            self.scan_results["scans"]["configuration"] = {
                "tool": "configuration_scanner",
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_infrastructure_scan(self) -> None:
        """Run infrastructure security scan."""
        logger.info("Running infrastructure security scan...")
        
        try:
            infra_issues = []
            
            # Check Docker configuration
            docker_compose_files = list(self.project_root.rglob("docker-compose*.yml"))
            for compose_file in docker_compose_files:
                content = compose_file.read_text()
                
                # Check for privileged containers
                if "privileged: true" in content:
                    infra_issues.append({
                        "type": "docker_security",
                        "file": str(compose_file),
                        "issue": "Privileged container configuration",
                        "severity": "high",
                        "recommendation": "Avoid privileged containers"
                    })
                
                # Check for host network mode
                if "network_mode: host" in content:
                    infra_issues.append({
                        "type": "docker_security",
                        "file": str(compose_file),
                        "issue": "Host network mode",
                        "severity": "medium",
                        "recommendation": "Use bridge networking"
                    })
            
            # Check Kubernetes configurations
            k8s_files = list(self.project_root.rglob("*.yaml")) + list(self.project_root.rglob("*.yml"))
            for k8s_file in k8s_files:
                if "deployment" in k8s_file.name.lower() or "pod" in k8s_file.name.lower():
                    content = k8s_file.read_text()
                    
                    # Check for privileged pods
                    if "privileged: true" in content:
                        infra_issues.append({
                            "type": "kubernetes_security",
                            "file": str(k8s_file),
                            "issue": "Privileged pod configuration",
                            "severity": "high",
                            "recommendation": "Remove privileged access"
                        })
                    
                    # Check for root user
                    if "runAsUser: 0" in content:
                        infra_issues.append({
                            "type": "kubernetes_security",
                            "file": str(k8s_file),
                            "issue": "Running as root user",
                            "severity": "medium",
                            "recommendation": "Use non-root user"
                        })
            
            self.scan_results["scans"]["infrastructure"] = {
                "tool": "infrastructure_scanner",
                "status": "completed",
                "results": infra_issues,
                "issues_found": len(infra_issues)
            }
            self.scan_results["tools_used"].append("infrastructure_scanner")
            
            logger.info(f"Infrastructure scan completed: {len(infra_issues)} issues found")
            
        except Exception as e:
            logger.error(f"Infrastructure scan failed: {e}")
            self.scan_results["scans"]["infrastructure"] = {
                "tool": "infrastructure_scanner",
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_secret_scan(self) -> None:
        """Run secret detection scan."""
        logger.info("Running secret detection scan...")
        
        try:
            secret_patterns = [
                (r"(?i)password\s*=\s*['\"][^'\"]{8,}['\"]", "hardcoded_password"),
                (r"(?i)secret\s*=\s*['\"][^'\"]{8,}['\"]", "hardcoded_secret"),
                (r"(?i)api_key\s*=\s*['\"][^'\"]{8,}['\"]", "hardcoded_api_key"),
                (r"(?i)token\s*=\s*['\"][^'\"]{8,}['\"]", "hardcoded_token"),
                (r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----", "private_key"),
                (r"sk_live_[0-9a-zA-Z]{24}", "stripe_secret_key"),
                (r"AIza[0-9A-Za-z\\-_]{35}", "google_api_key"),
                (r"AKIA[0-9A-Z]{16}", "aws_access_key"),
                (r"mongodb://[^\\s]+", "mongodb_connection"),
                (r"postgres://[^\\s]+", "postgres_connection"),
            ]
            
            secret_issues = []
            
            # Scan Python files
            python_files = list(self.project_root.rglob("*.py"))
            for py_file in python_files:
                if "venv" in str(py_file) or "node_modules" in str(py_file):
                    continue
                
                try:
                    content = py_file.read_text()
                    
                    for pattern, secret_type in secret_patterns:
                        import re
                        matches = re.finditer(pattern, content)
                        
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            secret_issues.append({
                                "type": "secret_detection",
                                "file": str(py_file),
                                "line": line_num,
                                "secret_type": secret_type,
                                "issue": f"Potential {secret_type} detected",
                                "severity": "high",
                                "recommendation": "Move secrets to environment variables"
                            })
                except Exception as e:
                    logger.warning(f"Failed to scan {py_file}: {e}")
            
            self.scan_results["scans"]["secrets"] = {
                "tool": "secret_scanner",
                "status": "completed",
                "results": secret_issues,
                "issues_found": len(secret_issues)
            }
            self.scan_results["tools_used"].append("secret_scanner")
            
            logger.info(f"Secret scan completed: {len(secret_issues)} issues found")
            
        except Exception as e:
            logger.error(f"Secret scan failed: {e}")
            self.scan_results["scans"]["secrets"] = {
                "tool": "secret_scanner",
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_anomaly_detection_vulnerability_scan(self) -> None:
        """Run anomaly_detection-specific vulnerability scan using existing infrastructure."""
        logger.info("Running anomaly_detection vulnerability scan...")
        
        try:
            # Import and use existing vulnerability scanner
            sys.path.append(str(self.project_root / "src"))
            
            from anomaly_detection.presentation.api.security.vulnerability_scanner import VulnerabilityScanner
            
            scanner = VulnerabilityScanner()
            
            # Configuration for scan
            config = {
                "DEBUG": False,
                "SECRET_KEY": "test-key",
                "SECURITY_HEADERS": {},
                "SSL_VERIFY": True,
                "MIN_PASSWORD_LENGTH": 12,
            }
            
            # Code paths to scan
            code_paths = [
                str(self.project_root / "src" / "anomaly_detection"),
                str(self.project_root / "src" / "packages"),
            ]
            
            # Requirements file
            requirements_file = str(self.project_root / "requirements.txt")
            
            # Run scan
            anomaly_detection_results = scanner.scan_all(config, code_paths, requirements_file)
            
            self.scan_results["scans"]["anomaly_detection_vulnerability"] = {
                "tool": "anomaly_detection_vulnerability_scanner",
                "status": "completed",
                "results": anomaly_detection_results,
                "issues_found": len(anomaly_detection_results.get("vulnerabilities", []))
            }
            self.scan_results["tools_used"].append("anomaly_detection_vulnerability_scanner")
            
            logger.info(f"anomaly_detection vulnerability scan completed: {len(anomaly_detection_results.get('vulnerabilities', []))} issues found")
            
        except Exception as e:
            logger.error(f"anomaly_detection vulnerability scan failed: {e}")
            self.scan_results["scans"]["anomaly_detection_vulnerability"] = {
                "tool": "anomaly_detection_vulnerability_scanner",
                "status": "failed",
                "error": str(e)
            }
    
    def _analyze_bandit_results(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Analyze bandit results to extract severity breakdown."""
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        
        for result in results.get("results", []):
            severity = result.get("issue_severity", "MEDIUM")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return severity_counts
    
    def _analyze_semgrep_results(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Analyze semgrep results to extract severity breakdown."""
        severity_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        
        for result in results.get("results", []):
            severity = result.get("extra", {}).get("severity", "INFO")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return severity_counts
    
    def _generate_scan_summary(self) -> None:
        """Generate overall scan summary."""
        total_issues = 0
        scans_completed = 0
        scans_failed = 0
        
        for scan_name, scan_data in self.scan_results["scans"].items():
            if scan_data["status"] == "completed":
                scans_completed += 1
                total_issues += scan_data.get("issues_found", 0)
            else:
                scans_failed += 1
        
        self.scan_results["summary"] = {
            "total_issues": total_issues,
            "scans_completed": scans_completed,
            "scans_failed": scans_failed,
            "tools_used": len(self.scan_results["tools_used"]),
            "scan_duration": "N/A"  # Will be calculated when scan completes
        }
    
    def _generate_recommendations(self) -> None:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        # High priority recommendations
        if any(scan.get("issues_found", 0) > 0 for scan in self.scan_results["scans"].values()):
            recommendations.append({
                "priority": "high",
                "category": "vulnerability_management",
                "recommendation": "Address all identified security vulnerabilities",
                "action": "Review and fix security issues found in scans"
            })
        
        # Tool-specific recommendations
        if "bandit" in self.scan_results["tools_used"]:
            bandit_issues = self.scan_results["scans"]["bandit"].get("issues_found", 0)
            if bandit_issues > 0:
                recommendations.append({
                    "priority": "medium",
                    "category": "static_analysis",
                    "recommendation": f"Fix {bandit_issues} static analysis issues",
                    "action": "Review bandit findings and implement fixes"
                })
        
        if "safety" in self.scan_results["tools_used"]:
            safety_issues = self.scan_results["scans"]["safety"].get("issues_found", 0)
            if safety_issues > 0:
                recommendations.append({
                    "priority": "high",
                    "category": "dependency_management",
                    "recommendation": f"Update {safety_issues} vulnerable dependencies",
                    "action": "Update dependencies to secure versions"
                })
        
        # General recommendations
        recommendations.extend([
            {
                "priority": "medium",
                "category": "automation",
                "recommendation": "Integrate security scanning into CI/CD pipeline",
                "action": "Set up automated security scans on every commit"
            },
            {
                "priority": "low",
                "category": "monitoring",
                "recommendation": "Set up security monitoring and alerting",
                "action": "Configure alerts for security events"
            },
            {
                "priority": "medium",
                "category": "training",
                "recommendation": "Provide security training for development team",
                "action": "Conduct security awareness training"
            }
        ])
        
        self.scan_results["recommendations"] = recommendations
    
    def _assess_compliance(self) -> None:
        """Assess compliance with security standards."""
        compliance_checks = {
            "OWASP_Top_10": {
                "status": "partial",
                "description": "Partial compliance with OWASP Top 10",
                "findings": []
            },
            "NIST_Cybersecurity_Framework": {
                "status": "partial",
                "description": "Partial compliance with NIST framework",
                "findings": []
            },
            "ISO_27001": {
                "status": "needs_review",
                "description": "Requires manual review for ISO 27001",
                "findings": []
            }
        }
        
        # Add specific findings based on scan results
        total_issues = self.scan_results["summary"]["total_issues"]
        
        if total_issues > 0:
            compliance_checks["OWASP_Top_10"]["findings"].append(
                f"{total_issues} security issues found that may affect OWASP compliance"
            )
        
        self.scan_results["compliance_status"] = compliance_checks
    
    def _save_scan_results(self) -> None:
        """Save scan results to files."""
        # Create output directory
        output_dir = self.project_root / "reports" / "security"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_file = output_dir / f"security_scan_{self.scan_id}.json"
        with open(json_file, 'w') as f:
            json.dump(self.scan_results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = output_dir / f"security_summary_{self.scan_id}.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_text_report())
        
        logger.info(f"Scan results saved to {output_dir}")
    
    def _generate_text_report(self) -> str:
        """Generate human-readable text report."""
        report = f"""
anomaly_detection SECURITY SCAN REPORT
============================

Scan ID: {self.scan_id}
Timestamp: {self.scan_timestamp}
Project: {self.project_root}

SUMMARY
-------
Total Issues Found: {self.scan_results['summary']['total_issues']}
Scans Completed: {self.scan_results['summary']['scans_completed']}
Scans Failed: {self.scan_results['summary']['scans_failed']}
Tools Used: {', '.join(self.scan_results['tools_used'])}

SCAN RESULTS
-----------
"""
        
        for scan_name, scan_data in self.scan_results["scans"].items():
            report += f"\n{scan_name.upper()}:\n"
            report += f"  Status: {scan_data['status']}\n"
            report += f"  Issues Found: {scan_data.get('issues_found', 0)}\n"
            
            if scan_data["status"] == "failed":
                report += f"  Error: {scan_data.get('error', 'Unknown error')}\n"
        
        report += "\nRECOMMENDATIONS\n"
        report += "---------------\n"
        
        for i, rec in enumerate(self.scan_results["recommendations"], 1):
            report += f"{i}. [{rec['priority'].upper()}] {rec['recommendation']}\n"
            report += f"   Action: {rec['action']}\n\n"
        
        return report
    
    def install_security_tools(self) -> None:
        """Install required security tools."""
        logger.info("Installing security tools...")
        
        tools_to_install = [
            ("bandit", "bandit[toml]"),
            ("safety", "safety"),
            ("semgrep", "semgrep"),
        ]
        
        for tool_name, package_name in tools_to_install:
            if not self.security_tools.get(tool_name, False):
                try:
                    logger.info(f"Installing {tool_name}...")
                    subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                                 check=True, capture_output=True)
                    logger.info(f"{tool_name} installed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {tool_name}: {e}")


async def main():
    """Main function to run comprehensive security scan."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Security Scanner for anomaly_detection")
    parser.add_argument("--project-root", help="Project root directory", default=None)
    parser.add_argument("--install-tools", action="store_true", help="Install security tools")
    parser.add_argument("--output-format", choices=["json", "text", "both"], default="both")
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = ComprehensiveSecurityScanner(args.project_root)
    
    # Install tools if requested
    if args.install_tools:
        scanner.install_security_tools()
    
    # Run comprehensive scan
    results = await scanner.run_comprehensive_scan()
    
    # Print summary
    print("\n" + "="*60)
    print("SECURITY SCAN COMPLETED")
    print("="*60)
    print(f"Total Issues Found: {results['summary']['total_issues']}")
    print(f"Scans Completed: {results['summary']['scans_completed']}")
    print(f"Tools Used: {', '.join(results['tools_used'])}")
    
    if results['summary']['total_issues'] > 0:
        print("\n⚠️  SECURITY ISSUES FOUND - Review the detailed report")
        return 1
    else:
        print("\n✅ No security issues found!")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))