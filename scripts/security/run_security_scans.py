#!/usr/bin/env python3
"""
Comprehensive Security Scanning Script

This script runs multiple security tools (safety, bandit, pip-audit) and aggregates
results into a unified report with GitHub Actions integration.

Features:
- Auto-detects CI environment and activates Hatch environment when needed
- Runs safety with full reporting
- Runs bandit with JSON and text outputs
- Runs pip-audit with CycloneDX and JSON formats
- Aggregates results into artifacts/security/
- Generates security_summary.md with severity counts and tables
- Produces JSON and SARIF outputs for GitHub upload
- Configurable exit codes based on findings severity
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import xml.etree.ElementTree as ET


class SecurityScanner:
    def __init__(self, soft_mode: bool = False):
        self.soft_mode = soft_mode
        self.project_root = Path(__file__).parent.parent.parent
        self.artifacts_dir = self.project_root / "artifacts" / "security"
        self.results = {
            "safety": None,
            "bandit": None,
            "pip_audit": None,
            "summary": {
                "total_issues": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0
            }
        }
        self.scan_timestamp = datetime.now().isoformat()
        
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def is_ci_environment(self) -> bool:
        """Check if running in CI environment"""
        ci_indicators = [
            "CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", 
            "GITLAB_CI", "JENKINS_URL", "TRAVIS", "CIRCLECI",
            "BUILDKITE", "AZURE_PIPELINES_BUILD_ID"
        ]
        return any(os.getenv(indicator) for indicator in ci_indicators)
    
    def activate_hatch_env(self) -> Optional[str]:
        """Activate Hatch environment if not in CI"""
        if self.is_ci_environment():
            print("CI environment detected, skipping Hatch environment activation")
            return None
            
        try:
            # Check if hatch is available
            result = subprocess.run(
                ["hatch", "env", "show"], 
                capture_output=True, 
                text=True, 
                cwd=self.project_root
            )
            if result.returncode != 0:
                print("Hatch not available or no environment configured")
                return None
                
            # Get the Python path from hatch environment
            result = subprocess.run(
                ["hatch", "env", "find"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                env_path = result.stdout.strip()
                python_path = Path(env_path) / "bin" / "python"
                if not python_path.exists():
                    python_path = Path(env_path) / "Scripts" / "python.exe"  # Windows
                    
                if python_path.exists():
                    print(f"Using Hatch environment: {env_path}")
                    return str(python_path)
                    
        except FileNotFoundError:
            print("Hatch not found in PATH")
            
        return None
    
    def run_command(self, cmd: List[str], python_path: Optional[str] = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        if python_path and cmd[0] == "python":
            cmd[0] = python_path
        elif python_path and cmd[0] in ["safety", "bandit", "pip-audit"]:
            cmd = [python_path, "-m"] + cmd
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600  # 10 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def run_safety_scan(self, python_path: Optional[str] = None) -> Dict[str, Any]:
        """Run safety vulnerability scan"""
        print("Running safety scan...")
        
        # Run safety with full report
        cmd = ["safety", "check", "--full-report", "--json"]
        exit_code, stdout, stderr = self.run_command(cmd, python_path)
        
        safety_results = {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "vulnerabilities": [],
            "summary": {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        # Parse JSON output if available
        if stdout:
            try:
                safety_data = json.loads(stdout)
                if isinstance(safety_data, list):
                    safety_results["vulnerabilities"] = safety_data
                    safety_results["summary"]["total"] = len(safety_data)
                    
                    # Count by severity (safety doesn't always provide severity)
                    for vuln in safety_data:
                        # Estimate severity based on CVE score or other indicators
                        severity = self._estimate_safety_severity(vuln)
                        safety_results["summary"][severity] += 1
                        self.results["summary"][severity] += 1
                        
            except json.JSONDecodeError:
                print("Warning: Could not parse safety JSON output")
        
        # Save results
        safety_file = self.artifacts_dir / "safety_results.json"
        with open(safety_file, 'w') as f:
            json.dump(safety_results, f, indent=2)
            
        # Also save raw output
        raw_file = self.artifacts_dir / "safety_raw.txt"
        with open(raw_file, 'w') as f:
            f.write(f"Exit Code: {exit_code}\n")
            f.write(f"STDOUT:\n{stdout}\n")
            f.write(f"STDERR:\n{stderr}\n")
            
        return safety_results
    
    def run_bandit_scan(self, python_path: Optional[str] = None) -> Dict[str, Any]:
        """Run bandit security scan"""
        print("Running bandit scan...")
        
        bandit_results = {
            "exit_code": 0,
            "json_output": None,
            "text_output": "",
            "sarif_output": None,
            "summary": {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        # Run bandit with JSON output
        json_file = self.artifacts_dir / "bandit_results.json"
        cmd = ["bandit", "-r", "src/", "-f", "json", "-o", str(json_file)]
        exit_code, stdout, stderr = self.run_command(cmd, python_path)
        bandit_results["exit_code"] = exit_code
        
        # Run bandit with text output
        text_file = self.artifacts_dir / "bandit_results.txt"
        cmd = ["bandit", "-r", "src/", "-f", "txt", "-o", str(text_file)]
        self.run_command(cmd, python_path)
        
        # Run bandit with SARIF output for GitHub
        sarif_file = self.artifacts_dir / "bandit_results.sarif"
        cmd = ["bandit", "-r", "src/", "-f", "sarif", "-o", str(sarif_file)]
        self.run_command(cmd, python_path)
        
        # Parse JSON results
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    bandit_data = json.load(f)
                    bandit_results["json_output"] = bandit_data
                    
                    # Count issues by severity
                    if "results" in bandit_data:
                        for issue in bandit_data["results"]:
                            severity = issue.get("issue_severity", "").lower()
                            if severity in ["critical", "high", "medium", "low"]:
                                bandit_results["summary"][severity] += 1
                                self.results["summary"][severity] += 1
                            else:
                                bandit_results["summary"]["low"] += 1
                                self.results["summary"]["low"] += 1
                                
                        bandit_results["summary"]["total"] = len(bandit_data["results"])
                        
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not parse bandit JSON output: {e}")
        
        # Read text output
        if text_file.exists():
            try:
                with open(text_file, 'r') as f:
                    bandit_results["text_output"] = f.read()
            except IOError:
                pass
                
        # Read SARIF output
        if sarif_file.exists():
            try:
                with open(sarif_file, 'r') as f:
                    bandit_results["sarif_output"] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return bandit_results
    
    def run_pip_audit_scan(self, python_path: Optional[str] = None) -> Dict[str, Any]:
        """Run pip-audit dependency scan"""
        print("Running pip-audit scan...")
        
        pip_audit_results = {
            "exit_code": 0,
            "json_output": None,
            "cyclonedx_output": None,
            "vulnerabilities": [],
            "summary": {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        # Run pip-audit with JSON output
        json_file = self.artifacts_dir / "pip_audit_results.json"
        cmd = ["pip-audit", "--format", "json", "--output", str(json_file)]
        exit_code, stdout, stderr = self.run_command(cmd, python_path)
        pip_audit_results["exit_code"] = exit_code
        
        # Run pip-audit with CycloneDX output
        cyclonedx_file = self.artifacts_dir / "pip_audit_cyclonedx.json"
        cmd = ["pip-audit", "--format", "cyclonedx", "--output", str(cyclonedx_file)]
        self.run_command(cmd, python_path)
        
        # Parse JSON results
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    pip_audit_data = json.load(f)
                    pip_audit_results["json_output"] = pip_audit_data
                    
                    # Extract vulnerabilities
                    if "vulnerabilities" in pip_audit_data:
                        pip_audit_results["vulnerabilities"] = pip_audit_data["vulnerabilities"]
                        
                        # Count by severity
                        for vuln in pip_audit_data["vulnerabilities"]:
                            severity = self._estimate_pip_audit_severity(vuln)
                            pip_audit_results["summary"][severity] += 1
                            self.results["summary"][severity] += 1
                            
                        pip_audit_results["summary"]["total"] = len(pip_audit_data["vulnerabilities"])
                        
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not parse pip-audit JSON output: {e}")
        
        # Read CycloneDX output
        if cyclonedx_file.exists():
            try:
                with open(cyclonedx_file, 'r') as f:
                    pip_audit_results["cyclonedx_output"] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return pip_audit_results
    
    def _estimate_safety_severity(self, vuln: Dict[str, Any]) -> str:
        """Estimate severity for safety vulnerabilities"""
        # Check if CVE score is available
        if "cve" in vuln:
            cve_score = vuln.get("cve", {}).get("score", 0)
            if cve_score >= 9.0:
                return "critical"
            elif cve_score >= 7.0:
                return "high"
            elif cve_score >= 4.0:
                return "medium"
            else:
                return "low"
        
        # Fall back to keywords in description
        description = vuln.get("advisory", "").lower()
        if any(word in description for word in ["critical", "remote code execution", "rce"]):
            return "critical"
        elif any(word in description for word in ["high", "privilege escalation", "sql injection"]):
            return "high"
        elif any(word in description for word in ["medium", "denial of service", "dos"]):
            return "medium"
        else:
            return "low"
    
    def _estimate_pip_audit_severity(self, vuln: Dict[str, Any]) -> str:
        """Estimate severity for pip-audit vulnerabilities"""
        # Check for severity in aliases (CVE data)
        for alias in vuln.get("aliases", []):
            if alias.startswith("CVE-"):
                # This is a simplified estimation - in practice, you'd want to query CVE databases
                return "medium"  # Default for CVEs
        
        # Check description for severity keywords
        description = vuln.get("description", "").lower()
        if any(word in description for word in ["critical", "remote code execution"]):
            return "critical"
        elif any(word in description for word in ["high", "privilege escalation"]):
            return "high"
        elif any(word in description for word in ["medium", "denial of service"]):
            return "medium"
        else:
            return "low"
    
    def generate_sarif_output(self) -> Dict[str, Any]:
        """Generate consolidated SARIF output for GitHub"""
        sarif_output = {
            "version": "2.1.0",
            "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
            "runs": []
        }
        
        # Add bandit SARIF if available
        if self.results["bandit"] and self.results["bandit"].get("sarif_output"):
            sarif_output["runs"].extend(self.results["bandit"]["sarif_output"]["runs"])
        
        # Convert other tools to SARIF format
        run = {
            "tool": {
                "driver": {
                    "name": "security-scan-suite",
                    "version": "1.0.0",
                    "informationUri": "https://github.com/your-org/pynomaly"
                }
            },
            "results": []
        }
        
        # Add safety results to SARIF
        if self.results["safety"] and self.results["safety"]["vulnerabilities"]:
            for vuln in self.results["safety"]["vulnerabilities"]:
                result = {
                    "ruleId": f"safety-{vuln.get('id', 'unknown')}",
                    "level": self._severity_to_sarif_level(self._estimate_safety_severity(vuln)),
                    "message": {
                        "text": vuln.get("advisory", "Security vulnerability detected")
                    },
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": "requirements.txt"
                            }
                        }
                    }]
                }
                run["results"].append(result)
        
        # Add pip-audit results to SARIF
        if self.results["pip_audit"] and self.results["pip_audit"]["vulnerabilities"]:
            for vuln in self.results["pip_audit"]["vulnerabilities"]:
                result = {
                    "ruleId": f"pip-audit-{vuln.get('id', 'unknown')}",
                    "level": self._severity_to_sarif_level(self._estimate_pip_audit_severity(vuln)),
                    "message": {
                        "text": vuln.get("description", "Dependency vulnerability detected")
                    },
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": "requirements.txt"
                            }
                        }
                    }]
                }
                run["results"].append(result)
        
        if run["results"]:
            sarif_output["runs"].append(run)
            
        return sarif_output
    
    def _severity_to_sarif_level(self, severity: str) -> str:
        """Convert severity to SARIF level"""
        mapping = {
            "critical": "error",
            "high": "error", 
            "medium": "warning",
            "low": "note",
            "info": "note"
        }
        return mapping.get(severity, "note")
    
    def generate_summary_report(self) -> None:
        """Generate security_summary.md report"""
        summary_file = self.artifacts_dir / "security_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# Security Scan Summary\n\n")
            f.write(f"**Scan Date**: {self.scan_timestamp}\n")
            f.write(f"**Project**: Pynomaly\n\n")
            
            # Overall summary
            f.write("## Overall Summary\n\n")
            f.write("| Severity | Count |\n")
            f.write("|----------|-------|\n")
            f.write(f"| Critical | {self.results['summary']['critical']} |\n")
            f.write(f"| High     | {self.results['summary']['high']} |\n")
            f.write(f"| Medium   | {self.results['summary']['medium']} |\n")
            f.write(f"| Low      | {self.results['summary']['low']} |\n")
            f.write(f"| Info     | {self.results['summary']['info']} |\n")
            f.write(f"| **Total** | **{self.results['summary']['total_issues']}** |\n\n")
            
            # Tool-specific summaries
            f.write("## Tool-Specific Results\n\n")
            
            # Safety results
            if self.results["safety"]:
                f.write("### Safety (Dependency Vulnerabilities)\n\n")
                safety_summary = self.results["safety"]["summary"]
                f.write(f"- **Total Issues**: {safety_summary['total']}\n")
                f.write(f"- **Exit Code**: {self.results['safety']['exit_code']}\n")
                if safety_summary['total'] > 0:
                    f.write(f"- **Critical**: {safety_summary['critical']}\n")
                    f.write(f"- **High**: {safety_summary['high']}\n")
                    f.write(f"- **Medium**: {safety_summary['medium']}\n")
                    f.write(f"- **Low**: {safety_summary['low']}\n")
                f.write("\n")
            
            # Bandit results
            if self.results["bandit"]:
                f.write("### Bandit (Source Code Analysis)\n\n")
                bandit_summary = self.results["bandit"]["summary"]
                f.write(f"- **Total Issues**: {bandit_summary['total']}\n")
                f.write(f"- **Exit Code**: {self.results['bandit']['exit_code']}\n")
                if bandit_summary['total'] > 0:
                    f.write(f"- **Critical**: {bandit_summary['critical']}\n")
                    f.write(f"- **High**: {bandit_summary['high']}\n")
                    f.write(f"- **Medium**: {bandit_summary['medium']}\n")
                    f.write(f"- **Low**: {bandit_summary['low']}\n")
                f.write("\n")
            
            # Pip-audit results
            if self.results["pip_audit"]:
                f.write("### Pip-Audit (Dependency Vulnerabilities)\n\n")
                pip_audit_summary = self.results["pip_audit"]["summary"]
                f.write(f"- **Total Issues**: {pip_audit_summary['total']}\n")
                f.write(f"- **Exit Code**: {self.results['pip_audit']['exit_code']}\n")
                if pip_audit_summary['total'] > 0:
                    f.write(f"- **Critical**: {pip_audit_summary['critical']}\n")
                    f.write(f"- **High**: {pip_audit_summary['high']}\n")
                    f.write(f"- **Medium**: {pip_audit_summary['medium']}\n")
                    f.write(f"- **Low**: {pip_audit_summary['low']}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if self.results['summary']['critical'] > 0:
                f.write("⚠️ **CRITICAL**: Immediate action required for critical vulnerabilities.\n")
            if self.results['summary']['high'] > 0:
                f.write("🔴 **HIGH**: High-priority vulnerabilities should be addressed soon.\n")
            if self.results['summary']['medium'] > 0:
                f.write("🟡 **MEDIUM**: Medium-priority vulnerabilities should be reviewed.\n")
            if self.results['summary']['low'] > 0:
                f.write("🟢 **LOW**: Low-priority vulnerabilities can be addressed in routine maintenance.\n")
            
            if self.results['summary']['total_issues'] == 0:
                f.write("✅ **No security issues found!**\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `safety_results.json` - Safety scan results\n")
            f.write("- `safety_raw.txt` - Raw safety output\n")
            f.write("- `bandit_results.json` - Bandit JSON results\n")
            f.write("- `bandit_results.txt` - Bandit text results\n")
            f.write("- `bandit_results.sarif` - Bandit SARIF results\n")
            f.write("- `pip_audit_results.json` - Pip-audit JSON results\n")
            f.write("- `pip_audit_cyclonedx.json` - Pip-audit CycloneDX results\n")
            f.write("- `consolidated_sarif.json` - Consolidated SARIF for GitHub\n")
            f.write("- `security_summary.md` - This summary report\n")
        
        print(f"Security summary report generated: {summary_file}")
    
    def run_all_scans(self) -> int:
        """Run all security scans and aggregate results"""
        print("Starting comprehensive security scan...")
        
        # Get Python path from Hatch environment
        python_path = self.activate_hatch_env()
        
        # Run all scans
        self.results["safety"] = self.run_safety_scan(python_path)
        self.results["bandit"] = self.run_bandit_scan(python_path)
        self.results["pip_audit"] = self.run_pip_audit_scan(python_path)
        
        # Calculate total issues
        self.results["summary"]["total_issues"] = sum([
            self.results["summary"]["critical"],
            self.results["summary"]["high"],
            self.results["summary"]["medium"],
            self.results["summary"]["low"],
            self.results["summary"]["info"]
        ])
        
        # Generate consolidated outputs
        sarif_output = self.generate_sarif_output()
        sarif_file = self.artifacts_dir / "consolidated_sarif.json"
        with open(sarif_file, 'w') as f:
            json.dump(sarif_output, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save consolidated results
        results_file = self.artifacts_dir / "consolidated_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nSecurity scan completed!")
        print(f"Total issues found: {self.results['summary']['total_issues']}")
        print(f"Critical: {self.results['summary']['critical']}")
        print(f"High: {self.results['summary']['high']}")
        print(f"Medium: {self.results['summary']['medium']}")
        print(f"Low: {self.results['summary']['low']}")
        print(f"Results saved to: {self.artifacts_dir}")
        
        # Determine exit code
        if not self.soft_mode:
            if self.results["summary"]["critical"] > 0 or self.results["summary"]["high"] > 0:
                print("\n⚠️  HIGH/CRITICAL findings detected. Exiting with non-zero code.")
                return 1
        
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive security scans for Pynomaly project"
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Don't exit with non-zero code on HIGH/CRITICAL findings"
    )
    
    args = parser.parse_args()
    
    scanner = SecurityScanner(soft_mode=args.soft)
    exit_code = scanner.run_all_scans()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
