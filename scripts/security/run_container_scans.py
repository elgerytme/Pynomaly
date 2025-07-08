#!/usr/bin/env python3
"""
Container security scanning automation script for C-004 implementation.
Integrates multiple security tools for comprehensive container analysis.
"""

import subprocess
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

class ContainerSecurityScanner:
    """Comprehensive container security scanner for C-004."""
    
    def __init__(self, image_name: str, output_dir: str = "security-reports"):
        self.image_name = image_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def check_dependencies(self) -> bool:
        """Check if required tools are installed."""
        required_tools = ["trivy", "docker"]
        missing_tools = []
        
        for tool in required_tools:
            try:
                subprocess.run(
                    [tool, "--version"],
                    capture_output=True,
                    check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            self.logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            return False
        
        return True
    
    def scan_vulnerabilities(self) -> dict:
        """Run Trivy vulnerability scan."""
        self.logger.info(f"Scanning vulnerabilities for {self.image_name}")
        
        trivy_output = self.output_dir / "trivy-vulnerabilities.json"
        trivy_sarif = self.output_dir / "trivy-results.sarif"
        
        # JSON output for processing
        cmd_json = [
            "trivy", "image",
            "--format", "json",
            "--output", str(trivy_output),
            "--severity", "CRITICAL,HIGH,MEDIUM,LOW",
            self.image_name
        ]
        
        # SARIF output for GitHub integration
        cmd_sarif = [
            "trivy", "image",
            "--format", "sarif",
            "--output", str(trivy_sarif),
            "--severity", "CRITICAL,HIGH,MEDIUM",
            self.image_name
        ]
        
        # Run JSON scan
        result = subprocess.run(cmd_json, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Trivy vulnerability scan failed: {result.stderr}")
            return {}
        
        # Run SARIF scan
        subprocess.run(cmd_sarif, capture_output=True, text=True)
        
        # Load and return results
        try:
            with open(trivy_output) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading vulnerability results: {e}")
            return {}
    
    def scan_secrets(self) -> dict:
        """Scan for secrets in container image."""
        self.logger.info(f"Scanning secrets for {self.image_name}")
        
        secrets_output = self.output_dir / "trivy-secrets.json"
        cmd = [
            "trivy", "image",
            "--scanners", "secret",
            "--format", "json",
            "--output", str(secrets_output),
            self.image_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Secret scan failed: {result.stderr}")
            return {}
        
        try:
            with open(secrets_output) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading secret scan results: {e}")
            return {}
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report."""
        self.logger.info("Generating comprehensive security report")
        
        # Check dependencies first
        if not self.check_dependencies():
            self.logger.error("Missing required dependencies")
            return ""
        
        # Run scans
        vulns = self.scan_vulnerabilities()
        secrets = self.scan_secrets()
        
        # Generate report
        report = {
            "image": self.image_name,
            "scan_timestamp": datetime.utcnow().isoformat() + "Z",
            "c004_reference": "Container Security Implementation",
            "vulnerabilities": vulns,
            "secrets": secrets,
            "summary": self._generate_summary(vulns, secrets)
        }
        
        # Save report
        report_file = self.output_dir / f"security-report-{self.image_name.replace(':', '_')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Security report saved to {report_file}")
        return str(report_file)
    
    def _generate_summary(self, vulns: dict, secrets: dict) -> dict:
        """Generate security summary."""
        summary = {
            "total_vulnerabilities": 0,
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": 0,
            "low_vulnerabilities": 0,
            "secrets_found": 0,
            "security_score": 0,
            "risk_level": "unknown"
        }
        
        # Count vulnerabilities
        if vulns and "Results" in vulns:
            for result in vulns["Results"]:
                if "Vulnerabilities" in result:
                    for vuln in result["Vulnerabilities"]:
                        summary["total_vulnerabilities"] += 1
                        severity = vuln.get("Severity", "").upper()
                        if severity == "CRITICAL":
                            summary["critical_vulnerabilities"] += 1
                        elif severity == "HIGH":
                            summary["high_vulnerabilities"] += 1
                        elif severity == "MEDIUM":
                            summary["medium_vulnerabilities"] += 1
                        elif severity == "LOW":
                            summary["low_vulnerabilities"] += 1
        
        # Count secrets
        if secrets and "Results" in secrets:
            for result in secrets["Results"]:
                if "Secrets" in result:
                    summary["secrets_found"] += len(result["Secrets"])
        
        # Calculate security score (0-100)
        penalty = (
            summary["critical_vulnerabilities"] * 15 +
            summary["high_vulnerabilities"] * 8 +
            summary["medium_vulnerabilities"] * 3 +
            summary["low_vulnerabilities"] * 1 +
            summary["secrets_found"] * 20
        )
        
        summary["security_score"] = max(0, 100 - penalty)
        
        # Determine risk level
        if summary["critical_vulnerabilities"] > 0 or summary["secrets_found"] > 0:
            summary["risk_level"] = "critical"
        elif summary["high_vulnerabilities"] > 5 or summary["security_score"] < 50:
            summary["risk_level"] = "high"
        elif summary["medium_vulnerabilities"] > 10 or summary["security_score"] < 70:
            summary["risk_level"] = "medium"
        else:
            summary["risk_level"] = "low"
        
        return summary

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Container security scanner - C-004 implementation")
    parser.add_argument("image", help="Container image to scan")
    parser.add_argument("--output-dir", default="security-reports", 
                       help="Output directory for reports")
    parser.add_argument("--fail-on-critical", action="store_true",
                       help="Exit with non-zero code if critical vulnerabilities found")
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = ContainerSecurityScanner(args.image, args.output_dir)
    
    # Generate report
    report_file = scanner.generate_security_report()
    
    if not report_file:
        print("❌ Failed to generate security report")
        sys.exit(1)
    
    # Load report for exit code determination
    with open(report_file) as f:
        report = json.load(f)
    
    # Print summary
    summary = report["summary"]
    print(f"\\n=== Security Scan Summary for {args.image} ===")
    print(f"C-004 Container Security Implementation")
    print(f"Security Score: {summary['security_score']}/100")
    print(f"Risk Level: {summary['risk_level'].upper()}")
    print(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
    print(f"  - Critical: {summary['critical_vulnerabilities']}")
    print(f"  - High: {summary['high_vulnerabilities']}")
    print(f"  - Medium: {summary['medium_vulnerabilities']}")
    print(f"  - Low: {summary['low_vulnerabilities']}")
    print(f"Secrets Found: {summary['secrets_found']}")
    print(f"\\nFull report: {report_file}")
    
    # Exit with appropriate code
    if args.fail_on_critical and summary['critical_vulnerabilities'] > 0:
        print("\\n❌ Critical vulnerabilities found - failing build")
        sys.exit(1)
    elif summary['security_score'] < 70:
        print("\\n⚠️  Security score below 70 - consider addressing findings")
        sys.exit(1)
    else:
        print("\\n✅ Security scan passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
