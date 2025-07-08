#!/usr/bin/env python3
"""
Container security scanning automation script.
Integrates multiple security tools for comprehensive container analysis.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import argparse
from datetime import datetime

class ContainerSecurityScanner:
    """Comprehensive container security scanner."""
    
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
    
    def scan_vulnerabilities(self) -> Dict[str, any]:
        """Run Trivy vulnerability scan."""
        self.logger.info(f"Scanning vulnerabilities for {self.image_name}")
        
        # Run Trivy scan
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
    
    def scan_secrets(self) -> Dict[str, any]:
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
    
    def scan_misconfigurations(self) -> Dict[str, any]:
        """Scan for container misconfigurations."""
        self.logger.info(f"Scanning misconfigurations for {self.image_name}")
        
        config_output = self.output_dir / "trivy-misconfig.json"
        cmd = [
            "trivy", "image",
            "--scanners", "misconfig",
            "--format", "json",
            "--output", str(config_output),
            self.image_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Misconfiguration scan failed: {result.stderr}")
            return {}
        
        try:
            with open(config_output) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading misconfiguration results: {e}")
            return {}
    
    def scan_dockerfile(self, dockerfile_path: str = None) -> Dict[str, any]:
        """Scan Dockerfile for security issues."""
        if not dockerfile_path:
            # Try to find Dockerfile in common locations
            possible_paths = [
                "Dockerfile",
                "deploy/docker/Dockerfile.hardened",
                "deploy/docker/Dockerfile.api",
                "deploy/docker/Dockerfile"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    dockerfile_path = path
                    break
        
        if not dockerfile_path or not Path(dockerfile_path).exists():
            self.logger.warning("No Dockerfile found for scanning")
            return {}
        
        self.logger.info(f"Scanning Dockerfile: {dockerfile_path}")
        
        dockerfile_output = self.output_dir / "trivy-dockerfile.json"
        cmd = [
            "trivy", "fs",
            "--scanners", "misconfig",
            "--format", "json",
            "--output", str(dockerfile_output),
            dockerfile_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Dockerfile scan failed: {result.stderr}")
            return {}
        
        try:
            with open(dockerfile_output) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading Dockerfile scan results: {e}")
            return {}
    
    def generate_security_report(self, dockerfile_path: str = None) -> str:
        """Generate comprehensive security report."""
        self.logger.info("Generating comprehensive security report")
        
        # Check dependencies first
        if not self.check_dependencies():
            self.logger.error("Missing required dependencies")
            return ""
        
        # Run all scans
        vulns = self.scan_vulnerabilities()
        secrets = self.scan_secrets()
        misconfigs = self.scan_misconfigurations()
        dockerfile_issues = self.scan_dockerfile(dockerfile_path)
        
        # Generate report
        report = {
            "image": self.image_name,
            "scan_timestamp": datetime.utcnow().isoformat() + "Z",
            "scanner_version": self._get_scanner_version(),
            "vulnerabilities": vulns,
            "secrets": secrets,
            "misconfigurations": misconfigs,
            "dockerfile_issues": dockerfile_issues,
            "summary": self._generate_summary(vulns, secrets, misconfigs, dockerfile_issues)
        }
        
        # Save report
        report_file = self.output_dir / f"security-report-{self.image_name.replace(':', '_')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Security report saved to {report_file}")
        return str(report_file)
    
    def _get_scanner_version(self) -> str:
        """Get Trivy scanner version."""
        try:
            result = subprocess.run(
                ["trivy", "--version"],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def _generate_summary(self, vulns: Dict, secrets: Dict, misconfigs: Dict, dockerfile_issues: Dict) -> Dict:
        """Generate security summary."""
        summary = {
            "total_vulnerabilities": 0,
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": 0,
            "low_vulnerabilities": 0,
            "secrets_found": 0,
            "misconfigurations_found": 0,
            "dockerfile_issues_found": 0,
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
        
        # Count misconfigurations
        if misconfigs and "Results" in misconfigs:
            for result in misconfigs["Results"]:
                if "Misconfigurations" in result:
                    summary["misconfigurations_found"] += len(result["Misconfigurations"])
        
        # Count Dockerfile issues
        if dockerfile_issues and "Results" in dockerfile_issues:
            for result in dockerfile_issues["Results"]:
                if "Misconfigurations" in result:
                    summary["dockerfile_issues_found"] += len(result["Misconfigurations"])
        
        # Calculate security score (0-100)
        penalty = (
            summary["critical_vulnerabilities"] * 15 +
            summary["high_vulnerabilities"] * 8 +
            summary["medium_vulnerabilities"] * 3 +
            summary["low_vulnerabilities"] * 1 +
            summary["secrets_found"] * 20 +
            summary["misconfigurations_found"] * 5 +
            summary["dockerfile_issues_found"] * 3
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
    
    def generate_html_report(self, json_report_file: str) -> str:
        """Generate HTML security report."""
        with open(json_report_file) as f:
            report = json.load(f)
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Container Security Report - {image}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; margin-bottom: 20px; }}
                .critical {{ color: #d32f2f; }}
                .high {{ color: #f57c00; }}
                .medium {{ color: #fbc02d; }}
                .low {{ color: #388e3c; }}
                .section {{ margin-bottom: 30px; }}
                .vulnerability {{ border: 1px solid #ddd; padding: 10px; margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Container Security Report</h1>
                <p><strong>Image:</strong> {image}</p>
                <p><strong>Scan Date:</strong> {scan_timestamp}</p>
                <p><strong>Scanner:</strong> {scanner_version}</p>
            </div>
            
            <div class="summary">
                <h2>Security Summary</h2>
                <p><strong>Security Score:</strong> {security_score}/100</p>
                <p><strong>Risk Level:</strong> <span class="{risk_level}">{risk_level}</span></p>
                <p><strong>Total Vulnerabilities:</strong> {total_vulnerabilities}</p>
                <ul>
                    <li class="critical">Critical: {critical_vulnerabilities}</li>
                    <li class="high">High: {high_vulnerabilities}</li>
                    <li class="medium">Medium: {medium_vulnerabilities}</li>
                    <li class="low">Low: {low_vulnerabilities}</li>
                </ul>
                <p><strong>Secrets Found:</strong> {secrets_found}</p>
                <p><strong>Misconfigurations:</strong> {misconfigurations_found}</p>
                <p><strong>Dockerfile Issues:</strong> {dockerfile_issues_found}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {recommendations}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Generate recommendations
        recommendations = []
        summary = report["summary"]
        
        if summary["critical_vulnerabilities"] > 0:
            recommendations.append("<li>🚨 Address critical vulnerabilities immediately</li>")
        if summary["high_vulnerabilities"] > 0:
            recommendations.append("<li>⚠️ Review and fix high-severity vulnerabilities</li>")
        if summary["secrets_found"] > 0:
            recommendations.append("<li>🔐 Remove exposed secrets from the image</li>")
        if summary["misconfigurations_found"] > 0:
            recommendations.append("<li>🔧 Fix container misconfigurations</li>")
        if summary["dockerfile_issues_found"] > 0:
            recommendations.append("<li>📝 Improve Dockerfile security practices</li>")
        
        if not recommendations:
            recommendations.append("<li>✅ No critical security issues found</li>")
        
        html_content = html_template.format(
            image=report["image"],
            scan_timestamp=report["scan_timestamp"],
            scanner_version=report["scanner_version"],
            security_score=summary["security_score"],
            risk_level=summary["risk_level"],
            total_vulnerabilities=summary["total_vulnerabilities"],
            critical_vulnerabilities=summary["critical_vulnerabilities"],
            high_vulnerabilities=summary["high_vulnerabilities"],
            medium_vulnerabilities=summary["medium_vulnerabilities"],
            low_vulnerabilities=summary["low_vulnerabilities"],
            secrets_found=summary["secrets_found"],
            misconfigurations_found=summary["misconfigurations_found"],
            dockerfile_issues_found=summary["dockerfile_issues_found"],
            recommendations="\\n".join(recommendations)
        )
        
        html_file = self.output_dir / f"security-report-{self.image_name.replace(':', '_')}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to {html_file}")
        return str(html_file)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Container security scanner")
    parser.add_argument("image", help="Container image to scan")
    parser.add_argument("--output-dir", default="security-reports", 
                       help="Output directory for reports")
    parser.add_argument("--dockerfile", 
                       help="Path to Dockerfile to scan")
    parser.add_argument("--fail-on-critical", action="store_true",
                       help="Exit with non-zero code if critical vulnerabilities found")
    parser.add_argument("--fail-on-high", action="store_true",
                       help="Exit with non-zero code if high vulnerabilities found")
    parser.add_argument("--min-score", type=int, default=70,
                       help="Minimum security score required (0-100)")
    parser.add_argument("--html", action="store_true",
                       help="Generate HTML report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize scanner
    scanner = ContainerSecurityScanner(args.image, args.output_dir)
    
    # Generate report
    report_file = scanner.generate_security_report(args.dockerfile)
    
    if not report_file:
        print("❌ Failed to generate security report")
        sys.exit(1)
    
    # Load report for exit code determination
    with open(report_file) as f:
        report = json.load(f)
    
    # Generate HTML report if requested
    if args.html:
        scanner.generate_html_report(report_file)
    
    # Print summary
    summary = report["summary"]
    print(f"\\n=== Security Scan Summary for {args.image} ===")
    print(f"Security Score: {summary['security_score']}/100")
    print(f"Risk Level: {summary['risk_level'].upper()}")
    print(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
    print(f"  - Critical: {summary['critical_vulnerabilities']}")
    print(f"  - High: {summary['high_vulnerabilities']}")
    print(f"  - Medium: {summary['medium_vulnerabilities']}")
    print(f"  - Low: {summary['low_vulnerabilities']}")
    print(f"Secrets Found: {summary['secrets_found']}")
    print(f"Misconfigurations: {summary['misconfigurations_found']}")
    print(f"Dockerfile Issues: {summary['dockerfile_issues_found']}")
    print(f"\\nFull report: {report_file}")
    
    # Determine exit code
    exit_code = 0
    
    if args.fail_on_critical and summary['critical_vulnerabilities'] > 0:
        print("\\n❌ Critical vulnerabilities found - failing build")
        exit_code = 1
    elif args.fail_on_high and summary['high_vulnerabilities'] > 0:
        print("\\n❌ High vulnerabilities found - failing build")
        exit_code = 1
    elif summary['secrets_found'] > 0:
        print("\\n❌ Secrets found in image - failing build")
        exit_code = 1
    elif summary['security_score'] < args.min_score:
        print(f"\\n⚠️  Security score {summary['security_score']} below minimum {args.min_score}")
        exit_code = 1
    else:
        print("\\n✅ Security scan passed")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
