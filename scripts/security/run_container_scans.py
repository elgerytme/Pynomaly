#!/usr/bin/env python3
"""
Container Security Scanning Script

This script runs container security scans using Trivy and Clair, generates
SBOMs, and aggregates results into a unified report.

Features:
- Accepts image list via CLI args or manifest file
- Builds local Dockerfiles when needed
- Runs Trivy with SARIF output format
- Runs Clair via clairctl or docker run
- Generates SBOMs using Trivy with CycloneDX format
- Normalizes outputs into security-results/
- Returns combined exit code (fail on HIGH/CRITICAL unless --soft flag)
- Reuses helper functions from run_security_scans.py
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


class ContainerScanner:
    def __init__(self, soft_mode: bool = False):
        self.soft_mode = soft_mode
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "security-results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scan_timestamp = datetime.now().isoformat()
        
        # Get severity threshold from environment variable
        self.severity_threshold = os.environ.get('SEVERITY_THRESHOLD', 'HIGH')
        self.threshold_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
        self.threshold_level = self.threshold_map.get(self.severity_threshold, 2)
        
        self.results = {
            "trivy": {},
            "clair": {},
            "sboms": {},
            "summary": {
                "total_issues": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0,
                "scanned_images": 0,
                "failed_scans": 0
            }
        }
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root,
                timeout=600  # 10 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def is_dockerfile(self, path: str) -> bool:
        """Check if a path points to a Dockerfile"""
        path_obj = Path(path)
        return path_obj.is_file() and (path_obj.name.lower().startswith('dockerfile') or path_obj.suffix.lower() == '.dockerfile')
    
    def build_dockerfile(self, dockerfile_path: Path) -> Tuple[bool, str]:
        """Build a Docker image from Dockerfile"""
        print(f"Building Docker image from Dockerfile: {dockerfile_path}")
        
        # Generate image tag from dockerfile path
        tag = f"container-scan-{dockerfile_path.stem.lower()}"
        
        cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path), "."]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        if exit_code == 0:
            print(f"Successfully built image: {tag}")
            return True, tag
        else:
            print(f"Failed to build image from {dockerfile_path}: {stderr}")
            return False, ""
    
    def run_trivy_scan(self, image: str) -> Dict[str, Any]:
        """Run Trivy vulnerability scan on image"""
        print(f"Running Trivy scan on: {image}")
        
        safe_image_name = image.replace('/', '_').replace(':', '_')
        sarif_file = self.results_dir / f"{safe_image_name}_trivy.sarif"
        json_file = self.results_dir / f"{safe_image_name}_trivy.json"
        
        trivy_results = {
            "image": image,
            "exit_code": 0,
            "sarif_file": str(sarif_file),
            "json_file": str(json_file),
            "vulnerabilities": [],
            "summary": {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        # Run Trivy with SARIF output
        cmd = ["trivy", "image", "--format", "sarif", "--output", str(sarif_file), image]
        exit_code, stdout, stderr = self.run_command(cmd)
        trivy_results["exit_code"] = exit_code
        
        if exit_code != 0:
            print(f"Trivy scan failed for {image}: {stderr}")
            return trivy_results
        
        # Also run with JSON output for easier parsing
        cmd = ["trivy", "image", "--format", "json", "--output", str(json_file), image]
        self.run_command(cmd)
        
        # Parse JSON results for summary
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    trivy_data = json.load(f)
                    trivy_results["vulnerabilities"] = trivy_data
                    
                    # Count vulnerabilities by severity
                    if "Results" in trivy_data:
                        for result in trivy_data["Results"]:
                            if "Vulnerabilities" in result:
                                for vuln in result["Vulnerabilities"]:
                                    severity = vuln.get("Severity", "unknown").lower()
                                    if severity in ["critical", "high", "medium", "low"]:
                                        trivy_results["summary"][severity] += 1
                                        self.results["summary"][severity] += 1
                                    else:
                                        trivy_results["summary"]["low"] += 1
                                        self.results["summary"]["low"] += 1
                                        
                        trivy_results["summary"]["total"] = sum(trivy_results["summary"].values())
                        
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not parse Trivy JSON output: {e}")
        
        return trivy_results
    
    def run_clair_scan(self, image: str) -> Dict[str, Any]:
        """Run Clair vulnerability scan on image"""
        print(f"Running Clair scan on: {image}")
        
        safe_image_name = image.replace('/', '_').replace(':', '_')
        clair_file = self.results_dir / f"{safe_image_name}_clair.json"
        
        clair_results = {
            "image": image,
            "exit_code": 0,
            "output_file": str(clair_file),
            "vulnerabilities": [],
            "summary": {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        # Try clairctl first
        cmd = ["clairctl", "analyze", "--format", "json", "--output", str(clair_file), image]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        if exit_code != 0:
            # Fall back to docker run with clair-local-scan
            print("clairctl not available, trying docker-based Clair scan...")
            cmd = [
                "docker", "run", "--rm", "--network", "host",
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-v", f"{self.results_dir}:/output",
                "arminc/clair-local-scan:latest",
                "--ip", "localhost", "--output", f"/output/{safe_image_name}_clair.json",
                image
            ]
            exit_code, stdout, stderr = self.run_command(cmd)
        
        clair_results["exit_code"] = exit_code
        
        if exit_code != 0:
            print(f"Clair scan failed for {image}: {stderr}")
            return clair_results
        
        # Parse Clair results (format varies by implementation)
        if clair_file.exists():
            try:
                with open(clair_file, 'r') as f:
                    clair_data = json.load(f)
                    clair_results["vulnerabilities"] = clair_data
                    
                    # Basic parsing - Clair output format varies
                    if isinstance(clair_data, dict) and "vulnerabilities" in clair_data:
                        for vuln in clair_data["vulnerabilities"]:
                            severity = self._estimate_clair_severity(vuln)
                            clair_results["summary"][severity] += 1
                            self.results["summary"][severity] += 1
                            
                        clair_results["summary"]["total"] = len(clair_data["vulnerabilities"])
                        
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not parse Clair JSON output: {e}")
        
        return clair_results
    
    def generate_sbom(self, image: str) -> Dict[str, Any]:
        """Generate SBOM for image using Trivy"""
        print(f"Generating SBOM for: {image}")
        
        safe_image_name = image.replace('/', '_').replace(':', '_')
        sbom_file = self.results_dir / f"{safe_image_name}_sbom.json"
        
        sbom_results = {
            "image": image,
            "exit_code": 0,
            "sbom_file": str(sbom_file),
            "sbom_data": None
        }
        
        # Generate SBOM with Trivy
        cmd = ["trivy", "image", "--format", "cyclonedx", "--output", str(sbom_file), image]
        exit_code, stdout, stderr = self.run_command(cmd)
        sbom_results["exit_code"] = exit_code
        
        if exit_code != 0:
            print(f"SBOM generation failed for {image}: {stderr}")
            return sbom_results
        
        # Load SBOM data
        if sbom_file.exists():
            try:
                with open(sbom_file, 'r') as f:
                    sbom_results["sbom_data"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not parse SBOM JSON output: {e}")
        
        return sbom_results
    
    def _estimate_clair_severity(self, vuln: Dict[str, Any]) -> str:
        """Estimate severity for Clair vulnerabilities"""
        # Clair severity mapping (varies by version)
        severity = vuln.get("severity", vuln.get("Severity", "")).lower()
        
        if severity in ["critical", "high", "medium", "low"]:
            return severity
        elif severity in ["defcon1", "critical"]:
            return "critical"
        elif severity in ["high", "important"]:
            return "high"
        elif severity in ["medium", "moderate"]:
            return "medium"
        else:
            return "low"
    
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
    
    def scan_image(self, image_or_dockerfile: str) -> Dict[str, Any]:
        """Scan a single image or build and scan a Dockerfile"""
        image = image_or_dockerfile
        
        # Check if it's a Dockerfile that needs building
        if self.is_dockerfile(image_or_dockerfile):
            dockerfile_path = Path(image_or_dockerfile)
            built, image = self.build_dockerfile(dockerfile_path)
            if not built:
                return {"error": f"Failed to build {dockerfile_path}"}
        
        # Run scans
        trivy_results = self.run_trivy_scan(image)
        clair_results = self.run_clair_scan(image)
        sbom_results = self.generate_sbom(image)
        
        # Store results
        self.results["trivy"][image] = trivy_results
        self.results["clair"][image] = clair_results
        self.results["sboms"][image] = sbom_results
        
        # Update summary
        self.results["summary"]["scanned_images"] += 1
        if trivy_results["exit_code"] != 0 or clair_results["exit_code"] != 0:
            self.results["summary"]["failed_scans"] += 1
        
        return {
            "image": image,
            "trivy": trivy_results,
            "clair": clair_results,
            "sbom": sbom_results
        }
    
    def generate_consolidated_sarif(self) -> Dict[str, Any]:
        """Generate consolidated SARIF output for all scans"""
        sarif_output = {
            "version": "2.1.0",
            "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
            "runs": []
        }
        
        # Create a consolidated run for all container scans
        run = {
            "tool": {
                "driver": {
                    "name": "container-security-scanner",
                    "version": "1.0.0",
                    "informationUri": "https://github.com/your-org/pynomaly"
                }
            },
            "results": []
        }
        
        # Add Trivy results to SARIF
        for image, trivy_data in self.results["trivy"].items():
            if trivy_data["vulnerabilities"] and "Results" in trivy_data["vulnerabilities"]:
                for result in trivy_data["vulnerabilities"]["Results"]:
                    if "Vulnerabilities" in result:
                        for vuln in result["Vulnerabilities"]:
                            sarif_result = {
                                "ruleId": f"trivy-{vuln.get('VulnerabilityID', 'unknown')}",
                                "level": self._severity_to_sarif_level(vuln.get('Severity', 'unknown').lower()),
                                "message": {
                                    "text": vuln.get('Description', f"Vulnerability {vuln.get('VulnerabilityID', 'unknown')} in {image}")
                                },
                                "locations": [{
                                    "physicalLocation": {
                                        "artifactLocation": {
                                            "uri": f"docker://{image}"
                                        }
                                    }
                                }],
                                "properties": {
                                    "image": image,
                                    "package": vuln.get('PkgName', 'unknown'),
                                    "installedVersion": vuln.get('InstalledVersion', 'unknown'),
                                    "fixedVersion": vuln.get('FixedVersion', 'unknown')
                                }
                            }
                            run["results"].append(sarif_result)
        
        # Add Clair results to SARIF
        for image, clair_data in self.results["clair"].items():
            if clair_data["vulnerabilities"] and isinstance(clair_data["vulnerabilities"], dict):
                if "vulnerabilities" in clair_data["vulnerabilities"]:
                    for vuln in clair_data["vulnerabilities"]["vulnerabilities"]:
                        sarif_result = {
                            "ruleId": f"clair-{vuln.get('Name', 'unknown')}",
                            "level": self._severity_to_sarif_level(self._estimate_clair_severity(vuln)),
                            "message": {
                                "text": vuln.get('Description', f"Vulnerability {vuln.get('Name', 'unknown')} in {image}")
                            },
                            "locations": [{
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": f"docker://{image}"
                                    }
                                }
                            }],
                            "properties": {
                                "image": image,
                                "severity": vuln.get('Severity', 'unknown')
                            }
                        }
                        run["results"].append(sarif_result)
        
        if run["results"]:
            sarif_output["runs"].append(run)
            
        return sarif_output
    
    def generate_summary_report(self) -> None:
        """Generate container security summary report"""
        summary_file = self.results_dir / "container_security_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# Container Security Scan Summary\n\n")
            f.write(f"**Scan Date**: {self.scan_timestamp}\n")
            f.write(f"**Project**: Pynomaly\n\n")
            
            # Overall summary
            f.write("## Overall Summary\n\n")
            f.write("| Metric | Count |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Scanned Images | {self.results['summary']['scanned_images']} |\n")
            f.write(f"| Failed Scans | {self.results['summary']['failed_scans']} |\n")
            f.write(f"| Total Issues | {self.results['summary']['total_issues']} |\n\n")
            
            f.write("## Vulnerability Summary\n\n")
            f.write("| Severity | Count |\n")
            f.write("|----------|-------|\n")
            f.write(f"| Critical | {self.results['summary']['critical']} |\n")
            f.write(f"| High     | {self.results['summary']['high']} |\n")
            f.write(f"| Medium   | {self.results['summary']['medium']} |\n")
            f.write(f"| Low      | {self.results['summary']['low']} |\n")
            f.write(f"| Info     | {self.results['summary']['info']} |\n\n")
            
            # Per-image results
            f.write("## Per-Image Results\n\n")
            for image in self.results["trivy"].keys():
                f.write(f"### {image}\n\n")
                
                # Trivy results
                trivy_data = self.results["trivy"][image]
                f.write(f"**Trivy Scan**: {trivy_data['summary']['total']} issues\n")
                f.write(f"- Critical: {trivy_data['summary']['critical']}\n")
                f.write(f"- High: {trivy_data['summary']['high']}\n")
                f.write(f"- Medium: {trivy_data['summary']['medium']}\n")
                f.write(f"- Low: {trivy_data['summary']['low']}\n\n")
                
                # Clair results
                clair_data = self.results["clair"][image]
                f.write(f"**Clair Scan**: {clair_data['summary']['total']} issues\n")
                f.write(f"- Critical: {clair_data['summary']['critical']}\n")
                f.write(f"- High: {clair_data['summary']['high']}\n")
                f.write(f"- Medium: {clair_data['summary']['medium']}\n")
                f.write(f"- Low: {clair_data['summary']['low']}\n\n")
                
                # SBOM info
                sbom_data = self.results["sboms"][image]
                f.write(f"**SBOM Generated**: {'Yes' if sbom_data['exit_code'] == 0 else 'No'}\n\n")
            
            # Files generated
            f.write("## Files Generated\n\n")
            f.write("### Trivy Results\n")
            for image in self.results["trivy"].keys():
                safe_name = image.replace('/', '_').replace(':', '_')
                f.write(f"- `{safe_name}_trivy.sarif` - Trivy SARIF results\n")
                f.write(f"- `{safe_name}_trivy.json` - Trivy JSON results\n")
            
            f.write("\n### Clair Results\n")
            for image in self.results["clair"].keys():
                safe_name = image.replace('/', '_').replace(':', '_')
                f.write(f"- `{safe_name}_clair.json` - Clair JSON results\n")
            
            f.write("\n### SBOMs\n")
            for image in self.results["sboms"].keys():
                safe_name = image.replace('/', '_').replace(':', '_')
                f.write(f"- `{safe_name}_sbom.json` - CycloneDX SBOM\n")
            
            f.write("\n### Consolidated Reports\n")
            f.write("- `container_security_summary.md` - This summary report\n")
            f.write("- `container_consolidated_sarif.json` - Consolidated SARIF for GitHub\n")
            f.write("- `container_consolidated_results.json` - Complete results JSON\n")
        
        print(f"Container security summary report generated: {summary_file}")
    
    def run_all_scans(self, images: List[str]) -> int:
        """Run all container scans and aggregate results"""
        print("Starting container security scans...")
        
        if not images:
            print("No images or Dockerfiles provided")
            return 0
        
        exit_code = 0
        
        for image_or_dockerfile in images:
            print(f"\nProcessing: {image_or_dockerfile}")
            results = self.scan_image(image_or_dockerfile)
            
            # Check for errors
            if "error" in results:
                print(f"Error: {results['error']}")
                continue
            
            # Check for findings that exceed severity threshold
            if not self.soft_mode:
                trivy_summary = results["trivy"]["summary"]
                clair_summary = results["clair"]["summary"]
                
                # Check if any findings exceed the threshold
                threshold_exceeded = False
                if (self.threshold_level <= 3 and (trivy_summary["critical"] > 0 or clair_summary["critical"] > 0)) or \
                   (self.threshold_level <= 2 and (trivy_summary["high"] > 0 or clair_summary["high"] > 0)) or \
                   (self.threshold_level <= 1 and (trivy_summary["medium"] > 0 or clair_summary["medium"] > 0)) or \
                   (self.threshold_level <= 0 and (trivy_summary["low"] > 0 or clair_summary["low"] > 0)):
                    threshold_exceeded = True
                
                if threshold_exceeded:
                    print(f"Findings exceed severity threshold ({self.severity_threshold}) for image: {image_or_dockerfile}")
                    exit_code = 1
        
        # Calculate total issues
        self.results["summary"]["total_issues"] = sum([
            self.results["summary"]["critical"],
            self.results["summary"]["high"],
            self.results["summary"]["medium"],
            self.results["summary"]["low"],
            self.results["summary"]["info"]
        ])
        
        # Generate consolidated outputs
        sarif_output = self.generate_consolidated_sarif()
        sarif_file = self.results_dir / "container_consolidated_sarif.json"
        with open(sarif_file, 'w') as f:
            json.dump(sarif_output, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save consolidated results
        results_file = self.results_dir / "container_consolidated_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nContainer security scan completed!")
        print(f"Severity threshold: {self.severity_threshold}")
        print(f"Total issues found: {self.results['summary']['total_issues']}")
        print(f"Critical: {self.results['summary']['critical']}")
        print(f"High: {self.results['summary']['high']}")
        print(f"Medium: {self.results['summary']['medium']}")
        print(f"Low: {self.results['summary']['low']}")
        print(f"Results saved to: {self.results_dir}")
        
        if not self.soft_mode and exit_code == 1:
            print(f"\n⚠️  Findings exceed severity threshold ({self.severity_threshold}). Exiting with non-zero code.")
        
        return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Run container security scans with Trivy and Clair",
        epilog="Uses SEVERITY_THRESHOLD environment variable (LOW, MEDIUM, HIGH, CRITICAL) to determine failure threshold. Defaults to HIGH."
    )
    parser.add_argument("--images", nargs='+', help="List of images or Dockerfiles to scan")
    parser.add_argument("--manifest", type=Path, help="Manifest file containing image or Dockerfile list")
    parser.add_argument("--soft", action="store_true", help="Don't exit with non-zero code on findings above threshold")

    args = parser.parse_args()

    # Retrieve the images list
    images = args.images or []
    if args.manifest and args.manifest.exists():
        with open(args.manifest) as f:
            images.extend(line.strip() for line in f if line.strip())

    if not images:
        print("No images or Dockerfiles provided. Use --images or --manifest to specify targets.")
        sys.exit(1)

    # Initialize scanner
    scanner = ContainerScanner(soft_mode=args.soft)

    # Run all scans
    exit_code = scanner.run_all_scans(images)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()

