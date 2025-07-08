#!/usr/bin/env python3
"""
Container Security Scanning Script

This script runs container vulnerability scans using Trivy to generate SARIF and SBOM files.
It supports both regular and soft scan modes for different use cases.

Usage:
    python run_container_scans.py [OPTIONS]

Options:
    --soft          Run in soft mode (ignores scan failures)
    --image IMAGE   Docker image to scan (default: pynomaly:latest)
    --output-dir    Directory to save scan results (default: ./security-reports)
    --help         Show this help message and exit

Examples:
    python run_container_scans.py
    python run_container_scans.py --soft
    python run_container_scans.py --image myapp:v1.0.0
    python run_container_scans.py --soft --output-dir /tmp/scan-results
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ContainerScanner:
    """Container security scanner using Trivy."""
    
    def __init__(self, image: str, output_dir: str = "./security-reports", soft_mode: bool = False):
        """
        Initialize the container scanner.
        
        Args:
            image: Docker image to scan
            output_dir: Directory to save scan results
            soft_mode: If True, ignore scan failures
        """
        self.image = image
        self.output_dir = Path(output_dir)
        self.soft_mode = soft_mode
        self.logger = self._setup_logging()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("container_scanner")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _check_prerequisites(self) -> bool:
        """Check if required tools are available."""
        required_tools = ["docker", "trivy"]
        missing_tools = []
        
        for tool in required_tools:
            try:
                result = subprocess.run(
                    [tool, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    missing_tools.append(tool)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            self.logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            return False
        
        return True
    
    def _check_docker_image(self) -> bool:
        """Check if the Docker image exists."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout checking Docker image: {self.image}")
            return False
    
    def _run_trivy_scan(self, format_type: str, output_file: str) -> Tuple[bool, str]:
        """
        Run Trivy scan with specified format.
        
        Args:
            format_type: Output format (sarif, cyclonedx, json, etc.)
            output_file: Output file path
            
        Returns:
            Tuple of (success, error_message)
        """
        output_path = self.output_dir / output_file
        
        # Base Trivy command
        cmd = [
            "trivy",
            "image",
            "--format", format_type,
            "--output", str(output_path),
            self.image
        ]
        
        # Add security-specific scanners
        if format_type == "sarif":
            cmd.extend(["--scanners", "vuln,config,secret"])
        
        try:
            self.logger.info(f"Running Trivy scan: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Successfully generated {format_type} report: {output_path}")
                return True, ""
            else:
                error_msg = f"Trivy scan failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f":\n{result.stderr}"
                
                if self.soft_mode:
                    self.logger.warning(f"⚠️  {error_msg} (continuing in soft mode)")
                    # Create empty file to indicate scan was attempted
                    output_path.touch()
                    return True, error_msg
                else:
                    self.logger.error(f"❌ {error_msg}")
                    return False, error_msg
        
        except subprocess.TimeoutExpired:
            error_msg = f"Trivy scan timed out after 5 minutes"
            if self.soft_mode:
                self.logger.warning(f"⚠️  {error_msg} (continuing in soft mode)")
                output_path.touch()
                return True, error_msg
            else:
                self.logger.error(f"❌ {error_msg}")
                return False, error_msg
        
        except Exception as e:
            error_msg = f"Unexpected error during Trivy scan: {str(e)}"
            if self.soft_mode:
                self.logger.warning(f"⚠️  {error_msg} (continuing in soft mode)")
                output_path.touch()
                return True, error_msg
            else:
                self.logger.error(f"❌ {error_msg}")
                return False, error_msg
    
    def generate_sarif_report(self) -> bool:
        """Generate SARIF (Static Analysis Results Interchange Format) report."""
        self.logger.info("Generating SARIF report...")
        success, error = self._run_trivy_scan("sarif", "container-vulnerabilities.sarif")
        return success
    
    def generate_sbom_report(self) -> bool:
        """Generate SBOM (Software Bill of Materials) report."""
        self.logger.info("Generating SBOM report...")
        success, error = self._run_trivy_scan("cyclonedx", "container-sbom.json")
        return success
    
    def generate_summary_report(self) -> bool:
        """Generate a human-readable summary report."""
        self.logger.info("Generating summary report...")
        success, error = self._run_trivy_scan("table", "container-summary.txt")
        return success
    
    def run_all_scans(self) -> Dict[str, bool]:
        """
        Run all container scans.
        
        Returns:
            Dictionary with scan results
        """
        self.logger.info(f"🔍 Starting container security scan for: {self.image}")
        
        if not self._check_prerequisites():
            return {"prerequisites": False}
        
        if not self._check_docker_image():
            error_msg = f"Docker image '{self.image}' not found locally"
            if self.soft_mode:
                self.logger.warning(f"⚠️  {error_msg} (continuing in soft mode)")
                # Create empty report files
                (self.output_dir / "container-vulnerabilities.sarif").touch()
                (self.output_dir / "container-sbom.json").touch()
                (self.output_dir / "container-summary.txt").touch()
                return {
                    "prerequisites": True,
                    "image_found": False,
                    "sarif": True,
                    "sbom": True,
                    "summary": True
                }
            else:
                self.logger.error(f"❌ {error_msg}")
                return {"prerequisites": True, "image_found": False}
        
        # Run scans
        results = {
            "prerequisites": True,
            "image_found": True,
            "sarif": self.generate_sarif_report(),
            "sbom": self.generate_sbom_report(),
            "summary": self.generate_summary_report()
        }
        
        # Summary
        if all(results.values()):
            self.logger.info("✅ All container scans completed successfully!")
            self.logger.info(f"📄 Reports saved to: {self.output_dir}")
        else:
            failed_scans = [k for k, v in results.items() if not v]
            if self.soft_mode:
                self.logger.warning(f"⚠️  Some scans failed but continuing in soft mode: {failed_scans}")
            else:
                self.logger.error(f"❌ Failed scans: {failed_scans}")
        
        return results
    
    def list_report_files(self) -> List[str]:
        """List generated report files."""
        if not self.output_dir.exists():
            return []
        
        report_files = []
        for pattern in ["*.sarif", "*.json", "*.txt"]:
            report_files.extend(self.output_dir.glob(pattern))
        
        return [str(f) for f in sorted(report_files)]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Container Security Scanner using Trivy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Scan pynomaly:latest
  %(prog)s --soft                             # Scan in soft mode
  %(prog)s --image myapp:v1.0.0               # Scan custom image
  %(prog)s --soft --output-dir /tmp/reports   # Custom output directory
        """
    )
    
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Run in soft mode (ignore scan failures)"
    )
    
    parser.add_argument(
        "--image",
        default="pynomaly:latest",
        help="Docker image to scan (default: pynomaly:latest)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./security-reports",
        help="Directory to save scan results (default: ./security-reports)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger("container_scanner").setLevel(logging.DEBUG)
    
    # Create scanner and run scans
    scanner = ContainerScanner(
        image=args.image,
        output_dir=args.output_dir,
        soft_mode=args.soft
    )
    
    results = scanner.run_all_scans()
    
    # Print report files
    report_files = scanner.list_report_files()
    if report_files:
        print(f"\n📄 Generated report files:")
        for file in report_files:
            print(f"  - {file}")
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        if args.soft:
            print("\n⚠️  Some scans failed but exiting successfully due to soft mode")
            sys.exit(0)
        else:
            print("\n❌ Some scans failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
