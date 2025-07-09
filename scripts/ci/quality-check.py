#!/usr/bin/env python3
"""
Quality Check Script for Pynomaly CI/CD Pipeline.
This script runs comprehensive code quality checks including linting, formatting, and type checking.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class QualityCheck:
    """Quality check result container."""
    
    def __init__(self, name: str, passed: bool, duration: float, output: str = "", error: str = ""):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.output = output
        self.error = error
        self.timestamp = datetime.now()


class QualityChecker:
    """Comprehensive quality checker for Pynomaly."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[QualityCheck] = []
        self.src_path = project_root / "src"
        self.tests_path = project_root / "tests"
        
        # Quality check configurations
        self.checks = {
            "ruff_lint": {
                "description": "Ruff linting checks",
                "command": ["ruff", "check", "src/", "tests/", "--output-format=json"],
                "timeout": 300,
                "required": True,
            },
            "ruff_format": {
                "description": "Ruff formatting checks",
                "command": ["ruff", "format", "src/", "tests/", "--check", "--diff"],
                "timeout": 300,
                "required": True,
            },
            "mypy": {
                "description": "MyPy type checking",
                "command": ["mypy", "src/pynomaly", "--config-file=pyproject.toml"],
                "timeout": 600,
                "required": True,
            },
            "bandit": {
                "description": "Bandit security analysis",
                "command": ["bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"],
                "timeout": 300,
                "required": False,
            },
            "safety": {
                "description": "Safety dependency vulnerability check",
                "command": ["safety", "check", "--json", "--output", "safety-report.json"],
                "timeout": 300,
                "required": False,
            },
            "complexity": {
                "description": "Code complexity analysis",
                "command": ["radon", "cc", "src/", "--json"],
                "timeout": 300,
                "required": False,
            },
            "duplication": {
                "description": "Code duplication detection",
                "command": ["vulture", "src/", "--min-confidence", "80"],
                "timeout": 300,
                "required": False,
            },
            "imports": {
                "description": "Import organization check",
                "command": ["isort", "src/", "tests/", "--check-only", "--diff"],
                "timeout": 300,
                "required": True,
            },
            "docstring": {
                "description": "Docstring coverage check",
                "command": ["pydocstyle", "src/pynomaly", "--config=pyproject.toml"],
                "timeout": 300,
                "required": False,
            },
            "pytest_collect": {
                "description": "Test collection validation",
                "command": ["python", "-m", "pytest", "--collect-only", "-q"],
                "timeout": 300,
                "required": True,
            },
        }
        
        logger.info("Quality checker initialized", project_root=str(project_root))
    
    def install_dependencies(self) -> bool:
        """Install required quality check dependencies."""
        try:
            logger.info("Installing quality check dependencies...")
            
            dependencies = [
                "ruff>=0.1.0",
                "mypy>=1.0.0",
                "bandit>=1.7.0",
                "safety>=2.0.0",
                "radon>=5.0.0",
                "vulture>=2.0.0",
                "isort>=5.0.0",
                "pydocstyle>=6.0.0",
                "black>=22.0.0",
            ]
            
            for dep in dependencies:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
            
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error("Failed to install dependencies", error=str(e))
            return False
    
    def run_quality_check(self, check_name: str) -> QualityCheck:
        """Run a specific quality check."""
        if check_name not in self.checks:
            raise ValueError(f"Unknown quality check: {check_name}")
        
        check_config = self.checks[check_name]
        
        logger.info(f"Running quality check: {check_name}")
        
        start_time = time.time()
        
        try:
            # Run the check
            result = subprocess.run(
                check_config["command"],
                capture_output=True,
                text=True,
                timeout=check_config["timeout"],
                cwd=self.project_root
            )
            
            duration = time.time() - start_time
            
            # Determine if check passed
            passed = self._evaluate_check_result(check_name, result)
            
            quality_check = QualityCheck(
                name=check_name,
                passed=passed,
                duration=duration,
                output=result.stdout,
                error=result.stderr if not passed else ""
            )
            
            logger.info(
                f"Quality check completed: {check_name}",
                passed=passed,
                duration=duration,
                returncode=result.returncode
            )
            
            return quality_check
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Quality check {check_name} timed out after {check_config['timeout']} seconds"
            
            logger.error(error_msg)
            
            return QualityCheck(
                name=check_name,
                passed=False,
                duration=duration,
                error=error_msg
            )
        
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Quality check {check_name} failed with error: {str(e)}"
            
            logger.error(error_msg)
            
            return QualityCheck(
                name=check_name,
                passed=False,
                duration=duration,
                error=error_msg
            )
    
    def _evaluate_check_result(self, check_name: str, result: subprocess.CompletedProcess) -> bool:
        """Evaluate if a quality check passed based on its result."""
        # Most quality checks pass if return code is 0
        if result.returncode == 0:
            return True
        
        # Special handling for specific checks
        if check_name == "bandit":
            # Bandit may have findings but still be acceptable
            if result.returncode == 1:
                try:
                    # Parse JSON output to check severity
                    report_file = self.project_root / "bandit-report.json"
                    if report_file.exists():
                        with open(report_file) as f:
                            bandit_data = json.load(f)
                        
                        # Only fail on high severity issues
                        high_severity = sum(1 for r in bandit_data.get("results", []) 
                                          if r.get("issue_severity") == "HIGH")
                        return high_severity == 0
                except:
                    pass
        
        elif check_name == "safety":
            # Safety may have vulnerabilities but still be acceptable
            if result.returncode == 255:  # Safety found vulnerabilities
                try:
                    # Parse JSON output to check severity
                    report_file = self.project_root / "safety-report.json"
                    if report_file.exists():
                        with open(report_file) as f:
                            safety_data = json.load(f)
                        
                        # Only fail on high severity vulnerabilities
                        high_severity = sum(1 for v in safety_data.get("vulnerabilities", [])
                                          if v.get("severity") == "high")
                        return high_severity == 0
                except:
                    pass
        
        elif check_name == "complexity":
            # Complexity check is informational, always pass
            return True
        
        elif check_name == "duplication":
            # Duplication check is informational, always pass
            return True
        
        return False
    
    def run_all_checks(self, checks: Optional[List[str]] = None, fail_fast: bool = False) -> Dict[str, QualityCheck]:
        """Run all quality checks or specified checks."""
        if checks is None:
            checks = list(self.checks.keys())
        
        logger.info(f"Running quality checks: {checks}")
        
        results = {}
        total_start_time = time.time()
        
        for check_name in checks:
            if check_name not in self.checks:
                logger.warning(f"Unknown quality check: {check_name}")
                continue
            
            result = self.run_quality_check(check_name)
            results[check_name] = result
            self.results.append(result)
            
            # Fail fast if requested and this is a required check
            if fail_fast and not result.passed and self.checks[check_name]["required"]:
                logger.error(f"Required quality check failed: {check_name}")
                break
        
        total_duration = time.time() - total_start_time
        
        # Generate summary
        passed_count = sum(1 for r in results.values() if r.passed)
        required_count = sum(1 for name, r in results.items() 
                           if r.passed and self.checks[name]["required"])
        total_required = sum(1 for name in results.keys() 
                           if self.checks[name]["required"])
        
        logger.info(
            "Quality checks completed",
            passed=passed_count,
            total=len(results),
            required_passed=required_count,
            required_total=total_required,
            duration=total_duration
        )
        
        return results
    
    def generate_report(self, results: Dict[str, QualityCheck], output_path: Path):
        """Generate comprehensive quality report."""
        try:
            # Create reports directory
            reports_dir = output_path / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate overall quality score
            total_checks = len(results)
            passed_checks = sum(1 for r in results.values() if r.passed)
            required_checks = sum(1 for name in results.keys() if self.checks[name]["required"])
            required_passed = sum(1 for name, r in results.items() 
                                if r.passed and self.checks[name]["required"])
            
            quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            required_score = (required_passed / required_checks) * 100 if required_checks > 0 else 0
            
            # Generate JSON report
            json_report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_checks": total_checks,
                    "passed_checks": passed_checks,
                    "failed_checks": total_checks - passed_checks,
                    "required_checks": required_checks,
                    "required_passed": required_passed,
                    "quality_score": f"{quality_score:.1f}%",
                    "required_score": f"{required_score:.1f}%",
                    "overall_status": "PASS" if required_passed == required_checks else "FAIL",
                    "total_duration": sum(r.duration for r in results.values())
                },
                "checks": {
                    name: {
                        "description": self.checks[name]["description"],
                        "passed": result.passed,
                        "required": self.checks[name]["required"],
                        "duration": result.duration,
                        "timestamp": result.timestamp.isoformat(),
                        "output": result.output,
                        "error": result.error
                    }
                    for name, result in results.items()
                }
            }
            
            with open(reports_dir / "quality_report.json", "w") as f:
                json.dump(json_report, f, indent=2)
            
            # Generate HTML report
            html_report = self._generate_html_report(json_report)
            with open(reports_dir / "quality_report.html", "w") as f:
                f.write(html_report)
            
            # Generate markdown summary
            md_report = self._generate_markdown_report(json_report)
            with open(reports_dir / "quality_summary.md", "w") as f:
                f.write(md_report)
            
            logger.info("Quality reports generated", output_dir=str(reports_dir))
            
        except Exception as e:
            logger.error("Failed to generate quality report", error=str(e))
    
    def _generate_html_report(self, json_report: Dict) -> str:
        """Generate HTML quality report."""
        status_color = "#28a745" if json_report["summary"]["overall_status"] == "PASS" else "#dc3545"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pynomaly Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .status {{ color: {status_color}; font-weight: bold; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .check {{ border: 1px solid #ddd; margin: 10px 0; border-radius: 5px; }}
                .check-header {{ background-color: #f9f9f9; padding: 10px; }}
                .check-content {{ padding: 10px; }}
                .passed {{ color: #28a745; }}
                .failed {{ color: #dc3545; }}
                .required {{ font-weight: bold; }}
                .output {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; white-space: pre-wrap; font-family: monospace; max-height: 200px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Pynomaly Quality Report</h1>
                <p class="status">Overall Status: {json_report['summary']['overall_status']}</p>
                <p>Generated: {json_report['timestamp']}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Quality Score</h3>
                    <p>{json_report['summary']['quality_score']}</p>
                </div>
                <div class="metric">
                    <h3>Required Score</h3>
                    <p>{json_report['summary']['required_score']}</p>
                </div>
                <div class="metric">
                    <h3>Passed</h3>
                    <p class="passed">{json_report['summary']['passed_checks']}</p>
                </div>
                <div class="metric">
                    <h3>Failed</h3>
                    <p class="failed">{json_report['summary']['failed_checks']}</p>
                </div>
                <div class="metric">
                    <h3>Duration</h3>
                    <p>{json_report['summary']['total_duration']:.2f}s</p>
                </div>
            </div>
            
            <h2>Quality Checks</h2>
        """
        
        for check_name, check_data in json_report['checks'].items():
            status_class = "passed" if check_data['passed'] else "failed"
            status_text = "✅ PASSED" if check_data['passed'] else "❌ FAILED"
            required_text = " (REQUIRED)" if check_data['required'] else ""
            
            html += f"""
            <div class="check">
                <div class="check-header">
                    <h3>{check_name}{required_text} <span class="{status_class}">{status_text}</span></h3>
                    <p>{check_data['description']}</p>
                    <p>Duration: {check_data['duration']:.2f}s</p>
                </div>
                <div class="check-content">
                    {f'<div class="output">{check_data["error"]}</div>' if check_data['error'] else ''}
                    {f'<div class="output">{check_data["output"][:1000]}...</div>' if check_data['output'] and len(check_data['output']) > 1000 else f'<div class="output">{check_data["output"]}</div>' if check_data['output'] else ''}
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_markdown_report(self, json_report: Dict) -> str:
        """Generate markdown quality report."""
        status_emoji = "✅" if json_report["summary"]["overall_status"] == "PASS" else "❌"
        
        md = f"""# 🔍 Pynomaly Quality Report

{status_emoji} **Overall Status:** {json_report['summary']['overall_status']}

**Generated:** {json_report['timestamp']}

## 📊 Summary

| Metric | Value |
|--------|-------|
| Quality Score | {json_report['summary']['quality_score']} |
| Required Score | {json_report['summary']['required_score']} |
| Total Checks | {json_report['summary']['total_checks']} |
| Passed | {json_report['summary']['passed_checks']} |
| Failed | {json_report['summary']['failed_checks']} |
| Required Passed | {json_report['summary']['required_passed']}/{json_report['summary']['required_checks']} |
| Total Duration | {json_report['summary']['total_duration']:.2f}s |

## 🔍 Quality Checks

"""
        
        for check_name, check_data in json_report['checks'].items():
            status_emoji = "✅" if check_data['passed'] else "❌"
            status_text = "PASSED" if check_data['passed'] else "FAILED"
            required_text = " (REQUIRED)" if check_data['required'] else ""
            
            md += f"""### {status_emoji} {check_name}{required_text} - {status_text}

**Description:** {check_data['description']}  
**Duration:** {check_data['duration']:.2f}s  
**Timestamp:** {check_data['timestamp']}

"""
            
            if check_data['error']:
                md += f"""**Error:**
```
{check_data['error']}
```

"""
        
        return md


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pynomaly Quality Checker")
    parser.add_argument(
        "--checks",
        nargs="+",
        help="Quality checks to run (default: all)",
        choices=list(QualityChecker(Path.cwd()).checks.keys()),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first required check failure"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quality-output"),
        help="Output directory for quality reports"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required dependencies"
    )
    
    args = parser.parse_args()
    
    # Initialize quality checker
    checker = QualityChecker(args.project_root)
    
    # Install dependencies if requested
    if args.install_deps:
        if not checker.install_dependencies():
            logger.error("Failed to install dependencies")
            sys.exit(1)
    
    # Run quality checks
    results = checker.run_all_checks(
        checks=args.checks,
        fail_fast=args.fail_fast
    )
    
    # Generate report
    checker.generate_report(results, args.output_dir)
    
    # Calculate exit code
    required_failed = sum(1 for name, r in results.items() 
                         if not r.passed and checker.checks[name]["required"])
    
    if required_failed > 0:
        logger.error(f"Required quality checks failed: {required_failed}")
        sys.exit(1)
    else:
        logger.info("All required quality checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()