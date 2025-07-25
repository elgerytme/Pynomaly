#!/usr/bin/env python3
"""
Integrated Testing Framework for Monorepo Packages

Orchestrates comprehensive testing across all packages with validation,
security scanning, and compliance checking.
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import concurrent.futures
import tempfile

# Import local package validator
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import with error handling for missing dependencies
try:
    from package_validator import PackageValidator, PackageValidationReport
except ImportError as e:
    print(f"Warning: Package validator not available: {e}")
    
    # Create mock classes for graceful degradation
    class PackageValidator:
        def __init__(self, *args, **kwargs):
            pass
        def validate_package(self, package_path):
            return type('MockReport', (), {
                'overall_status': 'skipped',
                'score': 0,
                'results': [],
                'recommendations': ['Package validator dependencies not available']
            })()
    
    class PackageValidationReport:
        pass

class TestSuite:
    """Represents a test suite for a package."""
    
    def __init__(self, package_path: str, suite_type: str):
        self.package_path = Path(package_path)
        self.suite_type = suite_type  # unit, integration, security, performance
        self.results = {}
        
    def run(self) -> Dict[str, Any]:
        """Run the test suite."""
        try:
            if self.suite_type == "unit":
                return self._run_unit_tests()
            elif self.suite_type == "integration":
                return self._run_integration_tests()
            elif self.suite_type == "security":
                return self._run_security_tests()
            elif self.suite_type == "performance":
                return self._run_performance_tests()
            else:
                return {"status": "error", "message": f"Unknown suite type: {self.suite_type}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/unit/", "-v", "--tb=short", "--maxfail=5",
            "--json-report", "--json-report-file=/tmp/pytest_report.json"
        ]
        
        result = subprocess.run(
            cmd, 
            cwd=self.package_path,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": 0  # Would parse from pytest output
        }
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/integration/", "-v", "--tb=short", "--maxfail=3"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=self.package_path,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def _run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        # Run bandit security scanner
        cmd = ["bandit", "-r", "src/", "-f", "json", "-o", "/tmp/bandit_report.json"]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.package_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                "status": "pass" if result.returncode == 0 else "warning",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "tool": "bandit"
            }
        except FileNotFoundError:
            return {
                "status": "skipped", 
                "message": "bandit not available",
                "tool": "bandit"
            }
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        performance_tests = list(self.package_path.glob("tests/performance/*.py"))
        
        if not performance_tests:
            return {"status": "skipped", "message": "No performance tests found"}
        
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/performance/", "-v", "--tb=short", "--maxfail=1"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=self.package_path,
            capture_output=True,
            text=True,
            timeout=900  # 15 minutes for performance tests
        )
        
        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

class IntegratedTestRunner:
    """Orchestrates comprehensive testing across all packages."""
    
    def __init__(self, monorepo_root: str = "."):
        self.monorepo_root = Path(monorepo_root).resolve()
        self.validator = PackageValidator(str(self.monorepo_root))
        self.results = {}
        
    def discover_packages(self) -> List[str]:
        """Discover all packages in the monorepo."""
        packages = []
        packages_dir = self.monorepo_root / "src" / "packages"
        
        if packages_dir.exists():
            for package_dir in packages_dir.rglob("*/"):
                if package_dir.is_dir() and (package_dir / "src").exists():
                    packages.append(str(package_dir))
        
        return packages
    
    async def run_comprehensive_tests(self, 
                                    packages: Optional[List[str]] = None,
                                    test_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive tests across packages."""
        
        if packages is None:
            packages = self.discover_packages()
        
        if test_types is None:
            test_types = ["validation", "unit", "integration", "security"]
        
        print(f"🧪 Running comprehensive tests on {len(packages)} packages")
        print(f"📋 Test types: {', '.join(test_types)}")
        print("=" * 60)
        
        overall_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "monorepo_root": str(self.monorepo_root),
            "packages_tested": len(packages),
            "test_types": test_types,
            "package_results": {},
            "summary": {
                "total_packages": len(packages),
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "skipped": 0
            }
        }
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all package test jobs
            future_to_package = {}
            
            for package_path in packages:
                future = executor.submit(self._test_package, package_path, test_types)
                future_to_package[future] = package_path
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_package):
                package_path = future_to_package[future]
                package_name = Path(package_path).name
                
                try:
                    package_result = future.result(timeout=1800)  # 30 minute timeout
                    overall_results["package_results"][package_name] = package_result
                    
                    # Update summary
                    if package_result["overall_status"] == "pass":
                        overall_results["summary"]["passed"] += 1
                    elif package_result["overall_status"] == "fail":
                        overall_results["summary"]["failed"] += 1
                    elif package_result["overall_status"] == "warning":
                        overall_results["summary"]["warnings"] += 1
                    else:
                        overall_results["summary"]["skipped"] += 1
                    
                    print(f"✅ {package_name}: {package_result['overall_status']}")
                    
                except Exception as e:
                    print(f"❌ {package_name}: error - {str(e)}")
                    overall_results["package_results"][package_name] = {
                        "overall_status": "error",
                        "error": str(e)
                    }
                    overall_results["summary"]["failed"] += 1
        
        # Calculate overall success rate
        total_tested = overall_results["summary"]["total_packages"]
        passed = overall_results["summary"]["passed"]
        overall_results["summary"]["success_rate"] = (passed / total_tested * 100) if total_tested > 0 else 0
        
        return overall_results
    
    def _test_package(self, package_path: str, test_types: List[str]) -> Dict[str, Any]:
        """Test a single package with all specified test types."""
        package_path = Path(package_path)
        package_name = package_path.name
        
        package_result = {
            "package_name": package_name,
            "package_path": str(package_path),
            "test_results": {},
            "overall_status": "pass",
            "issues": [],
            "recommendations": []
        }
        
        # Run validation
        if "validation" in test_types:
            try:
                validation_report = self.validator.validate_package(str(package_path))
                package_result["test_results"]["validation"] = {
                    "status": validation_report.overall_status,
                    "score": validation_report.score,
                    "issues": [r.message for r in validation_report.results if r.status == "fail"],
                    "recommendations": validation_report.recommendations
                }
                
                if validation_report.overall_status in ["poor", "fair"]:
                    package_result["overall_status"] = "warning"
                    package_result["issues"].extend(validation_report.recommendations)
                    
            except Exception as e:
                package_result["test_results"]["validation"] = {
                    "status": "error",
                    "error": str(e)
                }
                package_result["overall_status"] = "fail"
        
        # Run test suites
        test_suite_types = [t for t in test_types if t != "validation"]
        
        for suite_type in test_suite_types:
            try:
                suite = TestSuite(str(package_path), suite_type)
                result = suite.run()
                package_result["test_results"][suite_type] = result
                
                if result["status"] == "fail":
                    package_result["overall_status"] = "fail"
                    package_result["issues"].append(f"{suite_type} tests failed")
                elif result["status"] == "warning" and package_result["overall_status"] != "fail":
                    package_result["overall_status"] = "warning"
                    
            except Exception as e:
                package_result["test_results"][suite_type] = {
                    "status": "error",
                    "error": str(e)
                }
                package_result["overall_status"] = "fail"
        
        return package_result
    
    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate a comprehensive test report."""
        
        report = f"""
Monorepo Comprehensive Test Report
==================================

Generated: {results['timestamp']}
Monorepo Root: {results['monorepo_root']}
Packages Tested: {results['packages_tested']}
Test Types: {', '.join(results['test_types'])}

Summary
-------
Total Packages: {results['summary']['total_packages']}
Passed: {results['summary']['passed']}
Failed: {results['summary']['failed']}
Warnings: {results['summary']['warnings']}
Skipped: {results['summary']['skipped']}
Success Rate: {results['summary']['success_rate']:.1f}%

Package Results
---------------
"""
        
        for package_name, package_result in results["package_results"].items():
            status_emoji = {
                "pass": "✅",
                "fail": "❌", 
                "warning": "⚠️",
                "error": "💥",
                "skipped": "⏭️"
            }.get(package_result["overall_status"], "❓")
            
            report += f"\n{status_emoji} {package_name}: {package_result['overall_status'].upper()}\n"
            
            if "validation" in package_result["test_results"]:
                val_result = package_result["test_results"]["validation"]
                if "score" in val_result:
                    report += f"   📊 Validation Score: {val_result['score']}/100\n"
            
            for test_type, test_result in package_result["test_results"].items():
                if test_type != "validation":
                    status = test_result.get("status", "unknown")
                    report += f"   🧪 {test_type.title()} Tests: {status}\n"
            
            if package_result["issues"]:
                report += f"   ⚠️  Issues: {len(package_result['issues'])}\n"
                for issue in package_result["issues"][:3]:  # Show first 3 issues
                    report += f"      - {issue}\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to: {output_file}")
        
        return report

async def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated testing framework for monorepo")
    parser.add_argument("--packages", nargs="*", help="Specific packages to test")
    parser.add_argument("--types", nargs="*", 
                       choices=["validation", "unit", "integration", "security", "performance"],
                       default=["validation", "unit", "security"],
                       help="Test types to run")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel test runners")
    
    args = parser.parse_args()
    
    runner = IntegratedTestRunner()
    
    # Run comprehensive tests
    results = await runner.run_comprehensive_tests(
        packages=args.packages,
        test_types=args.types
    )
    
    # Generate and display report
    report = runner.generate_report(results, args.output)
    print(report)
    
    # Exit with appropriate code
    success_rate = results["summary"]["success_rate"]
    if success_rate >= 90:
        print(f"\n🎉 Excellent! {success_rate:.1f}% success rate")
        return 0
    elif success_rate >= 75:
        print(f"\n👍 Good! {success_rate:.1f}% success rate")
        return 0
    elif success_rate >= 50:
        print(f"\n⚠️  Needs improvement: {success_rate:.1f}% success rate")
        return 1
    else:
        print(f"\n❌ Critical issues: {success_rate:.1f}% success rate")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)