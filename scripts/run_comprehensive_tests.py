#!/usr/bin/env python3
"""
Comprehensive test execution script for Pynomaly monorepo.
Runs algorithm validation, security testing, and integration tests with detailed reporting.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import json


class TestRunner:
    """Comprehensive test runner for Pynomaly packages."""
    
    def __init__(self, monorepo_root: str):
        self.monorepo_root = Path(monorepo_root)
        self.test_results = {}
        self.start_time = time.time()
        
    def run_package_tests(self, package_path: str, test_markers: List[str] = None) -> Dict[str, Any]:
        """Run tests for a specific package."""
        package_dir = self.monorepo_root / package_path
        
        if not (package_dir / "tests").exists():
            return {
                'status': 'skipped',
                'reason': 'No tests directory found',
                'package': package_path
            }
        
        # Build pytest command
        cmd = [
            sys.executable, '-m', 'pytest',
            str(package_dir / "tests"),
            '-v',
            '--tb=short',
            '--disable-warnings'
        ]
        
        # Add markers if specified
        if test_markers:
            cmd.extend(['-m', ' and '.join(test_markers)])
        
        # Add coverage if available
        try:
            import pytest_cov
            cmd.extend([
                '--cov=' + str(package_dir / "src"),
                '--cov-report=term-missing',
                '--cov-report=json:' + str(package_dir / "coverage.json")
            ])
        except ImportError:
            pass
        
        try:
            print(f"\n🧪 Running tests for {package_path}...")
            result = subprocess.run(
                cmd,
                cwd=package_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'package': package_path,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'markers': test_markers or []
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'package': package_path,
                'error': 'Test execution exceeded 5 minutes'
            }
        except Exception as e:
            return {
                'status': 'error',
                'package': package_path,
                'error': str(e)
            }
    
    def run_algorithm_validation_tests(self) -> Dict[str, Any]:
        """Run algorithm validation tests."""
        print("\n" + "="*50)
        print("🔬 ALGORITHM VALIDATION TESTS")
        print("="*50)
        
        results = {}
        
        # Machine Learning algorithm tests
        ml_result = self.run_package_tests(
            "src/packages/ai/machine_learning",
            ["algorithm_validation"]
        )
        results['machine_learning'] = ml_result
        
        # Anomaly Detection algorithm tests
        anomaly_result = self.run_package_tests(
            "src/packages/data/anomaly_detection", 
            ["accuracy"]
        )
        results['anomaly_detection'] = anomaly_result
        
        return results
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        print("\n" + "="*50)
        print("🔒 SECURITY TESTS")
        print("="*50)
        
        results = {}
        
        # Enterprise Authentication security tests
        auth_result = self.run_package_tests(
            "src/packages/enterprise/enterprise_auth",
            ["security"]
        )
        results['enterprise_auth'] = auth_result
        
        return results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n" + "="*50)
        print("🔗 INTEGRATION TESTS") 
        print("="*50)
        
        results = {}
        
        # Cross-package integration tests
        integration_result = self.run_package_tests(
            "src/packages/system_tests",
            ["integration"]
        )
        results['system_integration'] = integration_result
        
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        print("\n" + "="*50)
        print("⚡ PERFORMANCE TESTS")
        print("="*50)
        
        results = {}
        
        # Performance tests across packages
        packages_with_performance_tests = [
            "src/packages/ai/machine_learning",
            "src/packages/data/anomaly_detection",
            "src/packages/system_tests"
        ]
        
        for package in packages_with_performance_tests:
            perf_result = self.run_package_tests(package, ["performance"])
            results[package.split('/')[-1]] = perf_result
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        print("🚀 Starting Comprehensive Test Suite for Pynomaly Monorepo")
        print(f"📁 Repository: {self.monorepo_root}")
        print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {}
        
        # Run different test categories
        all_results['algorithm_validation'] = self.run_algorithm_validation_tests()
        all_results['security'] = self.run_security_tests()
        all_results['integration'] = self.run_integration_tests()
        all_results['performance'] = self.run_performance_tests()
        
        return all_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        report = []
        report.append("# Pynomaly Comprehensive Test Report")
        report.append("=" * 50)
        report.append(f"**Execution Time**: {total_duration:.2f} seconds")
        report.append(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for category, category_results in results.items():
            report.append(f"## {category.title().replace('_', ' ')} Tests")
            report.append("")
            
            for package, result in category_results.items():
                status = result['status']
                status_emoji = {
                    'passed': '✅',
                    'failed': '❌', 
                    'skipped': '⏭️',
                    'timeout': '⏰',
                    'error': '💥'
                }.get(status, '❓')
                
                report.append(f"### {status_emoji} {package}")
                report.append(f"**Status**: {status}")
                
                if status == 'passed':
                    passed_tests += 1
                elif status == 'failed':
                    failed_tests += 1
                    if 'stderr' in result and result['stderr']:
                        report.append("**Errors**:")
                        report.append(f"```\n{result['stderr'][:500]}...\n```")
                elif status == 'skipped':
                    skipped_tests += 1
                    if 'reason' in result:
                        report.append(f"**Reason**: {result['reason']}")
                
                total_tests += 1
                report.append("")
        
        # Overall summary
        report.insert(4, f"**Total Tests**: {total_tests}")
        report.insert(5, f"**Passed**: {passed_tests} ✅")
        report.insert(6, f"**Failed**: {failed_tests} ❌")
        report.insert(7, f"**Skipped**: {skipped_tests} ⏭️")
        report.insert(8, f"**Success Rate**: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "**Success Rate**: N/A")
        report.insert(9, "")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if failed_tests > 0:
            report.append("### ⚠️ Action Required")
            report.append("- Review and fix failing tests before production deployment")
            report.append("- Focus on algorithm validation and security test failures")
            report.append("")
        
        if skipped_tests > 0:
            report.append("### 📋 Test Coverage Improvements")
            report.append("- Implement tests for packages with skipped test suites")
            report.append("- Priority: packages with existing implementation but no tests")
            report.append("")
        
        success_rate = (passed_tests/total_tests*100) if total_tests > 0 else 0
        if success_rate < 80:
            report.append("### 🎯 Coverage Goals")
            report.append(f"- Current success rate: {success_rate:.1f}%")
            report.append("- Target: >80% for production readiness")
            report.append("- Focus on algorithm accuracy validation and security testing")
            report.append("")
        
        report.append("---")
        report.append("*Generated by Pynomaly Comprehensive Test Suite*")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], report: str):
        """Save test results and report."""
        # Save JSON results
        results_file = self.monorepo_root / "TEST_RESULTS.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save markdown report
        report_file = self.monorepo_root / "TEST_EXECUTION_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Results saved to: {results_file}")
        print(f"📄 Report saved to: {report_file}")


def main():
    """Main test execution function."""
    # Determine monorepo root
    script_dir = Path(__file__).parent
    monorepo_root = script_dir.parent
    
    # Initialize test runner
    runner = TestRunner(str(monorepo_root))
    
    try:
        # Run comprehensive tests
        results = runner.run_all_tests()
        
        # Generate report
        report = runner.generate_report(results)
        
        # Save results
        runner.save_results(results, report)
        
        # Print summary to console
        print("\n" + "="*50)
        print("📊 TEST EXECUTION SUMMARY")
        print("="*50)
        print(report.split("## Recommendations")[0])  # Print up to recommendations
        
        # Determine exit code
        failed_categories = 0
        for category_results in results.values():
            category_failed = any(
                result['status'] == 'failed' 
                for result in category_results.values()
            )
            if category_failed:
                failed_categories += 1
        
        if failed_categories > 0:
            print(f"\n❌ {failed_categories} test categories failed")
            return 1
        else:
            print("\n✅ All test categories passed")
            return 0
            
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())