#!/usr/bin/env python3
"""Comprehensive test runner for achieving 90%+ coverage with Docker and dependency-aware testing."""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TestResult:
    """Test result summary."""
    coverage: float
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    execution_time: float
    errors: List[str]


class ComprehensiveTestRunner:
    """Comprehensive test runner with Docker and dependency management."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.results = {}
        
    def check_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def check_local_dependencies(self) -> Dict[str, bool]:
        """Check available local dependencies."""
        dependencies = {
            'numpy': False,
            'pandas': False,
            'scikit-learn': False,
            'torch': False,
            'tensorflow': False,
            'jax': False,
            'fastapi': False,
            'redis': False,
            'pytest': False,
            'hypothesis': False,
            'sqlalchemy': False,
            'psycopg2': False
        }
        
        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
            except ImportError:
                try:
                    # Handle alternate import names
                    if dep == 'scikit-learn':
                        __import__('sklearn')
                        dependencies[dep] = True
                except ImportError:
                    pass
        
        return dependencies
    
    def run_local_tests(self) -> TestResult:
        """Run tests locally with available dependencies."""
        print("🧪 Running local tests with available dependencies...")
        
        start_time = time.time()
        
        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root / 'src')
        
        # Run pytest with coverage
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '--cov=pynomaly',
            '--cov-report=term-missing:skip-covered',
            '--cov-report=html:local-coverage-reports',
            '--cov-report=xml:local-coverage-reports/coverage.xml',
            '--junit-xml=local-test-results/junit.xml',
            '-v',
            '--tb=short',
            '--maxfail=20'
        ]
        
        try:
            # Create output directories
            (self.project_root / 'local-test-results').mkdir(exist_ok=True)
            (self.project_root / 'local-coverage-reports').mkdir(exist_ok=True)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            execution_time = time.time() - start_time
            
            # Parse output for test counts
            output = result.stdout + result.stderr
            
            # Extract coverage percentage
            coverage = self._extract_coverage_from_output(output)
            
            # Extract test counts
            tests_passed, tests_failed, tests_skipped = self._extract_test_counts_from_output(output)
            
            errors = []
            if result.returncode != 0:
                errors.append(f"Tests failed with return code {result.returncode}")
                if "FAILED" in output:
                    errors.append("Some tests failed")
            
            return TestResult(
                coverage=coverage,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_skipped=tests_skipped,
                execution_time=execution_time,
                errors=errors
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                coverage=0.0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=time.time() - start_time,
                errors=["Tests timed out after 10 minutes"]
            )
        except Exception as e:
            return TestResult(
                coverage=0.0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=time.time() - start_time,
                errors=[f"Error running local tests: {e}"]
            )
    
    def run_docker_tests(self) -> TestResult:
        """Run comprehensive tests in Docker with all dependencies."""
        print("🐳 Running Docker tests with all dependencies...")
        
        start_time = time.time()
        
        try:
            # Build Docker image
            print("Building Docker test image...")
            build_result = subprocess.run(
                ["docker", "build", "-f", "Dockerfile.testing", "-t", "pynomaly-test", "."],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if build_result.returncode != 0:
                return TestResult(
                    coverage=0.0,
                    tests_passed=0,
                    tests_failed=0,
                    tests_skipped=0,
                    execution_time=time.time() - start_time,
                    errors=[f"Docker build failed: {build_result.stderr}"]
                )
            
            # Run tests in Docker
            print("Running comprehensive test suite in Docker...")
            test_result = subprocess.run([
                "docker", "run", "--rm",
                "-v", f"{self.project_root}/test-results:/app/test-results",
                "-v", f"{self.project_root}/coverage-reports:/app/coverage-reports",
                "pynomaly-test"
            ], capture_output=True, text=True, timeout=1200)
            
            execution_time = time.time() - start_time
            
            # Parse output
            output = test_result.stdout + test_result.stderr
            coverage = self._extract_coverage_from_output(output)
            tests_passed, tests_failed, tests_skipped = self._extract_test_counts_from_output(output)
            
            errors = []
            if test_result.returncode != 0:
                errors.append(f"Docker tests failed with return code {test_result.returncode}")
            
            return TestResult(
                coverage=coverage,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_skipped=tests_skipped,
                execution_time=execution_time,
                errors=errors
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                coverage=0.0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=time.time() - start_time,
                errors=["Docker tests timed out after 20 minutes"]
            )
        except Exception as e:
            return TestResult(
                coverage=0.0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=time.time() - start_time,
                errors=[f"Error running Docker tests: {e}"]
            )
    
    def run_docker_compose_tests(self) -> TestResult:
        """Run tests using Docker Compose with services."""
        print("🐙 Running Docker Compose tests with services...")
        
        start_time = time.time()
        
        try:
            # Start services
            print("Starting test services...")
            up_result = subprocess.run([
                "docker-compose", "-f", "docker-compose.testing.yml",
                "up", "--build", "-d", "postgres-test", "redis-test"
            ], cwd=self.project_root, capture_output=True, text=True, timeout=300)
            
            if up_result.returncode != 0:
                return TestResult(
                    coverage=0.0,
                    tests_passed=0,
                    tests_failed=0,
                    tests_skipped=0,
                    execution_time=time.time() - start_time,
                    errors=[f"Failed to start services: {up_result.stderr}"]
                )
            
            # Wait for services to be ready
            print("Waiting for services to be ready...")
            time.sleep(15)
            
            # Run tests
            print("Running tests with services...")
            test_result = subprocess.run([
                "docker-compose", "-f", "docker-compose.testing.yml",
                "run", "--rm", "pynomaly-test"
            ], cwd=self.project_root, capture_output=True, text=True, timeout=1200)
            
            # Clean up services
            subprocess.run([
                "docker-compose", "-f", "docker-compose.testing.yml",
                "down", "-v"
            ], cwd=self.project_root, capture_output=True)
            
            execution_time = time.time() - start_time
            
            # Parse output
            output = test_result.stdout + test_result.stderr
            coverage = self._extract_coverage_from_output(output)
            tests_passed, tests_failed, tests_skipped = self._extract_test_counts_from_output(output)
            
            errors = []
            if test_result.returncode != 0:
                errors.append(f"Docker Compose tests failed with return code {test_result.returncode}")
            
            return TestResult(
                coverage=coverage,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_skipped=tests_skipped,
                execution_time=execution_time,
                errors=errors
            )
            
        except Exception as e:
            # Clean up on error
            subprocess.run([
                "docker-compose", "-f", "docker-compose.testing.yml",
                "down", "-v"
            ], cwd=self.project_root, capture_output=True)
            
            return TestResult(
                coverage=0.0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=time.time() - start_time,
                errors=[f"Error running Docker Compose tests: {e}"]
            )
    
    def _extract_coverage_from_output(self, output: str) -> float:
        """Extract coverage percentage from test output."""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                # Look for pattern like "TOTAL ... 85%"
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            return float(part[:-1])
                        except ValueError:
                            continue
        return 0.0
    
    def _extract_test_counts_from_output(self, output: str) -> tuple:
        """Extract test counts from pytest output."""
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # Look for summary line like "25 passed, 5 failed, 10 skipped"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        try:
                            tests_passed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'failed' and i > 0:
                        try:
                            tests_failed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'skipped' and i > 0:
                        try:
                            tests_skipped = int(parts[i-1])
                        except ValueError:
                            pass
                break
        
        return tests_passed, tests_failed, tests_skipped
    
    def run_comprehensive_testing(self) -> Dict[str, TestResult]:
        """Run comprehensive testing strategy."""
        results = {}
        
        print("🚀 Starting Comprehensive Test Coverage Journey")
        print("=" * 60)
        
        # Check environment
        docker_available = self.check_docker_available()
        local_deps = self.check_local_dependencies()
        
        print(f"🐳 Docker Available: {'✅' if docker_available else '❌'}")
        print(f"📦 Local Dependencies Available: {sum(local_deps.values())}/{len(local_deps)}")
        
        available_deps = [dep for dep, available in local_deps.items() if available]
        print(f"✅ Available: {', '.join(available_deps)}")
        
        missing_deps = [dep for dep, available in local_deps.items() if not available]
        if missing_deps:
            print(f"❌ Missing: {', '.join(missing_deps)}")
        
        print("\n" + "=" * 60)
        
        # Run local tests first
        print("Phase 1: Local Testing with Available Dependencies")
        results['local'] = self.run_local_tests()
        
        print(f"📊 Local Results:")
        print(f"   Coverage: {results['local'].coverage:.1f}%")
        print(f"   Tests: {results['local'].tests_passed} passed, {results['local'].tests_failed} failed, {results['local'].tests_skipped} skipped")
        print(f"   Time: {results['local'].execution_time:.1f}s")
        
        if results['local'].errors:
            print(f"   Errors: {'; '.join(results['local'].errors)}")
        
        # Run Docker tests if available
        if docker_available:
            print("\n" + "-" * 40)
            print("Phase 2: Docker Testing with All Dependencies")
            results['docker'] = self.run_docker_tests()
            
            print(f"📊 Docker Results:")
            print(f"   Coverage: {results['docker'].coverage:.1f}%")
            print(f"   Tests: {results['docker'].tests_passed} passed, {results['docker'].tests_failed} failed, {results['docker'].tests_skipped} skipped")
            print(f"   Time: {results['docker'].execution_time:.1f}s")
            
            if results['docker'].errors:
                print(f"   Errors: {'; '.join(results['docker'].errors)}")
            
            # Run Docker Compose tests for full integration
            print("\n" + "-" * 40)
            print("Phase 3: Docker Compose Integration Testing")
            results['docker_compose'] = self.run_docker_compose_tests()
            
            print(f"📊 Docker Compose Results:")
            print(f"   Coverage: {results['docker_compose'].coverage:.1f}%")
            print(f"   Tests: {results['docker_compose'].tests_passed} passed, {results['docker_compose'].tests_failed} failed, {results['docker_compose'].tests_skipped} skipped")
            print(f"   Time: {results['docker_compose'].execution_time:.1f}s")
            
            if results['docker_compose'].errors:
                print(f"   Errors: {'; '.join(results['docker_compose'].errors)}")
        else:
            print("\n⚠️  Docker not available - skipping comprehensive dependency testing")
        
        return results
    
    def generate_final_report(self, results: Dict[str, TestResult]) -> None:
        """Generate final comprehensive test report."""
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE TEST COVERAGE FINAL REPORT")
        print("=" * 60)
        
        # Find best result
        best_coverage = 0
        best_test_name = "local"
        
        for test_name, result in results.items():
            if result.coverage > best_coverage:
                best_coverage = result.coverage
                best_test_name = test_name
        
        print(f"🏆 Best Coverage Achieved: {best_coverage:.1f}% ({best_test_name} testing)")
        print(f"📈 Starting Coverage: 18%")
        print(f"📊 Coverage Improvement: +{best_coverage - 18:.1f} percentage points")
        
        if best_coverage >= 90:
            print("🎉 SUCCESS: 90%+ coverage target ACHIEVED!")
        elif best_coverage >= 70:
            print("✅ EXCELLENT: 70%+ coverage achieved - production ready!")
        elif best_coverage >= 50:
            print("✅ GOOD: 50%+ coverage achieved - significant improvement!")
        elif best_coverage >= 30:
            print("✅ PROGRESS: 30%+ coverage achieved - good foundation!")
        else:
            print("⚠️  NEEDS WORK: Coverage still needs improvement")
        
        print("\n📋 Detailed Results:")
        for test_name, result in results.items():
            print(f"\n{test_name.upper()} TESTING:")
            print(f"  📊 Coverage: {result.coverage:.1f}%")
            print(f"  ✅ Passed: {result.tests_passed}")
            print(f"  ❌ Failed: {result.tests_failed}")
            print(f"  ⏭️  Skipped: {result.tests_skipped}")
            print(f"  ⏱️  Time: {result.execution_time:.1f}s")
            
            if result.errors:
                print(f"  🚨 Errors: {'; '.join(result.errors)}")
        
        # Save results to file
        report_data = {
            "timestamp": time.time(),
            "best_coverage": best_coverage,
            "best_test_type": best_test_name,
            "coverage_improvement": best_coverage - 18,
            "results": {
                name: {
                    "coverage": result.coverage,
                    "tests_passed": result.tests_passed,
                    "tests_failed": result.tests_failed,
                    "tests_skipped": result.tests_skipped,
                    "execution_time": result.execution_time,
                    "errors": result.errors
                }
                for name, result in results.items()
            }
        }
        
        with open(self.project_root / "comprehensive_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Full report saved to: comprehensive_test_report.json")
        
        # Generate next steps
        print("\n🎯 NEXT STEPS:")
        if best_coverage < 90:
            print("1. Install missing dependencies for higher coverage")
            print("2. Run tests in Docker environment for full dependency access")
            print("3. Add more integration tests for uncovered code paths")
            print("4. Consider property-based testing for edge cases")
        else:
            print("1. 🎉 CONGRATULATIONS! You've achieved excellent test coverage!")
            print("2. Consider adding performance tests and benchmarks")
            print("3. Set up continuous integration with this test suite")
            print("4. Monitor coverage in production deployments")


def main():
    """Main entry point."""
    runner = ComprehensiveTestRunner()
    results = runner.run_comprehensive_testing()
    runner.generate_final_report(results)


if __name__ == "__main__":
    main()