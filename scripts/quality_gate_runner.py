#!/usr/bin/env python3
"""
Local Quality Gate Runner

This script runs the same quality gate checks locally that are executed in CI/CD,
allowing developers to validate their changes before pushing to remote.
"""

import sys
import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class QualityGateRunner:
    """
    Local quality gate runner that mirrors CI/CD pipeline validation.
    """
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "phases": {},
            "overall_status": "UNKNOWN",
            "execution_time_seconds": 0
        }
        
    def run_command(self, command: List[str], timeout: int = 300, cwd: str = None) -> Tuple[int, str, str]:
        """Run command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or str(self.base_dir)
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return 1, "", str(e)
    
    def check_dependencies(self) -> bool:
        """Check if required testing dependencies are installed."""
        print("🔍 Checking testing dependencies...")
        
        required_packages = [
            "pytest", "pytest-cov", "pytest-timeout", "numpy", "pandas", 
            "scikit-learn", "structlog", "pydantic"
        ]
        
        missing_packages = []
        for package in required_packages:
            returncode, _, _ = self.run_command([sys.executable, "-c", f"import {package.replace('-', '_')}"])
            if returncode != 0:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All testing dependencies are installed")
        return True
    
    def run_phase1_tests(self) -> Dict[str, any]:
        """Run Phase 1: Critical Algorithm & Security Tests."""
        print("\n🚀 Phase 1: Critical Algorithm & Security Tests")
        
        phase_results = {
            "status": "UNKNOWN",
            "packages": {},
            "execution_time": 0,
            "coverage": {}
        }
        
        start_time = time.time()
        
        test_configs = [
            {
                "name": "machine_learning",
                "path": "src/packages/ai/machine_learning/tests/",
                "coverage_target": 80,
                "timeout": 300
            },
            {
                "name": "anomaly_detection", 
                "path": "src/packages/data/anomaly_detection/tests/",
                "coverage_target": 75,
                "timeout": 180
            },
            {
                "name": "enterprise_auth",
                "path": "src/packages/enterprise/enterprise_auth/tests/",
                "coverage_target": 85,
                "timeout": 120
            },
            {
                "name": "system_integration",
                "path": "src/packages/system_tests/integration/",
                "coverage_target": 70,
                "timeout": 600
            }
        ]
        
        all_passed = True
        
        for config in test_configs:
            print(f"  📊 Testing {config['name']}...")
            
            if not Path(self.base_dir / config["path"]).exists():
                print(f"    ⚠️  Test path not found: {config['path']}")
                phase_results["packages"][config["name"]] = {"status": "SKIPPED", "reason": "path_not_found"}
                continue
            
            # Run tests with coverage
            command = [
                sys.executable, "-m", "pytest",
                config["path"],
                f"--cov={config['path'].replace('/tests/', '').replace('/tests', '')}",
                "--cov-report=term-missing",
                f"--cov-fail-under={config['coverage_target']}",
                f"--timeout={config['timeout']}",
                "-v", "--tb=short"
            ]
            
            returncode, stdout, stderr = self.run_command(command, timeout=config["timeout"] + 60)
            
            # Parse results
            if returncode == 0:
                phase_results["packages"][config["name"]] = {"status": "PASSED"}
                print(f"    ✅ {config['name']} tests passed")
            else:
                phase_results["packages"][config["name"]] = {
                    "status": "FAILED", 
                    "error": stderr[:200] if stderr else "Unknown error"
                }
                print(f"    ❌ {config['name']} tests failed")
                all_passed = False
        
        phase_results["status"] = "PASSED" if all_passed else "FAILED"
        phase_results["execution_time"] = time.time() - start_time
        
        return phase_results
    
    def run_phase2_tests(self) -> Dict[str, any]:
        """Run Phase 2: Domain-Specific & Load Tests."""
        print("\n🚀 Phase 2: Domain-Specific & Load Tests")
        
        phase_results = {
            "status": "UNKNOWN",
            "packages": {},
            "execution_time": 0
        }
        
        start_time = time.time()
        
        test_configs = [
            {
                "name": "data_quality",
                "path": "src/packages/data/quality/tests/",
                "coverage_target": 75,
                "timeout": 240
            },
            {
                "name": "data_observability",
                "path": "src/packages/data/observability/tests/",
                "coverage_target": 70,
                "timeout": 180
            }
        ]
        
        all_passed = True
        
        for config in test_configs:
            print(f"  📊 Testing {config['name']}...")
            
            if not Path(self.base_dir / config["path"]).exists():
                phase_results["packages"][config["name"]] = {"status": "SKIPPED", "reason": "path_not_found"}
                continue
            
            command = [
                sys.executable, "-m", "pytest",
                config["path"],
                f"--cov={config['path'].replace('/tests/', '').replace('/tests', '')}",
                "--cov-report=term-missing",
                f"--cov-fail-under={config['coverage_target']}",
                f"--timeout={config['timeout']}",
                "-v", "--tb=short"
            ]
            
            returncode, stdout, stderr = self.run_command(command, timeout=config["timeout"] + 60)
            
            if returncode == 0:
                phase_results["packages"][config["name"]] = {"status": "PASSED"}
                print(f"    ✅ {config['name']} tests passed")
            else:
                phase_results["packages"][config["name"]] = {
                    "status": "FAILED",
                    "error": stderr[:200] if stderr else "Unknown error"
                }
                print(f"    ❌ {config['name']} tests failed")
                all_passed = False
        
        # Test load testing framework
        print("  📈 Testing load testing framework...")
        load_test_script = self.base_dir / "scripts/load_testing_framework.py"
        
        if load_test_script.exists():
            command = [sys.executable, str(load_test_script), "--test-mode", "--duration=60", "--max-load=100"]
            returncode, stdout, stderr = self.run_command(command, timeout=120)
            
            if returncode == 0:
                phase_results["packages"]["load_testing"] = {"status": "PASSED"}
                print("    ✅ Load testing framework tests passed")
            else:
                phase_results["packages"]["load_testing"] = {"status": "FAILED", "error": stderr[:200]}
                print("    ❌ Load testing framework tests failed")
                all_passed = False
        else:
            phase_results["packages"]["load_testing"] = {"status": "SKIPPED", "reason": "script_not_found"}
        
        phase_results["status"] = "PASSED" if all_passed else "FAILED"
        phase_results["execution_time"] = time.time() - start_time
        
        return phase_results
    
    def run_phase3_tests(self) -> Dict[str, any]:
        """Run Phase 3: Comprehensive Package Tests."""
        print("\n🚀 Phase 3: Comprehensive Package Tests")
        
        phase_results = {
            "status": "UNKNOWN",
            "packages": {},
            "execution_time": 0
        }
        
        start_time = time.time()
        
        test_configs = [
            {
                "name": "mlops",
                "path": "src/packages/ai/mlops/tests/",
                "coverage_target": 70,
                "timeout": 300
            },
            {
                "name": "enterprise_governance",
                "path": "src/packages/enterprise/enterprise_governance/tests/",
                "coverage_target": 75,
                "timeout": 180
            },
            {
                "name": "statistics",
                "path": "src/packages/data/statistics/tests/",
                "coverage_target": 70,
                "timeout": 240
            },
            {
                "name": "data_architecture",
                "path": "src/packages/data/data_architecture/tests/",
                "coverage_target": 70,
                "timeout": 300
            },
            {
                "name": "enterprise_scalability",
                "path": "src/packages/enterprise/enterprise_scalability/tests/",
                "coverage_target": 65,
                "timeout": 360
            }
        ]
        
        all_passed = True
        
        for config in test_configs:
            print(f"  📊 Testing {config['name']}...")
            
            if not Path(self.base_dir / config["path"]).exists():
                phase_results["packages"][config["name"]] = {"status": "SKIPPED", "reason": "path_not_found"}
                continue
            
            command = [
                sys.executable, "-m", "pytest",
                config["path"],
                f"--cov={config['path'].replace('/tests/', '').replace('/tests', '')}",
                "--cov-report=term-missing",
                f"--cov-fail-under={config['coverage_target']}",
                f"--timeout={config['timeout']}",
                "-v", "--tb=short"
            ]
            
            returncode, stdout, stderr = self.run_command(command, timeout=config["timeout"] + 60)
            
            if returncode == 0:
                phase_results["packages"][config["name"]] = {"status": "PASSED"}
                print(f"    ✅ {config['name']} tests passed")
            else:
                phase_results["packages"][config["name"]] = {
                    "status": "FAILED",
                    "error": stderr[:200] if stderr else "Unknown error"
                }
                print(f"    ❌ {config['name']} tests failed")
                all_passed = False
        
        phase_results["status"] = "PASSED" if all_passed else "FAILED"
        phase_results["execution_time"] = time.time() - start_time
        
        return phase_results
    
    def generate_report(self) -> None:
        """Generate quality gate report."""
        print("\n📋 Generating Quality Gate Report...")
        
        # Determine overall status
        phase_statuses = [phase["status"] for phase in self.results["phases"].values()]
        
        if all(status == "PASSED" for status in phase_statuses):
            self.results["overall_status"] = "PASS"
        elif self.results["phases"].get("phase1", {}).get("status") == "PASSED":
            self.results["overall_status"] = "PARTIAL_PASS"
        else:
            self.results["overall_status"] = "FAIL"
        
        # Save JSON report
        with open("local-quality-gate-report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary
        print("\n" + "="*60)
        print(f"🎯 QUALITY GATE REPORT")
        print("="*60)
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Total Execution Time: {self.results['execution_time_seconds']:.1f} seconds")
        print(f"Generated: {self.results['timestamp']}")
        
        print("\n📊 Phase Results:")
        for phase_name, phase_data in self.results["phases"].items():
            status_emoji = "✅" if phase_data["status"] == "PASSED" else "❌"
            print(f"  {status_emoji} {phase_name}: {phase_data['status']} ({phase_data['execution_time']:.1f}s)")
            
            for package_name, package_data in phase_data["packages"].items():
                pkg_emoji = "✅" if package_data["status"] == "PASSED" else "⚠️" if package_data["status"] == "SKIPPED" else "❌"
                print(f"    {pkg_emoji} {package_name}: {package_data['status']}")
        
        print("\n🎯 Quality Gate Decision:")
        if self.results["overall_status"] == "PASS":
            print("✅ PASS - All tests passed, ready for production deployment")
        elif self.results["overall_status"] == "PARTIAL_PASS":
            print("⚠️  PARTIAL PASS - Critical tests passed, some optional tests failed")
        else:
            print("❌ FAIL - Critical tests failed, deployment blocked")
        
        print(f"\nDetailed report saved to: local-quality-gate-report.json")
    
    def run_all_phases(self, phases: List[str] = None) -> None:
        """Run all quality gate phases."""
        start_time = time.time()
        
        phases_to_run = phases or ["phase1", "phase2", "phase3"]
        
        print("🚀 Starting Local Quality Gate Validation")
        print(f"📁 Working directory: {self.base_dir}")
        print(f"🎯 Running phases: {', '.join(phases_to_run)}")
        
        if not self.check_dependencies():
            print("❌ Dependency check failed. Please install required packages.")
            sys.exit(1)
        
        if "phase1" in phases_to_run:
            self.results["phases"]["phase1"] = self.run_phase1_tests()
        
        if "phase2" in phases_to_run:
            self.results["phases"]["phase2"] = self.run_phase2_tests()
        
        if "phase3" in phases_to_run:
            self.results["phases"]["phase3"] = self.run_phase3_tests()
        
        self.results["execution_time_seconds"] = time.time() - start_time
        self.generate_report()


def main():
    """Main entry point for local quality gate runner."""
    parser = argparse.ArgumentParser(description="Local Quality Gate Runner")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["phase1", "phase2", "phase3"],
        default=["phase1", "phase2", "phase3"],
        help="Phases to run (default: all phases)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (default: current working directory)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only Phase 1 (critical tests)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        args.phases = ["phase1"]
    
    runner = QualityGateRunner(base_dir=args.base_dir)
    runner.run_all_phases(phases=args.phases)
    
    # Exit with appropriate code
    exit_code = 0 if runner.results["overall_status"] in ["PASS", "PARTIAL_PASS"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()