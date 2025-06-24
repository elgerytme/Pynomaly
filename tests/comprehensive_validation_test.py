#!/usr/bin/env python3
"""
Comprehensive Validation Test Suite
Executes CLI, API, and UI testing with centralized reporting
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import asyncio
import pytest
import requests
import tempfile
import os

class ComprehensiveValidationTester:
    """Orchestrates comprehensive testing across all components"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "cli_tests": {},
            "api_tests": {},
            "ui_tests": {},
            "summary": {}
        }
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
    
    def run_cli_tests(self) -> Dict[str, Any]:
        """Execute comprehensive CLI testing"""
        print("🔍 Starting CLI Testing...")
        
        cli_results = {
            "basic_commands": self._test_basic_cli_commands(),
            "data_commands": self._test_data_cli_commands(),
            "model_commands": self._test_model_cli_commands(),
            "export_commands": self._test_export_cli_commands(),
            "server_commands": self._test_server_cli_commands()
        }
        
        self.results["cli_tests"] = cli_results
        return cli_results
    
    def _test_basic_cli_commands(self) -> Dict[str, Any]:
        """Test basic CLI functionality"""
        tests = {}
        
        # Test help command
        try:
            result = subprocess.run(
                ["poetry", "run", "pynomaly", "--help"],
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            tests["help_command"] = {
                "success": result.returncode == 0,
                "output_length": len(result.stdout),
                "has_usage": "usage:" in result.stdout.lower() or "Usage:" in result.stdout
            }
        except Exception as e:
            tests["help_command"] = {"success": False, "error": str(e)}
        
        # Test version command
        try:
            result = subprocess.run(
                ["poetry", "run", "pynomaly", "--version"],
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            tests["version_command"] = {
                "success": result.returncode == 0,
                "has_version": len(result.stdout.strip()) > 0
            }
        except Exception as e:
            tests["version_command"] = {"success": False, "error": str(e)}
        
        return tests
    
    def _test_data_cli_commands(self) -> Dict[str, Any]:
        """Test data-related CLI commands"""
        tests = {}
        
        # Create sample test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,feature3\n1,2,3\n4,5,6\n7,8,9\n")
            test_file = f.name
        
        try:
            # Test data quality command
            result = subprocess.run(
                ["poetry", "run", "pynomaly", "dataset", "quality", test_file],
                capture_output=True, text=True, timeout=60,
                cwd=self.project_root
            )
            tests["data_quality"] = {
                "success": result.returncode == 0,
                "has_output": len(result.stdout) > 0
            }
            
            # Test data info command
            result = subprocess.run(
                ["poetry", "run", "pynomaly", "dataset", "info", test_file],
                capture_output=True, text=True, timeout=60,
                cwd=self.project_root
            )
            tests["data_info"] = {
                "success": result.returncode == 0,
                "has_output": len(result.stdout) > 0
            }
            
        except Exception as e:
            tests["data_commands"] = {"success": False, "error": str(e)}
        finally:
            os.unlink(test_file)
        
        return tests
    
    def _test_model_cli_commands(self) -> Dict[str, Any]:
        """Test model-related CLI commands"""
        tests = {}
        
        try:
            # Test detector list command
            result = subprocess.run(
                ["poetry", "run", "pynomaly", "detector", "list"],
                capture_output=True, text=True, timeout=60,
                cwd=self.project_root
            )
            tests["detector_list"] = {
                "success": result.returncode == 0,
                "has_detectors": "isolation" in result.stdout.lower() or "lof" in result.stdout.lower()
            }
            
        except Exception as e:
            tests["model_commands"] = {"success": False, "error": str(e)}
        
        return tests
    
    def _test_export_cli_commands(self) -> Dict[str, Any]:
        """Test export-related CLI commands"""
        tests = {}
        
        try:
            # Test export formats command
            result = subprocess.run(
                ["poetry", "run", "pynomaly", "export", "formats"],
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            tests["export_formats"] = {
                "success": result.returncode == 0,
                "has_formats": len(result.stdout) > 0
            }
            
        except Exception as e:
            tests["export_commands"] = {"success": False, "error": str(e)}
        
        return tests
    
    def _test_server_cli_commands(self) -> Dict[str, Any]:
        """Test server-related CLI commands"""
        tests = {}
        
        try:
            # Test server help
            result = subprocess.run(
                ["poetry", "run", "pynomaly", "server", "--help"],
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            tests["server_help"] = {
                "success": result.returncode == 0,
                "has_options": "--port" in result.stdout or "--host" in result.stdout
            }
            
        except Exception as e:
            tests["server_commands"] = {"success": False, "error": str(e)}
        
        return tests
    
    def run_api_tests(self) -> Dict[str, Any]:
        """Execute comprehensive API testing"""
        print("🌐 Starting API Testing...")
        
        # Start server for testing
        server_process = None
        try:
            server_process = subprocess.Popen(
                ["poetry", "run", "uvicorn", "pynomaly.presentation.api:app", "--port", "8899"],
                cwd=self.project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server startup
            time.sleep(5)
            
            api_results = {
                "health_endpoints": self._test_health_endpoints(),
                "detection_endpoints": self._test_detection_endpoints(),
                "dataset_endpoints": self._test_dataset_endpoints(),
                "detector_endpoints": self._test_detector_endpoints()
            }
            
        except Exception as e:
            api_results = {"error": f"Failed to start server: {str(e)}"}
        finally:
            if server_process:
                server_process.terminate()
                server_process.wait()
        
        self.results["api_tests"] = api_results
        return api_results
    
    def _test_health_endpoints(self) -> Dict[str, Any]:
        """Test health and status endpoints"""
        tests = {}
        base_url = "http://localhost:8899"
        
        endpoints = [
            "/health",
            "/health/ready",
            "/health/live",
            "/",
            "/docs"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                tests[endpoint] = {
                    "success": response.status_code < 400,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                tests[endpoint] = {"success": False, "error": str(e)}
        
        return tests
    
    def _test_detection_endpoints(self) -> Dict[str, Any]:
        """Test detection-related endpoints"""
        tests = {}
        base_url = "http://localhost:8899"
        
        try:
            # Test detection endpoint with sample data
            sample_data = {
                "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [100, 200, 300]],
                "detector_type": "isolation_forest",
                "contamination": 0.1
            }
            
            response = requests.post(
                f"{base_url}/api/v1/detect",
                json=sample_data,
                timeout=30
            )
            
            tests["detection_post"] = {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "has_results": "anomalies" in response.text if response.status_code < 400 else False
            }
            
        except Exception as e:
            tests["detection_endpoints"] = {"success": False, "error": str(e)}
        
        return tests
    
    def _test_dataset_endpoints(self) -> Dict[str, Any]:
        """Test dataset-related endpoints"""
        tests = {}
        base_url = "http://localhost:8899"
        
        try:
            # Test dataset list endpoint
            response = requests.get(f"{base_url}/api/v1/datasets", timeout=10)
            tests["datasets_list"] = {
                "success": response.status_code < 400,
                "status_code": response.status_code
            }
            
        except Exception as e:
            tests["dataset_endpoints"] = {"success": False, "error": str(e)}
        
        return tests
    
    def _test_detector_endpoints(self) -> Dict[str, Any]:
        """Test detector-related endpoints"""
        tests = {}
        base_url = "http://localhost:8899"
        
        try:
            # Test detector list endpoint
            response = requests.get(f"{base_url}/api/v1/detectors", timeout=10)
            tests["detectors_list"] = {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "has_detectors": "isolation" in response.text.lower() if response.status_code < 400 else False
            }
            
        except Exception as e:
            tests["detector_endpoints"] = {"success": False, "error": str(e)}
        
        return tests
    
    def run_ui_tests(self) -> Dict[str, Any]:
        """Execute comprehensive UI testing using existing Playwright infrastructure"""
        print("🎨 Starting UI Testing...")
        
        try:
            # Use existing UI test runner
            result = subprocess.run(
                ["python", "ui/run_comprehensive_ui_tests.py"],
                capture_output=True, text=True, timeout=300,
                cwd=self.test_dir
            )
            
            ui_results = {
                "execution_success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "test_completion": "test completed" in result.stdout.lower()
            }
            
            # Try to load UI test results if available
            ui_results_file = self.test_dir / "ui" / "ui_test_results.json"
            if ui_results_file.exists():
                with open(ui_results_file) as f:
                    ui_results["detailed_results"] = json.load(f)
            
        except Exception as e:
            ui_results = {"success": False, "error": str(e)}
        
        self.results["ui_tests"] = ui_results
        return ui_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            "total_tests_run": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "issues_found": [],
            "recommendations": []
        }
        
        # Analyze CLI results
        for category, tests in self.results["cli_tests"].items():
            for test_name, result in tests.items():
                summary["total_tests_run"] += 1
                if result.get("success", False):
                    summary["successful_tests"] += 1
                else:
                    summary["failed_tests"] += 1
                    summary["issues_found"].append(f"CLI {category}.{test_name}: {result.get('error', 'Failed')}")
        
        # Analyze API results
        for category, tests in self.results["api_tests"].items():
            if isinstance(tests, dict) and "error" not in tests:
                for test_name, result in tests.items():
                    summary["total_tests_run"] += 1
                    if result.get("success", False):
                        summary["successful_tests"] += 1
                    else:
                        summary["failed_tests"] += 1
                        summary["issues_found"].append(f"API {category}.{test_name}: {result.get('error', 'Failed')}")
            elif "error" in tests:
                summary["issues_found"].append(f"API {category}: {tests['error']}")
        
        # Analyze UI results
        if not self.results["ui_tests"].get("execution_success", False):
            summary["issues_found"].append("UI testing execution failed")
        
        # Generate recommendations
        if summary["failed_tests"] > 0:
            summary["recommendations"].extend([
                "Review failed test details for specific issues",
                "Check dependency installation and environment setup",
                "Verify server startup and port availability",
                "Validate CLI command registration and imports"
            ])
        
        self.results["summary"] = summary
        return summary
    
    def save_results(self, filename: str = "comprehensive_validation_results.json"):
        """Save test results to file"""
        results_file = self.test_dir / filename
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📊 Results saved to: {results_file}")
        return results_file

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Validation Testing...")
    
    tester = ComprehensiveValidationTester()
    
    # Execute all test suites
    tester.run_cli_tests()
    tester.run_api_tests()
    tester.run_ui_tests()
    
    # Generate summary and save results
    summary = tester.generate_summary()
    results_file = tester.save_results()
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Total Tests: {summary['total_tests_run']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    
    if summary["issues_found"]:
        print(f"\n❌ Issues Found ({len(summary['issues_found'])}):")
        for issue in summary["issues_found"][:10]:  # Show first 10
            print(f"  • {issue}")
        if len(summary["issues_found"]) > 10:
            print(f"  • ... and {len(summary['issues_found']) - 10} more issues")
    
    if summary["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"  • {rec}")
    
    print(f"\n📁 Detailed results: {results_file}")
    
    return summary["failed_tests"] == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)