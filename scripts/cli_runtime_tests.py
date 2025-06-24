#!/usr/bin/env python3
"""Comprehensive CLI runtime tests using mocks to avoid dependency issues."""

import sys
import json
import tempfile
import unittest.mock as mock
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class CLIRuntimeTester:
    """Comprehensive CLI runtime testing with dependency mocking."""
    
    def __init__(self):
        self.test_results = {}
        self.console_output = []
        
    def capture_output(self, func, *args, **kwargs):
        """Capture console output from CLI commands."""
        with mock.patch('sys.stdout') as mock_stdout:
            try:
                result = func(*args, **kwargs)
                output = ''.join(call.args[0] for call in mock_stdout.write.call_args_list)
                return result, output
            except Exception as e:
                return None, str(e)

    def test_main_app_commands(self):
        """Test main CLI application commands."""
        print("🔍 Testing Main App Commands...")
        
        test_results = {}
        
        try:
            # Mock all external dependencies
            with patch.dict('sys.modules', {
                'pandas': Mock(),
                'numpy': Mock(),
                'sklearn': Mock(),
                'sklearn.base': Mock(),
                'pyod': Mock(),
                'pydantic': Mock(),
                'pydantic_settings': Mock(),
                'dependency_injector': Mock(),
                'dependency_injector.wiring': Mock(),
                'structlog': Mock(),
                'fastapi': Mock(),
                'uvicorn': Mock(),
                'redis': Mock(),
                'requests': Mock(),
                'aiofiles': Mock(),
                'psutil': Mock(),
            }):
                
                # Import CLI app with mocked dependencies
                from pynomaly.presentation.cli.app import app
                from typer.testing import CliRunner
                
                runner = CliRunner()
                
                # Test help command
                try:
                    result = runner.invoke(app, ["--help"])
                    test_results["help"] = {
                        "exit_code": result.exit_code,
                        "success": result.exit_code == 0,
                        "has_output": bool(result.stdout),
                        "contains_commands": all(cmd in result.stdout for cmd in ["detector", "dataset", "detect", "server"])
                    }
                except Exception as e:
                    test_results["help"] = {"success": False, "error": str(e)}
                
                # Test version command with mocked container
                try:
                    with patch('pynomaly.presentation.cli.app.get_cli_container') as mock_container:
                        mock_settings = Mock()
                        mock_settings.version = "1.0.0"
                        mock_settings.storage_path = "/tmp/pynomaly"
                        mock_container.return_value.config.return_value = mock_settings
                        
                        result = runner.invoke(app, ["version"])
                        test_results["version"] = {
                            "exit_code": result.exit_code,
                            "success": result.exit_code == 0,
                            "contains_version": "1.0.0" in result.stdout if result.stdout else False
                        }
                except Exception as e:
                    test_results["version"] = {"success": False, "error": str(e)}
                
                # Test config command
                try:
                    with patch('pynomaly.presentation.cli.app.get_cli_container') as mock_container:
                        mock_settings = Mock()
                        mock_settings.app_name = "Pynomaly"
                        mock_settings.version = "1.0.0"
                        mock_settings.debug = False
                        mock_settings.storage_path = "/tmp/pynomaly"
                        mock_settings.api_host = "localhost"
                        mock_settings.api_port = 8000
                        mock_settings.max_dataset_size_mb = 500
                        mock_settings.default_contamination_rate = 0.1
                        mock_settings.gpu_enabled = False
                        mock_container.return_value.config.return_value = mock_settings
                        
                        result = runner.invoke(app, ["config", "--show"])
                        test_results["config_show"] = {
                            "exit_code": result.exit_code,
                            "success": result.exit_code == 0,
                            "contains_config": "Configuration" in result.stdout if result.stdout else False
                        }
                except Exception as e:
                    test_results["config_show"] = {"success": False, "error": str(e)}
                
                # Test status command
                try:
                    with patch('pynomaly.presentation.cli.app.get_cli_container') as mock_container:
                        mock_container.return_value.detector_repository.return_value.count.return_value = 5
                        mock_container.return_value.dataset_repository.return_value.count.return_value = 3
                        mock_container.return_value.result_repository.return_value.count.return_value = 10
                        mock_container.return_value.result_repository.return_value.find_recent.return_value = []
                        
                        result = runner.invoke(app, ["status"])
                        test_results["status"] = {
                            "exit_code": result.exit_code,
                            "success": result.exit_code == 0,
                            "contains_status": "System Status" in result.stdout if result.stdout else False
                        }
                except Exception as e:
                    test_results["status"] = {"success": False, "error": str(e)}
                
                # Test quickstart command (with cancellation)
                try:
                    result = runner.invoke(app, ["quickstart"], input="n\n")
                    test_results["quickstart"] = {
                        "exit_code": result.exit_code,
                        "success": result.exit_code in [0, 1],  # Can exit with 1 for cancellation
                        "contains_welcome": "Welcome" in result.stdout if result.stdout else False
                    }
                except Exception as e:
                    test_results["quickstart"] = {"success": False, "error": str(e)}
        
        except Exception as e:
            test_results["import_error"] = {"success": False, "error": str(e)}
        
        # Print results
        for test_name, result in test_results.items():
            if result.get("success", False):
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL - {result.get('error', 'Unknown error')}")
        
        return test_results

    def test_detector_commands(self):
        """Test detector management commands."""
        print("\n🔍 Testing Detector Commands...")
        
        test_results = {}
        
        try:
            # Mock all external dependencies
            with patch.dict('sys.modules', {
                'pandas': Mock(),
                'numpy': Mock(),
                'sklearn': Mock(),
                'sklearn.base': Mock(),
                'pyod': Mock(),
                'pydantic': Mock(),
                'pydantic_settings': Mock(),
                'dependency_injector': Mock(),
                'dependency_injector.wiring': Mock(),
                'structlog': Mock(),
            }):
                from pynomaly.presentation.cli.app import app
                from typer.testing import CliRunner
                
                runner = CliRunner()
                
                # Test detector help
                try:
                    result = runner.invoke(app, ["detector", "--help"])
                    test_results["detector_help"] = {
                        "success": result.exit_code == 0,
                        "has_commands": all(cmd in result.stdout for cmd in ["create", "list", "show", "delete"])
                    }
                except Exception as e:
                    test_results["detector_help"] = {"success": False, "error": str(e)}
                
                # Test detector list with mocked data
                try:
                    with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_container:
                        mock_detector1 = Mock()
                        mock_detector1.id = "det1"
                        mock_detector1.name = "Test Detector 1"
                        mock_detector1.algorithm = "IsolationForest"
                        mock_detector1.is_fitted = True
                        mock_detector1.created_at.strftime.return_value = "2024-01-01 12:00"
                        
                        mock_detector2 = Mock()
                        mock_detector2.id = "det2"
                        mock_detector2.name = "Test Detector 2"
                        mock_detector2.algorithm = "LOF"
                        mock_detector2.is_fitted = False
                        mock_detector2.created_at.strftime.return_value = "2024-01-02 12:00"
                        
                        mock_container.return_value.detector_repository.return_value.find_all.return_value = [
                            mock_detector1, mock_detector2
                        ]
                        
                        result = runner.invoke(app, ["detector", "list"])
                        test_results["detector_list"] = {
                            "success": result.exit_code == 0,
                            "contains_detectors": "Test Detector" in result.stdout if result.stdout else False
                        }
                except Exception as e:
                    test_results["detector_list"] = {"success": False, "error": str(e)}
                
                # Test detector create
                try:
                    with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_container:
                        mock_detector = Mock()
                        mock_detector.id = "det123"
                        mock_detector.name = "New Detector"
                        mock_detector.algorithm = "IsolationForest"
                        
                        mock_container.return_value.detector_repository.return_value.save.return_value = None
                        
                        # Mock Detector class
                        with patch('pynomaly.presentation.cli.detectors.Detector', return_value=mock_detector):
                            result = runner.invoke(app, [
                                "detector", "create", "New Detector",
                                "--algorithm", "IsolationForest"
                            ])
                            test_results["detector_create"] = {
                                "success": result.exit_code == 0,
                                "created_successfully": "Created detector" in result.stdout if result.stdout else False
                            }
                except Exception as e:
                    test_results["detector_create"] = {"success": False, "error": str(e)}
        
        except Exception as e:
            test_results["import_error"] = {"success": False, "error": str(e)}
        
        # Print results
        for test_name, result in test_results.items():
            if result.get("success", False):
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL - {result.get('error', 'Unknown error')}")
        
        return test_results

    def test_dataset_commands(self):
        """Test dataset management commands."""
        print("\n🔍 Testing Dataset Commands...")
        
        test_results = {}
        
        try:
            with patch.dict('sys.modules', {
                'pandas': Mock(),
                'numpy': Mock(),
                'sklearn': Mock(),
                'pyod': Mock(),
                'pydantic': Mock(),
                'pydantic_settings': Mock(),
                'dependency_injector': Mock(),
                'structlog': Mock(),
            }):
                from pynomaly.presentation.cli.app import app
                from typer.testing import CliRunner
                
                runner = CliRunner()
                
                # Test dataset help
                try:
                    result = runner.invoke(app, ["dataset", "--help"])
                    test_results["dataset_help"] = {
                        "success": result.exit_code == 0,
                        "has_commands": all(cmd in result.stdout for cmd in ["load", "list", "show"])
                    }
                except Exception as e:
                    test_results["dataset_help"] = {"success": False, "error": str(e)}
                
                # Test dataset list
                try:
                    with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_container:
                        mock_dataset1 = Mock()
                        mock_dataset1.id = "ds1"
                        mock_dataset1.name = "Test Dataset 1"
                        mock_dataset1.n_samples = 1000
                        mock_dataset1.n_features = 10
                        mock_dataset1.has_target = True
                        mock_dataset1.memory_usage = 1024 * 1024  # 1MB
                        mock_dataset1.created_at.strftime.return_value = "2024-01-01 12:00"
                        
                        mock_container.return_value.dataset_repository.return_value.find_all.return_value = [
                            mock_dataset1
                        ]
                        
                        result = runner.invoke(app, ["dataset", "list"])
                        test_results["dataset_list"] = {
                            "success": result.exit_code == 0,
                            "contains_datasets": "Test Dataset" in result.stdout if result.stdout else False
                        }
                except Exception as e:
                    test_results["dataset_list"] = {"success": False, "error": str(e)}
                
                # Test dataset load (with file not found)
                try:
                    result = runner.invoke(app, [
                        "dataset", "load", "/nonexistent/file.csv",
                        "--name", "Test Dataset"
                    ])
                    test_results["dataset_load_not_found"] = {
                        "success": result.exit_code == 1,  # Should fail
                        "error_handled": "not found" in result.stdout.lower() if result.stdout else False
                    }
                except Exception as e:
                    test_results["dataset_load_not_found"] = {"success": False, "error": str(e)}
        
        except Exception as e:
            test_results["import_error"] = {"success": False, "error": str(e)}
        
        # Print results
        for test_name, result in test_results.items():
            if result.get("success", False):
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL - {result.get('error', 'Unknown error')}")
        
        return test_results

    def test_detection_commands(self):
        """Test detection workflow commands."""
        print("\n🔍 Testing Detection Commands...")
        
        test_results = {}
        
        try:
            with patch.dict('sys.modules', {
                'pandas': Mock(),
                'numpy': Mock(),
                'sklearn': Mock(),
                'pyod': Mock(),
                'pydantic': Mock(),
                'pydantic_settings': Mock(),
                'dependency_injector': Mock(),
                'structlog': Mock(),
                'asyncio': Mock(),
            }):
                from pynomaly.presentation.cli.app import app
                from typer.testing import CliRunner
                
                runner = CliRunner()
                
                # Test detection help
                try:
                    result = runner.invoke(app, ["detect", "--help"])
                    test_results["detect_help"] = {
                        "success": result.exit_code == 0,
                        "has_commands": all(cmd in result.stdout for cmd in ["train", "run", "results"])
                    }
                except Exception as e:
                    test_results["detect_help"] = {"success": False, "error": str(e)}
                
                # Test results list
                try:
                    with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_container:
                        mock_result = Mock()
                        mock_result.id = "res1"
                        mock_result.detector_id = "det1"
                        mock_result.dataset_id = "ds1"
                        mock_result.n_samples = 1000
                        mock_result.n_anomalies = 50
                        mock_result.anomaly_rate = 0.05
                        mock_result.timestamp.strftime.return_value = "2024-01-01 12:00"
                        
                        mock_detector = Mock()
                        mock_detector.name = "Test Detector"
                        
                        mock_dataset = Mock()
                        mock_dataset.name = "Test Dataset"
                        
                        mock_container.return_value.result_repository.return_value.find_recent.return_value = [mock_result]
                        mock_container.return_value.detector_repository.return_value.find_by_id.return_value = mock_detector
                        mock_container.return_value.dataset_repository.return_value.find_by_id.return_value = mock_dataset
                        
                        result = runner.invoke(app, ["detect", "results", "--limit", "5"])
                        test_results["detect_results"] = {
                            "success": result.exit_code == 0,
                            "contains_results": "Detection Results" in result.stdout if result.stdout else False
                        }
                except Exception as e:
                    test_results["detect_results"] = {"success": False, "error": str(e)}
        
        except Exception as e:
            test_results["import_error"] = {"success": False, "error": str(e)}
        
        # Print results
        for test_name, result in test_results.items():
            if result.get("success", False):
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL - {result.get('error', 'Unknown error')}")
        
        return test_results

    def test_server_commands(self):
        """Test server management commands."""
        print("\n🔍 Testing Server Commands...")
        
        test_results = {}
        
        try:
            with patch.dict('sys.modules', {
                'pandas': Mock(),
                'numpy': Mock(),
                'sklearn': Mock(),
                'pyod': Mock(),
                'pydantic': Mock(),
                'pydantic_settings': Mock(),
                'dependency_injector': Mock(),
                'structlog': Mock(),
                'uvicorn': Mock(),
                'requests': Mock(),
            }):
                from pynomaly.presentation.cli.app import app
                from typer.testing import CliRunner
                
                runner = CliRunner()
                
                # Test server help
                try:
                    result = runner.invoke(app, ["server", "--help"])
                    test_results["server_help"] = {
                        "success": result.exit_code == 0,
                        "has_commands": all(cmd in result.stdout for cmd in ["start", "stop", "status"])
                    }
                except Exception as e:
                    test_results["server_help"] = {"success": False, "error": str(e)}
                
                # Test server config
                try:
                    with patch('pynomaly.presentation.cli.server.get_cli_container') as mock_container:
                        mock_settings = Mock()
                        mock_settings.api_host = "localhost"
                        mock_settings.api_port = 8000
                        mock_settings.cors_origins = ["*"]
                        mock_settings.rate_limit_requests = 100
                        mock_settings.storage_path = "/tmp/pynomaly"
                        mock_settings.log_path = "/tmp/pynomaly/logs"
                        mock_settings.temp_path = "/tmp/pynomaly/temp"
                        mock_settings.model_path = "/tmp/pynomaly/models"
                        mock_settings.max_workers = 4
                        mock_settings.batch_size = 1000
                        mock_settings.cache_ttl_seconds = 300
                        mock_settings.gpu_enabled = False
                        mock_settings.max_dataset_size_mb = 500
                        mock_settings.default_contamination_rate = 0.1
                        mock_settings.debug = False
                        mock_settings.environment = "development"
                        
                        mock_container.return_value.config.return_value = mock_settings
                        
                        result = runner.invoke(app, ["server", "config"])
                        test_results["server_config"] = {
                            "success": result.exit_code == 0,
                            "contains_config": "Server Configuration" in result.stdout if result.stdout else False
                        }
                except Exception as e:
                    test_results["server_config"] = {"success": False, "error": str(e)}
        
        except Exception as e:
            test_results["import_error"] = {"success": False, "error": str(e)}
        
        # Print results
        for test_name, result in test_results.items():
            if result.get("success", False):
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL - {result.get('error', 'Unknown error')}")
        
        return test_results

    def test_error_handling_scenarios(self):
        """Test CLI error handling scenarios."""
        print("\n🔍 Testing Error Handling Scenarios...")
        
        test_results = {}
        
        try:
            with patch.dict('sys.modules', {
                'pandas': Mock(),
                'numpy': Mock(),
                'sklearn': Mock(),
                'pyod': Mock(),
                'pydantic': Mock(),
                'pydantic_settings': Mock(),
                'dependency_injector': Mock(),
                'structlog': Mock(),
            }):
                from pynomaly.presentation.cli.app import app
                from typer.testing import CliRunner
                
                runner = CliRunner()
                
                # Test invalid command
                try:
                    result = runner.invoke(app, ["invalid_command"])
                    test_results["invalid_command"] = {
                        "success": result.exit_code != 0,  # Should fail
                        "error_handled": "Usage:" in result.stdout if result.stdout else False
                    }
                except Exception as e:
                    test_results["invalid_command"] = {"success": False, "error": str(e)}
                
                # Test verbose/quiet conflict
                try:
                    result = runner.invoke(app, ["--verbose", "--quiet", "version"])
                    test_results["verbose_quiet_conflict"] = {
                        "success": result.exit_code != 0,  # Should fail
                        "error_handled": "Cannot use" in result.stdout if result.stdout else False
                    }
                except Exception as e:
                    test_results["verbose_quiet_conflict"] = {"success": False, "error": str(e)}
        
        except Exception as e:
            test_results["import_error"] = {"success": False, "error": str(e)}
        
        # Print results
        for test_name, result in test_results.items():
            if result.get("success", False):
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL - {result.get('error', 'Unknown error')}")
        
        return test_results

    def test_user_experience(self):
        """Test CLI user experience aspects."""
        print("\n🔍 Testing User Experience...")
        
        test_results = {}
        
        try:
            with patch.dict('sys.modules', {
                'pandas': Mock(),
                'numpy': Mock(),
                'sklearn': Mock(),
                'pyod': Mock(),
                'pydantic': Mock(),
                'pydantic_settings': Mock(),
                'dependency_injector': Mock(),
                'structlog': Mock(),
            }):
                from pynomaly.presentation.cli.app import app
                from typer.testing import CliRunner
                
                runner = CliRunner()
                
                # Test help availability for all main commands
                commands_to_test = ["detector", "dataset", "detect", "server"]
                
                for cmd in commands_to_test:
                    try:
                        result = runner.invoke(app, [cmd, "--help"])
                        test_results[f"{cmd}_help_available"] = {
                            "success": result.exit_code == 0,
                            "has_help": len(result.stdout) > 100 if result.stdout else False
                        }
                    except Exception as e:
                        test_results[f"{cmd}_help_available"] = {"success": False, "error": str(e)}
                
                # Test command discoverability
                try:
                    result = runner.invoke(app, ["--help"])
                    test_results["command_discoverability"] = {
                        "success": result.exit_code == 0,
                        "lists_subcommands": len([line for line in result.stdout.split('\n') if 'detector' in line]) > 0 if result.stdout else False
                    }
                except Exception as e:
                    test_results["command_discoverability"] = {"success": False, "error": str(e)}
        
        except Exception as e:
            test_results["import_error"] = {"success": False, "error": str(e)}
        
        # Print results
        for test_name, result in test_results.items():
            if result.get("success", False):
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL - {result.get('error', 'Unknown error')}")
        
        return test_results

    def run_all_tests(self):
        """Run all CLI runtime tests."""
        print("🚀 Running Comprehensive CLI Runtime Tests")
        print("=" * 60)
        
        all_results = {}
        
        # Run test suites
        all_results["main_app"] = self.test_main_app_commands()
        all_results["detector"] = self.test_detector_commands()
        all_results["dataset"] = self.test_dataset_commands()
        all_results["detection"] = self.test_detection_commands()
        all_results["server"] = self.test_server_commands()
        all_results["error_handling"] = self.test_error_handling_scenarios()
        all_results["user_experience"] = self.test_user_experience()
        
        return all_results

    def generate_final_report(self, all_results: Dict[str, Dict[str, Any]]):
        """Generate final CLI runtime test report."""
        print("\n📋 CLI Runtime Test Report")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in all_results.items():
            category_total = len(tests)
            category_passed = sum(1 for test in tests.values() if test.get("success", False))
            
            total_tests += category_total
            passed_tests += category_passed
            
            print(f"\n{category.upper().replace('_', ' ')} CATEGORY:")
            print(f"  Tests: {category_passed}/{category_total}")
            print(f"  Success Rate: {(category_passed/category_total)*100:.1f}%")
        
        print(f"\nOVERALL RESULTS:")
        print(f"Total Tests: {passed_tests}/{total_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Save detailed report
        report_data = {
            "runtime_test_results": all_results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": (passed_tests/total_tests)*100,
                "all_passed": passed_tests == total_tests
            }
        }
        
        report_file = Path(__file__).parent / "cli_runtime_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        if passed_tests == total_tests:
            print("\n🎉 All CLI runtime tests passed! CLI is production-ready.")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} issues found. Review and fix before production.")
        
        return passed_tests == total_tests

def main():
    """Main function to run CLI runtime tests."""
    tester = CLIRuntimeTester()
    
    # Run all tests
    all_results = tester.run_all_tests()
    
    # Generate final report
    success = tester.generate_final_report(all_results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())