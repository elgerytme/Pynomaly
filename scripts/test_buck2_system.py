#!/usr/bin/env python3
"""
Buck2 System Validation and Testing Script
Comprehensive test suite to validate the Buck2 incremental testing system.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Buck2SystemTester:
    """Comprehensive tester for the Buck2 incremental testing system."""
    
    def __init__(self, repo_root: Path = None, dry_run: bool = False):
        self.repo_root = repo_root or Path.cwd()
        self.dry_run = dry_run
        self.test_results = {}
        
        # Validate that all required scripts exist
        self.required_scripts = [
            "scripts/buck2_change_detector.py",
            "scripts/buck2_incremental_test.py",
            "scripts/buck2_git_integration.py",
            "scripts/buck2_impact_analyzer.py",
            "scripts/buck2_workflow.py",
        ]
        
        self.validate_prerequisites()
    
    def validate_prerequisites(self):
        """Validate that all required files and tools are available."""
        logger.info("Validating prerequisites...")
        
        # Check required scripts
        for script in self.required_scripts:
            script_path = self.repo_root / script
            if not script_path.exists():
                raise FileNotFoundError(f"Required script not found: {script}")
            logger.debug(f"✓ Found script: {script}")
        
        # Check Buck2 configuration
        buck_file = self.repo_root / "BUCK"
        buckconfig_file = self.repo_root / ".buckconfig"
        
        if not buck_file.exists():
            raise FileNotFoundError("BUCK file not found")
        if not buckconfig_file.exists():
            raise FileNotFoundError(".buckconfig file not found")
        
        logger.info("✓ All prerequisites validated")
    
    def test_change_detector(self) -> Dict:
        """Test the Buck2 change detector functionality."""
        logger.info("Testing Buck2 change detector...")
        
        test_result = {
            "test_name": "change_detector",
            "status": "pending",
            "details": {}
        }
        
        try:
            # Test 1: Basic change detection
            cmd = ["python", "scripts/buck2_change_detector.py", "--format", "json", "--output", "test_change_analysis.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                # Check if output file was created
                output_file = self.repo_root / "test_change_analysis.json"
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        analysis_data = json.load(f)
                    
                    test_result["details"]["basic_detection"] = {
                        "status": "passed",
                        "files_changed": len(analysis_data.get("changed_files", [])),
                        "targets_affected": len(analysis_data.get("affected_targets", [])),
                    }
                    
                    # Cleanup
                    output_file.unlink()
                else:
                    test_result["details"]["basic_detection"] = {
                        "status": "failed",
                        "error": "Output file not created"
                    }
            else:
                test_result["details"]["basic_detection"] = {
                    "status": "failed",
                    "error": result.stderr
                }
            
            # Test 2: Summary format
            cmd = ["python", "scripts/buck2_change_detector.py", "--format", "summary"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            test_result["details"]["summary_format"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output_length": len(result.stdout),
                "error": result.stderr if result.returncode != 0 else None
            }
            
            # Determine overall status
            all_passed = all(
                detail.get("status") == "passed" 
                for detail in test_result["details"].values()
            )
            test_result["status"] = "passed" if all_passed else "failed"
            
        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_incremental_runner(self) -> Dict:
        """Test the Buck2 incremental test runner."""
        logger.info("Testing Buck2 incremental test runner...")
        
        test_result = {
            "test_name": "incremental_runner",
            "status": "pending",
            "details": {}
        }
        
        try:
            # Test 1: Dry run mode
            cmd = ["python", "scripts/buck2_incremental_test.py", "--dry-run", "--output", "test_results.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            test_result["details"]["dry_run"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output_length": len(result.stdout),
                "error": result.stderr if result.returncode != 0 else None
            }
            
            # Check if results file was created
            results_file = self.repo_root / "test_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                test_result["details"]["results_file"] = {
                    "status": "passed",
                    "total_targets": results_data.get("total_targets", 0),
                    "dry_run_confirmed": results_data.get("run_metadata", {}).get("dry_run", False)
                }
                
                # Cleanup
                results_file.unlink()
            else:
                test_result["details"]["results_file"] = {
                    "status": "failed",
                    "error": "Results file not created"
                }
            
            # Test 2: Verbose mode
            cmd = ["python", "scripts/buck2_incremental_test.py", "--dry-run", "--verbose"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            test_result["details"]["verbose_mode"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "has_debug_output": "DEBUG" in result.stderr or "INFO" in result.stderr,
                "error": result.stderr if result.returncode != 0 else None
            }
            
            # Determine overall status
            all_passed = all(
                detail.get("status") == "passed" 
                for detail in test_result["details"].values()
            )
            test_result["status"] = "passed" if all_passed else "failed"
            
        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_git_integration(self) -> Dict:
        """Test the Git integration functionality."""
        logger.info("Testing Git integration...")
        
        test_result = {
            "test_name": "git_integration",
            "status": "pending",
            "details": {}
        }
        
        try:
            # Test 1: Branch info
            cmd = ["python", "scripts/buck2_git_integration.py", "branch-info"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            test_result["details"]["branch_info"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "has_output": len(result.stdout) > 0,
                "error": result.stderr if result.returncode != 0 else None
            }
            
            # Test 2: Test branch (dry run equivalent)
            cmd = ["python", "scripts/buck2_git_integration.py", "test-branch", "--dry-run"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            test_result["details"]["test_branch"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output_length": len(result.stdout),
                "error": result.stderr if result.returncode != 0 else None
            }
            
            # Determine overall status
            all_passed = all(
                detail.get("status") == "passed" 
                for detail in test_result["details"].values()
            )
            test_result["status"] = "passed" if all_passed else "failed"
            
        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_impact_analyzer(self) -> Dict:
        """Test the impact analyzer functionality."""
        logger.info("Testing impact analyzer...")
        
        test_result = {
            "test_name": "impact_analyzer",
            "status": "pending",
            "details": {}
        }
        
        try:
            # Test 1: Basic impact analysis
            cmd = ["python", "scripts/buck2_impact_analyzer.py", "--format", "json", "--output", "test_impact.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                output_file = self.repo_root / "test_impact.json"
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        impact_data = json.load(f)
                    
                    test_result["details"]["basic_analysis"] = {
                        "status": "passed",
                        "has_risk_assessment": "risk_assessment" in impact_data,
                        "has_test_strategy": "test_strategy" in impact_data,
                        "risk_level": impact_data.get("risk_assessment", {}).get("level", "unknown")
                    }
                    
                    # Cleanup
                    output_file.unlink()
                else:
                    test_result["details"]["basic_analysis"] = {
                        "status": "failed",
                        "error": "Output file not created"
                    }
            else:
                test_result["details"]["basic_analysis"] = {
                    "status": "failed",
                    "error": result.stderr
                }
            
            # Test 2: Summary format
            cmd = ["python", "scripts/buck2_impact_analyzer.py", "--format", "summary"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            test_result["details"]["summary_analysis"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "has_risk_output": "Risk Assessment" in result.stdout,
                "has_strategy_output": "Test Strategy" in result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
            # Determine overall status
            all_passed = all(
                detail.get("status") == "passed" 
                for detail in test_result["details"].values()
            )
            test_result["status"] = "passed" if all_passed else "failed"
            
        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_workflow_orchestrator(self) -> Dict:
        """Test the workflow orchestrator."""
        logger.info("Testing workflow orchestrator...")
        
        test_result = {
            "test_name": "workflow_orchestrator",
            "status": "pending",
            "details": {}
        }
        
        try:
            # Test 1: Standard workflow with dry run
            cmd = ["python", "scripts/buck2_workflow.py", "standard", "--dry-run", "--strategy", "minimal"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            test_result["details"]["standard_workflow"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "has_summary": "Buck2 Workflow Summary" in result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
            # Test 2: Branch workflow
            cmd = ["python", "scripts/buck2_workflow.py", "branch", "--dry-run", "--strategy", "auto"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            test_result["details"]["branch_workflow"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "has_summary": "Buck2 Workflow Summary" in result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
            # Determine overall status
            all_passed = all(
                detail.get("status") == "passed" 
                for detail in test_result["details"].values()
            )
            test_result["status"] = "passed" if all_passed else "failed"
            
        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_file_imports(self) -> Dict:
        """Test that all Python files can be imported without errors."""
        logger.info("Testing Python imports...")
        
        test_result = {
            "test_name": "file_imports",
            "status": "pending",
            "details": {}
        }
        
        try:
            for script in self.required_scripts:
                script_name = Path(script).stem
                
                # Test import
                cmd = ["python", "-c", f"import sys; sys.path.append('scripts'); import {script_name}"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
                
                test_result["details"][script_name] = {
                    "status": "passed" if result.returncode == 0 else "failed",
                    "error": result.stderr if result.returncode != 0 else None
                }
            
            # Determine overall status
            all_passed = all(
                detail.get("status") == "passed" 
                for detail in test_result["details"].values()
            )
            test_result["status"] = "passed" if all_passed else "failed"
            
        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
        
        return test_result
    
    def run_comprehensive_test(self) -> Dict:
        """Run all tests and return comprehensive results."""
        logger.info("Starting comprehensive Buck2 system testing...")
        start_time = time.time()
        
        test_suite = [
            self.test_file_imports,
            self.test_change_detector,
            self.test_incremental_runner,
            self.test_git_integration,
            self.test_impact_analyzer,
            self.test_workflow_orchestrator,
        ]
        
        results = {
            "test_suite": "buck2_system_validation",
            "start_time": start_time,
            "tests": [],
            "summary": {}
        }
        
        for test_func in test_suite:
            try:
                test_result = test_func()
                results["tests"].append(test_result)
                logger.info(f"✓ {test_result['test_name']}: {test_result['status']}")
            except Exception as e:
                error_result = {
                    "test_name": test_func.__name__,
                    "status": "error",
                    "error": str(e)
                }
                results["tests"].append(error_result)
                logger.error(f"✗ {test_func.__name__}: error - {e}")
        
        # Calculate summary
        total_tests = len(results["tests"])
        passed_tests = sum(1 for t in results["tests"] if t["status"] == "passed")
        failed_tests = sum(1 for t in results["tests"] if t["status"] == "failed")
        error_tests = sum(1 for t in results["tests"] if t["status"] == "error")
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "duration": time.time() - start_time
        }
        
        results["overall_status"] = "passed" if failed_tests + error_tests == 0 else "failed"
        
        return results
    
    def print_test_summary(self, results: Dict):
        """Print a human-readable test summary."""
        print(f"\n=== Buck2 System Validation Results ===")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Duration: {results['summary']['duration']:.2f}s")
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")
        
        print(f"\nTest Summary:")
        print(f"  Total: {results['summary']['total_tests']}")
        print(f"  Passed: {results['summary']['passed']}")
        print(f"  Failed: {results['summary']['failed']}")
        print(f"  Errors: {results['summary']['errors']}")
        
        print(f"\nDetailed Results:")
        for test in results["tests"]:
            status_symbol = {"passed": "✓", "failed": "✗", "error": "!"}[test["status"]]
            print(f"  {status_symbol} {test['test_name']}: {test['status']}")
            
            if test["status"] in ["failed", "error"] and "error" in test:
                print(f"    Error: {test['error']}")
            
            # Show details for passed tests
            if test.get("details") and test["status"] == "passed":
                detail_count = len(test["details"])
                passed_details = sum(1 for d in test["details"].values() if d.get("status") == "passed")
                print(f"    Details: {passed_details}/{detail_count} sub-tests passed")
    
    def save_results(self, results: Dict, output_file: Path = None) -> Path:
        """Save test results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.repo_root / f"buck2_test_validation_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {output_file}")
        return output_file

def main():
    """Main entry point for Buck2 system testing."""
    parser = argparse.ArgumentParser(description="Buck2 System Validation")
    parser.add_argument("--output", type=Path, help="Output file for test results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize tester
        tester = Buck2SystemTester(dry_run=args.dry_run)
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test()
        
        # Print summary
        tester.print_test_summary(results)
        
        # Save results
        if args.output:
            tester.save_results(results, args.output)
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_status"] == "passed" else 1)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()