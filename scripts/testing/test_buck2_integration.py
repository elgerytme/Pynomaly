#!/usr/bin/env python3
"""Buck2 Integration Test Script for Phase 2 validation.

This script tests the Buck2 + Hatch integration by validating:
1. Buck2 installation and accessibility
2. Buck2 configuration files
3. Build target validation
4. Basic build operations
5. Hatch integration readiness
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

class Buck2IntegrationTester:
    """Test Buck2 integration with Pynomaly project."""
    
    def __init__(self):
        self.root_path = Path.cwd()
        self.buck2_executable = "/mnt/c/Users/andre/buck2.exe"
        self.test_results = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete Buck2 integration test suite."""
        print("🚀 Buck2 + Hatch Integration Test Suite")
        print("=" * 50)
        
        test_methods = [
            ("Buck2 Installation Test", self.test_buck2_installation),
            ("Buck2 Configuration Test", self.test_buck2_configuration),
            ("Build Target Discovery", self.test_build_target_discovery),
            ("Simple Build Test", self.test_simple_build),
            ("Clean Architecture Validation", self.test_clean_architecture),
            ("Web Assets Pipeline", self.test_web_assets_pipeline),
            ("Hatch Integration Readiness", self.test_hatch_integration_readiness)
        ]
        
        results = {}
        total_start = time.time()
        
        for test_name, test_method in test_methods:
            print(f"\n📋 {test_name}")
            print("-" * 40)
            
            start_time = time.time()
            try:
                result = test_method()
                execution_time = time.time() - start_time
                
                results[test_name] = {
                    "status": "success" if result.get("success", False) else "failed",
                    "result": result,
                    "execution_time": execution_time
                }
                
                if result.get("success", False):
                    print(f"✅ {test_name} - PASSED ({execution_time:.2f}s)")
                else:
                    print(f"❌ {test_name} - FAILED ({execution_time:.2f}s)")
                    if result.get("error"):
                        print(f"   Error: {result['error']}")
                        
            except Exception as e:
                execution_time = time.time() - start_time
                results[test_name] = {
                    "status": "error",
                    "result": {"error": str(e)},
                    "execution_time": execution_time
                }
                print(f"💥 {test_name} - ERROR ({execution_time:.2f}s): {e}")
        
        total_time = time.time() - total_start
        
        # Generate summary
        passed = sum(1 for r in results.values() if r["status"] == "success")
        failed = sum(1 for r in results.values() if r["status"] == "failed")
        errors = sum(1 for r in results.values() if r["status"] == "error")
        
        print(f"\n🎯 Integration Test Summary")
        print("=" * 50)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"📊 Success Rate: {passed}/{len(test_methods)} ({100*passed/len(test_methods):.1f}%)")
        
        return {
            "summary": {
                "passed": passed,
                "failed": failed, 
                "errors": errors,
                "total_time": total_time,
                "success_rate": passed / len(test_methods)
            },
            "results": results
        }
    
    def test_buck2_installation(self) -> Dict[str, Any]:
        """Test Buck2 installation and basic functionality."""
        try:
            # Test Buck2 version
            result = subprocess.run(
                [self.buck2_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"   Buck2 Version: {version}")
                
                return {
                    "success": True,
                    "version": version,
                    "executable": self.buck2_executable
                }
            else:
                return {
                    "success": False,
                    "error": f"Buck2 returned error code {result.returncode}",
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Buck2 command timed out"}
        except FileNotFoundError:
            return {"success": False, "error": "Buck2 executable not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_buck2_configuration(self) -> Dict[str, Any]:
        """Test Buck2 configuration files."""
        config_files = {
            ".buckconfig": self.root_path / ".buckconfig",
            "BUCK": self.root_path / "BUCK",
            "toolchains/BUCK": self.root_path / "toolchains" / "BUCK"
        }
        
        results = {}
        all_present = True
        
        for name, path in config_files.items():
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            
            results[name] = {
                "exists": exists,
                "size": size,
                "path": str(path)
            }
            
            if exists:
                print(f"   ✅ {name}: {size} bytes")
            else:
                print(f"   ❌ {name}: Missing")
                all_present = False
        
        return {
            "success": all_present,
            "config_files": results
        }
    
    def test_build_target_discovery(self) -> Dict[str, Any]:
        """Test Buck2 build target discovery."""
        try:
            # Query available targets
            result = subprocess.run(
                [self.buck2_executable, "query", "//..."],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.root_path
            )
            
            if result.returncode == 0:
                targets = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                
                # Expected key targets
                expected_targets = [
                    "//:pynomaly-lib",
                    "//:pynomaly-cli",
                    "//:pynomaly-api",
                    "//:pynomaly-web",
                    "//:domain",
                    "//:application",
                    "//:infrastructure",
                    "//:presentation"
                ]
                
                found_targets = []
                missing_targets = []
                
                for target in expected_targets:
                    if target in targets:
                        found_targets.append(target)
                        print(f"   ✅ {target}")
                    else:
                        missing_targets.append(target)
                        print(f"   ❌ {target} (missing)")
                
                return {
                    "success": len(missing_targets) == 0,
                    "total_targets": len(targets),
                    "found_targets": found_targets,
                    "missing_targets": missing_targets,
                    "all_targets": targets[:10]  # First 10 for brevity
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to query targets",
                    "stderr": result.stderr
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_simple_build(self) -> Dict[str, Any]:
        """Test simple Buck2 build operation."""
        try:
            # Try building the domain layer (should be simplest)
            target = "//:domain"
            
            print(f"   Building {target}...")
            result = subprocess.run(
                [self.buck2_executable, "build", target],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.root_path
            )
            
            if result.returncode == 0:
                print(f"   ✅ Build successful")
                
                # Check for build artifacts
                buck_out_dir = self.root_path / "buck-out"
                artifacts_found = buck_out_dir.exists()
                
                return {
                    "success": True,
                    "target": target,
                    "artifacts_created": artifacts_found,
                    "build_output": result.stdout[:500] if result.stdout else ""
                }
            else:
                return {
                    "success": False,
                    "error": f"Build failed with code {result.returncode}",
                    "stderr": result.stderr[:500] if result.stderr else "",
                    "stdout": result.stdout[:500] if result.stdout else ""
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_clean_architecture(self) -> Dict[str, Any]:
        """Test clean architecture layer dependencies."""
        layer_tests = [
            ("//:domain", "Domain layer should build independently"),
            ("//:application", "Application layer should build with domain"),
            ("//:infrastructure", "Infrastructure layer should build with domain+application"),
            ("//:presentation", "Presentation layer should build with all layers")
        ]
        
        results = {}
        all_success = True
        
        for target, description in layer_tests:
            try:
                print(f"   Testing {target}...")
                result = subprocess.run(
                    [self.buck2_executable, "build", target, "--dry-run"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.root_path
                )
                
                success = result.returncode == 0
                results[target] = {
                    "success": success,
                    "description": description
                }
                
                if success:
                    print(f"     ✅ {description}")
                else:
                    print(f"     ❌ {description}")
                    all_success = False
                    
            except Exception as e:
                results[target] = {
                    "success": False,
                    "description": description,
                    "error": str(e)
                }
                all_success = False
        
        return {
            "success": all_success,
            "layer_results": results
        }
    
    def test_web_assets_pipeline(self) -> Dict[str, Any]:
        """Test web assets build pipeline."""
        web_targets = [
            "//:tailwind-build",
            "//:pynomaly-js",
            "//:web-assets"
        ]
        
        results = {}
        any_success = False
        
        for target in web_targets:
            try:
                print(f"   Testing {target}...")
                result = subprocess.run(
                    [self.buck2_executable, "build", target, "--dry-run"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.root_path
                )
                
                success = result.returncode == 0
                results[target] = {"success": success}
                
                if success:
                    print(f"     ✅ {target} - Ready")
                    any_success = True
                else:
                    print(f"     ⚠️  {target} - Not ready")
                    
            except Exception as e:
                results[target] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "success": any_success,  # Web assets are optional for Phase 2
            "web_targets": results,
            "note": "Web assets pipeline is optional for basic integration"
        }
    
    def test_hatch_integration_readiness(self) -> Dict[str, Any]:
        """Test Hatch integration readiness."""
        try:
            # Test basic Hatch functionality
            print("   Testing Hatch version...")
            hatch_result = subprocess.run(
                ["python3", "-m", "hatch", "version"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.root_path
            )
            
            hatch_works = hatch_result.returncode == 0
            
            # Check for Buck2 plugin
            plugin_path = self.root_path / "hatch_buck2_plugin"
            plugin_exists = plugin_path.exists()
            
            # Check pyproject.toml configuration
            pyproject_path = self.root_path / "pyproject.toml"
            pyproject_exists = pyproject_path.exists()
            
            buck2_config_ready = False
            if pyproject_exists:
                content = pyproject_path.read_text()
                buck2_config_ready = "hatch.build.hooks.buck2" in content
            
            print(f"   ✅ Hatch working: {hatch_works}")
            print(f"   ✅ Buck2 plugin exists: {plugin_exists}")
            print(f"   ✅ Buck2 config ready: {buck2_config_ready}")
            
            return {
                "success": hatch_works and plugin_exists,
                "hatch_working": hatch_works,
                "plugin_exists": plugin_exists,
                "config_ready": buck2_config_ready
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """Run Buck2 integration tests."""
    tester = Buck2IntegrationTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results["summary"]["success_rate"] >= 0.8:  # 80% pass rate
        print(f"\n🎉 Buck2 Integration Phase 2: READY FOR DEPLOYMENT")
        sys.exit(0)
    else:
        print(f"\n⚠️  Buck2 Integration Phase 2: NEEDS ATTENTION")
        sys.exit(1)

if __name__ == "__main__":
    main()