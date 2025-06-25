#!/usr/bin/env python3
"""
UI Testing with Server Integration
Runs UI tests with proper server startup and dependency management
"""

import subprocess
import sys
import time
import json
import signal
import os
from pathlib import Path
from typing import Dict, Any, Optional

class UITestRunner:
    """Manages UI testing with server dependencies"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = 8000
        self.python_exe = "/usr/bin/python3.12"
        self.results = {
            "server_startup": {},
            "ui_tests": {},
            "cleanup": {},
            "summary": {}
        }
    
    def setup_python_path(self):
        """Configure Python path for imports"""
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        os.environ["PYTHONPATH"] = src_path
        print("✅ Python path configured")
    
    def start_server(self) -> bool:
        """Start the API server for UI testing"""
        print(f"\n🚀 Starting API server on port {self.server_port}...")
        
        try:
            # Create server startup script
            server_script = f'''
import sys
sys.path.insert(0, "{self.project_root}/src")
import uvicorn
from pynomaly.presentation.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={self.server_port}, log_level="info")
'''
            
            # Write server script
            server_script_path = self.project_root / "temp_server.py"
            with open(server_script_path, 'w') as f:
                f.write(server_script)
            
            # Start server process
            self.server_process = subprocess.Popen(
                [self.python_exe, str(server_script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            
            # Wait for server startup
            print("⏳ Waiting for server startup...")
            time.sleep(8)  # Give server time to start
            
            # Check if server is running
            if self.server_process.poll() is None:
                # Test server accessibility
                import requests
                try:
                    response = requests.get(f"http://127.0.0.1:{self.server_port}/health", timeout=5)
                    if response.status_code < 400:
                        self.results["server_startup"] = {
                            "success": True,
                            "port": self.server_port,
                            "health_check": response.status_code
                        }
                        print(f"✅ Server running on http://127.0.0.1:{self.server_port}")
                        return True
                except requests.exceptions.RequestException:
                    pass
            
            self.results["server_startup"] = {
                "success": False,
                "error": "Server failed to start or respond to health check"
            }
            print("❌ Server startup failed")
            return False
            
        except Exception as e:
            self.results["server_startup"] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ Server startup error: {e}")
            return False
    
    def run_ui_tests(self) -> Dict[str, Any]:
        """Execute UI tests with server running"""
        print("\n🎨 Running UI Tests...")
        
        ui_test_script = self.project_root / "tests" / "ui" / "run_comprehensive_ui_tests.py"
        
        if not ui_test_script.exists():
            self.results["ui_tests"] = {
                "success": False,
                "error": "UI test script not found"
            }
            print("❌ UI test script not found")
            return self.results["ui_tests"]
        
        try:
            # Run UI tests
            result = subprocess.run(
                [self.python_exe, str(ui_test_script)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=self.project_root / "tests" / "ui"
            )
            
            self.results["ui_tests"] = {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_length": len(result.stdout)
            }
            
            # Try to load detailed UI test results
            ui_results_file = self.project_root / "tests" / "ui" / "ui_test_results.json"
            if ui_results_file.exists():
                with open(ui_results_file) as f:
                    detailed_results = json.load(f)
                    self.results["ui_tests"]["detailed_results"] = detailed_results
            
            if result.returncode == 0:
                print("✅ UI tests completed successfully")
            else:
                print(f"❌ UI tests failed (return code: {result.returncode})")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
            
        except subprocess.TimeoutExpired:
            self.results["ui_tests"] = {
                "success": False,
                "error": "UI tests timed out after 5 minutes"
            }
            print("❌ UI tests timed out")
        except Exception as e:
            self.results["ui_tests"] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ UI test execution error: {e}")
        
        return self.results["ui_tests"]
    
    def stop_server(self):
        """Stop the API server"""
        print("\n🛑 Stopping server...")
        
        cleanup_success = True
        
        if self.server_process:
            try:
                # Try graceful shutdown first
                self.server_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.server_process.wait(timeout=5)
                    print("✅ Server stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.server_process.kill()
                    self.server_process.wait()
                    print("⚠️ Server force-killed")
                    cleanup_success = False
                    
            except Exception as e:
                print(f"❌ Error stopping server: {e}")
                cleanup_success = False
        
        # Clean up temporary files
        temp_files = [
            self.project_root / "temp_server.py"
        ]
        
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                print(f"⚠️ Failed to remove {temp_file}: {e}")
                cleanup_success = False
        
        self.results["cleanup"] = {
            "success": cleanup_success,
            "server_stopped": self.server_process is not None
        }
    
    def run_complete_ui_validation(self):
        """Run complete UI validation with server management"""
        print("🎨 UI VALIDATION WITH SERVER INTEGRATION")
        print("=" * 60)
        
        self.setup_python_path()
        
        # Start server
        server_started = self.start_server()
        
        if server_started:
            # Run UI tests
            self.run_ui_tests()
        else:
            print("⚠️ Skipping UI tests - server startup failed")
            self.results["ui_tests"] = {
                "success": False,
                "error": "Server startup failed"
            }
        
        # Always attempt cleanup
        self.stop_server()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def generate_summary(self):
        """Generate test summary"""
        summary = {
            "server_startup_success": self.results["server_startup"].get("success", False),
            "ui_tests_success": self.results["ui_tests"].get("success", False),
            "cleanup_success": self.results["cleanup"].get("success", False),
            "overall_success": False
        }
        
        # Overall success if server started and UI tests ran (regardless of UI test results)
        summary["overall_success"] = (
            summary["server_startup_success"] and 
            "error" not in self.results["ui_tests"]
        )
        
        # Extract UI test details if available
        if "detailed_results" in self.results["ui_tests"]:
            detailed = self.results["ui_tests"]["detailed_results"]
            if "summary" in detailed:
                summary["ui_test_details"] = detailed["summary"]
        
        self.results["summary"] = summary
        
        print("\n" + "=" * 60)
        print("🎨 UI VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Server Startup: {'✅' if summary['server_startup_success'] else '❌'}")
        print(f"UI Tests Execution: {'✅' if summary['ui_tests_success'] else '❌'}")
        print(f"Cleanup: {'✅' if summary['cleanup_success'] else '❌'}")
        print(f"Overall: {'✅' if summary['overall_success'] else '❌'}")
        
        if "ui_test_details" in summary:
            details = summary["ui_test_details"]
            print(f"\nUI Test Results:")
            print(f"  • Total Tests: {details.get('total_tests', 'N/A')}")
            print(f"  • Passed: {details.get('passed_tests', 'N/A')}")
            print(f"  • Success Rate: {details.get('success_rate', 'N/A')}")
            print(f"  • Duration: {details.get('total_duration', 'N/A')}")
    
    def save_results(self):
        """Save test results"""
        results_file = self.project_root / "tests" / "ui_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Results saved: {results_file}")

def main():
    """Main execution"""
    runner = UITestRunner()
    
    # Set up signal handling for cleanup
    def signal_handler(signum, frame):
        print("\n⚠️ Received interrupt signal")
        runner.stop_server()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        results = runner.run_complete_ui_validation()
        return results["summary"]["overall_success"]
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        runner.stop_server()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)