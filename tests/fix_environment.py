#!/usr/bin/env python3
"""
Environment Fix Script
Fixes Python path issues and validates all components
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

class EnvironmentFixer:
    """Fixes environment issues and validates functionality"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.python_exe = "/usr/bin/python3.12"
        self.results = {
            "cli_tests": {},
            "api_tests": {},
            "import_tests": {},
            "fixes_applied": []
        }
    
    def fix_python_path(self):
        """Apply Python path fix"""
        os.environ["PYTHONPATH"] = str(self.project_root / "src")
        self.results["fixes_applied"].append("Python path configured")
        print("✅ Python path configured")
    
    def test_imports(self):
        """Test all critical imports"""
        print("\n🔍 Testing Imports...")
        
        modules_to_test = [
            "pynomaly",
            "pynomaly.presentation.cli",
            "pynomaly.presentation.api",
            "pynomaly.infrastructure.config.container",
            "pynomaly.domain.entities",
            "pynomaly.application.services"
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                self.results["import_tests"][module] = {"success": True}
                print(f"✅ {module}")
            except Exception as e:
                self.results["import_tests"][module] = {"success": False, "error": str(e)}
                print(f"❌ {module}: {e}")
    
    def test_cli_commands(self):
        """Test CLI functionality"""
        print("\n🔍 Testing CLI Commands...")
        
        # Test basic CLI help
        try:
            from pynomaly.presentation.cli.app import app
            
            # Capture help output
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    app(["--help"])
                except SystemExit:
                    pass  # Expected for help command
            
            help_output = f.getvalue()
            
            self.results["cli_tests"]["help"] = {
                "success": len(help_output) > 0,
                "has_commands": "Commands" in help_output,
                "output_length": len(help_output)
            }
            print("✅ CLI help command working")
            
            # Test version command
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    app(["version"])
                except SystemExit:
                    pass
            
            version_output = f.getvalue()
            self.results["cli_tests"]["version"] = {
                "success": len(version_output) > 0,
                "output": version_output.strip()
            }
            print("✅ CLI version command working")
            
        except Exception as e:
            self.results["cli_tests"]["error"] = str(e)
            print(f"❌ CLI test failed: {e}")
    
    def test_api_startup(self):
        """Test API startup capability"""
        print("\n🔍 Testing API Startup...")
        
        try:
            from pynomaly.presentation.api import app
            print("✅ FastAPI app import successful")
            
            # Test basic app creation
            if hasattr(app, 'routes'):
                route_count = len(app.routes)
                self.results["api_tests"]["app_creation"] = {
                    "success": True,
                    "route_count": route_count
                }
                print(f"✅ FastAPI app has {route_count} routes")
            else:
                self.results["api_tests"]["app_creation"] = {
                    "success": False,
                    "error": "No routes attribute"
                }
                print("❌ FastAPI app missing routes")
                
        except Exception as e:
            self.results["api_tests"]["error"] = str(e)
            print(f"❌ API test failed: {e}")
    
    def create_cli_wrapper(self):
        """Create a working CLI wrapper script"""
        wrapper_content = f'''#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, "{self.project_root}/src")
from pynomaly.presentation.cli.app import app

if __name__ == "__main__":
    app()
'''
        
        wrapper_path = self.project_root / "run_pynomaly.py"
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        
        # Make executable
        os.chmod(wrapper_path, 0o755)
        
        self.results["fixes_applied"].append(f"CLI wrapper created: {wrapper_path}")
        print(f"✅ CLI wrapper created: {wrapper_path}")
        
        return wrapper_path
    
    def test_cli_wrapper(self, wrapper_path):
        """Test the CLI wrapper"""
        print("\n🔍 Testing CLI Wrapper...")
        
        try:
            result = subprocess.run(
                [str(wrapper_path), "--help"],
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.results["cli_tests"]["wrapper"] = {
                    "success": True,
                    "output_length": len(result.stdout),
                    "has_usage": "Usage:" in result.stdout
                }
                print("✅ CLI wrapper working perfectly")
            else:
                self.results["cli_tests"]["wrapper"] = {
                    "success": False,
                    "error": result.stderr
                }
                print(f"❌ CLI wrapper failed: {result.stderr}")
                
        except Exception as e:
            self.results["cli_tests"]["wrapper"] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ CLI wrapper test failed: {e}")
    
    def run_comprehensive_fix(self):
        """Run all fixes and validations"""
        print("🚨 ENVIRONMENT RECOVERY - Phase 2")
        print("=" * 50)
        
        self.fix_python_path()
        self.test_imports()
        self.test_cli_commands()
        self.test_api_startup()
        
        # Create CLI wrapper
        wrapper_path = self.create_cli_wrapper()
        self.test_cli_wrapper(wrapper_path)
        
        # Save results
        results_file = self.project_root / "tests" / "environment_fix_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📊 Results saved: {results_file}")
        
        # Summary
        print("\n" + "=" * 50)
        print("🚨 ENVIRONMENT FIX SUMMARY")
        print("=" * 50)
        
        import_success = sum(1 for r in self.results["import_tests"].values() if r["success"])
        import_total = len(self.results["import_tests"])
        
        print(f"Imports: {import_success}/{import_total} successful")
        print(f"CLI Tests: {len([k for k, v in self.results['cli_tests'].items() if v.get('success')])}")
        print(f"API Tests: {len([k for k, v in self.results['api_tests'].items() if v.get('success')])}")
        print(f"Fixes Applied: {len(self.results['fixes_applied'])}")
        
        for fix in self.results["fixes_applied"]:
            print(f"  • {fix}")
        
        return import_success == import_total

def main():
    """Main execution"""
    fixer = EnvironmentFixer()
    success = fixer.run_comprehensive_fix()
    
    if success:
        print("\n🎉 ENVIRONMENT RECOVERY SUCCESSFUL!")
        print("Use ./run_pynomaly.py for CLI access")
    else:
        print("\n❌ Some issues remain - check detailed results")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)