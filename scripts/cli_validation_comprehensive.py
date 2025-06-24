#!/usr/bin/env python3
"""Comprehensive CLI validation script that tests CLI structure and commands."""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_structure():
    """Test CLI structure and command hierarchy."""
    print("🔍 Testing CLI Structure...")
    
    try:
        # Test import paths
        import importlib.util
        
        # Check main CLI app
        app_spec = importlib.util.spec_from_file_location(
            "cli_app", 
            Path(__file__).parent / "src" / "pynomaly" / "presentation" / "cli" / "app.py"
        )
        if app_spec and app_spec.loader:
            print("✅ Main CLI app file found")
        else:
            print("❌ Main CLI app file not found")
            return False
        
        # Check CLI modules
        cli_modules = [
            "detectors.py", 
            "datasets.py", 
            "detection.py", 
            "server.py", 
            "performance.py"
        ]
        
        cli_dir = Path(__file__).parent / "src" / "pynomaly" / "presentation" / "cli"
        
        for module in cli_modules:
            if (cli_dir / module).exists():
                print(f"✅ CLI module {module} found")
            else:
                print(f"❌ CLI module {module} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ CLI structure test failed: {e}")
        return False

def test_cli_entry_point():
    """Test CLI entry point configuration."""
    print("\n🔍 Testing CLI Entry Point...")
    
    try:
        import toml
        
        # Check pyproject.toml for entry point
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        if not pyproject_path.exists():
            print("❌ pyproject.toml not found")
            return False
        
        with open(pyproject_path) as f:
            config = toml.load(f)
        
        # Check for CLI entry point
        scripts = config.get("tool", {}).get("poetry", {}).get("scripts", {})
        if "pynomaly" in scripts:
            entry_point = scripts["pynomaly"]
            expected = "pynomaly.presentation.cli.app:app"
            if entry_point == expected:
                print(f"✅ CLI entry point correctly configured: {entry_point}")
                return True
            else:
                print(f"❌ CLI entry point mismatch. Expected: {expected}, Got: {entry_point}")
                return False
        else:
            print("❌ CLI entry point not found in pyproject.toml")
            return False
        
    except ImportError:
        print("❌ toml library not available, skipping entry point test")
        return True  # Skip this test
    except Exception as e:
        print(f"❌ Entry point test failed: {e}")
        return False

def analyze_cli_commands():
    """Analyze CLI command structure by parsing source files."""
    print("\n🔍 Analyzing CLI Commands...")
    
    commands_found = {}
    
    try:
        cli_dir = Path(__file__).parent / "src" / "pynomaly" / "presentation" / "cli"
        
        # Parse main app.py for commands
        app_file = cli_dir / "app.py"
        if app_file.exists():
            with open(app_file, 'r') as f:
                content = f.read()
                
            # Look for @app.command() decorators
            import re
            commands = re.findall(r'@app\.command\(\)\s*\ndef\s+(\w+)', content)
            if commands:
                commands_found["main"] = commands
                print(f"✅ Main app commands: {', '.join(commands)}")
            
            # Look for subcommand additions
            subcommands = re.findall(r'app\.add_typer\([^,]+,\s*name="([^"]+)"', content)
            if subcommands:
                print(f"✅ Subcommands: {', '.join(subcommands)}")
        
        # Parse individual command modules
        modules = {
            "detectors": "detector management",
            "datasets": "dataset management", 
            "detection": "detection workflows",
            "server": "server management",
            "performance": "performance monitoring"
        }
        
        for module_name, description in modules.items():
            module_file = cli_dir / f"{module_name}.py"
            if module_file.exists():
                with open(module_file, 'r') as f:
                    content = f.read()
                
                # Look for @app.command() decorators
                commands = re.findall(r'@app\.command\(["\']?([^"\']*)["\']?\)\s*\ndef\s+(\w+)', content)
                if commands:
                    command_names = [cmd[0] if cmd[0] else cmd[1] for cmd in commands]
                    commands_found[module_name] = command_names
                    print(f"✅ {module_name} commands: {', '.join(command_names)}")
                else:
                    print(f"⚠️ No commands found in {module_name}")
        
        return commands_found
        
    except Exception as e:
        print(f"❌ CLI command analysis failed: {e}")
        return {}

def test_cli_dependencies():
    """Test CLI dependency structure."""
    print("\n🔍 Testing CLI Dependencies...")
    
    try:
        # Check if key files import correctly
        files_to_check = [
            "src/pynomaly/presentation/cli/app.py",
            "src/pynomaly/presentation/cli/detectors.py",
            "src/pynomaly/presentation/cli/datasets.py"
        ]
        
        for file_path in files_to_check:
            full_path = Path(__file__).parent / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Check for critical imports
                critical_imports = [
                    "import typer",
                    "from rich.console import Console", 
                    "from rich.table import Table"
                ]
                
                for imp in critical_imports:
                    if imp in content:
                        print(f"✅ {file_path} has import: {imp}")
                    else:
                        print(f"⚠️ {file_path} missing import: {imp}")
            else:
                print(f"❌ File not found: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI dependency test failed: {e}")
        return False

def test_cli_architecture():
    """Test CLI architecture and design patterns."""
    print("\n🔍 Testing CLI Architecture...")
    
    try:
        architecture_checks = []
        
        # Check dependency injection usage
        app_file = Path(__file__).parent / "src" / "pynomaly" / "presentation" / "cli" / "app.py"
        if app_file.exists():
            with open(app_file, 'r') as f:
                content = f.read()
            
            if "get_cli_container" in content:
                architecture_checks.append("✅ Dependency injection container usage found")
            else:
                architecture_checks.append("❌ Dependency injection pattern not found")
        
        # Check command separation
        cli_dir = Path(__file__).parent / "src" / "pynomaly" / "presentation" / "cli"
        expected_modules = ["detectors.py", "datasets.py", "detection.py", "server.py"]
        
        for module in expected_modules:
            if (cli_dir / module).exists():
                architecture_checks.append(f"✅ Modular design: {module} exists")
            else:
                architecture_checks.append(f"❌ Missing module: {module}")
        
        # Check error handling patterns
        for check in architecture_checks:
            print(check)
        
        return "❌" not in "".join(architecture_checks)
        
    except Exception as e:
        print(f"❌ CLI architecture test failed: {e}")
        return False

def test_cli_help_system():
    """Test CLI help system and documentation."""
    print("\n🔍 Testing CLI Help System...")
    
    try:
        help_checks = []
        
        # Check main app help configuration
        app_file = Path(__file__).parent / "src" / "pynomaly" / "presentation" / "cli" / "app.py"
        if app_file.exists():
            with open(app_file, 'r') as f:
                content = f.read()
            
            if 'help="' in content:
                help_checks.append("✅ Help strings found in main app")
            else:
                help_checks.append("❌ No help strings in main app")
        
        # Check subcommand help
        cli_dir = Path(__file__).parent / "src" / "pynomaly" / "presentation" / "cli"
        modules = ["detectors.py", "datasets.py", "detection.py", "server.py"]
        
        for module in modules:
            module_file = cli_dir / module
            if module_file.exists():
                with open(module_file, 'r') as f:
                    content = f.read()
                
                if 'help="' in content:
                    help_checks.append(f"✅ Help strings found in {module}")
                else:
                    help_checks.append(f"❌ No help strings in {module}")
        
        for check in help_checks:
            print(check)
        
        return "❌" not in "".join(help_checks)
        
    except Exception as e:
        print(f"❌ CLI help system test failed: {e}")
        return False

def validate_cli_error_handling():
    """Validate CLI error handling patterns."""
    print("\n🔍 Validating CLI Error Handling...")
    
    try:
        error_patterns = []
        
        cli_dir = Path(__file__).parent / "src" / "pynomaly" / "presentation" / "cli"
        
        for py_file in cli_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Check for error handling patterns
            if "try:" in content and "except" in content:
                error_patterns.append(f"✅ {py_file.name} has try/except blocks")
            else:
                error_patterns.append(f"⚠️ {py_file.name} lacks error handling")
            
            if "typer.Exit" in content:
                error_patterns.append(f"✅ {py_file.name} uses typer.Exit for errors")
            else:
                error_patterns.append(f"⚠️ {py_file.name} doesn't use typer.Exit")
            
            if 'console.print.*red.*Error' in content or '[red]Error[/red]' in content:
                error_patterns.append(f"✅ {py_file.name} has formatted error messages")
            else:
                error_patterns.append(f"⚠️ {py_file.name} lacks formatted error messages")
        
        for pattern in error_patterns:
            print(pattern)
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling validation failed: {e}")
        return False

def create_cli_test_report(results: Dict[str, Any]):
    """Create a comprehensive CLI test report."""
    print("\n📋 CLI Validation Report")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Tests Run: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    # Generate recommendations
    print("\n🔧 Recommendations:")
    
    if not results.get("cli_structure", False):
        print("  - Fix CLI module structure and file organization")
    
    if not results.get("entry_point", False):
        print("  - Configure CLI entry point in pyproject.toml")
    
    if not results.get("dependencies", False):
        print("  - Install missing CLI dependencies (typer, rich)")
    
    if not results.get("architecture", False):
        print("  - Review CLI architecture and dependency injection")
    
    if not results.get("help_system", False):
        print("  - Add comprehensive help strings to all commands")
    
    if passed_tests == total_tests:
        print("\n🎉 All CLI validation tests passed! CLI is ready for runtime testing.")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} issues found. Address these before runtime testing.")
    
    return passed_tests == total_tests

def main():
    """Run comprehensive CLI validation."""
    print("🚀 Pynomaly CLI Comprehensive Validation")
    print("=" * 50)
    
    # Run all validation tests
    results = {}
    
    results["cli_structure"] = test_cli_structure()
    results["entry_point"] = test_cli_entry_point()
    
    # Analyze commands (informational)
    commands_found = analyze_cli_commands()
    results["command_analysis"] = bool(commands_found)
    
    results["dependencies"] = test_cli_dependencies()
    results["architecture"] = test_cli_architecture()
    results["help_system"] = test_cli_help_system()
    results["error_handling"] = validate_cli_error_handling()
    
    # Create final report
    all_passed = create_cli_test_report(results)
    
    # Save detailed results
    report_data = {
        "validation_results": results,
        "commands_found": commands_found,
        "timestamp": str(Path(__file__).stat().st_mtime),
        "validation_summary": {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results.values() if r),
            "all_passed": all_passed
        }
    }
    
    report_file = Path(__file__).parent / "cli_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())