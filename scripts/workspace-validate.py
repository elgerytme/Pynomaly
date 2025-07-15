#!/usr/bin/env python3
"""
Workspace Validation Script

Validates workspace integrity, dependencies, and configuration.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import subprocess
import tomllib


class WorkspaceValidator:
    """Validates workspace configuration and integrity."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.workspace_config = self._load_workspace_config()
        self.errors = []
        self.warnings = []
    
    def _load_workspace_config(self) -> dict:
        """Load workspace configuration."""
        config_path = self.root_path / "workspace.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
    
    def validate_structure(self) -> bool:
        """Validate workspace directory structure."""
        print("🏗️  Validating workspace structure...")
        
        required_dirs = [
            "src/pynomaly",
            "tests",
            "docs",
            "scripts"
        ]
        
        for dir_path in required_dirs:
            full_path = self.root_path / dir_path
            if not full_path.exists():
                self.errors.append(f"Missing required directory: {dir_path}")
            elif not full_path.is_dir():
                self.errors.append(f"Path is not a directory: {dir_path}")
        
        # Check for workspace configuration
        if not (self.root_path / "workspace.json").exists():
            self.errors.append("Missing workspace.json configuration")
        
        # Check for pyproject.toml
        if not (self.root_path / "pyproject.toml").exists():
            self.errors.append("Missing pyproject.toml configuration")
        
        return len(self.errors) == 0
    
    def validate_packages(self) -> bool:
        """Validate all packages in workspace."""
        print("📦 Validating packages...")
        
        if "packages" not in self.workspace_config:
            self.errors.append("No packages defined in workspace.json")
            return False
        
        packages = self.workspace_config["packages"]
        
        for package_name, package_info in packages.items():
            package_path = Path(package_info["path"])
            full_path = self.root_path / package_path
            
            # Check package exists
            if not full_path.exists():
                self.errors.append(f"Package directory missing: {package_name} at {package_path}")
                continue
            
            # Check for __init__.py
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                self.warnings.append(f"Package missing __init__.py: {package_name}")
            
            # Check pyproject.toml if it should exist
            pyproject_file = full_path / "pyproject.toml"
            if package_info.get("type") == "package" and not pyproject_file.exists():
                self.warnings.append(f"Package missing pyproject.toml: {package_name}")
        
        return len(self.errors) == 0
    
    def validate_dependencies(self) -> bool:
        """Validate package dependencies."""
        print("🔗 Validating dependencies...")
        
        if "packages" not in self.workspace_config:
            return False
        
        packages = self.workspace_config["packages"]
        
        # Check dependency references
        for package_name, package_info in packages.items():
            dependencies = package_info.get("dependencies", [])
            
            for dep in dependencies:
                if dep not in packages and not dep.startswith("external:"):
                    self.errors.append(f"Package {package_name} depends on unknown package: {dep}")
        
        # Check for circular dependencies
        cycles = self._detect_circular_dependencies(packages)
        for cycle in cycles:
            self.errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        return len(self.errors) == 0
    
    def _detect_circular_dependencies(self, packages: Dict) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(package: str, path: List[str]):
            if package in rec_stack:
                # Found a cycle
                cycle_start = path.index(package)
                cycles.append(path[cycle_start:] + [package])
                return
            
            if package in visited:
                return
            
            visited.add(package)
            rec_stack.add(package)
            
            if package in packages:
                for dep in packages[package].get("dependencies", []):
                    if dep in packages:  # Only check internal dependencies
                        dfs(dep, path + [package])
            
            rec_stack.remove(package)
        
        for package in packages:
            if package not in visited:
                dfs(package, [])
        
        return cycles
    
    def validate_configuration(self) -> bool:
        """Validate workspace configuration files."""
        print("⚙️  Validating configuration...")
        
        # Validate pyproject.toml
        pyproject_path = self.root_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, "rb") as f:
                    tomllib.load(f)
            except Exception as e:
                self.errors.append(f"Invalid pyproject.toml: {e}")
        
        # Validate workspace.json
        if self.workspace_config:
            required_sections = ["workspace", "packages"]
            for section in required_sections:
                if section not in self.workspace_config:
                    self.errors.append(f"Missing section in workspace.json: {section}")
        
        return len(self.errors) == 0
    
    def validate_tools(self) -> bool:
        """Validate development tools and their configuration."""
        print("🔧 Validating tools...")
        
        tools_to_check = [
            ("python", "Python interpreter"),
            ("hatch", "Build system"),
            ("pytest", "Testing framework"),
            ("black", "Code formatter"),
            ("ruff", "Linter")
        ]
        
        for tool, description in tools_to_check:
            try:
                result = subprocess.run([tool, "--version"], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    self.warnings.append(f"Tool not working properly: {tool} ({description})")
            except FileNotFoundError:
                self.warnings.append(f"Tool not found: {tool} ({description})")
        
        return True  # Tool issues are warnings, not errors
    
    def validate_tests(self) -> bool:
        """Validate test structure and configuration."""
        print("🧪 Validating tests...")
        
        tests_dir = self.root_path / "tests"
        if not tests_dir.exists():
            self.errors.append("Tests directory not found")
            return False
        
        # Check for pytest configuration
        pytest_configs = [
            "pytest.ini",
            "pyproject.toml",
            "setup.cfg"
        ]
        
        has_pytest_config = any(
            (self.root_path / config).exists() for config in pytest_configs
        )
        
        if not has_pytest_config:
            self.warnings.append("No pytest configuration found")
        
        # Check for test files
        test_files = list(tests_dir.rglob("test_*.py"))
        if not test_files:
            self.warnings.append("No test files found in tests directory")
        
        return True
    
    def validate_documentation(self) -> bool:
        """Validate documentation structure."""
        print("📚 Validating documentation...")
        
        docs_dir = self.root_path / "docs"
        if not docs_dir.exists():
            self.warnings.append("Documentation directory not found")
            return True
        
        # Check for essential documentation files
        essential_docs = [
            "README.md",
            "docs/installation.md",
            "docs/getting-started/quickstart.md"
        ]
        
        for doc_path in essential_docs:
            full_path = self.root_path / doc_path
            if not full_path.exists():
                self.warnings.append(f"Missing documentation file: {doc_path}")
        
        return True
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🔍 Running workspace validation...")
        print("=" * 50)
        
        validation_steps = [
            ("Structure", self.validate_structure),
            ("Packages", self.validate_packages),
            ("Dependencies", self.validate_dependencies),
            ("Configuration", self.validate_configuration),
            ("Tools", self.validate_tools),
            ("Tests", self.validate_tests),
            ("Documentation", self.validate_documentation)
        ]
        
        all_passed = True
        
        for step_name, step_func in validation_steps:
            try:
                step_passed = step_func()
                if not step_passed:
                    all_passed = False
            except Exception as e:
                self.errors.append(f"Validation step '{step_name}' failed: {e}")
                all_passed = False
        
        return all_passed
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 50)
        print("📋 Validation Report")
        print("=" * 50)
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All validation checks passed!")
        elif not self.errors:
            print(f"\n✅ Validation passed with {len(self.warnings)} warnings")
        else:
            print(f"\n❌ Validation failed with {len(self.errors)} errors and {len(self.warnings)} warnings")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Status: {'✅ PASSED' if not self.errors else '❌ FAILED'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Pynomaly workspace")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                       help="Workspace root directory")
    parser.add_argument("--fail-on-warnings", action="store_true",
                       help="Treat warnings as errors")
    
    args = parser.parse_args()
    
    validator = WorkspaceValidator(args.root)
    
    # Run validation
    passed = validator.run_validation()
    
    # Print report
    validator.print_report()
    
    # Determine exit code
    if args.fail_on_warnings:
        exit_code = 0 if (not validator.errors and not validator.warnings) else 1
    else:
        exit_code = 0 if not validator.errors else 1
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())