#!/usr/bin/env python3
"""
Package Structure Enforcement Script
Validates and enforces consistent package structure across the monorepo.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


class PackageStructureEnforcer:
    """Enforces consistent package structure across the monorepo."""
    
    REQUIRED_PACKAGE_FILES = {
        "__init__.py": "Package initialization file",
        "pyproject.toml": "Package configuration and dependencies", 
        "BUCK": "Buck2 build configuration",
        "README.md": "Package documentation"
    }
    
    REQUIRED_PACKAGE_DIRS = {
        "{package_name}/": "Main package source code",
        "tests/": "Package test suite",
        "docs/": "Package-specific documentation"
    }
    
    ALLOWED_ROOT_FILES = {
        "README.md", "pyproject.toml", "nx.json", "workspace.json",
        "BUCK", "BUCK2_MIGRATION_COMPLETE.md", ".gitignore", ".gitattributes",
        "LICENSE", "CONTRIBUTING.md", ".pre-commit-config.yaml"
    }
    
    LAYER_DEPENDENCIES = {
        "domain": [],
        "infrastructure": ["domain"],
        "application": ["domain", "infrastructure"], 
        "presentation": ["domain", "infrastructure", "application"],
        "shared": []
    }
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.packages_dir = repo_root / "src" / "packages"
        self.violations: List[Dict] = []
        self.warnings: List[Dict] = []
        
    def validate_package_structure(self, package_path: Path) -> List[Dict]:
        """Validate a single package's structure."""
        violations = []
        package_name = package_path.name
        
        # Check required files
        for file_name, description in self.REQUIRED_PACKAGE_FILES.items():
            file_path = package_path / file_name
            if not file_path.exists():
                violations.append({
                    "type": "missing_file",
                    "package": package_name,
                    "file": file_name,
                    "description": description,
                    "severity": "error"
                })
        
        # Check required directories
        for dir_template, description in self.REQUIRED_PACKAGE_DIRS.items():
            dir_name = dir_template.format(package_name=package_name)
            dir_path = package_path / dir_name
            if not dir_path.exists():
                violations.append({
                    "type": "missing_directory", 
                    "package": package_name,
                    "directory": dir_name,
                    "description": description,
                    "severity": "error"
                })
        
        # Check for unwanted files
        unwanted_patterns = [
            "*.pyc", "__pycache__", ".pytest_cache", 
            "dist", "build", ".coverage*", "htmlcov"
        ]
        
        for pattern in unwanted_patterns:
            for unwanted in package_path.rglob(pattern):
                violations.append({
                    "type": "unwanted_file",
                    "package": package_name, 
                    "file": str(unwanted.relative_to(package_path)),
                    "pattern": pattern,
                    "severity": "warning"
                })
        
        return violations
    
    def validate_dependency_layers(self) -> List[Dict]:
        """Validate that package dependencies respect layer boundaries."""
        violations = []
        
        # Load workspace configuration to get dependencies
        workspace_path = self.repo_root / "workspace.json"
        if not workspace_path.exists():
            return violations
            
        try:
            with open(workspace_path) as f:
                workspace = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            violations.append({
                "type": "workspace_config_error",
                "description": f"Could not parse workspace.json: {e}",
                "severity": "error"
            })
            return violations
        
        packages = workspace.get("packages", {})
        
        for package_name, package_config in packages.items():
            package_layer = package_config.get("layer")
            if not package_layer:
                continue
                
            dependencies = package_config.get("dependencies", [])
            allowed_layers = self.LAYER_DEPENDENCIES.get(package_layer, [])
            
            for dep in dependencies:
                dep_config = packages.get(dep, {})
                dep_layer = dep_config.get("layer")
                
                if dep_layer and dep_layer not in allowed_layers:
                    violations.append({
                        "type": "layer_violation",
                        "package": package_name,
                        "package_layer": package_layer,
                        "dependency": dep,
                        "dependency_layer": dep_layer,
                        "allowed_layers": allowed_layers,
                        "severity": "error"
                    })
        
        return violations
    
    def validate_root_directory(self) -> List[Dict]:
        """Validate root directory organization."""
        violations = []
        
        # Check for files that shouldn't be in root
        for item in self.repo_root.iterdir():
            if item.is_file() and item.name not in self.ALLOWED_ROOT_FILES:
                violations.append({
                    "type": "misplaced_file",
                    "file": item.name,
                    "location": "root",
                    "severity": "warning",
                    "suggestion": self._suggest_location(item.name)
                })
        
        return violations
    
    def _suggest_location(self, filename: str) -> str:
        """Suggest appropriate location for misplaced files."""
        if filename.endswith(('.md', '.txt')) and any(
            keyword in filename.lower() 
            for keyword in ['analysis', 'report', 'assessment']
        ):
            return "reports/analysis/"
        elif filename.endswith('.py') and any(
            keyword in filename.lower()
            for keyword in ['script', 'analyze', 'debug', 'test']
        ):
            return "scripts/analysis/"
        elif filename.endswith(('.yml', '.yaml')):
            return "configs/"
        elif 'docker' in filename.lower():
            return "deployment/docker/"
        else:
            return "appropriate subdirectory"
    
    def validate_naming_conventions(self) -> List[Dict]:
        """Validate naming conventions across packages."""
        violations = []
        
        for package_path in self.packages_dir.glob("*"):
            if not package_path.is_dir():
                continue
                
            package_name = package_path.name
            
            # Check for consistent naming (snake_case for directories)
            if "-" in package_name:
                violations.append({
                    "type": "naming_violation",
                    "package": package_name,
                    "issue": "Package name contains hyphens, should use snake_case",
                    "suggested": package_name.replace("-", "_"),
                    "severity": "warning"
                })
            
            # Check for proper Python module naming
            if not package_name.islower():
                violations.append({
                    "type": "naming_violation", 
                    "package": package_name,
                    "issue": "Package name should be lowercase",
                    "suggested": package_name.lower(),
                    "severity": "warning"
                })
        
        return violations
    
    def generate_package_template(self, package_name: str, package_type: str = "library") -> None:
        """Generate standardized package structure."""
        package_path = self.packages_dir / package_name
        package_path.mkdir(exist_ok=True)
        
        # Create package source directory
        source_dir = package_path / package_name
        source_dir.mkdir(exist_ok=True)
        
        # Create standard directories
        (package_path / "tests").mkdir(exist_ok=True)
        (package_path / "docs").mkdir(exist_ok=True)
        
        # Create __init__.py files
        (source_dir / "__init__.py").touch()
        (package_path / "tests" / "__init__.py").touch()
        
        print(f"✅ Generated package template for: {package_name}")
    
    def run_validation(self) -> bool:
        """Run complete package structure validation."""
        print("🔍 Validating package structure...")
        
        all_violations = []
        
        # Validate individual packages
        if self.packages_dir.exists():
            for package_path in self.packages_dir.glob("*"):
                if package_path.is_dir():
                    violations = self.validate_package_structure(package_path)
                    all_violations.extend(violations)
        
        # Validate dependency layers
        layer_violations = self.validate_dependency_layers()
        all_violations.extend(layer_violations)
        
        # Validate root directory
        root_violations = self.validate_root_directory()
        all_violations.extend(root_violations)
        
        # Validate naming conventions
        naming_violations = self.validate_naming_conventions()
        all_violations.extend(naming_violations)
        
        # Report results
        errors = [v for v in all_violations if v.get("severity") == "error"]
        warnings = [v for v in all_violations if v.get("severity") == "warning"]
        
        if errors:
            print(f"❌ Found {len(errors)} error(s):")
            for error in errors:
                print(f"  • {error}")
        
        if warnings:
            print(f"⚠️  Found {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"  • {warning}")
        
        if not errors and not warnings:
            print("✅ All package structures are valid!")
        
        return len(errors) == 0
    
    def auto_fix_violations(self) -> None:
        """Automatically fix common violations."""
        print("🔧 Auto-fixing violations...")
        
        # Fix missing package directories
        for package_path in self.packages_dir.glob("*"):
            if package_path.is_dir():
                package_name = package_path.name
                
                # Create missing directories
                for dir_template in self.REQUIRED_PACKAGE_DIRS:
                    dir_name = dir_template.format(package_name=package_name)
                    dir_path = package_path / dir_name
                    if not dir_path.exists():
                        dir_path.mkdir(parents=True, exist_ok=True)
                        print(f"  ✅ Created {package_name}/{dir_name}")
                
                # Create missing __init__.py files
                init_files = [
                    package_path / package_name / "__init__.py",
                    package_path / "tests" / "__init__.py"
                ]
                
                for init_file in init_files:
                    if not init_file.exists() and init_file.parent.exists():
                        init_file.touch()
                        print(f"  ✅ Created {init_file.relative_to(self.repo_root)}")


def main():
    """Main entry point for package structure enforcement."""
    repo_root = Path(__file__).parent.parent.parent
    enforcer = PackageStructureEnforcer(repo_root)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        enforcer.auto_fix_violations()
    
    is_valid = enforcer.run_validation()
    
    if not is_valid:
        print("\n💡 Run with --fix to automatically fix common issues")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()