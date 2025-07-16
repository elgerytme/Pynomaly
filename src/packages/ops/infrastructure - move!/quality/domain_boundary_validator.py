"""Domain boundary validation for clean architecture enforcement."""

import ast
import os
from typing import Dict, List, Set, Tuple
from pathlib import Path


class DomainBoundaryValidator:
    """Validates domain boundaries and detects violations."""
    
    ALLOWED_DEPENDENCIES = {
        # Core domain packages - these form the foundation
        "core": [],  # Core should have no dependencies on other packages
        "mathematics": [],  # Mathematics is pure computational logic
        
        # Infrastructure layer - can depend on core domains only
        "infrastructure": ["core", "mathematics"],
        "interfaces": ["core", "mathematics"],  # Interfaces define generic contracts
        
        # Isolated domain packages - can depend on core and infrastructure ONLY
        "anomaly_detection": ["core", "mathematics", "infrastructure"],  # ISOLATED: No other packages can depend on this
        "fraud_detection": ["core", "mathematics", "infrastructure"],  # Future domain package
        "intrusion_detection": ["core", "mathematics", "infrastructure"],  # Future domain package
        
        # Generic business domain packages - CANNOT depend on specific detection domains
        "machine_learning": ["core", "mathematics", "infrastructure"],  # Generic ML, no specific detection types
        "data_platform": ["core", "mathematics", "infrastructure"],  # Generic data processing
        
        # Application layer - can depend on generic domains but NOT specific detection domains
        "services": ["core", "mathematics", "infrastructure", "machine_learning", "data_platform"],
        "enterprise": ["core", "mathematics", "infrastructure", "services"],
        "mlops": ["core", "mathematics", "infrastructure", "machine_learning", "services"],
        
        # Presentation layer - can depend on generic layers only (uses interfaces for detection)
        "interfaces": ["core", "mathematics", "infrastructure", "services"],
        
        # Utility packages
        "testing": ["core", "mathematics", "infrastructure"],
        "tools": ["core", "mathematics", "infrastructure"],
        "documentation": [],  # Documentation should be independent
        
        # Domain-agnostic packages
        "people_ops": ["core", "infrastructure"],
        "domain_library": ["core"],
        "data_observability": ["data_platform", "infrastructure"],
        "mobile": ["interfaces", "infrastructure"],
        "integration": ["interfaces"],
    }
    
    # Packages that are completely isolated (no other packages can depend on them)
    ISOLATED_PACKAGES = {
        "anomaly_detection",
        "fraud_detection", 
        "intrusion_detection",
    }
    
    def __init__(self, packages_root: str):
        """Initialize with packages root directory."""
        self.packages_root = Path(packages_root)
    
    def validate_all_packages(self) -> Dict[str, List[str]]:
        """Validate all packages for domain boundary violations."""
        violations = {}
        
        for package_dir in self.packages_root.iterdir():
            if package_dir.is_dir() and package_dir.name not in [".git", "__pycache__"]:
                package_violations = self.validate_package(package_dir.name)
                if package_violations:
                    violations[package_dir.name] = package_violations
        
        return violations
    
    def validate_package(self, package_name: str) -> List[str]:
        """Validate a specific package for domain boundary violations."""
        violations = []
        package_path = self.packages_root / package_name
        
        if not package_path.exists():
            return violations
        
        allowed_deps = self.ALLOWED_DEPENDENCIES.get(package_name, [])
        
        # Find all Python files in the package
        python_files = list(package_path.rglob("*.py"))
        
        for py_file in python_files:
            file_violations = self._check_file_imports(py_file, package_name, allowed_deps)
            violations.extend(file_violations)
        
        return violations
    
    def _check_file_imports(self, file_path: Path, package_name: str, allowed_deps: List[str]) -> List[str]:
        """Check imports in a specific file for violations."""
        violations = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        violation = self._check_import_path(alias.name, package_name, allowed_deps, file_path)
                        if violation:
                            violations.append(violation)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        violation = self._check_import_path(node.module, package_name, allowed_deps, file_path)
                        if violation:
                            violations.append(violation)
        
        except Exception as e:
            violations.append(f"Error parsing {file_path}: {str(e)}")
        
        return violations
    
    def _check_import_path(self, import_path: str, package_name: str, allowed_deps: List[str], file_path: Path) -> str:
        """Check if an import path violates domain boundaries."""
        if not import_path or import_path.startswith("."):
            return ""  # Relative imports are OK within package
        
        # Check if import is from another package
        package_imports = self._extract_package_imports(import_path)
        
        for imported_package in package_imports:
            if imported_package == package_name:
                continue  # Self-imports are OK
            
            # Check isolation violation: no package can depend on isolated packages
            if imported_package in self.ISOLATED_PACKAGES and package_name != imported_package:
                return (
                    f"ISOLATION VIOLATION: {file_path} imports from ISOLATED package '{imported_package}'. "
                    f"Isolated packages like anomaly_detection cannot be imported by other packages. "
                    f"Use generic interfaces instead."
                )
            
            # Check standard dependency violations
            if imported_package in self.ALLOWED_DEPENDENCIES and imported_package not in allowed_deps:
                return (
                    f"DEPENDENCY VIOLATION: {file_path} imports from '{imported_package}' "
                    f"but '{package_name}' is not allowed to depend on '{imported_package}'"
                )
        
        return ""
    
    def _extract_package_imports(self, import_path: str) -> List[str]:
        """Extract package names from import path."""
        packages = []
        
        # Split by dots and check each part
        parts = import_path.split(".")
        
        for part in parts:
            if part in self.ALLOWED_DEPENDENCIES:
                packages.append(part)
        
        return packages
    
    def generate_dependency_report(self) -> Dict[str, any]:
        """Generate a comprehensive dependency report."""
        violations = self.validate_all_packages()
        
        report = {
            "total_packages": len(list(self.packages_root.iterdir())),
            "packages_with_violations": len(violations),
            "violations": violations,
            "allowed_dependencies": self.ALLOWED_DEPENDENCIES,
            "clean_packages": [],
            "summary": {
                "total_violations": sum(len(v) for v in violations.values()),
                "most_violated_package": max(violations.keys(), key=lambda k: len(violations[k])) if violations else None,
                "cleanest_packages": [],
            }
        }
        
        # Find clean packages
        for package_dir in self.packages_root.iterdir():
            if package_dir.is_dir() and package_dir.name not in violations:
                report["clean_packages"].append(package_dir.name)
        
        report["summary"]["cleanest_packages"] = report["clean_packages"]
        
        return report
    
    def generate_fix_recommendations(self) -> List[str]:
        """Generate recommendations for fixing domain boundary violations."""
        violations = self.validate_all_packages()
        recommendations = []
        
        for package, package_violations in violations.items():
            recommendations.append(f"\n## Package: {package}")
            
            # Group violations by type
            import_violations = [v for v in package_violations if "imports from" in v]
            
            if import_violations:
                recommendations.append("### Import Violations:")
                for violation in import_violations:
                    recommendations.append(f"- {violation}")
                    
                    # Extract the violating package
                    if "imports from" in violation:
                        violating_package = violation.split("imports from '")[1].split("'")[0]
                        recommendations.append(f"  → Use interface from 'interfaces' package instead of direct import from '{violating_package}'")
                        recommendations.append(f"  → Create adapter in current package to bridge the interface")
        
        return recommendations