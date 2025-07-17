"""
Package Isolation Validator for Architecture Package

Validates that packages adhere to domain isolation principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set
import ast
import re


@dataclass
class IsolationViolation:
    """Represents a package isolation violation."""
    package: str
    file_path: str
    line_number: int
    violation_type: str
    description: str
    severity: str


class PackageIsolationValidator:
    """Validates package isolation according to architectural rules."""
    
    def __init__(self, packages_root: Path):
        self.packages_root = packages_root
        self.violations: List[IsolationViolation] = []
    
    def validate_package_isolation(self) -> List[IsolationViolation]:
        """Validate all packages for isolation violations."""
        self.violations = []
        
        for package_path in self.packages_root.rglob("*"):
            if package_path.is_dir() and (package_path / "__init__.py").exists():
                self._validate_package(package_path)
        
        return self.violations
    
    def _validate_package(self, package_path: Path) -> None:
        """Validate a single package."""
        relative_path = package_path.relative_to(self.packages_root)
        package_name = str(relative_path).replace("/", ".")
        
        # Check README for domain boundary violations
        readme_path = package_path / "README.md"
        if readme_path.exists():
            self._validate_readme_isolation(package_name, readme_path)
        
        # Check Python files for import violations
        for py_file in package_path.rglob("*.py"):
            self._validate_python_file_isolation(package_name, py_file)
    
    def _validate_readme_isolation(self, package_name: str, readme_path: Path) -> None:
        """Validate README file for domain isolation."""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for cross-domain references
            self._check_cross_domain_references(package_name, readme_path, content)
            
        except Exception as e:
            self.violations.append(IsolationViolation(
                package=package_name,
                file_path=str(readme_path),
                line_number=0,
                violation_type="file_error",
                description=f"Error reading README: {e}",
                severity="warning"
            ))
    
    def _validate_python_file_isolation(self, package_name: str, file_path: Path) -> None:
        """Validate Python file for import isolation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse and check imports
            try:
                tree = ast.parse(content)
                self._check_domain_imports(package_name, file_path, tree)
            except SyntaxError:
                # Skip files with syntax errors
                pass
            
        except Exception as e:
            self.violations.append(IsolationViolation(
                package=package_name,
                file_path=str(file_path),
                line_number=0,
                violation_type="file_error",
                description=f"Error reading Python file: {e}",
                severity="warning"
            ))
    
    def _check_cross_domain_references(self, package_name: str, file_path: Path, content: str) -> None:
        """Check for cross-domain references in documentation."""
        domain_terms = [
            "anomaly", "detection", "fraud", "intrusion", "outlier",
            "machine learning", "ml", "ai", "artificial intelligence",
            "statistics", "mathematical", "algorithm", "computation",
            "enterprise", "business", "authentication", "dashboard"
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for term in domain_terms:
                if term.lower() in line.lower():
                    # Check if this violates isolation
                    if self._is_cross_domain_violation(package_name, line, term):
                        self.violations.append(IsolationViolation(
                            package=package_name,
                            file_path=str(file_path),
                            line_number=i,
                            violation_type="cross_domain_reference",
                            description=f"References '{term}' domain in documentation",
                            severity="warning"
                        ))
    
    def _check_domain_imports(self, package_name: str, file_path: Path, tree: ast.AST) -> None:
        """Check for inappropriate domain imports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module
                
                # Check for domain-specific imports
                if self._is_domain_import_violation(package_name, module_name):
                    self.violations.append(IsolationViolation(
                        package=package_name,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        violation_type="domain_import",
                        description=f"Imports domain-specific module: {module_name}",
                        severity="error"
                    ))
    
    def _is_cross_domain_violation(self, package_name: str, line: str, term: str) -> bool:
        """Check if a line contains a cross-domain violation."""
        # Simple heuristic: if the term is not in the package name/path
        # and appears in a descriptive context, it's likely a violation
        
        if term.lower() in package_name.lower():
            return False  # Own domain
        
        violation_patterns = [
            f"for {term}",
            f"with {term}",
            f"{term} system",
            f"{term} platform",
            f"provides {term}",
            f"supports {term}"
        ]
        
        line_lower = line.lower()
        return any(pattern in line_lower for pattern in violation_patterns)
    
    def _is_domain_import_violation(self, package_name: str, module_name: str) -> bool:
        """Check if an import violates domain boundaries."""
        # Skip imports from same package
        if module_name.startswith(f"pynomaly.{package_name}"):
            return False
        
        # Check for imports from other domain internals
        forbidden_paths = [
            ".domain.", ".entities.", ".models.", ".business_logic.",
            ".use_cases.", ".services.", ".repositories."
        ]
        
        return any(forbidden_path in module_name for forbidden_path in forbidden_paths)


class ArchitectureIsolationEnforcer:
    """Enforces package isolation rules in architecture validation."""
    
    def __init__(self, packages_root: Path):
        self.validator = PackageIsolationValidator(packages_root)
    
    def validate_architecture_isolation(self) -> Dict[str, List[IsolationViolation]]:
        """Validate architecture-level isolation."""
        violations = self.validator.validate_package_isolation()
        
        # Group by package for reporting
        grouped_violations = {}
        for violation in violations:
            if violation.package not in grouped_violations:
                grouped_violations[violation.package] = []
            grouped_violations[violation.package].append(violation)
        
        return grouped_violations
    
    def generate_isolation_report(self, violations: Dict[str, List[IsolationViolation]]) -> str:
        """Generate a comprehensive isolation report."""
        if not violations:
            return "✅ All packages comply with isolation rules!"
        
        report = "🚨 Package Isolation Violations\n"
        report += "=" * 50 + "\n\n"
        
        total_violations = sum(len(v) for v in violations.values())
        report += f"Total violations: {total_violations}\n"
        report += f"Affected packages: {len(violations)}\n\n"
        
        for package, package_violations in violations.items():
            report += f"Package: {package} ({len(package_violations)} violations)\n"
            report += "-" * 40 + "\n"
            
            for violation in package_violations:
                report += f"  {violation.severity.upper()}: {violation.description}\n"
                report += f"  File: {violation.file_path}:{violation.line_number}\n"
                report += f"  Type: {violation.violation_type}\n\n"
        
        return report