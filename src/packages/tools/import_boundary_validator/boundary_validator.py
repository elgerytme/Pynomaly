"""
Automated import boundary validator for package interaction guidelines.

This tool validates that all imports in the monorepo follow the established
architectural boundaries and interaction patterns defined in IMPORT_GUIDELINES.md.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import json
import argparse


class ViolationType(Enum):
    """Types of boundary violations."""
    FORBIDDEN_CROSS_DOMAIN = "forbidden_cross_domain"
    DOMAIN_TO_ENTERPRISE = "domain_to_enterprise"
    DOMAIN_TO_INTEGRATIONS = "domain_to_integrations"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    INVALID_LAYER_IMPORT = "invalid_layer_import"
    MISSING_INTERFACES = "missing_interfaces"


@dataclass
class ImportViolation:
    """Represents an import boundary violation."""
    file_path: str
    line_number: int
    import_statement: str
    violation_type: ViolationType
    source_package: str
    target_package: str
    description: str
    suggestion: Optional[str] = None


class PackageType(Enum):
    """Types of packages in the monorepo."""
    DOMAIN = "domain"           # ai/, data/
    INTERFACES = "interfaces"   # interfaces/
    SHARED = "shared"          # shared/
    ENTERPRISE = "enterprise"  # enterprise/
    INTEGRATIONS = "integrations"  # integrations/
    CONFIGURATIONS = "configurations"  # configurations/
    TOOLS = "tools"            # tools/
    CLIENTS = "clients"        # clients/
    TEMPLATES = "templates"    # templates/


class ArchitecturalLayer(Enum):
    """Architectural layers within packages."""
    DOMAIN = "domain"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    PRESENTATION = "presentation"


class ImportBoundaryValidator:
    """
    Validates import boundaries across the monorepo.
    
    This validator checks that all imports follow the architectural
    guidelines defined in IMPORT_GUIDELINES.md.
    """
    
    def __init__(self, monorepo_root: Path):
        self.monorepo_root = monorepo_root
        self.packages_root = monorepo_root / "src" / "packages"
        self.violations: List[ImportViolation] = []
        
        # Package classification
        self.domain_packages = {"ai", "data"}
        self.special_packages = {
            "interfaces": PackageType.INTERFACES,
            "shared": PackageType.SHARED,
            "enterprise": PackageType.ENTERPRISE,
            "integrations": PackageType.INTEGRATIONS,
            "configurations": PackageType.CONFIGURATIONS,
            "tools": PackageType.TOOLS,
            "clients": PackageType.CLIENTS,
            "templates": PackageType.TEMPLATES,
        }
        
        # Allowed import patterns
        self.allowed_imports = self._build_allowed_imports_matrix()
    
    def _build_allowed_imports_matrix(self) -> Dict[PackageType, Set[PackageType]]:
        """Build matrix of allowed import relationships."""
        return {
            PackageType.DOMAIN: {PackageType.INTERFACES, PackageType.SHARED},
            PackageType.INTERFACES: set(),  # No imports from other packages
            PackageType.SHARED: {PackageType.INTERFACES},
            PackageType.ENTERPRISE: {PackageType.INTERFACES, PackageType.SHARED},
            PackageType.INTEGRATIONS: {PackageType.INTERFACES, PackageType.SHARED},
            PackageType.CONFIGURATIONS: {  # Can import everything
                PackageType.DOMAIN, PackageType.INTERFACES, PackageType.SHARED,
                PackageType.ENTERPRISE, PackageType.INTEGRATIONS
            },
            PackageType.TOOLS: {PackageType.INTERFACES, PackageType.SHARED},
            PackageType.CLIENTS: {PackageType.INTERFACES, PackageType.SHARED},
            PackageType.TEMPLATES: {PackageType.INTERFACES, PackageType.SHARED},
        }
    
    def validate_all_packages(self) -> List[ImportViolation]:
        """Validate all packages in the monorepo."""
        self.violations.clear()
        
        for package_dir in self.packages_root.iterdir():
            if package_dir.is_dir() and not package_dir.name.startswith('.'):
                self._validate_package(package_dir)
        
        return self.violations
    
    def _validate_package(self, package_path: Path) -> None:
        """Validate a single package."""
        package_name = package_path.name
        package_type = self._get_package_type(package_name)
        
        # Find all Python files in the package
        python_files = self._find_python_files(package_path)
        
        for py_file in python_files:
            self._validate_file(py_file, package_name, package_type)
    
    def _find_python_files(self, package_path: Path) -> List[Path]:
        """Find all Python files in a package."""
        python_files = []
        for root, dirs, files in os.walk(package_path):
            # Skip __pycache__ and .git directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _validate_file(self, file_path: Path, source_package: str, source_package_type: PackageType) -> None:
        """Validate imports in a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Validate each import
            for import_info in imports:
                self._validate_import(file_path, import_info, source_package, source_package_type)
                
        except Exception as e:
            # Skip files that can't be parsed
            pass
    
    def _extract_imports(self, tree: ast.AST) -> List[Tuple[int, str, str]]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((node.lineno, alias.name, f"import {alias.name}"))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append((node.lineno, node.module, f"from {node.module} import ..."))
        
        return imports
    
    def _validate_import(self, file_path: Path, import_info: Tuple[int, str, str], 
                        source_package: str, source_package_type: PackageType) -> None:
        """Validate a single import statement."""
        line_number, module_name, import_statement = import_info
        
        # Skip standard library and third-party imports
        if not self._is_monorepo_import(module_name):
            return
        
        # Parse the import to understand the target
        target_info = self._parse_import_target(module_name)
        if not target_info:
            return
        
        target_package, target_subpath = target_info
        target_package_type = self._get_package_type(target_package)
        
        # Check various violation types
        self._check_forbidden_cross_domain(
            file_path, line_number, import_statement, 
            source_package, target_package, source_package_type, target_package_type
        )
        
        self._check_forbidden_enterprise_integrations(
            file_path, line_number, import_statement,
            source_package, target_package, source_package_type, target_package_type
        )
        
        self._check_layer_violations(
            file_path, line_number, import_statement,
            source_package, target_package, target_subpath
        )
    
    def _is_monorepo_import(self, module_name: str) -> bool:
        """Check if import is from within the monorepo."""
        # Check for explicit package imports
        monorepo_patterns = [
            # Direct package imports
            r'^(ai|data|interfaces|shared|enterprise|integrations|configurations|tools|clients|templates)\.',
            # Src.packages imports
            r'^src\.packages\.',
            # Package-specific imports
            r'^(anomaly_detection|data_quality|machine_learning|mlops)\.',
        ]
        
        return any(re.match(pattern, module_name) for pattern in monorepo_patterns)
    
    def _parse_import_target(self, module_name: str) -> Optional[Tuple[str, str]]:
        """Parse import target to extract package and subpath."""
        # Handle src.packages.* imports
        if module_name.startswith('src.packages.'):
            parts = module_name.split('.')
            if len(parts) >= 3:
                return parts[2], '.'.join(parts[3:])
        
        # Handle direct package imports
        parts = module_name.split('.')
        if parts[0] in self.domain_packages or parts[0] in self.special_packages:
            return parts[0], '.'.join(parts[1:])
        
        # Handle package-specific imports (e.g., anomaly_detection.*)
        package_mappings = {
            'anomaly_detection': 'data',
            'data_quality': 'data',
            'machine_learning': 'ai',
            'mlops': 'ai',
        }
        
        if parts[0] in package_mappings:
            return package_mappings[parts[0]], module_name
        
        return None
    
    def _get_package_type(self, package_name: str) -> PackageType:
        """Get the type of a package."""
        if package_name in self.domain_packages:
            return PackageType.DOMAIN
        return self.special_packages.get(package_name, PackageType.DOMAIN)
    
    def _check_forbidden_cross_domain(self, file_path: Path, line_number: int, import_statement: str,
                                    source_package: str, target_package: str,
                                    source_type: PackageType, target_type: PackageType) -> None:
        """Check for forbidden cross-domain imports."""
        # Domain packages cannot import from other domain packages
        if (source_type == PackageType.DOMAIN and target_type == PackageType.DOMAIN and 
            source_package != target_package):
            
            violation = ImportViolation(
                file_path=str(file_path),
                line_number=line_number,
                import_statement=import_statement,
                violation_type=ViolationType.FORBIDDEN_CROSS_DOMAIN,
                source_package=source_package,
                target_package=target_package,
                description=f"Domain package '{source_package}' cannot import from domain package '{target_package}'",
                suggestion="Use event-driven communication via interfaces/events.py or dependency injection"
            )
            self.violations.append(violation)
    
    def _check_forbidden_enterprise_integrations(self, file_path: Path, line_number: int, import_statement: str,
                                               source_package: str, target_package: str,
                                               source_type: PackageType, target_type: PackageType) -> None:
        """Check for forbidden enterprise/integrations imports."""
        # Domain packages cannot import from enterprise or integrations
        if source_type == PackageType.DOMAIN and target_type in {PackageType.ENTERPRISE, PackageType.INTEGRATIONS}:
            violation = ImportViolation(
                file_path=str(file_path),
                line_number=line_number,
                import_statement=import_statement,
                violation_type=ViolationType.DOMAIN_TO_ENTERPRISE if target_type == PackageType.ENTERPRISE else ViolationType.DOMAIN_TO_INTEGRATIONS,
                source_package=source_package,
                target_package=target_package,
                description=f"Domain package '{source_package}' cannot import from {target_type.value} package '{target_package}'",
                suggestion="Enterprise and integration concerns should be handled in the configurations layer"
            )
            self.violations.append(violation)
        
        # Check allowed imports matrix
        allowed_targets = self.allowed_imports.get(source_type, set())
        if target_type not in allowed_targets:
            violation = ImportViolation(
                file_path=str(file_path),
                line_number=line_number,
                import_statement=import_statement,
                violation_type=ViolationType.INVALID_LAYER_IMPORT,
                source_package=source_package,
                target_package=target_package,
                description=f"Package type '{source_type.value}' cannot import from package type '{target_type.value}'",
                suggestion=f"Allowed imports for {source_type.value}: {', '.join(t.value for t in allowed_targets)}"
            )
            self.violations.append(violation)
    
    def _check_layer_violations(self, file_path: Path, line_number: int, import_statement: str,
                              source_package: str, target_package: str, target_subpath: str) -> None:
        """Check for architectural layer violations."""
        # Determine source layer from file path
        source_layer = self._get_layer_from_path(file_path)
        if not source_layer:
            return
        
        # Check if importing from domain layer of other packages
        if 'domain' in target_subpath and source_package != target_package:
            violation = ImportViolation(
                file_path=str(file_path),
                line_number=line_number,
                import_statement=import_statement,
                violation_type=ViolationType.INVALID_LAYER_IMPORT,
                source_package=source_package,
                target_package=target_package,
                description=f"Cannot import from domain layer of another package",
                suggestion="Import from application layer or use interfaces/events for communication"
            )
            self.violations.append(violation)
    
    def _get_layer_from_path(self, file_path: Path) -> Optional[ArchitecturalLayer]:
        """Determine architectural layer from file path."""
        path_str = str(file_path)
        
        if '/domain/' in path_str:
            return ArchitecturalLayer.DOMAIN
        elif '/application/' in path_str:
            return ArchitecturalLayer.APPLICATION
        elif '/infrastructure/' in path_str:
            return ArchitecturalLayer.INFRASTRUCTURE
        elif '/presentation/' in path_str:
            return ArchitecturalLayer.PRESENTATION
        
        return None
    
    def generate_report(self, output_format: str = "text") -> str:
        """Generate a validation report."""
        if output_format == "json":
            return self._generate_json_report()
        else:
            return self._generate_text_report()
    
    def _generate_text_report(self) -> str:
        """Generate a text-based report."""
        if not self.violations:
            return "✅ No import boundary violations found!"
        
        report = [
            "❌ Import Boundary Violations Found",
            "=" * 50,
            f"Total violations: {len(self.violations)}",
            ""
        ]
        
        # Group by violation type
        violations_by_type = {}
        for violation in self.violations:
            vtype = violation.violation_type
            if vtype not in violations_by_type:
                violations_by_type[vtype] = []
            violations_by_type[vtype].append(violation)
        
        for vtype, violations in violations_by_type.items():
            report.append(f"🚫 {vtype.value.replace('_', ' ').title()} ({len(violations)} violations)")
            report.append("-" * 40)
            
            for violation in violations:
                report.append(f"File: {violation.file_path}:{violation.line_number}")
                report.append(f"Import: {violation.import_statement}")
                report.append(f"Issue: {violation.description}")
                if violation.suggestion:
                    report.append(f"Suggestion: {violation.suggestion}")
                report.append("")
        
        return "\n".join(report)
    
    def _generate_json_report(self) -> str:
        """Generate a JSON-based report."""
        violations_data = []
        for violation in self.violations:
            violations_data.append({
                "file_path": violation.file_path,
                "line_number": violation.line_number,
                "import_statement": violation.import_statement,
                "violation_type": violation.violation_type.value,
                "source_package": violation.source_package,
                "target_package": violation.target_package,
                "description": violation.description,
                "suggestion": violation.suggestion
            })
        
        report = {
            "summary": {
                "total_violations": len(self.violations),
                "violations_by_type": {
                    vtype.value: len([v for v in self.violations if v.violation_type == vtype])
                    for vtype in ViolationType
                }
            },
            "violations": violations_data
        }
        
        return json.dumps(report, indent=2)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Validate import boundaries in the monorepo")
    parser.add_argument("--root", default=".", help="Path to monorepo root")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--fail-on-violations", action="store_true", help="Exit with code 1 if violations found")
    parser.add_argument("--output", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Create validator
    validator = ImportBoundaryValidator(Path(args.root))
    
    # Run validation
    violations = validator.validate_all_packages()
    
    # Generate report
    report = validator.generate_report(args.format)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)
    
    # Exit with appropriate code
    if args.fail_on_violations and violations:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()