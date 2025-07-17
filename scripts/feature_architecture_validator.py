#!/usr/bin/env python3
"""
Feature Architecture Validator

Validates the domain → package → feature → layer architecture ensuring:
1. Proper directory structure
2. Layer dependency rules
3. Feature boundary isolation
4. Consistent naming conventions
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class LayerType(Enum):
    DOMAIN = "domain"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    SHARED = "shared"

class ViolationType(Enum):
    STRUCTURE = "structure"
    DEPENDENCY = "dependency"
    NAMING = "naming"
    BOUNDARY = "boundary"

@dataclass
class ArchitectureViolation:
    """Represents an architecture violation"""
    file_path: str
    violation_type: ViolationType
    layer: Optional[LayerType]
    message: str
    severity: str = "error"  # error, warning, info
    line_number: Optional[int] = None
    suggestion: Optional[str] = None

class FeatureArchitectureValidator:
    """Validates feature-based architecture"""
    
    def __init__(self, packages_root: str = "src/packages_new"):
        self.packages_root = Path(packages_root)
        self.violations: List[ArchitectureViolation] = []
        
        # Expected directory structure
        self.required_layer_dirs = {
            LayerType.DOMAIN: ["entities", "services", "repositories", "value_objects"],
            LayerType.APPLICATION: ["use_cases", "user_stories", "story_maps", "services", "dto"],
            LayerType.INFRASTRUCTURE: ["api", "cli", "gui", "adapters", "repositories"]
        }
        
        # Dependency rules (what each layer can depend on)
        self.allowed_dependencies = {
            LayerType.DOMAIN: [],  # Domain depends on nothing
            LayerType.APPLICATION: [LayerType.DOMAIN],  # Application can depend on domain
            LayerType.INFRASTRUCTURE: [LayerType.DOMAIN, LayerType.APPLICATION],  # Infrastructure can depend on both
            LayerType.SHARED: [LayerType.DOMAIN, LayerType.APPLICATION, LayerType.INFRASTRUCTURE]
        }
        
        # Prohibited imports for each layer
        self.prohibited_imports = {
            LayerType.DOMAIN: [
                "fastapi", "flask", "django", "sqlalchemy", "psycopg2", "pymongo",
                "requests", "httpx", "click", "typer", "pytest", "unittest"
            ],
            LayerType.APPLICATION: [
                "fastapi", "flask", "django", "sqlalchemy", "psycopg2", "pymongo",
                "requests", "httpx", "click", "typer"
            ],
            LayerType.INFRASTRUCTURE: []  # Infrastructure can import anything
        }
    
    def validate_all_features(self) -> Dict[str, List[ArchitectureViolation]]:
        """Validate all features in the packages"""
        results = {}
        
        if not self.packages_root.exists():
            return {"error": [ArchitectureViolation(
                file_path=str(self.packages_root),
                violation_type=ViolationType.STRUCTURE,
                layer=None,
                message=f"Packages root directory does not exist: {self.packages_root}"
            )]}
        
        # Walk through domain/package/feature structure
        for domain_dir in self.packages_root.iterdir():
            if not domain_dir.is_dir() or domain_dir.name.startswith('.'):
                continue
                
            for package_dir in domain_dir.iterdir():
                if not package_dir.is_dir() or package_dir.name.startswith('.'):
                    continue
                    
                for feature_dir in package_dir.iterdir():
                    if not feature_dir.is_dir() or feature_dir.name in ['docs', 'shared'] or feature_dir.name.startswith('.'):
                        continue
                    
                    feature_key = f"{domain_dir.name}/{package_dir.name}/{feature_dir.name}"
                    violations = self.validate_feature(feature_dir)
                    
                    if violations:
                        results[feature_key] = violations
        
        return results
    
    def validate_feature(self, feature_path: Path) -> List[ArchitectureViolation]:
        """Validate a single feature"""
        violations = []
        
        # 1. Validate directory structure
        violations.extend(self._validate_directory_structure(feature_path))
        
        # 2. Validate layer dependencies
        violations.extend(self._validate_layer_dependencies(feature_path))
        
        # 3. Validate naming conventions
        violations.extend(self._validate_naming_conventions(feature_path))
        
        # 4. Validate feature boundaries
        violations.extend(self._validate_feature_boundaries(feature_path))
        
        # 5. Validate prohibited imports
        violations.extend(self._validate_prohibited_imports(feature_path))
        
        return violations
    
    def _validate_directory_structure(self, feature_path: Path) -> List[ArchitectureViolation]:
        """Validate feature directory structure"""
        violations = []
        
        # Check required top-level directories
        required_dirs = ["domain", "application", "infrastructure", "docs", "tests", "scripts"]
        for required_dir in required_dirs:
            dir_path = feature_path / required_dir
            if not dir_path.exists():
                violations.append(ArchitectureViolation(
                    file_path=str(dir_path),
                    violation_type=ViolationType.STRUCTURE,
                    layer=None,
                    message=f"Missing required directory: {required_dir}",
                    severity="warning",
                    suggestion=f"Create directory: mkdir -p {dir_path}"
                ))
        
        # Check layer-specific subdirectories
        for layer_type, subdirs in self.required_layer_dirs.items():
            layer_path = feature_path / layer_type.value
            if layer_path.exists():
                for subdir in subdirs:
                    subdir_path = layer_path / subdir
                    if not subdir_path.exists():
                        violations.append(ArchitectureViolation(
                            file_path=str(subdir_path),
                            violation_type=ViolationType.STRUCTURE,
                            layer=layer_type,
                            message=f"Missing {layer_type.value} subdirectory: {subdir}",
                            severity="info",
                            suggestion=f"Create directory: mkdir -p {subdir_path}"
                        ))
        
        return violations
    
    def _validate_layer_dependencies(self, feature_path: Path) -> List[ArchitectureViolation]:
        """Validate layer dependency rules"""
        violations = []
        
        for layer_type in LayerType:
            layer_path = feature_path / layer_type.value
            if not layer_path.exists():
                continue
            
            # Check all Python files in the layer
            for py_file in layer_path.rglob("*.py"):
                file_violations = self._validate_file_dependencies(py_file, layer_type, feature_path)
                violations.extend(file_violations)
        
        return violations
    
    def _validate_file_dependencies(self, file_path: Path, layer_type: LayerType, feature_path: Path) -> List[ArchitectureViolation]:
        """Validate dependencies in a single file"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            # Extract all imports
            imports = self._extract_imports(tree)
            
            # Check each import
            for import_info in imports:
                import_path, line_number = import_info
                
                # Check if import violates layer dependency rules
                violation = self._check_import_violation(import_path, layer_type, feature_path, line_number)
                if violation:
                    violations.append(violation)
        
        except Exception as e:
            violations.append(ArchitectureViolation(
                file_path=str(file_path),
                violation_type=ViolationType.DEPENDENCY,
                layer=layer_type,
                message=f"Error parsing file: {str(e)}",
                severity="warning"
            ))
        
        return violations
    
    def _extract_imports(self, tree: ast.AST) -> List[Tuple[str, int]]:
        """Extract all imports from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append((node.module, node.lineno))
        
        return imports
    
    def _check_import_violation(self, import_path: str, layer_type: LayerType, feature_path: Path, line_number: int) -> Optional[ArchitectureViolation]:
        """Check if an import violates layer dependency rules"""
        # Skip standard library and third-party imports for now
        if not import_path.startswith("src.packages"):
            return None
        
        # Extract layer information from import path
        imported_layer = self._extract_layer_from_import(import_path)
        
        if imported_layer and imported_layer not in self.allowed_dependencies[layer_type]:
            return ArchitectureViolation(
                file_path=str(feature_path),
                violation_type=ViolationType.DEPENDENCY,
                layer=layer_type,
                message=f"Layer {layer_type.value} cannot import from {imported_layer.value}",
                line_number=line_number,
                suggestion=f"Use dependency injection or move code to appropriate layer"
            )
        
        return None
    
    def _extract_layer_from_import(self, import_path: str) -> Optional[LayerType]:
        """Extract layer type from import path"""
        parts = import_path.split('.')
        
        # Look for layer indicators in the import path
        for i, part in enumerate(parts):
            if part in [layer.value for layer in LayerType]:
                return LayerType(part)
        
        return None
    
    def _validate_naming_conventions(self, feature_path: Path) -> List[ArchitectureViolation]:
        """Validate naming conventions"""
        violations = []
        
        # Check feature name (should be snake_case)
        feature_name = feature_path.name
        if not re.match(r'^[a-z][a-z0-9_]*$', feature_name):
            violations.append(ArchitectureViolation(
                file_path=str(feature_path),
                violation_type=ViolationType.NAMING,
                layer=None,
                message=f"Feature name should be snake_case: {feature_name}",
                suggestion=f"Rename to: {self._to_snake_case(feature_name)}"
            ))
        
        # Check Python file naming
        for py_file in feature_path.rglob("*.py"):
            filename = py_file.stem
            if not re.match(r'^[a-z][a-z0-9_]*$', filename) and filename != '__init__':
                violations.append(ArchitectureViolation(
                    file_path=str(py_file),
                    violation_type=ViolationType.NAMING,
                    layer=self._get_layer_from_path(py_file, feature_path),
                    message=f"Python file should be snake_case: {filename}",
                    suggestion=f"Rename to: {self._to_snake_case(filename)}.py"
                ))
        
        return violations
    
    def _validate_feature_boundaries(self, feature_path: Path) -> List[ArchitectureViolation]:
        """Validate feature boundary isolation"""
        violations = []
        
        # Check that feature doesn't import from other features in same package
        for py_file in feature_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for imports from other features
                feature_name = feature_path.name
                package_name = feature_path.parent.name
                domain_name = feature_path.parent.parent.name
                
                # Pattern to match imports from other features
                other_feature_pattern = rf'from src\.packages\.{re.escape(domain_name)}\.{re.escape(package_name)}\.(?!{re.escape(feature_name)})(\w+)'
                
                matches = re.finditer(other_feature_pattern, content, re.MULTILINE)
                for match in matches:
                    other_feature = match.group(1)
                    if other_feature != "shared":  # shared is allowed
                        line_number = content[:match.start()].count('\n') + 1
                        violations.append(ArchitectureViolation(
                            file_path=str(py_file),
                            violation_type=ViolationType.BOUNDARY,
                            layer=self._get_layer_from_path(py_file, feature_path),
                            message=f"Feature {feature_name} imports from feature {other_feature}",
                            line_number=line_number,
                            suggestion="Use shared components or define proper interfaces"
                        ))
            
            except Exception:
                continue
        
        return violations
    
    def _validate_prohibited_imports(self, feature_path: Path) -> List[ArchitectureViolation]:
        """Validate prohibited imports for each layer"""
        violations = []
        
        for layer_type in LayerType:
            layer_path = feature_path / layer_type.value
            if not layer_path.exists():
                continue
            
            prohibited = self.prohibited_imports.get(layer_type, [])
            if not prohibited:
                continue
            
            for py_file in layer_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for prohibited_import in prohibited:
                        if f"import {prohibited_import}" in content or f"from {prohibited_import}" in content:
                            violations.append(ArchitectureViolation(
                                file_path=str(py_file),
                                violation_type=ViolationType.DEPENDENCY,
                                layer=layer_type,
                                message=f"Layer {layer_type.value} cannot import {prohibited_import}",
                                suggestion=f"Move code using {prohibited_import} to infrastructure layer"
                            ))
                
                except Exception:
                    continue
        
        return violations
    
    def _get_layer_from_path(self, file_path: Path, feature_path: Path) -> Optional[LayerType]:
        """Get layer type from file path"""
        relative_path = file_path.relative_to(feature_path)
        
        if len(relative_path.parts) > 0:
            first_part = relative_path.parts[0]
            try:
                return LayerType(first_part)
            except ValueError:
                return None
        
        return None
    
    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case"""
        # Insert underscores before uppercase letters
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        # Convert to lowercase
        return name.lower()
    
    def generate_validation_report(self, results: Dict[str, List[ArchitectureViolation]]) -> str:
        """Generate comprehensive validation report"""
        report = "# Feature Architecture Validation Report\n\n"
        
        total_violations = sum(len(violations) for violations in results.values())
        
        if total_violations == 0:
            report += "✅ **All features pass architecture validation!**\n\n"
            report += "The codebase follows proper domain → package → feature → layer architecture.\n"
            return report
        
        report += f"❌ **Found {total_violations} architecture violations across {len(results)} features.**\n\n"
        
        # Summary by violation type
        violation_counts = {}
        for violations in results.values():
            for violation in violations:
                violation_type = violation.violation_type.value
                violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        report += "## Summary by Violation Type\n\n"
        for violation_type, count in sorted(violation_counts.items()):
            report += f"- **{violation_type.title()}**: {count} violations\n"
        
        report += "\n## Detailed Violations\n\n"
        
        # Group violations by feature
        for feature_name, violations in sorted(results.items()):
            report += f"### {feature_name}\n\n"
            
            # Group by violation type
            violations_by_type = {}
            for violation in violations:
                vtype = violation.violation_type.value
                if vtype not in violations_by_type:
                    violations_by_type[vtype] = []
                violations_by_type[vtype].append(violation)
            
            for violation_type, type_violations in sorted(violations_by_type.items()):
                report += f"#### {violation_type.title()} Violations\n\n"
                
                for violation in type_violations:
                    severity_icon = "🔴" if violation.severity == "error" else "🟡" if violation.severity == "warning" else "ℹ️"
                    
                    report += f"{severity_icon} **{violation.message}**\n"
                    report += f"   - File: `{violation.file_path}`\n"
                    
                    if violation.line_number:
                        report += f"   - Line: {violation.line_number}\n"
                    
                    if violation.layer:
                        report += f"   - Layer: {violation.layer.value}\n"
                    
                    if violation.suggestion:
                        report += f"   - Suggestion: {violation.suggestion}\n"
                    
                    report += "\n"
        
        # Add recommendations
        report += "## Recommendations\n\n"
        report += "1. **Structure Violations**: Create missing directories using the suggested mkdir commands\n"
        report += "2. **Dependency Violations**: Refactor code to follow layer dependency rules\n"
        report += "3. **Naming Violations**: Rename files and directories to follow snake_case convention\n"
        report += "4. **Boundary Violations**: Use shared components or define proper interfaces\n"
        report += "5. **Import Violations**: Move framework-specific code to infrastructure layer\n"
        
        return report
    
    def generate_json_report(self, results: Dict[str, List[ArchitectureViolation]]) -> str:
        """Generate JSON report for programmatic consumption"""
        json_results = {}
        
        for feature_name, violations in results.items():
            json_results[feature_name] = [
                {
                    "file_path": v.file_path,
                    "violation_type": v.violation_type.value,
                    "layer": v.layer.value if v.layer else None,
                    "message": v.message,
                    "severity": v.severity,
                    "line_number": v.line_number,
                    "suggestion": v.suggestion
                }
                for v in violations
            ]
        
        return json.dumps(json_results, indent=2)

def main():
    """Main entry point"""
    validator = FeatureArchitectureValidator()
    
    # Validate all features
    results = validator.validate_all_features()
    
    # Generate and save reports
    markdown_report = validator.generate_validation_report(results)
    with open("feature_architecture_validation_report.md", "w") as f:
        f.write(markdown_report)
    
    json_report = validator.generate_json_report(results)
    with open("feature_architecture_validation_report.json", "w") as f:
        f.write(json_report)
    
    # Print summary
    total_violations = sum(len(violations) for violations in results.values())
    if total_violations == 0:
        print("✅ All features pass architecture validation!")
    else:
        print(f"❌ Found {total_violations} architecture violations across {len(results)} features.")
        print("See feature_architecture_validation_report.md for details.")
    
    # Exit with appropriate code
    return 1 if total_violations > 0 else 0

if __name__ == "__main__":
    exit(main())