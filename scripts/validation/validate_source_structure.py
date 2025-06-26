#!/usr/bin/env python3
"""
Source Code Structure Validation Script

This script validates that the source code follows Clean Architecture principles
and maintains consistent module organization as defined in the project plan.
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse
from dataclasses import dataclass
import json


@dataclass
class ValidationResult:
    """Results of source code structure validation."""
    passed: bool
    violations: List[str]
    warnings: List[str]
    metrics: Dict[str, any]


@dataclass
class ArchitectureRule:
    """Defines an architecture rule and its validation logic."""
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    check_function: str


class SourceStructureValidator:
    """Validates source code structure against Clean Architecture principles."""
    
    def __init__(self, src_root: Path):
        self.src_root = src_root
        self.pynomaly_root = src_root / "pynomaly"
        self.violations = []
        self.warnings = []
        self.metrics = {}
        
        # Define the expected Clean Architecture structure
        self.expected_structure = {
            "domain": {
                "required": True,
                "description": "Pure business logic layer",
                "subdirs": ["entities", "value_objects", "services", "exceptions"],
                "dependencies": []  # No external dependencies allowed
            },
            "application": {
                "required": True,
                "description": "Application logic layer",
                "subdirs": ["use_cases", "services", "dto"],
                "dependencies": ["domain"]  # Can only depend on domain
            },
            "infrastructure": {
                "required": True,
                "description": "External integrations layer",
                "subdirs": ["adapters", "persistence", "config", "monitoring"],
                "dependencies": ["domain", "application"]  # Can depend on inner layers
            },
            "presentation": {
                "required": True,
                "description": "User interface layer",
                "subdirs": ["api", "cli", "sdk", "web"],
                "dependencies": ["domain", "application", "infrastructure"]
            },
            "shared": {
                "required": False,
                "description": "Shared utilities and protocols",
                "subdirs": ["protocols", "utils"],
                "dependencies": []  # Utility layer
            }
        }
        
        # Define naming conventions
        self.naming_conventions = {
            "modules": r"^[a-z][a-z0-9_]*[a-z0-9]$",
            "classes": r"^[A-Z][a-zA-Z0-9]*$",
            "functions": r"^[a-z][a-z0-9_]*[a-z0-9]$",
            "constants": r"^[A-Z][A-Z0-9_]*[A-Z0-9]$",
            "private": r"^_[a-z][a-z0-9_]*[a-z0-9]$"
        }
        
        # Define architecture rules
        self.architecture_rules = [
            ArchitectureRule(
                "domain_purity",
                "Domain layer must not import from external libraries (except typing, abc, dataclasses)",
                "error",
                "_check_domain_purity"
            ),
            ArchitectureRule(
                "dependency_direction",
                "Dependencies must flow inward (outer layers depend on inner layers)",
                "error",
                "_check_dependency_direction"
            ),
            ArchitectureRule(
                "naming_conventions",
                "All modules, classes, and functions must follow Python naming conventions",
                "warning",
                "_check_naming_conventions"
            ),
            ArchitectureRule(
                "module_organization",
                "Each layer must have proper subdirectory organization",
                "error",
                "_check_module_organization"
            ),
            ArchitectureRule(
                "circular_imports",
                "No circular imports within or between layers",
                "error",
                "_check_circular_imports"
            ),
            ArchitectureRule(
                "interface_segregation",
                "Use protocols for interface definitions",
                "warning",
                "_check_interface_segregation"
            )
        ]
    
    def validate(self) -> ValidationResult:
        """
        Perform comprehensive source code structure validation.
        
        Returns:
            ValidationResult with validation status and details
        """
        print("🏗️ Starting source code structure validation...")
        
        if not self.pynomaly_root.exists():
            self.violations.append(f"Source root not found: {self.pynomaly_root}")
            return ValidationResult(False, self.violations, self.warnings, self.metrics)
        
        # Run all validation checks
        self._validate_basic_structure()
        self._validate_architecture_rules()
        self._collect_metrics()
        
        # Determine overall result
        passed = len(self.violations) == 0
        
        print(f"✅ Structure validation completed")
        print(f"   Violations: {len(self.violations)}")
        print(f"   Warnings: {len(self.warnings)}")
        
        return ValidationResult(passed, self.violations, self.warnings, self.metrics)
    
    def _validate_basic_structure(self):
        """Validate basic directory structure requirements."""
        print("📁 Validating basic structure...")
        
        # Check main package exists
        if not (self.pynomaly_root / "__init__.py").exists():
            self.violations.append("Missing __init__.py in main package")
        
        # Check required layers
        for layer, config in self.expected_structure.items():
            layer_path = self.pynomaly_root / layer
            
            if config["required"] and not layer_path.exists():
                self.violations.append(f"Missing required layer: {layer}")
                continue
            
            if not layer_path.exists():
                continue
                
            # Check layer has __init__.py
            if not (layer_path / "__init__.py").exists():
                self.violations.append(f"Missing __init__.py in layer: {layer}")
            
            # Check required subdirectories
            for subdir in config["subdirs"]:
                subdir_path = layer_path / subdir
                if not subdir_path.exists():
                    self.warnings.append(f"Missing subdirectory: {layer}/{subdir}")
                elif not (subdir_path / "__init__.py").exists():
                    self.violations.append(f"Missing __init__.py in: {layer}/{subdir}")
    
    def _validate_architecture_rules(self):
        """Validate all architecture rules."""
        print("🏛️ Validating architecture rules...")
        
        for rule in self.architecture_rules:
            print(f"  📋 Checking: {rule.name}")
            try:
                check_method = getattr(self, rule.check_function)
                check_method(rule)
            except Exception as e:
                self.violations.append(f"Error checking {rule.name}: {e}")
    
    def _check_domain_purity(self, rule: ArchitectureRule):
        """Check that domain layer has no external dependencies."""
        domain_path = self.pynomaly_root / "domain"
        if not domain_path.exists():
            return
        
        allowed_imports = {
            # Python standard library
            "typing", "abc", "dataclasses", "enum", "functools", 
            "itertools", "collections", "datetime", "uuid", "math",
            "re", "json", "pathlib", "logging", "time", "os", "sys",
            "warnings", "weakref", "copy", "decimal", "fractions",
            "operator", "random", "string", "textwrap", "unicodedata",
            # Special cases
            "__future__",  # Future annotations are allowed
        }
        
        for py_file in domain_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name.split('.')[0]
                            if module not in allowed_imports:
                                self.violations.append(
                                    f"Domain layer imports external dependency: "
                                    f"{py_file.relative_to(self.src_root)} imports {module}"
                                )
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            module = node.module.split('.')[0]
                            if (module not in allowed_imports and 
                                not module.startswith('pynomaly.domain') and
                                not module.startswith('pynomaly.shared')):  # Allow shared utilities
                                self.violations.append(
                                    f"Domain layer imports external dependency: "
                                    f"{py_file.relative_to(self.src_root)} imports from {node.module}"
                                )
            
            except Exception as e:
                self.warnings.append(f"Could not parse {py_file}: {e}")
    
    def _check_dependency_direction(self, rule: ArchitectureRule):
        """Check that dependencies flow in the correct direction."""
        layer_order = ["domain", "application", "infrastructure", "presentation"]
        
        for i, layer in enumerate(layer_order):
            layer_path = self.pynomaly_root / layer
            if not layer_path.exists():
                continue
            
            allowed_dependencies = set(layer_order[:i+1])  # Current and inner layers
            allowed_dependencies.add("shared")  # Shared is always allowed
            
            for py_file in layer_path.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            if node.module.startswith('pynomaly.'):
                                imported_layer = node.module.split('.')[1] if len(node.module.split('.')) > 1 else None
                                
                                if imported_layer and imported_layer in layer_order:
                                    if imported_layer not in allowed_dependencies:
                                        self.violations.append(
                                            f"Invalid dependency direction: {layer} layer "
                                            f"imports from {imported_layer} layer in "
                                            f"{py_file.relative_to(self.src_root)}"
                                        )
                
                except Exception as e:
                    self.warnings.append(f"Could not analyze dependencies in {py_file}: {e}")
    
    def _check_naming_conventions(self, rule: ArchitectureRule):
        """Check naming conventions for modules, classes, and functions."""
        for py_file in self.pynomaly_root.rglob("*.py"):
            # Check module name
            module_name = py_file.stem
            if module_name != "__init__" and not re.match(self.naming_conventions["modules"], module_name):
                self.warnings.append(
                    f"Module name violates convention: {py_file.relative_to(self.src_root)}"
                )
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if not re.match(self.naming_conventions["classes"], node.name):
                            self.warnings.append(
                                f"Class name violates convention: {node.name} in "
                                f"{py_file.relative_to(self.src_root)}"
                            )
                    
                    elif isinstance(node, ast.FunctionDef):
                        if node.name.startswith('_'):
                            if not re.match(self.naming_conventions["private"], node.name):
                                self.warnings.append(
                                    f"Private function name violates convention: {node.name} in "
                                    f"{py_file.relative_to(self.src_root)}"
                                )
                        else:
                            if not re.match(self.naming_conventions["functions"], node.name):
                                self.warnings.append(
                                    f"Function name violates convention: {node.name} in "
                                    f"{py_file.relative_to(self.src_root)}"
                                )
            
            except Exception as e:
                self.warnings.append(f"Could not check naming in {py_file}: {e}")
    
    def _check_module_organization(self, rule: ArchitectureRule):
        """Check that modules are properly organized within layers."""
        for layer, config in self.expected_structure.items():
            layer_path = self.pynomaly_root / layer
            if not layer_path.exists():
                continue
            
            # Check for files directly in layer root (should be minimal)
            python_files = [f for f in layer_path.glob("*.py") if f.name != "__init__.py"]
            if len(python_files) > 2:  # Allow __init__.py and maybe one other
                self.warnings.append(
                    f"Too many files in layer root: {layer}/ has {len(python_files)} Python files"
                )
            
            # Check subdirectory organization
            for subdir in config["subdirs"]:
                subdir_path = layer_path / subdir
                if subdir_path.exists():
                    py_files = list(subdir_path.glob("*.py"))
                    if len(py_files) == 0:
                        self.warnings.append(f"Empty subdirectory: {layer}/{subdir}")
    
    def _check_circular_imports(self, rule: ArchitectureRule):
        """Check for circular imports (simplified check)."""
        # This is a simplified check - a full check would require import graph analysis
        import_graph = {}
        
        for py_file in self.pynomaly_root.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            module_path = str(py_file.relative_to(self.src_root)).replace('/', '.').replace('.py', '')
            imports = set()
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith('pynomaly.'):
                            imports.add(node.module)
                
                import_graph[module_path] = imports
            
            except Exception:
                continue
        
        # Simple circular import detection
        for module, imports in import_graph.items():
            for imported in imports:
                if imported in import_graph:
                    if module in import_graph[imported]:
                        self.violations.append(
                            f"Potential circular import: {module} <-> {imported}"
                        )
    
    def _check_interface_segregation(self, rule: ArchitectureRule):
        """Check for proper use of protocols for interfaces."""
        protocol_count = 0
        abc_count = 0
        
        for py_file in self.pynomaly_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check for Protocol usage
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id == "Protocol":
                                protocol_count += 1
                            elif isinstance(base, ast.Attribute) and base.attr == "Protocol":
                                protocol_count += 1
                        
                        # Check for ABC usage
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id in ["ABC", "AbstractBaseClass"]:
                                abc_count += 1
            
            except Exception:
                continue
        
        if protocol_count == 0 and abc_count > 0:
            self.warnings.append(
                "Consider using typing.Protocol instead of ABC for interface definitions"
            )
        
        self.metrics["protocol_count"] = protocol_count
        self.metrics["abc_count"] = abc_count
    
    def _collect_metrics(self):
        """Collect structural metrics."""
        print("📊 Collecting metrics...")
        
        # Count files and directories
        total_files = len(list(self.pynomaly_root.rglob("*.py")))
        total_dirs = len([d for d in self.pynomaly_root.rglob("*") if d.is_dir()])
        
        # Count by layer
        layer_metrics = {}
        for layer in self.expected_structure.keys():
            layer_path = self.pynomaly_root / layer
            if layer_path.exists():
                layer_files = len(list(layer_path.rglob("*.py")))
                layer_dirs = len([d for d in layer_path.rglob("*") if d.is_dir()])
                layer_metrics[layer] = {"files": layer_files, "dirs": layer_dirs}
        
        # Calculate depth
        max_depth = 0
        for path in self.pynomaly_root.rglob("*"):
            depth = len(path.relative_to(self.pynomaly_root).parts)
            max_depth = max(max_depth, depth)
        
        self.metrics.update({
            "total_python_files": total_files,
            "total_directories": total_dirs,
            "max_directory_depth": max_depth,
            "layer_metrics": layer_metrics,
            "violations_count": len(self.violations),
            "warnings_count": len(self.warnings)
        })


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate source code structure against Clean Architecture"
    )
    parser.add_argument(
        '--src-root',
        type=Path,
        default=Path('src'),
        help="Path to source code root directory"
    )
    parser.add_argument(
        '--output',
        type=Path,
        help="Output file for detailed results (JSON format)"
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    if not args.src_root.exists():
        print(f"❌ Source root not found: {args.src_root}")
        return 1
    
    validator = SourceStructureValidator(args.src_root)
    result = validator.validate()
    
    # Output detailed results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "passed": result.passed,
                "violations": result.violations,
                "warnings": result.warnings,
                "metrics": result.metrics
            }, f, indent=2)
    
    # Print summary
    print(f"\n📊 Source Structure Validation Summary:")
    print(f"  • Total Python files: {result.metrics.get('total_python_files', 0)}")
    print(f"  • Total directories: {result.metrics.get('total_directories', 0)}")
    print(f"  • Max directory depth: {result.metrics.get('max_directory_depth', 0)}")
    print(f"  • Violations: {len(result.violations)}")
    print(f"  • Warnings: {len(result.warnings)}")
    
    if result.violations:
        print(f"\n❌ Violations found:")
        for violation in result.violations[:10]:  # Show first 10
            print(f"  • {violation}")
        if len(result.violations) > 10:
            print(f"  ... and {len(result.violations) - 10} more")
    
    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings[:5]:  # Show first 5
            print(f"  • {warning}")
        if len(result.warnings) > 5:
            print(f"  ... and {len(result.warnings) - 5} more")
    
    # Print layer-specific metrics
    if "layer_metrics" in result.metrics:
        print(f"\n📁 Layer Metrics:")
        for layer, metrics in result.metrics["layer_metrics"].items():
            print(f"  • {layer}: {metrics['files']} files, {metrics['dirs']} directories")
    
    # Determine exit code
    if result.violations or (args.strict and result.warnings):
        print("❌ Source structure validation failed!")
        return 1
    else:
        print("✅ Source structure validation passed!")
        return 0


if __name__ == '__main__':
    exit(main())