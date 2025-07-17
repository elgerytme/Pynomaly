"""
Architecture checker for repository governance.
Checks for clean architecture compliance and design patterns.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .base_checker import BaseChecker


class ArchitectureChecker(BaseChecker):
    """Checker for architectural compliance and design patterns."""
    
    def __init__(self, root_path: Path):
        """Initialize the architecture checker."""
        super().__init__(root_path)
        
        # Clean architecture layers
        self.layers = {
            "domain": {"weight": 1, "depends_on": []},
            "application": {"weight": 2, "depends_on": ["domain"]},
            "infrastructure": {"weight": 3, "depends_on": ["domain", "application"]},
            "interfaces": {"weight": 4, "depends_on": ["domain", "application", "infrastructure"]}
        }
        
        # Design patterns to check for
        self.design_patterns = {
            "repository_pattern": r"class.*Repository.*:",
            "service_pattern": r"class.*Service.*:",
            "factory_pattern": r"class.*Factory.*:",
            "builder_pattern": r"class.*Builder.*:",
            "strategy_pattern": r"class.*Strategy.*:",
            "observer_pattern": r"class.*Observer.*:",
            "command_pattern": r"class.*Command.*:"
        }
        
        # Anti-patterns to detect
        self.anti_patterns = {
            "god_class": {"max_methods": 20, "max_lines": 500},
            "long_method": {"max_lines": 50},
            "long_parameter_list": {"max_params": 5},
            "feature_envy": {"max_external_calls": 10}
        }
    
    def check(self) -> Dict:
        """Run architecture checks."""
        violations = []
        
        # Check clean architecture layer violations
        layer_violations = self._check_layer_violations()
        if layer_violations:
            violations.append({
                "type": "layer_violations",
                "severity": "high",
                "message": f"Found {len(layer_violations)} clean architecture layer violations",
                "violations": layer_violations,
                "total_count": len(layer_violations)
            })
        
        # Check for missing design patterns
        missing_patterns = self._check_missing_patterns()
        if missing_patterns:
            violations.append({
                "type": "missing_design_patterns",
                "severity": "medium",
                "message": f"Found {len(missing_patterns)} packages missing recommended design patterns",
                "patterns": missing_patterns,
                "total_count": len(missing_patterns)
            })
        
        # Check for anti-patterns
        anti_pattern_violations = self._check_anti_patterns()
        if anti_pattern_violations:
            violations.append({
                "type": "anti_patterns",
                "severity": "medium",
                "message": f"Found {len(anti_pattern_violations)} anti-pattern violations",
                "violations": anti_pattern_violations,
                "total_count": len(anti_pattern_violations)
            })
        
        # Check dependency injection usage
        di_violations = self._check_dependency_injection()
        if di_violations:
            violations.append({
                "type": "dependency_injection_violations",
                "severity": "medium",
                "message": f"Found {len(di_violations)} dependency injection violations",
                "violations": di_violations,
                "total_count": len(di_violations)
            })
        
        # Check for proper abstraction usage
        abstraction_violations = self._check_abstraction_violations()
        if abstraction_violations:
            violations.append({
                "type": "abstraction_violations",
                "severity": "medium",
                "message": f"Found {len(abstraction_violations)} abstraction violations",
                "violations": abstraction_violations,
                "total_count": len(abstraction_violations)
            })
        
        # Check for SOLID principles violations
        solid_violations = self._check_solid_principles()
        if solid_violations:
            violations.append({
                "type": "solid_violations",
                "severity": "high",
                "message": f"Found {len(solid_violations)} SOLID principle violations",
                "violations": solid_violations,
                "total_count": len(solid_violations)
            })
        
        return {
            "violations": violations,
            "total_violations": len(violations),
            "score": self.calculate_score(violations, penalty_per_violation=8),
            "recommendations": self._generate_recommendations(violations)
        }
    
    def _check_layer_violations(self) -> List[Dict]:
        """Check for clean architecture layer violations."""
        violations = []
        
        for python_file in self.get_python_files():
            file_layer = self._get_file_layer(python_file)
            if not file_layer:
                continue
            
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            import_layer = self._get_import_layer(node)
                            if import_layer and self._is_layer_violation(file_layer, import_layer):
                                violations.append({
                                    "file": str(python_file.relative_to(self.root_path)),
                                    "file_layer": file_layer,
                                    "import_layer": import_layer,
                                    "line": node.lineno,
                                    "import_statement": self._format_import(node)
                                })
                except SyntaxError:
                    continue
                    
            except Exception as e:
                continue
        
        return violations
    
    def _check_missing_patterns(self) -> List[Dict]:
        """Check for missing design patterns in packages."""
        missing_patterns = []
        
        for package_dir in self.get_package_directories():
            found_patterns = set()
            
            for python_file in package_dir.rglob("*.py"):
                try:
                    with open(python_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern_name, pattern_regex in self.design_patterns.items():
                        if re.search(pattern_regex, content):
                            found_patterns.add(pattern_name)
                            
                except Exception as e:
                    continue
            
            # Check for expected patterns based on package type
            expected_patterns = self._get_expected_patterns(package_dir)
            missing = expected_patterns - found_patterns
            
            if missing:
                missing_patterns.append({
                    "package": str(package_dir.relative_to(self.root_path)),
                    "missing_patterns": list(missing),
                    "found_patterns": list(found_patterns),
                    "expected_patterns": list(expected_patterns)
                })
        
        return missing_patterns
    
    def _check_anti_patterns(self) -> List[Dict]:
        """Check for anti-patterns in code."""
        violations = []
        
        for python_file in self.get_python_files():
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Check for god class
                            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                            if len(methods) > self.anti_patterns["god_class"]["max_methods"]:
                                violations.append({
                                    "type": "god_class",
                                    "file": str(python_file.relative_to(self.root_path)),
                                    "class": node.name,
                                    "line": node.lineno,
                                    "method_count": len(methods),
                                    "max_allowed": self.anti_patterns["god_class"]["max_methods"]
                                })
                        
                        elif isinstance(node, ast.FunctionDef):
                            # Check for long method
                            if hasattr(node, 'end_lineno') and node.end_lineno:
                                method_lines = node.end_lineno - node.lineno
                                if method_lines > self.anti_patterns["long_method"]["max_lines"]:
                                    violations.append({
                                        "type": "long_method",
                                        "file": str(python_file.relative_to(self.root_path)),
                                        "method": node.name,
                                        "line": node.lineno,
                                        "line_count": method_lines,
                                        "max_allowed": self.anti_patterns["long_method"]["max_lines"]
                                    })
                            
                            # Check for long parameter list
                            param_count = len(node.args.args)
                            if param_count > self.anti_patterns["long_parameter_list"]["max_params"]:
                                violations.append({
                                    "type": "long_parameter_list",
                                    "file": str(python_file.relative_to(self.root_path)),
                                    "method": node.name,
                                    "line": node.lineno,
                                    "param_count": param_count,
                                    "max_allowed": self.anti_patterns["long_parameter_list"]["max_params"]
                                })
                                
                except SyntaxError:
                    continue
                    
            except Exception as e:
                continue
        
        return violations
    
    def _check_dependency_injection(self) -> List[Dict]:
        """Check for proper dependency injection usage."""
        violations = []
        
        for python_file in self.get_python_files():
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for direct instantiation in constructors
                if "def __init__" in content and "= " in content:
                    # Check for direct instantiation patterns
                    direct_instantiation_patterns = [
                        r"self\.\w+\s*=\s*\w+\(\)",
                        r"self\.\w+\s*=\s*\w+\.\w+\(\)"
                    ]
                    
                    for pattern in direct_instantiation_patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            violations.append({
                                "file": str(python_file.relative_to(self.root_path)),
                                "line": line_num,
                                "pattern": match.group(),
                                "suggestion": "Consider using dependency injection"
                            })
                            
            except Exception as e:
                continue
        
        return violations
    
    def _check_abstraction_violations(self) -> List[Dict]:
        """Check for proper abstraction usage."""
        violations = []
        
        for python_file in self.get_python_files():
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    
                    # Check for missing abstract base classes
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Check if class looks like it should be abstract
                            if (node.name.endswith('Interface') or 
                                node.name.endswith('Protocol') or
                                node.name.endswith('Abstract')):
                                
                                # Check if it actually uses ABC
                                has_abc = any(
                                    isinstance(base, ast.Name) and base.id == 'ABC'
                                    for base in node.bases
                                )
                                
                                if not has_abc:
                                    violations.append({
                                        "type": "missing_abc",
                                        "file": str(python_file.relative_to(self.root_path)),
                                        "class": node.name,
                                        "line": node.lineno,
                                        "suggestion": "Consider using ABC for abstract classes"
                                    })
                                    
                except SyntaxError:
                    continue
                    
            except Exception as e:
                continue
        
        return violations
    
    def _check_solid_principles(self) -> List[Dict]:
        """Check for SOLID principle violations."""
        violations = []
        
        for python_file in self.get_python_files():
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Single Responsibility Principle
                            responsibilities = self._count_responsibilities(node)
                            if responsibilities > 3:
                                violations.append({
                                    "principle": "Single Responsibility",
                                    "file": str(python_file.relative_to(self.root_path)),
                                    "class": node.name,
                                    "line": node.lineno,
                                    "responsibilities": responsibilities,
                                    "suggestion": "Consider splitting into smaller classes"
                                })
                                
                except SyntaxError:
                    continue
                    
            except Exception as e:
                continue
        
        return violations
    
    def _get_file_layer(self, file_path: Path) -> str:
        """Get the architectural layer of a file."""
        parts = file_path.parts
        for layer in self.layers:
            if layer in parts:
                return layer
        return ""
    
    def _get_import_layer(self, node: ast.AST) -> str:
        """Get the layer from an import statement."""
        if isinstance(node, ast.ImportFrom):
            if node.module:
                for layer in self.layers:
                    if f".{layer}." in node.module or node.module.endswith(f".{layer}"):
                        return layer
        return ""
    
    def _is_layer_violation(self, from_layer: str, to_layer: str) -> bool:
        """Check if importing from one layer to another is a violation."""
        if not from_layer or not to_layer:
            return False
        
        from_weight = self.layers[from_layer]["weight"]
        to_weight = self.layers[to_layer]["weight"]
        
        # Lower layers should not depend on higher layers
        return from_weight < to_weight
    
    def _format_import(self, node: ast.AST) -> str:
        """Format an import statement for display."""
        if isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            return f"from {node.module} import {names}"
        elif isinstance(node, ast.Import):
            names = ", ".join(alias.name for alias in node.names)
            return f"import {names}"
        return ""
    
    def _get_expected_patterns(self, package_dir: Path) -> Set[str]:
        """Get expected design patterns for a package."""
        expected = set()
        
        # Check for domain layer
        if (package_dir / "domain").exists():
            expected.add("repository_pattern")
        
        # Check for application layer
        if (package_dir / "application").exists():
            expected.add("service_pattern")
        
        # Check for infrastructure layer
        if (package_dir / "infrastructure").exists():
            expected.add("factory_pattern")
        
        return expected
    
    def _count_responsibilities(self, class_node: ast.ClassDef) -> int:
        """Count the number of responsibilities in a class."""
        responsibilities = 0
        
        # Count different types of methods
        method_types = set()
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("get_"):
                    method_types.add("getter")
                elif node.name.startswith("set_"):
                    method_types.add("setter")
                elif node.name.startswith("create_"):
                    method_types.add("creator")
                elif node.name.startswith("update_"):
                    method_types.add("updater")
                elif node.name.startswith("delete_"):
                    method_types.add("deleter")
                elif node.name.startswith("validate_"):
                    method_types.add("validator")
                elif node.name.startswith("calculate_"):
                    method_types.add("calculator")
                else:
                    method_types.add("other")
        
        return len(method_types)
    
    def _generate_recommendations(self, violations: List[Dict]) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        for violation in violations:
            if violation["type"] == "layer_violations":
                recommendations.append(
                    f"Fix {violation['total_count']} clean architecture layer violations: "
                    f"ensure lower layers don't depend on higher layers"
                )
            elif violation["type"] == "missing_design_patterns":
                recommendations.append(
                    f"Implement missing design patterns in {violation['total_count']} packages: "
                    f"Repository, Service, Factory patterns"
                )
            elif violation["type"] == "anti_patterns":
                recommendations.append(
                    f"Refactor {violation['total_count']} anti-pattern violations: "
                    f"break down god classes, long methods, and parameter lists"
                )
            elif violation["type"] == "dependency_injection_violations":
                recommendations.append(
                    f"Implement dependency injection for {violation['total_count']} violations: "
                    f"avoid direct instantiation in constructors"
                )
            elif violation["type"] == "abstraction_violations":
                recommendations.append(
                    f"Add proper abstractions for {violation['total_count']} violations: "
                    f"use ABC for abstract classes and interfaces"
                )
            elif violation["type"] == "solid_violations":
                recommendations.append(
                    f"Address {violation['total_count']} SOLID principle violations: "
                    f"ensure single responsibility and proper abstractions"
                )
        
        return recommendations