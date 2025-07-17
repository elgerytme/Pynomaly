"""
Domain leakage checker for repository governance.
Detects architectural violations and improper cross-domain dependencies.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .base_checker import BaseChecker


class DomainLeakageChecker(BaseChecker):
    """Checker for domain leakage and architectural violations."""
    
    def __init__(self, root_path: Path):
        """Initialize the domain leakage checker."""
        super().__init__(root_path)
        self.domain_boundaries = {
            "ai": ["machine_learning"],
            "data": ["anomaly_detection", "data_observability", "data_platform"],
            "business": ["administration", "analytics", "compliance"],
            "software": ["core", "interfaces", "enterprise"]
        }
        
        self.forbidden_imports = [
            r"from monorepo\.",
            r"import monorepo\.",
            r"from packages\.",
            r"import packages\.",
        ]
        
        self.allowed_cross_domain_imports = [
            "typing",
            "datetime",
            "pathlib",
            "logging",
            "json",
            "os",
            "sys",
            "dataclasses",
            "enum",
            "abc",
            "uuid",
            "functools",
            "collections",
            "itertools",
        ]
    
    def check(self) -> Dict:
        """Run domain leakage checks."""
        violations = []
        
        # Check for monorepo imports
        monorepo_imports = self._find_monorepo_imports()
        if monorepo_imports:
            violations.append({
                "type": "monorepo_imports",
                "severity": "high",
                "message": f"Found {len(monorepo_imports)} monorepo imports",
                "imports": monorepo_imports[:30],  # Limit for readability
                "total_count": len(monorepo_imports)
            })
        
        # Check for cross-domain imports
        cross_domain_imports = self._find_cross_domain_imports()
        if cross_domain_imports:
            violations.append({
                "type": "cross_domain_imports",
                "severity": "medium",
                "message": f"Found {len(cross_domain_imports)} cross-domain imports",
                "imports": cross_domain_imports[:20],
                "total_count": len(cross_domain_imports)
            })
        
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies()
        if circular_deps:
            violations.append({
                "type": "circular_dependencies",
                "severity": "high",
                "message": f"Found {len(circular_deps)} circular dependencies",
                "dependencies": circular_deps,
                "total_count": len(circular_deps)
            })
        
        # Check for missing domain boundaries
        missing_boundaries = self._find_missing_domain_boundaries()
        if missing_boundaries:
            violations.append({
                "type": "missing_domain_boundaries",
                "severity": "medium",
                "message": f"Found {len(missing_boundaries)} packages missing proper domain structure",
                "packages": missing_boundaries,
                "total_count": len(missing_boundaries)
            })
        
        # Check for shared business logic violations
        shared_logic_violations = self._find_shared_logic_violations()
        if shared_logic_violations:
            violations.append({
                "type": "shared_logic_violations",
                "severity": "medium",
                "message": f"Found {len(shared_logic_violations)} shared business logic violations",
                "violations": shared_logic_violations,
                "total_count": len(shared_logic_violations)
            })
        
        return {
            "violations": violations,
            "total_violations": len(violations),
            "score": self.calculate_score(violations, penalty_per_violation=15),
            "recommendations": self._generate_recommendations(violations)
        }
    
    def _find_monorepo_imports(self) -> List[Dict]:
        """Find all monorepo imports in Python files."""
        monorepo_imports = []
        
        for python_file in self.get_python_files():
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for forbidden import patterns
                for pattern in self.forbidden_imports:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = content.split('\n')[line_num - 1].strip()
                        
                        monorepo_imports.append({
                            "file": str(python_file.relative_to(self.root_path)),
                            "line": line_num,
                            "content": line_content,
                            "pattern": pattern
                        })
                        
            except Exception as e:
                continue  # Skip files that can't be read
        
        return monorepo_imports
    
    def _find_cross_domain_imports(self) -> List[Dict]:
        """Find cross-domain imports that violate boundaries."""
        cross_domain_imports = []
        
        for python_file in self.get_python_files():
            try:
                file_domain = self._get_file_domain(python_file)
                if not file_domain:
                    continue
                
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            import_domain = self._get_import_domain(node)
                            if (import_domain and 
                                import_domain != file_domain and
                                not self._is_allowed_cross_domain_import(node)):
                                
                                cross_domain_imports.append({
                                    "file": str(python_file.relative_to(self.root_path)),
                                    "line": node.lineno,
                                    "from_domain": file_domain,
                                    "to_domain": import_domain,
                                    "import_statement": self._format_import(node)
                                })
                except SyntaxError:
                    continue  # Skip files with syntax errors
                    
            except Exception as e:
                continue
        
        return cross_domain_imports
    
    def _find_circular_dependencies(self) -> List[Dict]:
        """Find circular dependencies between packages."""
        # Build dependency graph
        dependency_graph = {}
        
        for python_file in self.get_python_files():
            try:
                file_package = self._get_file_package(python_file)
                if not file_package:
                    continue
                
                if file_package not in dependency_graph:
                    dependency_graph[file_package] = set()
                
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            import_package = self._get_import_package(node)
                            if import_package and import_package != file_package:
                                dependency_graph[file_package].add(import_package)
                except SyntaxError:
                    continue
                    
            except Exception as e:
                continue
        
        # Find cycles
        return self._find_cycles_in_graph(dependency_graph)
    
    def _find_missing_domain_boundaries(self) -> List[Dict]:
        """Find packages missing proper domain structure."""
        missing_boundaries = []
        required_dirs = ["domain", "application", "infrastructure", "interfaces"]
        
        for package_dir in self.get_package_directories():
            existing_dirs = [d.name for d in package_dir.iterdir() if d.is_dir()]
            missing_dirs = [d for d in required_dirs if d not in existing_dirs]
            
            if missing_dirs:
                missing_boundaries.append({
                    "package": str(package_dir.relative_to(self.root_path)),
                    "missing_directories": missing_dirs,
                    "existing_directories": existing_dirs
                })
        
        return missing_boundaries
    
    def _find_shared_logic_violations(self) -> List[Dict]:
        """Find instances of shared business logic that should be abstracted."""
        violations = []
        
        # Look for duplicate entity/service names across domains
        entity_names = {}
        service_names = {}
        
        for python_file in self.get_python_files():
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for class definitions
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name
                            file_domain = self._get_file_domain(python_file)
                            
                            if class_name.endswith('Entity') or class_name.endswith('Service'):
                                container = entity_names if class_name.endswith('Entity') else service_names
                                
                                if class_name not in container:
                                    container[class_name] = []
                                container[class_name].append({
                                    "file": str(python_file.relative_to(self.root_path)),
                                    "domain": file_domain,
                                    "line": node.lineno
                                })
                except SyntaxError:
                    continue
                    
            except Exception as e:
                continue
        
        # Find duplicates across domains
        for class_name, occurrences in {**entity_names, **service_names}.items():
            if len(occurrences) > 1:
                domains = set(occ["domain"] for occ in occurrences if occ["domain"])
                if len(domains) > 1:
                    violations.append({
                        "class_name": class_name,
                        "domains": list(domains),
                        "occurrences": occurrences
                    })
        
        return violations
    
    def _get_file_domain(self, file_path: Path) -> str:
        """Get the domain of a file based on its path."""
        parts = file_path.parts
        if "packages" in parts:
            try:
                packages_index = parts.index("packages")
                if packages_index + 1 < len(parts):
                    return parts[packages_index + 1]
            except ValueError:
                pass
        return ""
    
    def _get_file_package(self, file_path: Path) -> str:
        """Get the package of a file based on its path."""
        parts = file_path.parts
        if "packages" in parts:
            try:
                packages_index = parts.index("packages")
                if packages_index + 2 < len(parts):
                    return f"{parts[packages_index + 1]}.{parts[packages_index + 2]}"
            except ValueError:
                pass
        return ""
    
    def _get_import_domain(self, node: ast.AST) -> str:
        """Get the domain from an import statement."""
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("packages."):
                parts = node.module.split(".")
                if len(parts) > 1:
                    return parts[1]
        return ""
    
    def _get_import_package(self, node: ast.AST) -> str:
        """Get the package from an import statement."""
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("packages."):
                parts = node.module.split(".")
                if len(parts) > 2:
                    return f"{parts[1]}.{parts[2]}"
        return ""
    
    def _is_allowed_cross_domain_import(self, node: ast.AST) -> bool:
        """Check if a cross-domain import is allowed."""
        if isinstance(node, ast.ImportFrom):
            if node.module:
                return any(node.module.startswith(allowed) for allowed in self.allowed_cross_domain_imports)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(allowed) for allowed in self.allowed_cross_domain_imports):
                    return True
        return False
    
    def _format_import(self, node: ast.AST) -> str:
        """Format an import statement for display."""
        if isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            return f"from {node.module} import {names}"
        elif isinstance(node, ast.Import):
            names = ", ".join(alias.name for alias in node.names)
            return f"import {names}"
        return ""
    
    def _find_cycles_in_graph(self, graph: Dict[str, Set[str]]) -> List[Dict]:
        """Find cycles in a dependency graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append({
                    "cycle": cycle,
                    "length": len(cycle) - 1
                })
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _generate_recommendations(self, violations: List[Dict]) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        for violation in violations:
            if violation["type"] == "monorepo_imports":
                recommendations.append(
                    f"Replace {violation['total_count']} monorepo imports with proper "
                    f"domain-specific imports or dependency injection"
                )
            elif violation["type"] == "cross_domain_imports":
                recommendations.append(
                    f"Refactor {violation['total_count']} cross-domain imports to use "
                    f"proper interfaces and dependency inversion"
                )
            elif violation["type"] == "circular_dependencies":
                recommendations.append(
                    f"Break {violation['total_count']} circular dependencies by "
                    f"introducing interfaces or extracting shared components"
                )
            elif violation["type"] == "missing_domain_boundaries":
                recommendations.append(
                    f"Add proper domain structure to {violation['total_count']} packages "
                    f"(domain, application, infrastructure, interfaces)"
                )
            elif violation["type"] == "shared_logic_violations":
                recommendations.append(
                    f"Extract {violation['total_count']} shared business logic violations "
                    f"into proper domain abstractions or shared kernel"
                )
        
        return recommendations