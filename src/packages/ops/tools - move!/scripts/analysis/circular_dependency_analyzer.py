#!/usr/bin/env python3
"""Circular dependency analyzer for Pynomaly codebase."""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract imports from a Python file."""
    
    def __init__(self):
        self.imports = []
    
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith('monorepo'):
                self.imports.append(alias.name)
    
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith('monorepo'):
            self.imports.append(node.module)


def get_imports_from_file(file_path: Path) -> List[str]:
    """Extract pynomaly imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def get_module_path(file_path: Path, src_root: Path) -> str:
    """Convert file path to module path."""
    relative_path = file_path.relative_to(src_root)
    module_parts = list(relative_path.parts[:-1])  # Remove filename
    if relative_path.stem != '__init__':
        module_parts.append(relative_path.stem)
    return '.'.join(module_parts)


def categorize_layer(module_path: str) -> str:
    """Categorize module into architectural layer."""
    if 'domain' in module_path:
        return 'domain'
    elif 'application' in module_path:
        return 'application'
    elif 'infrastructure' in module_path:
        return 'infrastructure'
    elif 'presentation' in module_path:
        return 'presentation'
    elif 'shared' in module_path:
        return 'shared'
    else:
        return 'other'


def is_violation(from_layer: str, to_layer: str) -> bool:
    """Check if import represents a Clean Architecture violation."""
    # Define allowed dependencies (from -> to)
    allowed = {
        'presentation': {'application', 'domain', 'shared'},
        'application': {'domain', 'shared'},
        'infrastructure': {'domain', 'shared'},
        'domain': {'shared'},
        'shared': set(),
        'other': {'domain', 'application', 'infrastructure', 'presentation', 'shared'}
    }
    
    return to_layer not in allowed.get(from_layer, set())


def find_circular_dependencies(dependencies: Dict[str, Set[str]]) -> List[List[str]]:
    """Find circular dependencies using DFS."""
    cycles = []
    visited = set()
    rec_stack = set()
    
    def dfs(node: str, path: List[str]):
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return
        
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in dependencies.get(node, set()):
            dfs(neighbor, path[:])
        
        rec_stack.remove(node)
    
    for node in dependencies:
        if node not in visited:
            dfs(node, [])
    
    return cycles


def analyze_codebase():
    """Analyze the Pynomaly codebase for circular dependencies."""
    src_root = Path('./src/pynomaly')
    if not src_root.exists():
        print("Error: src/pynomaly directory not found")
        return
    
    # Collect all Python files and their imports
    module_imports = {}
    layer_violations = []
    
    for py_file in src_root.rglob('*.py'):
        if py_file.name == '__init__.py' and len(list(py_file.parent.glob('*.py'))) == 1:
            continue  # Skip empty __init__.py files
        
        module_path = get_module_path(py_file, src_root.parent)
        imports = get_imports_from_file(py_file)
        
        if imports:
            module_imports[module_path] = set(imports)
            
            # Check for layer violations
            from_layer = categorize_layer(module_path)
            for imported_module in imports:
                to_layer = categorize_layer(imported_module)
                if is_violation(from_layer, to_layer):
                    layer_violations.append({
                        'file': str(py_file),
                        'module': module_path,
                        'from_layer': from_layer,
                        'to_layer': to_layer,
                        'imported_module': imported_module
                    })
    
    # Find circular dependencies
    cycles = find_circular_dependencies(module_imports)
    
    # Report results
    print("=" * 80)
    print("CIRCULAR DEPENDENCY ANALYSIS REPORT")
    print("=" * 80)
    
    print("\n🔍 LAYER VIOLATIONS (Clean Architecture)")
    print("-" * 50)
    if layer_violations:
        layer_counts = defaultdict(int)
        for violation in layer_violations:
            layer_key = f"{violation['from_layer']} -> {violation['to_layer']}"
            layer_counts[layer_key] += 1
        
        print("Summary by layer violation type:")
        for layer_violation, count in sorted(layer_counts.items()):
            print(f"  {layer_violation}: {count} violations")
        
        print("\nDetailed violations:")
        for violation in layer_violations[:20]:  # Show first 20
            print(f"  ❌ {violation['module']} ({violation['from_layer']})")
            print(f"     imports {violation['imported_module']} ({violation['to_layer']})")
            print(f"     in file: {violation['file']}")
            print()
        
        if len(layer_violations) > 20:
            print(f"  ... and {len(layer_violations) - 20} more violations")
    else:
        print("  ✅ No layer violations found!")
    
    print(f"\n🔄 CIRCULAR DEPENDENCIES")
    print("-" * 50)
    if cycles:
        print(f"Found {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles[:10], 1):  # Show first 10
            print(f"  {i}. {' -> '.join(cycle)}")
        
        if len(cycles) > 10:
            print(f"  ... and {len(cycles) - 10} more cycles")
    else:
        print("  ✅ No circular dependencies found!")
    
    print(f"\n📊 SUMMARY")
    print("-" * 50)
    print(f"  Total modules analyzed: {len(module_imports)}")
    print(f"  Layer violations: {len(layer_violations)}")
    print(f"  Circular dependencies: {len(cycles)}")
    
    # Most problematic files
    if layer_violations:
        violation_files = defaultdict(int)
        for v in layer_violations:
            violation_files[v['file']] += 1
        
        print(f"\n🚨 MOST PROBLEMATIC FILES")
        print("-" * 50)
        for file_path, count in sorted(violation_files.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {count} violations: {file_path}")


if __name__ == '__main__':
    analyze_codebase()