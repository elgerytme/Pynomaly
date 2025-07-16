#!/usr/bin/env python3
"""
Comprehensive circular dependency analysis for Pynomaly project.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import re

def extract_imports_from_file(file_path):
    """Extract all imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return []
    
    # Extract both 'from' and 'import' statements
    imports = []
    
    # Pattern for 'from X import Y' statements
    from_pattern = r'^from\s+([^\s]+)\s+import'
    
    # Pattern for 'import X' statements
    import_pattern = r'^import\s+([^\s,]+)'
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('#') or not line:
            continue
            
        # Check for 'from' imports
        from_match = re.match(from_pattern, line)
        if from_match:
            module = from_match.group(1)
            if module.startswith('monorepo') or module.startswith('packages') or module.startswith('.'):
                imports.append(module)
        
        # Check for 'import' statements
        import_match = re.match(import_pattern, line)
        if import_match:
            module = import_match.group(1)
            if module.startswith('monorepo') or module.startswith('packages'):
                imports.append(module)
    
    return imports

def normalize_module_path(module_name, file_path):
    """Normalize relative imports to absolute module paths."""
    if module_name.startswith('.'):
        # Handle relative imports
        file_dir = Path(file_path).parent
        rel_path = str(file_dir).replace('/', '.').replace('\\', '.')
        # Remove leading 'src.'
        if rel_path.startswith('src.'):
            rel_path = rel_path[4:]
        
        # Calculate the absolute import
        dots = len(module_name) - len(module_name.lstrip('.'))
        parent_parts = rel_path.split('.')
        
        if dots == 1:
            return rel_path + '.' + module_name[1:]
        else:
            # Go up directories for multiple dots
            target_parts = parent_parts[:-dots+1] if dots > 1 else parent_parts
            return '.'.join(target_parts) + ('.' + module_name.lstrip('.') if module_name.lstrip('.') else '')
    
    return module_name

def get_package_from_module(module_name):
    """Extract the main package from a module name."""
    parts = module_name.split('.')
    if len(parts) >= 2:
        if parts[0] == 'monorepo':
            return '.'.join(parts[:2])  # e.g., 'monorepo.domain'
        elif parts[0] == 'packages':
            return '.'.join(parts[:2])  # e.g., 'packages.core'
    return parts[0]

def find_python_files(directory):
    """Find all Python files in a directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def build_dependency_graph():
    """Build a dependency graph of all modules."""
    src_dir = 'src'
    python_files = find_python_files(src_dir)
    
    # Module to package mapping
    module_to_package = {}
    package_dependencies = defaultdict(set)
    module_dependencies = defaultdict(set)
    
    print(f'Analyzing {len(python_files)} Python files...')
    
    for file_path in python_files:
        # Convert file path to module name
        rel_path = os.path.relpath(file_path, src_dir)
        module_parts = rel_path.replace(os.sep, '.').replace('.py', '').split('.')
        
        # Skip __init__ files for module name
        if module_parts[-1] == '__init__':
            module_name = '.'.join(module_parts[:-1])
        else:
            module_name = '.'.join(module_parts)
        
        # Skip empty module names
        if not module_name:
            continue
            
        # Get package name
        package_name = get_package_from_module(module_name)
        module_to_package[module_name] = package_name
        
        # Extract imports
        imports = extract_imports_from_file(file_path)
        
        for imp in imports:
            normalized_imp = normalize_module_path(imp, file_path)
            
            # Skip self-imports and standard library
            if normalized_imp == module_name or not (normalized_imp.startswith('monorepo') or normalized_imp.startswith('packages')):
                continue
            
            # Add to module dependencies
            module_dependencies[module_name].add(normalized_imp)
            
            # Add to package dependencies
            imp_package = get_package_from_module(normalized_imp)
            if package_name != imp_package:
                package_dependencies[package_name].add(imp_package)
    
    return module_dependencies, package_dependencies, module_to_package

def detect_cycles(graph):
    """Detect cycles in a dependency graph using DFS."""
    color = {}  # 0: white, 1: gray, 2: black
    cycles = []
    
    def dfs(node, path):
        if node in color:
            if color[node] == 1:  # gray (currently being processed)
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True
            elif color[node] == 2:  # black (already processed)
                return False
        
        color[node] = 1  # mark as gray
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if dfs(neighbor, path):
                break
        
        path.pop()
        color[node] = 2  # mark as black
        return False
    
    for node in graph:
        if node not in color:
            dfs(node, [])
    
    return cycles

def main():
    """Main analysis function."""
    # Main analysis
    module_deps, package_deps, mod_to_pkg = build_dependency_graph()

    print('\n=== PACKAGE DEPENDENCY ANALYSIS ===')
    print('\nPackage Dependencies:')
    for pkg, deps in sorted(package_deps.items()):
        if deps:
            print(f'{pkg} -> {sorted(deps)}')

    print('\n=== CIRCULAR DEPENDENCY DETECTION ===')

    # Detect package-level cycles
    package_cycles = detect_cycles(package_deps)
    if package_cycles:
        print('\nPACKAGE-LEVEL CIRCULAR DEPENDENCIES FOUND:')
        for i, cycle in enumerate(package_cycles, 1):
            print(f'{i}. {" -> ".join(cycle)}')
    else:
        print('\nNo package-level circular dependencies found.')

    # Detect module-level cycles
    module_cycles = detect_cycles(module_deps)
    if module_cycles:
        print(f'\nMODULE-LEVEL CIRCULAR DEPENDENCIES FOUND ({len(module_cycles)} cycles):')
        for i, cycle in enumerate(module_cycles[:10], 1):  # Show first 10
            print(f'{i}. {" -> ".join(cycle)}')
        if len(module_cycles) > 10:
            print(f'... and {len(module_cycles) - 10} more cycles')
    else:
        print('\nNo module-level circular dependencies found.')

    print('\n=== CLEAN ARCHITECTURE VIOLATIONS ===')

    # Check for Clean Architecture violations
    violations = []

    # Domain should not import from Application or Infrastructure
    domain_violations = []
    for module, deps in module_deps.items():
        if module.startswith('monorepo.domain'):
            for dep in deps:
                if dep.startswith('monorepo.application') or dep.startswith('monorepo.infrastructure') or dep.startswith('monorepo.presentation'):
                    domain_violations.append((module, dep))

    if domain_violations:
        print('\nDOMAIN LAYER VIOLATIONS (importing from outer layers):')
        for module, dep in domain_violations:
            print(f'  {module} -> {dep}')

    # Application should not import from Infrastructure (except interfaces)
    app_violations = []
    for module, deps in module_deps.items():
        if module.startswith('monorepo.application'):
            for dep in deps:
                if dep.startswith('monorepo.infrastructure') and 'interfaces' not in dep:
                    app_violations.append((module, dep))

    if app_violations:
        print('\nAPPLICATION LAYER VIOLATIONS (importing from infrastructure):')
        for module, dep in app_violations:
            print(f'  {module} -> {dep}')

    print('\n=== SUMMARY ===')
    print(f'Total packages analyzed: {len(package_deps)}')
    print(f'Total modules analyzed: {len(module_deps)}')
    print(f'Package-level cycles: {len(package_cycles)}')
    print(f'Module-level cycles: {len(module_cycles)}')
    print(f'Domain layer violations: {len(domain_violations)}')
    print(f'Application layer violations: {len(app_violations)}')

if __name__ == '__main__':
    main()