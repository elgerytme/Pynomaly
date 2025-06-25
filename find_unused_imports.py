#!/usr/bin/env python3
"""Simple script to find potential unused imports in Python files."""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check for unused imports."""
    
    def __init__(self):
        self.imports: Dict[str, int] = {}  # name -> line number
        self.from_imports: Dict[str, int] = {}  # name -> line number
        self.used_names: Set[str] = set()
        self.all_names: Set[str] = set()
        
    def visit_Import(self, node):
        """Handle regular imports like 'import foo'."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            # For dotted imports like 'import foo.bar', use the root name
            root_name = name.split('.')[0]
            self.imports[root_name] = node.lineno
            
    def visit_ImportFrom(self, node):
        """Handle from imports like 'from foo import bar'."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            if name != '*':
                self.from_imports[name] = node.lineno
                
    def visit_Name(self, node):
        """Track name usage."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
            
    def visit_Attribute(self, node):
        """Track attribute access like obj.attr."""
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        """Handle assignments to track __all__ usage."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '__all__':
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Str):
                            self.all_names.add(elt.s)
                        elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.all_names.add(elt.value)
        self.generic_visit(node)
        
    def get_unused_imports(self) -> List[Tuple[str, int]]:
        """Return list of (import_name, line_number) for unused imports."""
        unused = []
        
        # Check regular imports
        for name, line in self.imports.items():
            if name not in self.used_names and name not in self.all_names:
                unused.append((name, line))
                
        # Check from imports
        for name, line in self.from_imports.items():
            if name not in self.used_names and name not in self.all_names:
                unused.append((name, line))
                
        return sorted(unused, key=lambda x: x[1])


def check_file(file_path: Path) -> List[Tuple[str, int]]:
    """Check a single Python file for unused imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content, filename=str(file_path))
        checker = ImportChecker()
        checker.visit(tree)
        
        return checker.get_unused_imports()
        
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return []


def main():
    """Main function to check all Python files in src/ directory."""
    src_dir = Path("src")
    if not src_dir.exists():
        print("src/ directory not found!")
        return
        
    python_files = list(src_dir.rglob("*.py"))
    total_files = len(python_files)
    files_with_unused = 0
    
    print(f"Checking {total_files} Python files for unused imports...\n")
    
    for file_path in python_files:
        unused_imports = check_file(file_path)
        
        if unused_imports:
            files_with_unused += 1
            print(f"{file_path}:")
            for name, line in unused_imports:
                print(f"  Line {line}: unused import '{name}'")
            print()
            
    if files_with_unused == 0:
        print("No unused imports found!")
    else:
        print(f"Found unused imports in {files_with_unused} out of {total_files} files.")


if __name__ == "__main__":
    main()