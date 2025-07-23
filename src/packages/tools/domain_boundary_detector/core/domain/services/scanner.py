"""AST-based scanner for detecting cross-package imports and references."""

import ast
import os
from pathlib import Path
from typing import List, Set, Dict, Any, Optional
import re
from dataclasses import dataclass, field


@dataclass
class Import:
    """Represents an import statement found in code."""
    module: str
    names: List[str]
    file_path: str
    line_number: int
    import_type: str  # 'import' or 'from'
    is_relative: bool = False
    level: int = 0  # For relative imports

    def __str__(self) -> str:
        if self.import_type == 'import':
            return f"import {self.module}"
        else:
            names_str = ', '.join(self.names) if self.names else '*'
            return f"from {self.module} import {names_str}"


@dataclass
class StringReference:
    """Represents a string that might be a module reference."""
    value: str
    file_path: str
    line_number: int
    context: str  # The surrounding code context


@dataclass
class ScanResult:
    """Results from scanning a file or directory."""
    imports: List[Import] = field(default_factory=list)
    string_references: List[StringReference] = field(default_factory=list)
    type_annotations: List[StringReference] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


class ImportVisitor(ast.NodeVisitor):
    """AST visitor for extracting imports and potential references."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.imports: List[Import] = []
        self.string_references: List[StringReference] = []
        self.type_annotations: List[StringReference] = []
        
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            import_obj = Import(
                module=alias.name,
                names=[],
                file_path=self.file_path,
                line_number=node.lineno,
                import_type='import',
                is_relative=False
            )
            self.imports.append(import_obj)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statements."""
        module = node.module or ''
        names = [alias.name for alias in node.names]
        
        import_obj = Import(
            module=module,
            names=names,
            file_path=self.file_path,
            line_number=node.lineno,
            import_type='from',
            is_relative=node.level > 0,
            level=node.level
        )
        self.imports.append(import_obj)
        self.generic_visit(node)
        
    def visit_Str(self, node: ast.Str) -> None:
        """Visit string literals that might be module references."""
        if self._is_module_like_string(node.s):
            ref = StringReference(
                value=node.s,
                file_path=self.file_path,
                line_number=node.lineno,
                context=ast.dump(node)
            )
            self.string_references.append(ref)
        self.generic_visit(node)
        
    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit constants (Python 3.8+)."""
        if isinstance(node.value, str) and self._is_module_like_string(node.value):
            ref = StringReference(
                value=node.value,
                file_path=self.file_path,
                line_number=node.lineno,
                context=ast.dump(node)
            )
            self.string_references.append(ref)
        self.generic_visit(node)
        
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignments to find type hints."""
        if isinstance(node.annotation, ast.Str):
            if self._is_module_like_string(node.annotation.s):
                ref = StringReference(
                    value=node.annotation.s,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    context="type_annotation"
                )
                self.type_annotations.append(ref)
        self.generic_visit(node)
        
    def _is_module_like_string(self, s: str) -> bool:
        """Check if a string looks like a module path."""
        # Pattern for module-like strings
        module_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$'
        
        # Common patterns to exclude
        exclude_patterns = [
            r'^test_',  # Test names
            r'^__',     # Dunder names
            r'^\d',     # Starts with number
            r'^https?://',  # URLs
            r'^[A-Z_]+$',  # All caps (constants)
        ]
        
        if not s or len(s) < 3:
            return False
            
        if not re.match(module_pattern, s):
            return False
            
        for pattern in exclude_patterns:
            if re.match(pattern, s):
                return False
                
        # Check if it contains package-like segments
        return '.' in s or s in ['numpy', 'pandas', 'torch', 'tensorflow']


class Scanner:
    """Main scanner for detecting imports and references."""
    
    def __init__(self, ignore_patterns: Optional[List[str]] = None):
        self.ignore_patterns = ignore_patterns or [
            '*.pyc',
            '__pycache__',
            '.git',
            '.pytest_cache',
            'venv',
            'env',
            '.env',
            'node_modules',
        ]
        
    def scan_file(self, file_path: Path) -> ScanResult:
        """Scan a single Python file."""
        result = ScanResult()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            visitor = ImportVisitor(str(file_path))
            visitor.visit(tree)
            
            result.imports = visitor.imports
            result.string_references = visitor.string_references
            result.type_annotations = visitor.type_annotations
            
        except Exception as e:
            result.errors.append({
                'file': str(file_path),
                'error': str(e),
                'type': type(e).__name__
            })
            
        return result
        
    def scan_directory(self, directory: Path) -> ScanResult:
        """Scan all Python files in a directory recursively."""
        result = ScanResult()
        
        for py_file in self._find_python_files(directory):
            file_result = self.scan_file(py_file)
            result.imports.extend(file_result.imports)
            result.string_references.extend(file_result.string_references)
            result.type_annotations.extend(file_result.type_annotations)
            result.errors.extend(file_result.errors)
            
        return result
        
    def _find_python_files(self, directory: Path) -> List[Path]:
        """Find all Python files in directory, respecting ignore patterns."""
        python_files = []
        
        for root, dirs, files in os.walk(directory):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(d)]
            
            for file in files:
                if file.endswith('.py') and not self._should_ignore(file):
                    python_files.append(Path(root) / file)
                    
        return python_files
        
    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        for pattern in self.ignore_patterns:
            if pattern in path:
                return True
        return False


def extract_package_from_import(import_str: str, file_path: str, monorepo_root: str) -> Optional[str]:
    """Extract the package name from an import string."""
    # Handle relative imports
    if import_str.startswith('.'):
        # Convert relative to absolute based on file location
        file_parts = Path(file_path).relative_to(monorepo_root).parts
        package_parts = list(file_parts[:-1])  # Remove filename
        
        # Apply relative import levels
        level = len(import_str) - len(import_str.lstrip('.'))
        if level <= len(package_parts):
            package_parts = package_parts[:-level]
            remaining = import_str.lstrip('.')
            if remaining:
                package_parts.append(remaining.split('.')[0])
            return '/'.join(package_parts[:3]) if len(package_parts) >= 3 else None
    
    # Handle absolute imports
    parts = import_str.split('.')
    if len(parts) >= 3 and parts[0] == 'src' and parts[1] == 'packages':
        # src.packages.domain.package -> domain/package
        if len(parts) >= 4:
            return f"{parts[2]}/{parts[3]}"
    
    return None