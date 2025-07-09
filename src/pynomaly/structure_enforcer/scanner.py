"""
Repository scanner implementation.
"""

import ast
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from .config import (
    EXPECTED_STRUCTURE,
    LAYER_ORDER,
    get_file_type_info,
    is_python_file,
)
from .models import DirectoryNode, FileNode, Model


class RepositoryScanner:
    """Scans repository structure and builds a model."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.total_files = 0
        self.total_directories = 0
        self.max_depth = 0
        self.layers: Dict[str, DirectoryNode] = {}
        self.dependencies: Dict[str, Set[str]] = {}
    
    def scan(self) -> Model:
        """Scan the repository and build a model."""
        if not self.root_path.exists():
            raise FileNotFoundError(f"Repository root not found: {self.root_path}")
        
        # Reset counters
        self.total_files = 0
        self.total_directories = 0
        self.max_depth = 0
        self.layers = {}
        self.dependencies = {}
        
        # Build the directory tree
        root_directory = self._scan_directory(self.root_path, 0)
        
        # Identify layers in the src/pynomaly directory
        self._identify_layers()
        
        # Analyze dependencies
        self._analyze_dependencies()
        
        return Model(
            root_path=self.root_path,
            root_directory=root_directory,
            total_files=self.total_files,
            total_directories=self.total_directories,
            max_depth=self.max_depth,
            layers=self.layers,
            dependencies=self.dependencies,
            scan_timestamp=datetime.now(),
        )
    
    def _scan_directory(self, dir_path: Path, depth: int) -> DirectoryNode:
        """Recursively scan a directory."""
        self.total_directories += 1
        self.max_depth = max(self.max_depth, depth)
        
        files = []
        subdirectories = []
        
        try:
            for item in dir_path.iterdir():
                if item.is_file():
                    files.append(self._scan_file(item))
                elif item.is_dir():
                    # Skip hidden directories except for some allowed ones
                    if item.name.startswith('.') and item.name not in {'.git', '.github'}:
                        continue
                    subdirectories.append(self._scan_directory(item, depth + 1))
        except PermissionError:
            # Skip directories we can't read
            pass
        
        # Determine if this is a Python package
        is_package = any(f.name == "__init__.py" for f in files)
        
        # Determine if this is a layer directory
        is_layer = False
        layer_name = None
        if depth == 2 and dir_path.parent.name == "pynomaly":
            if dir_path.name in EXPECTED_STRUCTURE:
                is_layer = True
                layer_name = dir_path.name
        
        return DirectoryNode(
            path=dir_path,
            name=dir_path.name,
            files=files,
            subdirectories=subdirectories,
            is_package=is_package,
            is_layer=is_layer,
            layer_name=layer_name,
        )
    
    def _scan_file(self, file_path: Path) -> FileNode:
        """Scan a single file."""
        self.total_files += 1
        
        # Get file stats
        try:
            stat = file_path.stat()
            size = stat.st_size
            created_at = datetime.fromtimestamp(stat.st_ctime)
            modified_at = datetime.fromtimestamp(stat.st_mtime)
        except OSError:
            size = 0
            created_at = datetime.now()
            modified_at = datetime.now()
        
        # Get file type information
        type_info = get_file_type_info(file_path)
        
        return FileNode(
            path=file_path,
            name=file_path.name,
            size=size,
            created_at=created_at,
            modified_at=modified_at,
            **type_info,
        )
    
    def _identify_layers(self):
        """Identify Clean Architecture layers in the codebase."""
        # Look for src/pynomaly directory
        src_path = self.root_path / "src"
        if not src_path.exists():
            return
        
        pynomaly_path = src_path / "pynomaly"
        if not pynomaly_path.exists():
            return
        
        # Find the pynomaly directory node
        pynomaly_node = self._find_directory_node(pynomaly_path)
        if not pynomaly_node:
            return
        
        # Extract layer directories
        for subdir in pynomaly_node.subdirectories:
            if subdir.name in EXPECTED_STRUCTURE:
                self.layers[subdir.name] = subdir
    
    def _find_directory_node(self, target_path: Path) -> DirectoryNode:
        """Find a directory node by path."""
        def search_directory(node: DirectoryNode) -> DirectoryNode:
            if node.path == target_path:
                return node
            for subdir in node.subdirectories:
                result = search_directory(subdir)
                if result:
                    return result
            return None
        
        return search_directory(self.root_path)
    
    def _analyze_dependencies(self):
        """Analyze dependencies between layers."""
        for layer_name, layer_node in self.layers.items():
            self.dependencies[layer_name] = set()
            
            # Analyze Python files in the layer
            for py_file in self._get_python_files_recursive(layer_node):
                deps = self._analyze_file_dependencies(py_file)
                self.dependencies[layer_name].update(deps)
    
    def _get_python_files_recursive(self, node: DirectoryNode) -> List[FileNode]:
        """Get all Python files recursively from a directory node."""
        files = []
        
        # Add Python files from this directory
        files.extend(node.get_python_files())
        
        # Recursively add from subdirectories
        for subdir in node.subdirectories:
            files.extend(self._get_python_files_recursive(subdir))
        
        return files
    
    def _analyze_file_dependencies(self, file_node: FileNode) -> Set[str]:
        """Analyze dependencies in a single Python file."""
        dependencies = set()
        
        if not file_node.is_python:
            return dependencies
        
        try:
            with open(file_node.path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("pynomaly."):
                        # Extract layer name from import
                        parts = node.module.split(".")
                        if len(parts) >= 2 and parts[1] in EXPECTED_STRUCTURE:
                            dependencies.add(parts[1])
        except (SyntaxError, UnicodeDecodeError, OSError):
            # Skip files that can't be parsed
            pass
        
        return dependencies


def scan_repository(root_path: Path = None) -> Model:
    """
    Scan the repository and build its model representation.
    
    Args:
        root_path: Path to the repository root. If None, uses current directory.
    
    Returns:
        Model: Repository structure model.
    """
    if root_path is None:
        root_path = Path.cwd()
    
    scanner = RepositoryScanner(root_path)
    return scanner.scan()
