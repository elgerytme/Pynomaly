"""File discovery utilities for comprehensive static analysis."""

import ast
import os
from pathlib import Path
from typing import List, Set, Optional
import logging
import fnmatch

logger = logging.getLogger(__name__)


class FileDiscovery:
    """Discovers and filters files for analysis."""
    
    def __init__(self, include_patterns: List[str], exclude_patterns: List[str]):
        self.include_patterns = include_patterns or ["**/*.py"]
        self.exclude_patterns = exclude_patterns or []
        
        # Compile patterns for better performance
        self._compiled_includes = [self._compile_pattern(p) for p in self.include_patterns]
        self._compiled_excludes = [self._compile_pattern(p) for p in self.exclude_patterns]
    
    def discover(self, paths: List[Path]) -> List[Path]:
        """Discover files from given paths."""
        discovered_files = []
        
        for path in paths:
            if path.is_file():
                if self._should_include(path):
                    discovered_files.append(path)
            elif path.is_dir():
                discovered_files.extend(self._discover_directory(path))
            else:
                logger.warning(f"Path does not exist: {path}")
        
        # Remove duplicates and sort
        unique_files = list(set(discovered_files))
        unique_files.sort()
        
        logger.info(f"Discovered {len(unique_files)} files for analysis")
        return unique_files
    
    def _discover_directory(self, directory: Path) -> List[Path]:
        """Discover files in a directory."""
        files = []
        
        try:
            for root, dirs, filenames in os.walk(directory):
                root_path = Path(root)
                
                # Filter directories early to avoid unnecessary traversal
                dirs[:] = [d for d in dirs if not self._should_exclude_dir(root_path / d)]
                
                for filename in filenames:
                    file_path = root_path / filename
                    
                    # Skip if file should be excluded
                    if self._should_include(file_path):
                        files.append(file_path)
                        
        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {directory}")
        except Exception as e:
            logger.error(f"Error discovering files in {directory}: {e}")
        
        return files
    
    def _should_include(self, path: Path) -> bool:
        """Check if a file should be included."""
        path_str = str(path)
        
        # First check if it matches include patterns
        included = False
        for pattern in self._compiled_includes:
            if pattern.match(path_str) or fnmatch.fnmatch(path_str, pattern.pattern):
                included = True
                break
        
        if not included:
            return False
        
        # Then check if it should be excluded
        for pattern in self._compiled_excludes:
            if pattern.match(path_str) or fnmatch.fnmatch(path_str, pattern.pattern):
                return False
        
        return True
    
    def _should_exclude_dir(self, path: Path) -> bool:
        """Check if a directory should be excluded from traversal."""
        path_str = str(path)
        
        # Common directories to exclude
        common_excludes = [
            "__pycache__",
            ".git",
            ".svn",
            ".hg",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
            ".coverage",
            "htmlcov",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "build",
            "dist",
            "*.egg-info",
        ]
        
        dir_name = path.name
        
        # Check common excludes
        for exclude in common_excludes:
            if fnmatch.fnmatch(dir_name, exclude):
                return True
        
        # Check configured exclude patterns
        for pattern in self._compiled_excludes:
            if pattern.match(path_str) or fnmatch.fnmatch(path_str, pattern.pattern):
                return True
        
        return False
    
    def _compile_pattern(self, pattern: str) -> 'PatternMatcher':
        """Compile a pattern for efficient matching."""
        return PatternMatcher(pattern)
    
    def get_file_stats(self, files: List[Path]) -> dict:
        """Get statistics about discovered files."""
        stats = {
            "total_files": len(files),
            "extensions": {},
            "directories": set(),
            "total_size": 0,
        }
        
        for file_path in files:
            # Count extensions
            ext = file_path.suffix.lower()
            stats["extensions"][ext] = stats["extensions"].get(ext, 0) + 1
            
            # Track directories
            stats["directories"].add(file_path.parent)
            
            # Calculate size
            try:
                stats["total_size"] += file_path.stat().st_size
            except OSError:
                pass
        
        stats["directories"] = len(stats["directories"])
        return stats


class PatternMatcher:
    """Efficient pattern matcher for file paths."""
    
    def __init__(self, pattern: str):
        self.pattern = pattern
        self._is_glob = "*" in pattern or "?" in pattern or "[" in pattern
    
    def match(self, path: str) -> bool:
        """Check if path matches pattern."""
        if self._is_glob:
            return fnmatch.fnmatch(path, self.pattern)
        else:
            return path == self.pattern or path.endswith(self.pattern)


class DependencyGraph:
    """Builds and manages dependency relationships between files."""
    
    def __init__(self):
        self.dependencies = {}  # file -> set of dependencies
        self.dependents = {}    # file -> set of dependents
    
    def add_dependency(self, file_path: Path, depends_on: Path):
        """Add a dependency relationship."""
        if file_path not in self.dependencies:
            self.dependencies[file_path] = set()
        if depends_on not in self.dependents:
            self.dependents[depends_on] = set()
        
        self.dependencies[file_path].add(depends_on)
        self.dependents[depends_on].add(file_path)
    
    def get_dependencies(self, file_path: Path) -> Set[Path]:
        """Get all dependencies for a file."""
        return self.dependencies.get(file_path, set())
    
    def get_dependents(self, file_path: Path) -> Set[Path]:
        """Get all dependents for a file."""
        return self.dependents.get(file_path, set())
    
    def get_transitive_dependencies(self, file_path: Path) -> Set[Path]:
        """Get all transitive dependencies for a file."""
        visited = set()
        to_visit = [file_path]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            dependencies = self.get_dependencies(current)
            to_visit.extend(dependencies - visited)
        
        visited.discard(file_path)  # Remove the starting file
        return visited
    
    def get_transitive_dependents(self, file_path: Path) -> Set[Path]:
        """Get all transitive dependents for a file."""
        visited = set()
        to_visit = [file_path]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            dependents = self.get_dependents(current)
            to_visit.extend(dependents - visited)
        
        visited.discard(file_path)  # Remove the starting file
        return visited
    
    def detect_cycles(self) -> List[List[Path]]:
        """Detect dependency cycles."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: Path, path: List[Path]):
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
            
            for dep in self.get_dependencies(node):
                dfs(dep, path + [node])
            
            rec_stack.remove(node)
        
        for file_path in self.dependencies:
            if file_path not in visited:
                dfs(file_path, [])
        
        return cycles
    
    def build_from_files(self, files: List[Path]) -> None:
        """Build dependency graph by analyzing import statements."""
        import ast
        
        for file_path in files:
            if file_path.suffix != ".py":
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                dependencies = self._extract_dependencies(tree, file_path)
                
                for dep in dependencies:
                    self.add_dependency(file_path, dep)
                    
            except Exception as e:
                logger.debug(f"Could not analyze dependencies for {file_path}: {e}")
    
    def _extract_dependencies(self, tree: ast.AST, file_path: Path) -> List[Path]:
        """Extract dependencies from AST."""
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dep_path = self._resolve_import(alias.name, file_path)
                    if dep_path:
                        dependencies.append(dep_path)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dep_path = self._resolve_import(node.module, file_path)
                    if dep_path:
                        dependencies.append(dep_path)
        
        return dependencies
    
    def _resolve_import(self, module_name: str, file_path: Path) -> Optional[Path]:
        """Resolve import to file path."""
        # Simple resolution - could be made more sophisticated
        if module_name.startswith("."):
            # Relative import
            return None  # Skip for now
        
        # Try to find the module file
        parts = module_name.split(".")
        
        # Look for the module in the same directory and parent directories
        search_dirs = [file_path.parent]
        current_dir = file_path.parent
        
        while current_dir.parent != current_dir:
            current_dir = current_dir.parent
            search_dirs.append(current_dir)
        
        for search_dir in search_dirs:
            module_path = search_dir
            for part in parts:
                module_path = module_path / part
            
            # Check for .py file
            if (module_path.with_suffix(".py")).exists():
                return module_path.with_suffix(".py")
            
            # Check for __init__.py in directory
            if (module_path / "__init__.py").exists():
                return module_path / "__init__.py"
        
        return None