"""Complexity monitoring system for maintainability tracking.

This module provides automated monitoring of codebase complexity to prevent
regression during feature reintroduction. Essential for maintaining the
benefits of Phase 1 simplification.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import psutil


@dataclass
class ComplexityMetrics:
    """Container for complexity measurement data."""
    
    # File and line metrics
    total_files: int = 0
    total_lines: int = 0
    python_files: int = 0
    python_lines: int = 0
    test_files: int = 0
    
    # Dependency metrics
    total_dependencies: int = 0
    optional_dependencies: int = 0
    dev_dependencies: int = 0
    
    # Code complexity metrics
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    maintainability_index: float = 0.0
    
    # Performance metrics
    startup_time: float = 0.0
    memory_usage: float = 0.0
    import_time: float = 0.0
    
    # Documentation metrics
    docstring_coverage: float = 0.0
    comment_ratio: float = 0.0
    
    # Architecture metrics
    layer_violations: int = 0
    circular_dependencies: int = 0
    
    # Timestamps
    measured_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "python_files": self.python_files,
            "python_lines": self.python_lines,
            "test_files": self.test_files,
            "total_dependencies": self.total_dependencies,
            "optional_dependencies": self.optional_dependencies,
            "dev_dependencies": self.dev_dependencies,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "maintainability_index": self.maintainability_index,
            "startup_time": self.startup_time,
            "memory_usage": self.memory_usage,
            "import_time": self.import_time,
            "docstring_coverage": self.docstring_coverage,
            "comment_ratio": self.comment_ratio,
            "layer_violations": self.layer_violations,
            "circular_dependencies": self.circular_dependencies,
            "measured_at": self.measured_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> ComplexityMetrics:
        """Create metrics from dictionary."""
        data = data.copy()
        data["measured_at"] = datetime.fromisoformat(data["measured_at"])
        return cls(**data)


class ComplexityMonitor:
    """Monitor and track codebase complexity over time."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize complexity monitor."""
        self.project_root = project_root or Path.cwd()
        self.src_path = self.project_root / "src" / "pynomaly"
        self.tests_path = self.project_root / "tests"
        self.metrics_file = self.project_root / "complexity_metrics.json"
        
        # Phase 1 baseline targets (post-simplification)
        self.targets = {
            "max_files": 5000,  # Down from 18,500+
            "max_dependencies": 50,  # Down from 100+
            "max_startup_time": 30.0,  # seconds
            "min_maintainability": 70.0,  # index score
            "max_cyclomatic_complexity": 10.0,  # average per function
            "min_docstring_coverage": 80.0,  # percentage
            "max_memory_usage": 500.0,  # MB for basic operations
        }
    
    def measure_file_metrics(self) -> Dict:
        """Measure file count and line metrics."""
        metrics = {
            "total_files": 0,
            "total_lines": 0,
            "python_files": 0,
            "python_lines": 0,
            "test_files": 0
        }
        
        # Count all files in project
        for path in self.project_root.rglob("*"):
            if path.is_file() and not self._should_ignore_path(path):
                metrics["total_files"] += 1
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        metrics["total_lines"] += lines
                        
                        if path.suffix == '.py':
                            metrics["python_files"] += 1
                            metrics["python_lines"] += lines
                            
                            if 'test' in str(path).lower():
                                metrics["test_files"] += 1
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        return metrics
    
    def measure_dependency_metrics(self) -> Dict:
        """Measure dependency complexity from pyproject.toml."""
        metrics = {
            "total_dependencies": 0,
            "optional_dependencies": 0,
            "dev_dependencies": 0
        }
        
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            return metrics
        
        try:
            import toml
            config = toml.load(pyproject_path)
            
            # Count main dependencies
            deps = config.get("tool", {}).get("poetry", {}).get("dependencies", {})
            for key, value in deps.items():
                if key != "python":
                    metrics["total_dependencies"] += 1
                    if isinstance(value, dict) and value.get("optional", False):
                        metrics["optional_dependencies"] += 1
            
            # Count dev dependencies
            dev_deps = config.get("tool", {}).get("poetry", {}).get("group", {}).get("dev", {}).get("dependencies", {})
            metrics["dev_dependencies"] = len(dev_deps)
            metrics["total_dependencies"] += metrics["dev_dependencies"]
            
        except ImportError:
            # Fallback: count lines in pyproject.toml
            with open(pyproject_path, 'r') as f:
                content = f.read()
                # Simple heuristic: count lines with '=' in dependencies section
                in_deps = False
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('[tool.poetry.dependencies]'):
                        in_deps = True
                    elif line.startswith('[') and in_deps:
                        in_deps = False
                    elif in_deps and '=' in line and not line.startswith('python'):
                        metrics["total_dependencies"] += 1
        
        return metrics
    
    def measure_code_complexity(self) -> Dict:
        """Measure code complexity metrics."""
        metrics = {
            "cyclomatic_complexity": 0.0,
            "cognitive_complexity": 0.0,
            "maintainability_index": 0.0
        }
        
        complexity_scores = []
        maintainability_scores = []
        
        for py_file in self.src_path.rglob("*.py"):
            if self._should_ignore_path(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                # Calculate cyclomatic complexity
                complexity = self._calculate_cyclomatic_complexity(tree)
                complexity_scores.append(complexity)
                
                # Simple maintainability index based on file size and complexity
                lines = len(open(py_file).readlines())
                mi = max(0, 100 - (complexity * 2) - (lines / 50))
                maintainability_scores.append(mi)
                
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if complexity_scores:
            metrics["cyclomatic_complexity"] = sum(complexity_scores) / len(complexity_scores)
        
        if maintainability_scores:
            metrics["maintainability_index"] = sum(maintainability_scores) / len(maintainability_scores)
        
        return metrics
    
    def measure_performance_metrics(self) -> Dict:
        """Measure performance-related metrics."""
        metrics = {
            "startup_time": 0.0,
            "memory_usage": 0.0,
            "import_time": 0.0
        }
        
        # Measure import time
        start_time = time.time()
        try:
            import pynomaly
            metrics["import_time"] = time.time() - start_time
        except ImportError:
            pass
        
        # Measure memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        metrics["memory_usage"] = memory_info.rss / 1024 / 1024  # MB
        
        # Startup time would require external process measurement
        # For now, approximate with import time
        metrics["startup_time"] = metrics["import_time"] * 10  # Rough approximation
        
        return metrics
    
    def measure_documentation_metrics(self) -> Dict:
        """Measure documentation quality metrics."""
        metrics = {
            "docstring_coverage": 0.0,
            "comment_ratio": 0.0
        }
        
        total_functions = 0
        documented_functions = 0
        total_lines = 0
        comment_lines = 0
        
        for py_file in self.src_path.rglob("*.py"):
            if self._should_ignore_path(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                # Count functions and docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        if (ast.get_docstring(node) is not None):
                            documented_functions += 1
                
                # Count comment lines
                lines = content.split('\n')
                total_lines += len(lines)
                comment_lines += sum(1 for line in lines if line.strip().startswith('#'))
                
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if total_functions > 0:
            metrics["docstring_coverage"] = (documented_functions / total_functions) * 100
        
        if total_lines > 0:
            metrics["comment_ratio"] = (comment_lines / total_lines) * 100
        
        return metrics
    
    def measure_architecture_metrics(self) -> Dict:
        """Measure architecture quality metrics."""
        metrics = {
            "layer_violations": 0,
            "circular_dependencies": 0
        }
        
        # Simple layer violation detection
        violations = self._detect_layer_violations()
        metrics["layer_violations"] = len(violations)
        
        return metrics
    
    def measure_all(self) -> ComplexityMetrics:
        """Measure all complexity metrics."""
        all_metrics = {}
        
        all_metrics.update(self.measure_file_metrics())
        all_metrics.update(self.measure_dependency_metrics())
        all_metrics.update(self.measure_code_complexity())
        all_metrics.update(self.measure_performance_metrics())
        all_metrics.update(self.measure_documentation_metrics())
        all_metrics.update(self.measure_architecture_metrics())
        
        return ComplexityMetrics(**all_metrics)
    
    def save_metrics(self, metrics: ComplexityMetrics) -> None:
        """Save metrics to file for tracking over time."""
        history = self.load_metrics_history()
        history.append(metrics.to_dict())
        
        # Keep only last 100 measurements
        if len(history) > 100:
            history = history[-100:]
        
        with open(self.metrics_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_metrics_history(self) -> List[Dict]:
        """Load historical metrics data."""
        if not self.metrics_file.exists():
            return []
        
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def check_targets(self, metrics: ComplexityMetrics) -> Dict[str, bool]:
        """Check if current metrics meet targets."""
        results = {}
        
        results["files_within_target"] = metrics.total_files <= self.targets["max_files"]
        results["dependencies_within_target"] = metrics.total_dependencies <= self.targets["max_dependencies"]
        results["startup_time_within_target"] = metrics.startup_time <= self.targets["max_startup_time"]
        results["maintainability_within_target"] = metrics.maintainability_index >= self.targets["min_maintainability"]
        results["complexity_within_target"] = metrics.cyclomatic_complexity <= self.targets["max_cyclomatic_complexity"]
        results["documentation_within_target"] = metrics.docstring_coverage >= self.targets["min_docstring_coverage"]
        results["memory_within_target"] = metrics.memory_usage <= self.targets["max_memory_usage"]
        
        return results
    
    def generate_report(self, metrics: ComplexityMetrics) -> str:
        """Generate a human-readable complexity report."""
        targets = self.check_targets(metrics)
        
        report = f"""
# Pynomaly Complexity Report
Generated: {metrics.measured_at.strftime('%Y-%m-%d %H:%M:%S')}

## File Metrics
- Total Files: {metrics.total_files:,} (Target: ≤{self.targets['max_files']:,}) {'✅' if targets['files_within_target'] else '❌'}
- Python Files: {metrics.python_files:,}
- Test Files: {metrics.test_files:,}
- Total Lines: {metrics.total_lines:,}

## Dependency Metrics  
- Total Dependencies: {metrics.total_dependencies} (Target: ≤{self.targets['max_dependencies']}) {'✅' if targets['dependencies_within_target'] else '❌'}
- Optional Dependencies: {metrics.optional_dependencies}
- Dev Dependencies: {metrics.dev_dependencies}

## Code Quality Metrics
- Cyclomatic Complexity: {metrics.cyclomatic_complexity:.1f} (Target: ≤{self.targets['max_cyclomatic_complexity']}) {'✅' if targets['complexity_within_target'] else '❌'}
- Maintainability Index: {metrics.maintainability_index:.1f} (Target: ≥{self.targets['min_maintainability']}) {'✅' if targets['maintainability_within_target'] else '❌'}
- Docstring Coverage: {metrics.docstring_coverage:.1f}% (Target: ≥{self.targets['min_docstring_coverage']}%) {'✅' if targets['documentation_within_target'] else '❌'}

## Performance Metrics
- Startup Time: {metrics.startup_time:.1f}s (Target: ≤{self.targets['max_startup_time']}s) {'✅' if targets['startup_time_within_target'] else '❌'}
- Memory Usage: {metrics.memory_usage:.1f}MB (Target: ≤{self.targets['max_memory_usage']}MB) {'✅' if targets['memory_within_target'] else '❌'}
- Import Time: {metrics.import_time:.3f}s

## Architecture Metrics
- Layer Violations: {metrics.layer_violations}
- Circular Dependencies: {metrics.circular_dependencies}

## Overall Status
{'✅ All targets met!' if all(targets.values()) else '❌ Some targets not met'}
"""
        
        return report
    
    def _should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored in metrics."""
        ignore_patterns = {
            '.git', '.venv', '__pycache__', '.pytest_cache',
            'node_modules', '.coverage', 'htmlcov', 'build',
            'dist', '.egg-info', '.mypy_cache'
        }
        
        return any(pattern in str(path) for pattern in ignore_patterns)
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of AST."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
        
        return complexity
    
    def _detect_layer_violations(self) -> List[str]:
        """Detect violations of clean architecture layer boundaries."""
        violations = []
        
        # Check for domain importing infrastructure
        domain_path = self.src_path / "domain"
        if domain_path.exists():
            for py_file in domain_path.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if 'from pynomaly.infrastructure' in content or 'import pynomaly.infrastructure' in content:
                            violations.append(f"Domain imports infrastructure: {py_file}")
                except (UnicodeDecodeError, FileNotFoundError):
                    continue
        
        return violations


def run_complexity_check() -> ComplexityMetrics:
    """Run complexity check and return metrics."""
    monitor = ComplexityMonitor()
    metrics = monitor.measure_all()
    monitor.save_metrics(metrics)
    return metrics


def print_complexity_report() -> None:
    """Print complexity report to console."""
    monitor = ComplexityMonitor()
    metrics = monitor.measure_all()
    report = monitor.generate_report(metrics)
    print(report)


if __name__ == "__main__":
    print_complexity_report()