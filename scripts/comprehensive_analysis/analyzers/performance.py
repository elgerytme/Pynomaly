"""Performance analysis for comprehensive static analysis."""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

from ..tools.adapter_base import Issue

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for code analysis."""
    cyclomatic_complexity: int
    cognitive_complexity: int
    function_count: int
    class_count: int
    line_count: int
    import_count: int
    nested_loops: int
    recursive_calls: int
    memory_patterns: List[str]
    performance_issues: List[Issue]


class PerformanceAnalyzer:
    """Analyzes code for performance issues and complexity."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_config = config.get("performance", {})
        self.complexity_threshold = self.performance_config.get("complexity_threshold", 10)
        self.cognitive_threshold = self.performance_config.get("cognitive_threshold", 15)
        self.max_nested_loops = self.performance_config.get("max_nested_loops", 3)
        
    def analyze_file(self, file_path: Path) -> PerformanceMetrics:
        """Analyze a single file for performance issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Calculate metrics
            cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
            cognitive_complexity = self._calculate_cognitive_complexity(tree)
            function_count = self._count_functions(tree)
            class_count = self._count_classes(tree)
            line_count = len(content.splitlines())
            import_count = self._count_imports(tree)
            nested_loops = self._count_nested_loops(tree)
            recursive_calls = self._count_recursive_calls(tree)
            memory_patterns = self._analyze_memory_patterns(tree)
            
            # Generate performance issues
            issues = self._generate_performance_issues(
                file_path, tree, cyclomatic_complexity, cognitive_complexity, 
                nested_loops, recursive_calls, memory_patterns
            )
            
            return PerformanceMetrics(
                cyclomatic_complexity=cyclomatic_complexity,
                cognitive_complexity=cognitive_complexity,
                function_count=function_count,
                class_count=class_count,
                line_count=line_count,
                import_count=import_count,
                nested_loops=nested_loops,
                recursive_calls=recursive_calls,
                memory_patterns=memory_patterns,
                performance_issues=issues
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return PerformanceMetrics(
                cyclomatic_complexity=0,
                cognitive_complexity=0,
                function_count=0,
                class_count=0,
                line_count=0,
                import_count=0,
                nested_loops=0,
                recursive_calls=0,
                memory_patterns=[],
                performance_issues=[]
            )
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of the code."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(node, ast.Assert):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # Each additional condition in boolean operations
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Lambda):
                complexity += 1
            elif isinstance(node, ast.ListComp):
                complexity += 1
            elif isinstance(node, ast.DictComp):
                complexity += 1
            elif isinstance(node, ast.SetComp):
                complexity += 1
            elif isinstance(node, ast.GeneratorExp):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (more human-friendly metric)."""
        complexity = 0
        nesting_level = 0
        
        def calculate_node_complexity(node, level=0):
            nonlocal complexity
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1 + level
            elif isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    complexity += 1
                elif isinstance(node.op, ast.Or):
                    complexity += 1
            elif isinstance(node, ast.Lambda):
                complexity += 1
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
            
            # Increase nesting level for certain constructs
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                level += 1
            
            for child in ast.iter_child_nodes(node):
                calculate_node_complexity(child, level)
        
        calculate_node_complexity(tree)
        return complexity
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count the number of functions in the code."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                count += 1
        return count
    
    def _count_classes(self, tree: ast.AST) -> int:
        """Count the number of classes in the code."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                count += 1
        return count
    
    def _count_imports(self, tree: ast.AST) -> int:
        """Count the number of import statements."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                count += len(node.names)
            elif isinstance(node, ast.ImportFrom):
                count += len(node.names)
        return count
    
    def _count_nested_loops(self, tree: ast.AST) -> int:
        """Count nested loops in the code."""
        max_nesting = 0
        
        def count_nesting(node, current_nesting=0):
            nonlocal max_nesting
            
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            
            for child in ast.iter_child_nodes(node):
                count_nesting(child, current_nesting)
        
        count_nesting(tree)
        return max_nesting
    
    def _count_recursive_calls(self, tree: ast.AST) -> int:
        """Count potential recursive function calls."""
        recursive_count = 0
        function_names = set()
        
        # First pass: collect function names
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_names.add(node.name)
        
        # Second pass: look for recursive calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in function_names:
                    recursive_count += 1
        
        return recursive_count
    
    def _analyze_memory_patterns(self, tree: ast.AST) -> List[str]:
        """Analyze memory usage patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            # Large list/dict comprehensions
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                patterns.append("comprehension")
            
            # String concatenation in loops
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        if isinstance(child.target, ast.Name):
                            patterns.append("string_concatenation_in_loop")
            
            # Global variables
            if isinstance(node, ast.Global):
                patterns.append("global_variable")
            
            # Deep recursion potential
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "deepcopy":
                    patterns.append("deep_copy")
        
        return list(set(patterns))  # Remove duplicates
    
    def _generate_performance_issues(
        self, file_path: Path, tree: ast.AST, cyclomatic_complexity: int,
        cognitive_complexity: int, nested_loops: int, recursive_calls: int,
        memory_patterns: List[str]
    ) -> List[Issue]:
        """Generate performance issues based on analysis."""
        issues = []
        
        # High cyclomatic complexity
        if cyclomatic_complexity > self.complexity_threshold:
            issues.append(Issue(
                file=file_path,
                line=1,
                column=1,
                message=f"High cyclomatic complexity ({cyclomatic_complexity}). Consider breaking down the code.",
                rule="PERF001",
                severity="warning",
                tool="performance",
                fixable=False,
                category="performance",
                suggestion=f"Break down complex functions. Target complexity: {self.complexity_threshold}"
            ))
        
        # High cognitive complexity
        if cognitive_complexity > self.cognitive_threshold:
            issues.append(Issue(
                file=file_path,
                line=1,
                column=1,
                message=f"High cognitive complexity ({cognitive_complexity}). Code may be hard to understand.",
                rule="PERF002",
                severity="warning",
                tool="performance",
                fixable=False,
                category="performance",
                suggestion=f"Simplify complex logic. Target complexity: {self.cognitive_threshold}"
            ))
        
        # Deep nested loops
        if nested_loops > self.max_nested_loops:
            issues.append(Issue(
                file=file_path,
                line=1,
                column=1,
                message=f"Deep nested loops detected ({nested_loops} levels). May impact performance.",
                rule="PERF003",
                severity="info",
                tool="performance",
                fixable=False,
                category="performance",
                suggestion="Consider using list comprehensions or extracting inner loops into functions"
            ))
        
        # Memory pattern issues
        if "string_concatenation_in_loop" in memory_patterns:
            issues.append(Issue(
                file=file_path,
                line=1,
                column=1,
                message="String concatenation in loop detected. Consider using join() or f-strings.",
                rule="PERF004",
                severity="warning",
                tool="performance",
                fixable=False,
                category="performance",
                suggestion="Use ''.join() for multiple string concatenations or f-strings for formatting"
            ))
        
        if "global_variable" in memory_patterns:
            issues.append(Issue(
                file=file_path,
                line=1,
                column=1,
                message="Global variables detected. May impact performance and maintainability.",
                rule="PERF005",
                severity="info",
                tool="performance",
                fixable=False,
                category="performance",
                suggestion="Consider using class attributes or function parameters instead of globals"
            ))
        
        return issues
    
    def analyze_multiple_files(self, files: List[Path]) -> Dict[Path, PerformanceMetrics]:
        """Analyze multiple files for performance issues."""
        results = {}
        
        for file_path in files:
            if file_path.suffix == '.py':
                results[file_path] = self.analyze_file(file_path)
        
        return results
    
    def generate_performance_report(self, metrics: Dict[Path, PerformanceMetrics]) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "summary": {
                "total_files": len(metrics),
                "total_issues": 0,
                "average_complexity": 0,
                "average_cognitive_complexity": 0,
                "total_functions": 0,
                "total_classes": 0,
                "total_lines": 0,
            },
            "complexity_distribution": {},
            "most_complex_files": [],
            "performance_patterns": {},
            "recommendations": [],
        }
        
        if not metrics:
            return report
        
        # Calculate summary statistics
        total_complexity = 0
        total_cognitive_complexity = 0
        total_issues = 0
        
        for file_path, metric in metrics.items():
            total_complexity += metric.cyclomatic_complexity
            total_cognitive_complexity += metric.cognitive_complexity
            total_issues += len(metric.performance_issues)
            report["summary"]["total_functions"] += metric.function_count
            report["summary"]["total_classes"] += metric.class_count
            report["summary"]["total_lines"] += metric.line_count
        
        report["summary"]["total_issues"] = total_issues
        report["summary"]["average_complexity"] = total_complexity / len(metrics)
        report["summary"]["average_cognitive_complexity"] = total_cognitive_complexity / len(metrics)
        
        # Find most complex files
        sorted_files = sorted(
            metrics.items(), 
            key=lambda x: x[1].cyclomatic_complexity, 
            reverse=True
        )
        
        report["most_complex_files"] = [
            {
                "file": str(file_path),
                "complexity": metric.cyclomatic_complexity,
                "cognitive_complexity": metric.cognitive_complexity,
                "issues": len(metric.performance_issues)
            }
            for file_path, metric in sorted_files[:10]
        ]
        
        # Analyze performance patterns
        all_patterns = []
        for metric in metrics.values():
            all_patterns.extend(metric.memory_patterns)
        
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        report["performance_patterns"] = pattern_counts
        
        # Generate recommendations
        if report["summary"]["average_complexity"] > self.complexity_threshold:
            report["recommendations"].append(
                "Consider breaking down complex functions to improve maintainability"
            )
        
        if "string_concatenation_in_loop" in pattern_counts:
            report["recommendations"].append(
                "Replace string concatenation in loops with join() for better performance"
            )
        
        if "global_variable" in pattern_counts:
            report["recommendations"].append(
                "Minimize global variable usage to improve code isolation"
            )
        
        return report