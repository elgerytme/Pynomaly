"""Documentation analysis for comprehensive static analysis."""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

from ..tools.adapter_base import Issue

logger = logging.getLogger(__name__)


@dataclass
class DocMetrics:
    """Documentation metrics for code analysis."""
    total_functions: int
    documented_functions: int
    total_classes: int
    documented_classes: int
    total_modules: int
    documented_modules: int
    docstring_coverage: float
    docstring_quality_score: float
    missing_docstrings: List[str]
    documentation_issues: List[Issue]


class DocumentationAnalyzer:
    """Analyzes code for documentation completeness and quality."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.doc_config = config.get("documentation", {})
        self.require_module_docstring = self.doc_config.get("require_module_docstring", True)
        self.require_class_docstring = self.doc_config.get("require_class_docstring", True)
        self.require_function_docstring = self.doc_config.get("require_function_docstring", True)
        self.require_public_only = self.doc_config.get("require_public_only", True)
        self.min_docstring_length = self.doc_config.get("min_docstring_length", 10)
        self.docstring_style = self.doc_config.get("docstring_style", "google")  # google, numpy, sphinx
        
    def analyze_file(self, file_path: Path) -> DocMetrics:
        """Analyze a single file for documentation issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze different components
            module_doc = self._analyze_module_docstring(tree)
            class_docs = self._analyze_class_docstrings(tree)
            function_docs = self._analyze_function_docstrings(tree)
            
            # Calculate metrics
            total_functions = len(function_docs)
            documented_functions = sum(1 for _, has_doc, _ in function_docs if has_doc)
            total_classes = len(class_docs)
            documented_classes = sum(1 for _, has_doc, _ in class_docs if has_doc)
            total_modules = 1
            documented_modules = 1 if module_doc[1] else 0
            
            # Calculate coverage
            total_items = total_functions + total_classes + total_modules
            documented_items = documented_functions + documented_classes + documented_modules
            docstring_coverage = (documented_items / total_items * 100) if total_items > 0 else 100
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                tree, module_doc, class_docs, function_docs
            )
            
            # Find missing docstrings
            missing_docstrings = []
            if not module_doc[1] and self.require_module_docstring:
                missing_docstrings.append("module")
            
            for name, has_doc, _ in class_docs:
                if not has_doc and self.require_class_docstring:
                    missing_docstrings.append(f"class {name}")
            
            for name, has_doc, _ in function_docs:
                if not has_doc and self.require_function_docstring:
                    if not self.require_public_only or not name.startswith('_'):
                        missing_docstrings.append(f"function {name}")
            
            # Generate documentation issues
            issues = self._generate_documentation_issues(
                file_path, tree, module_doc, class_docs, function_docs
            )
            
            return DocMetrics(
                total_functions=total_functions,
                documented_functions=documented_functions,
                total_classes=total_classes,
                documented_classes=documented_classes,
                total_modules=total_modules,
                documented_modules=documented_modules,
                docstring_coverage=docstring_coverage,
                docstring_quality_score=quality_score,
                missing_docstrings=missing_docstrings,
                documentation_issues=issues
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return DocMetrics(
                total_functions=0,
                documented_functions=0,
                total_classes=0,
                documented_classes=0,
                total_modules=0,
                documented_modules=0,
                docstring_coverage=0,
                docstring_quality_score=0,
                missing_docstrings=[],
                documentation_issues=[]
            )
    
    def _analyze_module_docstring(self, tree: ast.AST) -> Tuple[Optional[str], bool, int]:
        """Analyze module-level docstring."""
        docstring = ast.get_docstring(tree)
        has_docstring = docstring is not None
        line_number = 1
        
        # Find the line number of the docstring
        if isinstance(tree, ast.Module) and tree.body:
            first_stmt = tree.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                line_number = first_stmt.lineno
        
        return docstring, has_docstring, line_number
    
    def _analyze_class_docstrings(self, tree: ast.AST) -> List[Tuple[str, bool, int]]:
        """Analyze class docstrings."""
        class_docs = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                has_docstring = docstring is not None
                class_docs.append((node.name, has_docstring, node.lineno))
        
        return class_docs
    
    def _analyze_function_docstrings(self, tree: ast.AST) -> List[Tuple[str, bool, int]]:
        """Analyze function docstrings."""
        function_docs = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                has_docstring = docstring is not None
                function_docs.append((node.name, has_docstring, node.lineno))
        
        return function_docs
    
    def _calculate_quality_score(
        self, tree: ast.AST, module_doc: Tuple[Optional[str], bool, int],
        class_docs: List[Tuple[str, bool, int]], 
        function_docs: List[Tuple[str, bool, int]]
    ) -> float:
        """Calculate documentation quality score."""
        score = 0
        total_points = 0
        
        # Module docstring quality
        if module_doc[1]:
            docstring = module_doc[0]
            if docstring and len(docstring) >= self.min_docstring_length:
                score += 20
            total_points += 20
        
        # Class docstring quality
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                total_points += 10
                docstring = ast.get_docstring(node)
                if docstring:
                    if len(docstring) >= self.min_docstring_length:
                        score += 5
                    if self._has_parameter_documentation(docstring):
                        score += 3
                    if self._has_return_documentation(docstring):
                        score += 2
        
        # Function docstring quality
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_points += 10
                docstring = ast.get_docstring(node)
                if docstring:
                    if len(docstring) >= self.min_docstring_length:
                        score += 5
                    if self._has_parameter_documentation(docstring):
                        score += 3
                    if self._has_return_documentation(docstring):
                        score += 2
        
        return (score / total_points * 100) if total_points > 0 else 100
    
    def _has_parameter_documentation(self, docstring: str) -> bool:
        """Check if docstring has parameter documentation."""
        if self.docstring_style == "google":
            return bool(re.search(r'Args:|Arguments:', docstring))
        elif self.docstring_style == "numpy":
            return bool(re.search(r'Parameters\s*\n\s*-+', docstring))
        elif self.docstring_style == "sphinx":
            return bool(re.search(r':param\s+\w+:', docstring))
        else:
            return False
    
    def _has_return_documentation(self, docstring: str) -> bool:
        """Check if docstring has return documentation."""
        if self.docstring_style == "google":
            return bool(re.search(r'Returns:', docstring))
        elif self.docstring_style == "numpy":
            return bool(re.search(r'Returns\s*\n\s*-+', docstring))
        elif self.docstring_style == "sphinx":
            return bool(re.search(r':returns?:', docstring))
        else:
            return False
    
    def _generate_documentation_issues(
        self, file_path: Path, tree: ast.AST,
        module_doc: Tuple[Optional[str], bool, int],
        class_docs: List[Tuple[str, bool, int]],
        function_docs: List[Tuple[str, bool, int]]
    ) -> List[Issue]:
        """Generate documentation issues based on analysis."""
        issues = []
        
        # Missing module docstring
        if not module_doc[1] and self.require_module_docstring:
            issues.append(Issue(
                file=file_path,
                line=1,
                column=1,
                message="Module is missing a docstring",
                rule="DOC001",
                severity="warning",
                tool="documentation",
                fixable=False,
                category="documentation",
                suggestion="Add a module-level docstring explaining the module's purpose"
            ))
        
        # Missing class docstrings
        for name, has_doc, line_no in class_docs:
            if not has_doc and self.require_class_docstring:
                issues.append(Issue(
                    file=file_path,
                    line=line_no,
                    column=1,
                    message=f"Class '{name}' is missing a docstring",
                    rule="DOC002",
                    severity="warning",
                    tool="documentation",
                    fixable=False,
                    category="documentation",
                    suggestion="Add a class docstring explaining the class's purpose and usage"
                ))
        
        # Missing function docstrings
        for name, has_doc, line_no in function_docs:
            if not has_doc and self.require_function_docstring:
                if not self.require_public_only or not name.startswith('_'):
                    issues.append(Issue(
                        file=file_path,
                        line=line_no,
                        column=1,
                        message=f"Function '{name}' is missing a docstring",
                        rule="DOC003",
                        severity="info",
                        tool="documentation",
                        fixable=False,
                        category="documentation",
                        suggestion="Add a function docstring explaining parameters, return value, and purpose"
                    ))
        
        # Check docstring quality
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if docstring and len(docstring) < self.min_docstring_length:
                    issues.append(Issue(
                        file=file_path,
                        line=node.lineno,
                        column=1,
                        message=f"Class '{node.name}' has a very short docstring",
                        rule="DOC004",
                        severity="info",
                        tool="documentation",
                        fixable=False,
                        category="documentation",
                        suggestion=f"Expand docstring to at least {self.min_docstring_length} characters"
                    ))
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if docstring and len(docstring) < self.min_docstring_length:
                    if not self.require_public_only or not node.name.startswith('_'):
                        issues.append(Issue(
                            file=file_path,
                            line=node.lineno,
                            column=1,
                            message=f"Function '{node.name}' has a very short docstring",
                            rule="DOC005",
                            severity="info",
                            tool="documentation",
                            fixable=False,
                            category="documentation",
                            suggestion=f"Expand docstring to at least {self.min_docstring_length} characters"
                        ))
        
        return issues
    
    def analyze_multiple_files(self, files: List[Path]) -> Dict[Path, DocMetrics]:
        """Analyze multiple files for documentation issues."""
        results = {}
        
        for file_path in files:
            if file_path.suffix == '.py':
                results[file_path] = self.analyze_file(file_path)
        
        return results
    
    def generate_documentation_report(self, metrics: Dict[Path, DocMetrics]) -> Dict[str, Any]:
        """Generate a comprehensive documentation report."""
        report = {
            "summary": {
                "total_files": len(metrics),
                "total_functions": 0,
                "documented_functions": 0,
                "total_classes": 0,
                "documented_classes": 0,
                "total_modules": 0,
                "documented_modules": 0,
                "overall_coverage": 0,
                "average_quality_score": 0,
                "total_issues": 0,
            },
            "coverage_by_file": {},
            "poorly_documented_files": [],
            "missing_documentation": {},
            "recommendations": [],
        }
        
        if not metrics:
            return report
        
        # Calculate summary statistics
        total_quality_score = 0
        total_issues = 0
        
        for file_path, metric in metrics.items():
            report["summary"]["total_functions"] += metric.total_functions
            report["summary"]["documented_functions"] += metric.documented_functions
            report["summary"]["total_classes"] += metric.total_classes
            report["summary"]["documented_classes"] += metric.documented_classes
            report["summary"]["total_modules"] += metric.total_modules
            report["summary"]["documented_modules"] += metric.documented_modules
            total_quality_score += metric.docstring_quality_score
            total_issues += len(metric.documentation_issues)
            
            # Coverage by file
            report["coverage_by_file"][str(file_path)] = {
                "coverage": metric.docstring_coverage,
                "quality_score": metric.docstring_quality_score,
                "issues": len(metric.documentation_issues)
            }
        
        # Calculate overall coverage
        total_items = (
            report["summary"]["total_functions"] + 
            report["summary"]["total_classes"] + 
            report["summary"]["total_modules"]
        )
        documented_items = (
            report["summary"]["documented_functions"] + 
            report["summary"]["documented_classes"] + 
            report["summary"]["documented_modules"]
        )
        
        report["summary"]["overall_coverage"] = (
            (documented_items / total_items * 100) if total_items > 0 else 100
        )
        report["summary"]["average_quality_score"] = (
            total_quality_score / len(metrics) if metrics else 0
        )
        report["summary"]["total_issues"] = total_issues
        
        # Find poorly documented files
        poorly_documented = [
            {
                "file": str(file_path),
                "coverage": metric.docstring_coverage,
                "quality_score": metric.docstring_quality_score,
                "missing_items": len(metric.missing_docstrings)
            }
            for file_path, metric in metrics.items()
            if metric.docstring_coverage < 50 or metric.docstring_quality_score < 50
        ]
        
        report["poorly_documented_files"] = sorted(
            poorly_documented, key=lambda x: x["coverage"]
        )
        
        # Collect missing documentation
        missing_types = {}
        for metric in metrics.values():
            for missing in metric.missing_docstrings:
                missing_type = missing.split()[0]
                missing_types[missing_type] = missing_types.get(missing_type, 0) + 1
        
        report["missing_documentation"] = missing_types
        
        # Generate recommendations
        if report["summary"]["overall_coverage"] < 80:
            report["recommendations"].append(
                "Improve documentation coverage by adding docstrings to undocumented items"
            )
        
        if report["summary"]["average_quality_score"] < 70:
            report["recommendations"].append(
                "Improve docstring quality by adding parameter and return documentation"
            )
        
        if "module" in missing_types:
            report["recommendations"].append(
                "Add module-level docstrings to explain the purpose of each module"
            )
        
        if "class" in missing_types:
            report["recommendations"].append(
                "Add class docstrings to explain the purpose and usage of classes"
            )
        
        if "function" in missing_types:
            report["recommendations"].append(
                "Add function docstrings with parameter and return value documentation"
            )
        
        return report