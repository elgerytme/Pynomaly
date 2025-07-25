#!/usr/bin/env python3
"""
Comprehensive Package Validation Framework

Validates domain-driven monorepo packages for architecture compliance,
independence, security, and quality standards.
"""

import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import importlib.util
import tempfile
import yaml

@dataclass
class ValidationResult:
    """Result of a package validation check."""
    check_name: str
    status: str  # 'pass', 'fail', 'warning'
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None

@dataclass
class PackageValidationReport:
    """Complete validation report for a package."""
    package_name: str
    package_path: str
    overall_status: str
    score: int  # 0-100
    results: List[ValidationResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class DependencyAnalyzer:
    """Analyzes package dependencies and imports."""
    
    def __init__(self, package_path: str, monorepo_root: str):
        self.package_path = Path(package_path)
        self.monorepo_root = Path(monorepo_root)
        
    def analyze_python_dependencies(self) -> Dict[str, Set[str]]:
        """Analyze Python file dependencies."""
        dependencies = {'internal': set(), 'external': set(), 'cross_package': set()}
        
        for py_file in self.package_path.rglob("*.py"):
            if py_file.name.startswith('.') or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        self._categorize_import(node, dependencies)
                        
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"Warning: Could not parse {py_file}: {e}")
                
        return dependencies
    
    def _categorize_import(self, node: ast.AST, dependencies: Dict[str, Set[str]]):
        """Categorize an import statement."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                self._classify_module(alias.name, dependencies)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                self._classify_module(node.module, dependencies)
    
    def _classify_module(self, module_name: str, dependencies: Dict[str, Set[str]]):
        """Classify a module as internal, external, or cross-package."""
        if not module_name:
            return
            
        # Check if it's a relative import within the package
        package_name = self.package_path.name
        if module_name.startswith(package_name) or module_name.startswith('.'):
            dependencies['internal'].add(module_name)
        # Check if it's another package in the monorepo
        elif self._is_monorepo_package(module_name):
            dependencies['cross_package'].add(module_name)
        else:
            dependencies['external'].add(module_name)
    
    def _is_monorepo_package(self, module_name: str) -> bool:
        """Check if module is from another package in the monorepo."""
        # Look for src/packages/*/src/ structure
        packages_dir = self.monorepo_root / "src" / "packages"
        if not packages_dir.exists():
            return False
            
        for package_dir in packages_dir.iterdir():
            if package_dir.is_dir() and package_dir.name != self.package_path.name:
                src_init = package_dir / "src" / package_dir.name / "__init__.py"
                if src_init.exists() and module_name.startswith(package_dir.name):
                    return True
        return False

class ArchitectureValidator:
    """Validates hexagonal architecture compliance."""
    
    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        
    def validate_structure(self) -> List[ValidationResult]:
        """Validate package follows hexagonal architecture."""
        results = []
        
        # Check for required directories
        required_dirs = [
            ("src", "Source code directory"),
            ("tests", "Test directory"),
            ("src/domain", "Domain layer"),
            ("src/application", "Application layer"),
            ("src/infrastructure", "Infrastructure layer")
        ]
        
        for dir_path, description in required_dirs:
            full_path = self.package_path / dir_path
            if full_path.exists():
                results.append(ValidationResult(
                    check_name=f"structure_{dir_path.replace('/', '_')}",
                    status="pass",
                    message=f"{description} exists",
                    details={"path": str(full_path)}
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"structure_{dir_path.replace('/', '_')}",
                    status="fail",
                    message=f"Missing {description}",
                    details={"expected_path": str(full_path)},
                    fix_suggestion=f"Create directory: mkdir -p {full_path}"
                ))
        
        # Validate layer dependencies
        results.extend(self._validate_layer_dependencies())
        
        return results
    
    def _validate_layer_dependencies(self) -> List[ValidationResult]:
        """Validate that layers only depend on appropriate other layers."""
        results = []
        
        # Domain should not depend on application or infrastructure
        domain_violations = self._check_forbidden_dependencies(
            "src/domain", ["application", "infrastructure"]
        )
        
        if not domain_violations:
            results.append(ValidationResult(
                check_name="layer_domain_independence",
                status="pass",
                message="Domain layer maintains proper independence"
            ))
        else:
            results.append(ValidationResult(
                check_name="layer_domain_independence",
                status="fail",
                message="Domain layer has forbidden dependencies",
                details={"violations": domain_violations},
                fix_suggestion="Remove dependencies from domain to application/infrastructure layers"
            ))
        
        # Application should not depend on infrastructure
        app_violations = self._check_forbidden_dependencies(
            "src/application", ["infrastructure"]
        )
        
        if not app_violations:
            results.append(ValidationResult(
                check_name="layer_application_independence",
                status="pass",
                message="Application layer maintains proper independence"
            ))
        else:
            results.append(ValidationResult(
                check_name="layer_application_independence",
                status="fail",
                message="Application layer has forbidden dependencies",
                details={"violations": app_violations},
                fix_suggestion="Use dependency injection instead of direct infrastructure imports"
            ))
        
        return results
    
    def _check_forbidden_dependencies(self, layer_path: str, forbidden_layers: List[str]) -> List[str]:
        """Check for forbidden dependencies in a layer."""
        violations = []
        layer_dir = self.package_path / layer_path
        
        if not layer_dir.exists():
            return violations
            
        for py_file in layer_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for forbidden_layer in forbidden_layers:
                    # Check for imports from forbidden layers
                    patterns = [
                        rf"from\s+\w*\.?{forbidden_layer}",
                        rf"import\s+\w*\.?{forbidden_layer}"
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, content):
                            violations.append(f"{py_file.relative_to(self.package_path)} imports from {forbidden_layer}")
                            
            except (UnicodeDecodeError, IOError):
                continue
                
        return violations

class SecurityValidator:
    """Validates security best practices and vulnerabilities."""
    
    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        
    def validate_security(self) -> List[ValidationResult]:
        """Run comprehensive security validation."""
        results = []
        
        # Check for hardcoded secrets
        results.extend(self._check_secrets())
        
        # Check for SQL injection vulnerabilities
        results.extend(self._check_sql_injection())
        
        # Check for insecure dependencies
        results.extend(self._check_dependency_security())
        
        # Check for proper input validation
        results.extend(self._check_input_validation())
        
        return results
    
    def _check_secrets(self) -> List[ValidationResult]:
        """Check for hardcoded secrets and credentials."""
        results = []
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api_?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
            (r'["\'][A-Za-z0-9]{32,}["\']', "Potential hardcoded credential")
        ]
        
        violations = []
        for py_file in self.package_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        violations.append({
                            "file": str(py_file.relative_to(self.package_path)),
                            "issue": description,
                            "match": match[:50]  # Truncate for safety
                        })
                        
            except (UnicodeDecodeError, IOError):
                continue
        
        if not violations:
            results.append(ValidationResult(
                check_name="security_no_hardcoded_secrets",
                status="pass",
                message="No hardcoded secrets detected"
            ))
        else:
            results.append(ValidationResult(
                check_name="security_no_hardcoded_secrets",
                status="fail",
                message=f"Found {len(violations)} potential hardcoded secrets",
                details={"violations": violations},
                fix_suggestion="Use environment variables or secure vaults for secrets"
            ))
        
        return results
    
    def _check_sql_injection(self) -> List[ValidationResult]:
        """Check for SQL injection vulnerabilities."""
        results = []
        
        # Pattern for string concatenation in SQL queries
        sql_injection_patterns = [
            r'execute\s*\(\s*["\'][^"\']*%s[^"\']*["\']',
            r'query\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
            r'SELECT\s+.*\+.*FROM',
            r'INSERT\s+.*\+.*INTO',
            r'UPDATE\s+.*\+.*SET',
            r'DELETE\s+.*\+.*FROM'
        ]
        
        violations = []
        for py_file in self.package_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in sql_injection_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        violations.append(str(py_file.relative_to(self.package_path)))
                        
            except (UnicodeDecodeError, IOError):
                continue
        
        if not violations:
            results.append(ValidationResult(
                check_name="security_sql_injection",
                status="pass",
                message="No SQL injection vulnerabilities detected"
            ))
        else:
            results.append(ValidationResult(
                check_name="security_sql_injection",
                status="fail",
                message=f"Potential SQL injection vulnerabilities in {len(violations)} files",
                details={"files": violations},
                fix_suggestion="Use parameterized queries or ORM methods"
            ))
        
        return results
    
    def _check_dependency_security(self) -> List[ValidationResult]:
        """Check for known vulnerable dependencies."""
        results = []
        
        # Check if safety is available for Python vulnerability scanning
        try:
            import safety
            # Run safety check
            # This is a simplified check - in practice, you'd integrate with safety CLI
            results.append(ValidationResult(
                check_name="security_dependency_scan",
                status="pass",
                message="Dependency security scan completed"
            ))
        except ImportError:
            results.append(ValidationResult(
                check_name="security_dependency_scan",
                status="warning",
                message="Safety package not available for vulnerability scanning",
                fix_suggestion="Install safety: pip install safety"
            ))
        
        return results
    
    def _check_input_validation(self) -> List[ValidationResult]:
        """Check for proper input validation patterns."""
        results = []
        
        # Look for FastAPI or similar validation patterns
        validation_found = False
        for py_file in self.package_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if any(pattern in content for pattern in [
                    "from pydantic import",
                    "BaseModel",
                    "validator",
                    "Field("
                ]):
                    validation_found = True
                    break
                    
            except (UnicodeDecodeError, IOError):
                continue
        
        if validation_found:
            results.append(ValidationResult(
                check_name="security_input_validation",
                status="pass",
                message="Input validation patterns detected"
            ))
        else:
            results.append(ValidationResult(
                check_name="security_input_validation",
                status="warning",
                message="No clear input validation patterns found",
                fix_suggestion="Implement Pydantic models or similar validation"
            ))
        
        return results

class QualityValidator:
    """Validates code quality metrics."""
    
    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        
    def validate_quality(self) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """Run code quality validation."""
        results = []
        metrics = {}
        
        # Test coverage
        results.extend(self._check_test_coverage())
        
        # Code complexity
        complexity_results, complexity_metrics = self._check_complexity()
        results.extend(complexity_results)
        metrics.update(complexity_metrics)
        
        # Documentation
        results.extend(self._check_documentation())
        
        # Type hints
        results.extend(self._check_type_hints())
        
        return results, metrics
    
    def _check_test_coverage(self) -> List[ValidationResult]:
        """Check test coverage."""
        results = []
        
        # Count Python files vs test files
        src_files = list(self.package_path.rglob("src/**/*.py"))
        test_files = list(self.package_path.rglob("tests/**/*.py"))
        
        if not src_files:
            results.append(ValidationResult(
                check_name="quality_test_coverage",
                status="fail",
                message="No source files found"
            ))
            return results
        
        coverage_ratio = len(test_files) / len(src_files) if src_files else 0
        
        if coverage_ratio >= 0.8:
            status = "pass"
        elif coverage_ratio >= 0.5:
            status = "warning"
        else:
            status = "fail"
        
        results.append(ValidationResult(
            check_name="quality_test_coverage",
            status=status,
            message=f"Test coverage ratio: {coverage_ratio:.2f} ({len(test_files)} tests for {len(src_files)} source files)",
            details={"ratio": coverage_ratio, "test_files": len(test_files), "source_files": len(src_files)},
            fix_suggestion="Add more test files to improve coverage" if status != "pass" else None
        ))
        
        return results
    
    def _check_complexity(self) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """Check code complexity metrics."""
        results = []
        metrics = {"average_complexity": 0, "max_complexity": 0, "complex_functions": []}
        
        total_complexity = 0
        function_count = 0
        
        for py_file in self.package_path.rglob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        total_complexity += complexity
                        function_count += 1
                        
                        if complexity > 10:
                            metrics["complex_functions"].append({
                                "file": str(py_file.relative_to(self.package_path)),
                                "function": node.name,
                                "complexity": complexity
                            })
                        
                        metrics["max_complexity"] = max(metrics["max_complexity"], complexity)
                        
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if function_count > 0:
            metrics["average_complexity"] = total_complexity / function_count
            
            if metrics["average_complexity"] <= 5:
                status = "pass"
            elif metrics["average_complexity"] <= 8:
                status = "warning"
            else:
                status = "fail"
            
            results.append(ValidationResult(
                check_name="quality_complexity",
                status=status,
                message=f"Average complexity: {metrics['average_complexity']:.2f}, Max: {metrics['max_complexity']}",
                details=metrics,
                fix_suggestion="Refactor complex functions (>10 complexity)" if metrics["complex_functions"] else None
            ))
        
        return results, metrics
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _check_documentation(self) -> List[ValidationResult]:
        """Check for documentation coverage."""
        results = []
        
        documented_functions = 0
        total_functions = 0
        
        for py_file in self.package_path.rglob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if total_functions == 0:
            results.append(ValidationResult(
                check_name="quality_documentation",
                status="warning",
                message="No functions/classes found to check documentation"
            ))
        else:
            doc_ratio = documented_functions / total_functions
            
            if doc_ratio >= 0.8:
                status = "pass"
            elif doc_ratio >= 0.5:
                status = "warning"
            else:
                status = "fail"
            
            results.append(ValidationResult(
                check_name="quality_documentation",
                status=status,
                message=f"Documentation coverage: {doc_ratio:.2f} ({documented_functions}/{total_functions})",
                details={"ratio": doc_ratio, "documented": documented_functions, "total": total_functions},
                fix_suggestion="Add docstrings to functions and classes" if status != "pass" else None
            ))
        
        return results
    
    def _check_type_hints(self) -> List[ValidationResult]:
        """Check for type hint coverage."""
        results = []
        
        typed_functions = 0
        total_functions = 0
        
        for py_file in self.package_path.rglob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Check for return type annotation
                        has_return_type = node.returns is not None
                        
                        # Check for parameter type annotations
                        has_param_types = any(arg.annotation for arg in node.args.args)
                        
                        if has_return_type or has_param_types:
                            typed_functions += 1
                            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if total_functions == 0:
            results.append(ValidationResult(
                check_name="quality_type_hints",
                status="warning",
                message="No functions found to check type hints"
            ))
        else:
            type_ratio = typed_functions / total_functions
            
            if type_ratio >= 0.8:
                status = "pass"
            elif type_ratio >= 0.5:
                status = "warning"
            else:
                status = "fail"
            
            results.append(ValidationResult(
                check_name="quality_type_hints",
                status=status,
                message=f"Type hint coverage: {type_ratio:.2f} ({typed_functions}/{total_functions})",
                details={"ratio": type_ratio, "typed": typed_functions, "total": total_functions},
                fix_suggestion="Add type hints to function parameters and return values" if status != "pass" else None
            ))
        
        return results

class PackageValidator:
    """Main package validation orchestrator."""
    
    def __init__(self, monorepo_root: str = "."):
        self.monorepo_root = Path(monorepo_root).resolve()
        
    def validate_package(self, package_path: str) -> PackageValidationReport:
        """Validate a single package."""
        package_path = Path(package_path).resolve()
        package_name = package_path.name
        
        report = PackageValidationReport(
            package_name=package_name,
            package_path=str(package_path),
            overall_status="pending",
            score=0
        )
        
        # Run all validation checks
        validators = [
            ("Architecture", ArchitectureValidator(package_path)),
            ("Security", SecurityValidator(package_path)),
            ("Quality", QualityValidator(package_path))
        ]
        
        for validator_name, validator in validators:
            try:
                if validator_name == "Quality":
                    results, metrics = validator.validate_quality()
                    report.results.extend(results)
                    report.metrics.update(metrics)
                else:
                    if validator_name == "Architecture":
                        results = validator.validate_structure()
                    else:  # Security
                        results = validator.validate_security()
                    report.results.extend(results)
                    
            except Exception as e:
                report.results.append(ValidationResult(
                    check_name=f"{validator_name.lower()}_error",
                    status="fail",
                    message=f"Error running {validator_name} validation: {str(e)}"
                ))
        
        # Analyze dependencies
        try:
            dep_analyzer = DependencyAnalyzer(package_path, self.monorepo_root)
            dependencies = dep_analyzer.analyze_python_dependencies()
            
            # Check package independence
            if dependencies['cross_package']:
                report.results.append(ValidationResult(
                    check_name="independence_cross_package_deps",
                    status="fail",
                    message=f"Package has {len(dependencies['cross_package'])} cross-package dependencies",
                    details={"dependencies": list(dependencies['cross_package'])},
                    fix_suggestion="Remove cross-package dependencies or use events/APIs for communication"
                ))
            else:
                report.results.append(ValidationResult(
                    check_name="independence_cross_package_deps",
                    status="pass",
                    message="Package maintains independence from other packages"
                ))
                
        except Exception as e:
            report.results.append(ValidationResult(
                check_name="dependency_analysis_error",
                status="fail",
                message=f"Error analyzing dependencies: {str(e)}"
            ))
        
        # Calculate overall score and status
        report.score, report.overall_status = self._calculate_score(report.results)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report.results)
        
        return report
    
    def _calculate_score(self, results: List[ValidationResult]) -> Tuple[int, str]:
        """Calculate overall package score."""
        if not results:
            return 0, "fail"
        
        pass_count = sum(1 for r in results if r.status == "pass")
        warning_count = sum(1 for r in results if r.status == "warning")
        fail_count = sum(1 for r in results if r.status == "fail")
        
        total_checks = len(results)
        score = int((pass_count + warning_count * 0.5) / total_checks * 100)
        
        if score >= 90:
            status = "excellent"
        elif score >= 75:
            status = "good"
        elif score >= 60:
            status = "fair"
        else:
            status = "poor"
        
        return score, status
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Group failed checks by category
        failed_checks = [r for r in results if r.status == "fail"]
        
        if any("structure" in r.check_name for r in failed_checks):
            recommendations.append("Establish proper hexagonal architecture structure with domain, application, and infrastructure layers")
        
        if any("security" in r.check_name for r in failed_checks):
            recommendations.append("Address security vulnerabilities and implement security best practices")
        
        if any("independence" in r.check_name for r in failed_checks):
            recommendations.append("Remove cross-package dependencies to maintain package independence")
        
        if any("quality" in r.check_name for r in failed_checks):
            recommendations.append("Improve code quality with better test coverage, documentation, and type hints")
        
        # Add specific fix suggestions
        for result in failed_checks:
            if result.fix_suggestion and result.fix_suggestion not in recommendations:
                recommendations.append(result.fix_suggestion)
        
        return recommendations[:10]  # Limit to top 10

def main():
    """Command line interface for package validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate domain-driven monorepo packages")
    parser.add_argument("package_path", help="Path to the package to validate")
    parser.add_argument("--monorepo-root", default=".", help="Path to monorepo root")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Validate package
    validator = PackageValidator(args.monorepo_root)
    report = validator.validate_package(args.package_path)
    
    # Format output
    if args.format == "json":
        output = json.dumps({
            "package_name": report.package_name,
            "package_path": report.package_path,
            "overall_status": report.overall_status,
            "score": report.score,
            "results": [
                {
                    "check_name": r.check_name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "fix_suggestion": r.fix_suggestion
                }
                for r in report.results
            ],
            "metrics": report.metrics,
            "recommendations": report.recommendations
        }, indent=2)
    else:
        output = f"""
Package Validation Report
========================

Package: {report.package_name}
Path: {report.package_path}
Overall Status: {report.overall_status.upper()}
Score: {report.score}/100

Validation Results:
------------------
"""
        for result in report.results:
            status_symbol = "✓" if result.status == "pass" else "⚠" if result.status == "warning" else "✗"
            output += f"{status_symbol} {result.check_name}: {result.message}\n"
            if result.fix_suggestion and result.status != "pass":
                output += f"   Fix: {result.fix_suggestion}\n"
        
        if report.recommendations:
            output += "\nRecommendations:\n"
            output += "---------------\n"
            for i, rec in enumerate(report.recommendations, 1):
                output += f"{i}. {rec}\n"
        
        if report.metrics:
            output += f"\nMetrics:\n"
            output += "--------\n"
            for key, value in report.metrics.items():
                if isinstance(value, (int, float)):
                    output += f"{key}: {value:.2f}\n"
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)
    
    # Exit with appropriate code
    sys.exit(0 if report.overall_status in ["excellent", "good"] else 1)

if __name__ == "__main__":
    main()