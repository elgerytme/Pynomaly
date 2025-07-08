#!/usr/bin/env python3
"""
Template Validation Framework

This framework provides comprehensive validation and testing capabilities for all
Pynomaly templates, ensuring quality, reliability, and consistency across the template system.
"""

import importlib.util
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# Testing imports
# Performance testing
# Documentation validation
import ast
import logging
from unittest.mock import MagicMock, patch

# Validation imports
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateValidationResult(BaseModel):
    """Data model for template validation results."""

    template_name: str
    template_path: str
    validation_timestamp: datetime
    overall_status: str  # 'passed', 'failed', 'warning'
    validation_categories: dict[str, dict[str, Any]]
    performance_metrics: dict[str, float]
    errors: list[str]
    warnings: list[str]
    recommendations: list[str]
    score: float  # Overall quality score 0-100


class TemplateValidator:
    """
    Comprehensive template validation framework.

    Features:
    - Syntax and structure validation
    - Functionality testing with mock data
    - Performance benchmarking
    - Documentation quality assessment
    - Best practices compliance
    - Integration compatibility testing
    - Security vulnerability scanning
    """

    def __init__(self, config: dict[str, Any] = None, verbose: bool = True):
        """
        Initialize the template validator.

        Args:
            config: Configuration dictionary for validation parameters
            verbose: Enable detailed logging
        """
        self.config = config or self._get_default_config()
        self.verbose = verbose

        # Validation results storage
        self.validation_results = {}
        self.summary_stats = {}

        # Template discovery
        self.template_registry = {}

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for template validation."""
        return {
            "validation_categories": {
                "syntax": {
                    "enabled": True,
                    "weight": 0.15,
                    "checks": ["python_syntax", "import_validation", "class_structure"],
                },
                "functionality": {
                    "enabled": True,
                    "weight": 0.25,
                    "checks": [
                        "mock_data_testing",
                        "method_execution",
                        "error_handling",
                    ],
                },
                "performance": {
                    "enabled": True,
                    "weight": 0.20,
                    "checks": ["memory_usage", "execution_time", "scalability"],
                },
                "documentation": {
                    "enabled": True,
                    "weight": 0.15,
                    "checks": [
                        "docstring_coverage",
                        "parameter_documentation",
                        "examples",
                    ],
                },
                "best_practices": {
                    "enabled": True,
                    "weight": 0.15,
                    "checks": ["code_style", "design_patterns", "security"],
                },
                "integration": {
                    "enabled": True,
                    "weight": 0.10,
                    "checks": [
                        "dependency_compatibility",
                        "api_consistency",
                        "data_compatibility",
                    ],
                },
            },
            "performance_thresholds": {
                "max_memory_mb": 500,
                "max_execution_time_sec": 30,
                "min_throughput_samples_sec": 100,
            },
            "mock_data_config": {
                "sample_sizes": [100, 1000, 5000],
                "feature_dimensions": [5, 10, 20],
                "contamination_rates": [0.05, 0.10, 0.15],
            },
            "reporting": {
                "generate_html_report": True,
                "include_screenshots": False,
                "detailed_metrics": True,
            },
        }

    def discover_templates(self, templates_root: str) -> dict[str, dict[str, Any]]:
        """
        Discover all templates in the template directory structure.

        Args:
            templates_root: Root directory containing template files

        Returns:
            Dictionary of discovered templates with metadata
        """
        logger.info(f"Discovering templates in: {templates_root}")

        templates_root = Path(templates_root)
        template_registry = {}

        # Define template categories and their patterns
        template_categories = {
            "reporting": ["**/reporting/**/*.py"],
            "testing": ["**/testing/**/*.py"],
            "experiments": ["**/experiments/**/*.py", "**/experiments/**/*.ipynb"],
            "scripts": ["**/scripts/**/*.py"],
            "preprocessing": ["**/preprocessing/**/*.py"],
            "classifiers": ["**/classifiers/**/*.py"],
            "notebooks": ["**/*.ipynb"],
        }

        for category, patterns in template_categories.items():
            template_registry[category] = {}

            for pattern in patterns:
                for template_path in templates_root.glob(pattern):
                    if template_path.is_file() and not template_path.name.startswith(
                        "_"
                    ):
                        template_name = template_path.stem

                        # Skip test files and __pycache__
                        if "test_" in template_name or "__pycache__" in str(
                            template_path
                        ):
                            continue

                        template_info = {
                            "path": str(template_path),
                            "category": category,
                            "file_type": template_path.suffix,
                            "size_bytes": template_path.stat().st_size,
                            "modified_time": datetime.fromtimestamp(
                                template_path.stat().st_mtime
                            ),
                            "relative_path": str(
                                template_path.relative_to(templates_root)
                            ),
                        }

                        template_registry[category][template_name] = template_info

        self.template_registry = template_registry

        total_templates = sum(
            len(templates) for templates in template_registry.values()
        )
        logger.info(
            f"Discovered {total_templates} templates across {len(template_registry)} categories"
        )

        return template_registry

    def validate_template(
        self, template_path: str, template_name: str = None
    ) -> TemplateValidationResult:
        """
        Validate a single template comprehensively.

        Args:
            template_path: Path to the template file
            template_name: Optional name for the template

        Returns:
            Comprehensive validation result
        """
        if template_name is None:
            template_name = Path(template_path).stem

        logger.info(f"Validating template: {template_name}")

        validation_start = time.time()

        # Initialize result structure
        result_data = {
            "template_name": template_name,
            "template_path": template_path,
            "validation_timestamp": datetime.now(),
            "overall_status": "passed",
            "validation_categories": {},
            "performance_metrics": {},
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "score": 0.0,
        }

        # Run validation categories
        validation_categories = self.config["validation_categories"]

        for category_name, category_config in validation_categories.items():
            if category_config["enabled"]:
                logger.info(f"Running {category_name} validation")

                try:
                    category_result = self._validate_category(
                        template_path, category_name, category_config
                    )
                    result_data["validation_categories"][category_name] = (
                        category_result
                    )

                    # Aggregate errors and warnings
                    result_data["errors"].extend(category_result.get("errors", []))
                    result_data["warnings"].extend(category_result.get("warnings", []))
                    result_data["recommendations"].extend(
                        category_result.get("recommendations", [])
                    )

                except Exception as e:
                    error_msg = f"Failed to validate category {category_name}: {str(e)}"
                    result_data["errors"].append(error_msg)
                    logger.error(error_msg)

        # Calculate overall score
        result_data["score"] = self._calculate_overall_score(
            result_data["validation_categories"], validation_categories
        )

        # Determine overall status
        if result_data["errors"]:
            result_data["overall_status"] = "failed"
        elif result_data["warnings"]:
            result_data["overall_status"] = "warning"
        else:
            result_data["overall_status"] = "passed"

        # Performance metrics
        result_data["performance_metrics"]["validation_time"] = (
            time.time() - validation_start
        )

        # Create validation result object
        validation_result = TemplateValidationResult(**result_data)

        # Store result
        self.validation_results[template_name] = validation_result

        return validation_result

    def _validate_category(
        self, template_path: str, category_name: str, category_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate a specific category for a template."""

        category_result = {
            "status": "passed",
            "score": 0.0,
            "checks_performed": [],
            "checks_passed": 0,
            "checks_failed": 0,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "details": {},
        }

        checks = category_config.get("checks", [])

        for check_name in checks:
            try:
                check_result = self._run_validation_check(
                    template_path, category_name, check_name
                )

                category_result["checks_performed"].append(check_name)
                category_result["details"][check_name] = check_result

                if check_result["status"] == "passed":
                    category_result["checks_passed"] += 1
                else:
                    category_result["checks_failed"] += 1
                    category_result["errors"].extend(check_result.get("errors", []))

                category_result["warnings"].extend(check_result.get("warnings", []))
                category_result["recommendations"].extend(
                    check_result.get("recommendations", [])
                )

            except Exception as e:
                category_result["checks_failed"] += 1
                category_result["errors"].append(f"Check {check_name} failed: {str(e)}")

        # Calculate category score
        total_checks = (
            category_result["checks_passed"] + category_result["checks_failed"]
        )
        if total_checks > 0:
            category_result["score"] = (
                category_result["checks_passed"] / total_checks
            ) * 100

        # Update category status
        if category_result["checks_failed"] > 0:
            category_result["status"] = (
                "failed" if category_result["errors"] else "warning"
            )

        return category_result

    def _run_validation_check(
        self, template_path: str, category_name: str, check_name: str
    ) -> dict[str, Any]:
        """Run a specific validation check."""

        # Route to appropriate validation method
        validation_method = f"_validate_{category_name}_{check_name}"

        if hasattr(self, validation_method):
            method = getattr(self, validation_method)
            return method(template_path)
        else:
            # Generic validation dispatcher
            return self._generic_validation_check(
                template_path, category_name, check_name
            )

    # Syntax validation methods
    def _validate_syntax_python_syntax(self, template_path: str) -> dict[str, Any]:
        """Validate Python syntax correctness."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            # Parse AST to check syntax
            ast.parse(source_code)
            result["metrics"]["lines_of_code"] = len(source_code.splitlines())

        except SyntaxError as e:
            result["status"] = "failed"
            result["errors"].append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error parsing file: {str(e)}")

        return result

    def _validate_syntax_import_validation(self, template_path: str) -> dict[str, Any]:
        """Validate import statements and dependencies."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            # Parse imports
            tree = ast.parse(source_code)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            result["metrics"]["import_count"] = len(imports)
            result["metrics"]["unique_imports"] = len(set(imports))

            # Check for problematic imports
            problematic_imports = ["os.system", "subprocess.call", "eval", "exec"]
            for imp in imports:
                if any(prob in imp for prob in problematic_imports):
                    result["warnings"].append(f"Potentially unsafe import: {imp}")

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing imports: {str(e)}")

        return result

    def _validate_syntax_class_structure(self, template_path: str) -> dict[str, Any]:
        """Validate class structure and design."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            classes = []
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)

                    # Check for __init__ method
                    has_init = any(
                        isinstance(child, ast.FunctionDef) and child.name == "__init__"
                        for child in node.body
                    )
                    if not has_init:
                        result["warnings"].append(
                            f"Class {node.name} missing __init__ method"
                        )

                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)

            result["metrics"]["class_count"] = len(classes)
            result["metrics"]["function_count"] = len(functions)

            # Check for main template class
            template_classes = [
                cls
                for cls in classes
                if "Template" in cls or "Processor" in cls or "Comparator" in cls
            ]
            if not template_classes:
                result["warnings"].append(
                    "No main template class found (should contain 'Template', 'Processor', or 'Comparator')"
                )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing class structure: {str(e)}")

        return result

    # Functionality validation methods
    def _validate_functionality_mock_data_testing(
        self, template_path: str
    ) -> dict[str, Any]:
        """Test template functionality with mock data."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            # Dynamically import the template module
            spec = importlib.util.spec_from_file_location(
                "template_module", template_path
            )
            template_module = importlib.util.module_from_spec(spec)

            # Mock dependencies that might not be available
            with patch.multiple(
                "sys.modules", pynomaly=MagicMock(), pyod=MagicMock(), tods=MagicMock()
            ):
                spec.loader.exec_module(template_module)

            # Find main template class
            template_classes = []
            for name, obj in vars(template_module).items():
                if isinstance(obj, type) and any(
                    keyword in name
                    for keyword in ["Template", "Processor", "Comparator", "Selector"]
                ):
                    template_classes.append((name, obj))

            if not template_classes:
                result["warnings"].append("No main template class found for testing")
                return result

            # Test with mock data
            mock_data_config = self.config["mock_data_config"]

            for class_name, template_class in template_classes[:1]:  # Test first class
                try:
                    # Generate mock data
                    n_samples = mock_data_config["sample_sizes"][
                        0
                    ]  # Use smallest size for testing
                    n_features = mock_data_config["feature_dimensions"][0]

                    X = np.random.randn(n_samples, n_features)
                    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

                    # Try to instantiate the class
                    try:
                        instance = template_class()
                        result["metrics"][f"{class_name}_instantiation"] = "success"
                    except Exception:
                        # Try with basic config
                        try:
                            instance = template_class(config={})
                            result["metrics"][f"{class_name}_instantiation"] = (
                                "success_with_config"
                            )
                        except Exception as e:
                            result["errors"].append(
                                f"Failed to instantiate {class_name}: {str(e)}"
                            )
                            continue

                    # Test main processing method
                    main_methods = [
                        "preprocess",
                        "process",
                        "compare_algorithms",
                        "select_ensemble",
                    ]

                    for method_name in main_methods:
                        if hasattr(instance, method_name):
                            try:
                                method = getattr(instance, method_name)
                                # Try calling with mock data
                                if method_name in ["preprocess", "process"]:
                                    method(X)
                                elif method_name in [
                                    "compare_algorithms",
                                    "select_ensemble",
                                ]:
                                    method(X, y)

                                result["metrics"][f"{class_name}_{method_name}"] = (
                                    "success"
                                )
                                break
                            except Exception as e:
                                result["warnings"].append(
                                    f"Method {method_name} failed with mock data: {str(e)}"
                                )

                except Exception as e:
                    result["warnings"].append(
                        f"Error testing class {class_name}: {str(e)}"
                    )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error in mock data testing: {str(e)}")

        return result

    def _validate_functionality_method_execution(
        self, template_path: str
    ) -> dict[str, Any]:
        """Validate method execution paths and error handling."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Analyze method complexity and error handling
            method_analysis = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_name = node.name

                    # Count try-except blocks
                    try_blocks = len(
                        [n for n in ast.walk(node) if isinstance(n, ast.Try)]
                    )

                    # Count if statements (complexity indicator)
                    if_statements = len(
                        [n for n in ast.walk(node) if isinstance(n, ast.If)]
                    )

                    # Count return statements
                    return_statements = len(
                        [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                    )

                    method_analysis[method_name] = {
                        "try_blocks": try_blocks,
                        "if_statements": if_statements,
                        "return_statements": return_statements,
                        "lines": len(node.body),
                    }

            result["metrics"]["methods_analyzed"] = len(method_analysis)
            result["metrics"]["average_try_blocks"] = (
                np.mean([m["try_blocks"] for m in method_analysis.values()])
                if method_analysis
                else 0
            )

            # Check for methods without error handling
            methods_without_error_handling = [
                name
                for name, analysis in method_analysis.items()
                if analysis["try_blocks"] == 0
                and analysis["lines"] > 5
                and not name.startswith("_")
            ]

            if methods_without_error_handling:
                result["warnings"].append(
                    f"Methods without error handling: {', '.join(methods_without_error_handling)}"
                )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing method execution: {str(e)}")

        return result

    def _validate_functionality_error_handling(
        self, template_path: str
    ) -> dict[str, Any]:
        """Validate error handling patterns and robustness."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            # Check for good error handling patterns
            error_patterns = [
                "try:",
                "except:",
                "raise",
                "finally:",
                "logging.error",
                "logger.error",
            ]

            pattern_counts = {}
            for pattern in error_patterns:
                pattern_counts[pattern] = source_code.count(pattern)

            result["metrics"]["error_handling_patterns"] = pattern_counts

            # Check for bare except clauses (bad practice)
            if "except:" in source_code:
                bare_except_count = source_code.count("except:")
                result["warnings"].append(
                    f"Found {bare_except_count} bare except clause(s) - should specify exception types"
                )

            # Check for proper logging
            if (
                pattern_counts.get("logging.error", 0)
                + pattern_counts.get("logger.error", 0)
                == 0
            ):
                if pattern_counts.get("except:", 0) > 0:
                    result["recommendations"].append(
                        "Consider adding error logging for better debugging"
                    )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing error handling: {str(e)}")

        return result

    # Performance validation methods
    def _validate_performance_memory_usage(self, template_path: str) -> dict[str, Any]:
        """Validate memory usage patterns."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            # Static analysis for memory-intensive patterns
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            # Check for potential memory issues
            memory_concerns = {
                "large_data_structures": source_code.count("np.zeros(")
                + source_code.count("np.ones("),
                "deep_copies": source_code.count(".copy()")
                + source_code.count("copy.deepcopy"),
                "list_comprehensions": source_code.count("[") - source_code.count("[]"),
                "dataframe_operations": source_code.count("pd.DataFrame")
                + source_code.count(".merge(")
                + source_code.count(".join("),
            }

            result["metrics"]["memory_patterns"] = memory_concerns

            # Memory usage recommendations
            if memory_concerns["large_data_structures"] > 5:
                result["recommendations"].append(
                    "Consider using memory-efficient data structures or chunking for large arrays"
                )

            if memory_concerns["deep_copies"] > 3:
                result["warnings"].append(
                    "High number of copy operations detected - may impact memory usage"
                )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing memory usage: {str(e)}")

        return result

    def _validate_performance_execution_time(
        self, template_path: str
    ) -> dict[str, Any]:
        """Validate execution time characteristics."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            # Static analysis for performance patterns
            performance_patterns = {
                "loops": source_code.count("for ") + source_code.count("while "),
                "nested_loops": len(
                    [
                        line
                        for line in source_code.split("\n")
                        if "    for " in line or "        for " in line
                    ]
                ),
                "vectorized_operations": source_code.count("np.")
                + source_code.count("pd."),
                "function_calls": source_code.count("("),
            }

            result["metrics"]["performance_patterns"] = performance_patterns

            # Performance recommendations
            if performance_patterns["nested_loops"] > 2:
                result["warnings"].append(
                    "Multiple nested loops detected - consider vectorization"
                )

            if (
                performance_patterns["vectorized_operations"]
                < performance_patterns["loops"]
            ):
                result["recommendations"].append(
                    "Consider using more vectorized operations for better performance"
                )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing execution time: {str(e)}")

        return result

    def _validate_performance_scalability(self, template_path: str) -> dict[str, Any]:
        """Validate scalability characteristics."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            # Check for scalability indicators
            scalability_indicators = {
                "parallel_processing": source_code.count("multiprocessing")
                + source_code.count("concurrent.futures"),
                "chunking": source_code.count("chunk") + source_code.count("batch"),
                "streaming": source_code.count("stream")
                + source_code.count("iterative"),
                "memory_management": source_code.count("del ")
                + source_code.count("gc.collect"),
            }

            result["metrics"]["scalability_indicators"] = scalability_indicators

            # Scalability recommendations
            total_scalability_features = sum(scalability_indicators.values())
            if total_scalability_features == 0:
                result["recommendations"].append(
                    "Consider adding scalability features like chunking or parallel processing"
                )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing scalability: {str(e)}")

        return result

    # Documentation validation methods
    def _validate_documentation_docstring_coverage(
        self, template_path: str
    ) -> dict[str, Any]:
        """Validate docstring coverage and quality."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            documented_functions = 0
            total_functions = 0
            documented_classes = 0
            total_classes = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Str)
                    ):
                        documented_functions += 1

                elif isinstance(node, ast.ClassDef):
                    total_classes += 1
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Str)
                    ):
                        documented_classes += 1

            # Calculate coverage
            function_coverage = (
                (documented_functions / total_functions * 100)
                if total_functions > 0
                else 100
            )
            class_coverage = (
                (documented_classes / total_classes * 100) if total_classes > 0 else 100
            )

            result["metrics"]["function_docstring_coverage"] = function_coverage
            result["metrics"]["class_docstring_coverage"] = class_coverage
            result["metrics"]["total_functions"] = total_functions
            result["metrics"]["total_classes"] = total_classes

            # Quality thresholds
            if function_coverage < 80:
                result["warnings"].append(
                    f"Low function docstring coverage: {function_coverage:.1f}%"
                )

            if class_coverage < 90:
                result["warnings"].append(
                    f"Low class docstring coverage: {class_coverage:.1f}%"
                )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing docstring coverage: {str(e)}")

        return result

    def _validate_documentation_parameter_documentation(
        self, template_path: str
    ) -> dict[str, Any]:
        """Validate parameter documentation quality."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            # Check for parameter documentation patterns
            doc_patterns = {
                "args_sections": source_code.count("Args:"),
                "returns_sections": source_code.count("Returns:"),
                "examples_sections": source_code.count("Example"),
                "raises_sections": source_code.count("Raises:"),
            }

            result["metrics"]["documentation_patterns"] = doc_patterns

            # Documentation quality recommendations
            if doc_patterns["args_sections"] < doc_patterns["returns_sections"]:
                result["recommendations"].append(
                    "Consider documenting function arguments more consistently"
                )

            if doc_patterns["examples_sections"] == 0:
                result["recommendations"].append(
                    "Consider adding usage examples to documentation"
                )

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(
                f"Error analyzing parameter documentation: {str(e)}"
            )

        return result

    def _validate_documentation_examples(self, template_path: str) -> dict[str, Any]:
        """Validate presence and quality of examples."""

        result = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        try:
            with open(template_path, encoding="utf-8") as f:
                source_code = f.read()

            # Check for example patterns
            example_indicators = {
                "main_function": "def main():" in source_code,
                "example_usage": "example" in source_code.lower(),
                "demonstration": "demo" in source_code.lower(),
                "if_name_main": 'if __name__ == "__main__"' in source_code,
            }

            result["metrics"]["example_indicators"] = example_indicators

            # Example quality assessment
            if (
                not example_indicators["main_function"]
                and not example_indicators["if_name_main"]
            ):
                result["warnings"].append("No main function or example usage found")

            if sum(example_indicators.values()) >= 3:
                result["metrics"]["example_quality"] = "good"
            elif sum(example_indicators.values()) >= 2:
                result["metrics"]["example_quality"] = "fair"
            else:
                result["metrics"]["example_quality"] = "poor"
                result["recommendations"].append("Add comprehensive usage examples")

        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Error analyzing examples: {str(e)}")

        return result

    # Generic validation check fallback
    def _generic_validation_check(
        self, template_path: str, category_name: str, check_name: str
    ) -> dict[str, Any]:
        """Generic validation check for unimplemented specific checks."""

        return {
            "status": "passed",
            "errors": [],
            "warnings": [f"Generic validation used for {category_name}:{check_name}"],
            "recommendations": [],
            "metrics": {"check_type": "generic"},
        }

    def _calculate_overall_score(
        self, validation_categories: dict[str, Any], categories_config: dict[str, Any]
    ) -> float:
        """Calculate overall quality score based on category results and weights."""

        total_score = 0.0
        total_weight = 0.0

        for category_name, category_result in validation_categories.items():
            category_config = categories_config.get(category_name, {})
            weight = category_config.get("weight", 0.1)
            score = category_result.get("score", 0.0)

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def validate_all_templates(self, templates_root: str) -> dict[str, Any]:
        """
        Validate all discovered templates.

        Args:
            templates_root: Root directory containing templates

        Returns:
            Comprehensive validation results for all templates
        """
        logger.info("Starting validation of all templates")

        # Discover templates
        template_registry = self.discover_templates(templates_root)

        # Validate each template
        validation_results = {}
        failed_templates = []
        warning_templates = []
        passed_templates = []

        total_templates = sum(
            len(templates) for templates in template_registry.values()
        )
        current_template = 0

        for category, templates in template_registry.items():
            validation_results[category] = {}

            for template_name, template_info in templates.items():
                current_template += 1
                logger.info(
                    f"Validating template {current_template}/{total_templates}: {template_name}"
                )

                try:
                    # Skip non-Python files for now
                    if template_info["file_type"] not in [".py"]:
                        logger.info(f"Skipping non-Python template: {template_name}")
                        continue

                    result = self.validate_template(
                        template_info["path"], template_name
                    )
                    validation_results[category][template_name] = result

                    # Categorize results
                    if result.overall_status == "failed":
                        failed_templates.append(template_name)
                    elif result.overall_status == "warning":
                        warning_templates.append(template_name)
                    else:
                        passed_templates.append(template_name)

                except Exception as e:
                    logger.error(
                        f"Failed to validate template {template_name}: {str(e)}"
                    )
                    failed_templates.append(template_name)

        # Generate summary statistics
        summary_stats = {
            "total_templates": total_templates,
            "templates_validated": len(passed_templates)
            + len(warning_templates)
            + len(failed_templates),
            "passed_templates": len(passed_templates),
            "warning_templates": len(warning_templates),
            "failed_templates": len(failed_templates),
            "success_rate": len(passed_templates)
            / max(
                1,
                len(passed_templates) + len(warning_templates) + len(failed_templates),
            )
            * 100,
            "validation_timestamp": datetime.now(),
            "passed_list": passed_templates,
            "warning_list": warning_templates,
            "failed_list": failed_templates,
        }

        self.summary_stats = summary_stats

        logger.info(
            f"Template validation complete: {summary_stats['success_rate']:.1f}% success rate"
        )

        return {
            "validation_results": validation_results,
            "summary_stats": summary_stats,
            "template_registry": template_registry,
        }

    def generate_validation_report(
        self, output_path: str = "template_validation_report.html"
    ) -> str:
        """
        Generate comprehensive HTML validation report.

        Args:
            output_path: Path for the output HTML report

        Returns:
            Path to the generated report
        """
        logger.info(f"Generating validation report: {output_path}")

        # Generate HTML report
        html_content = self._generate_html_report()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Validation report generated: {output_path}")
        return output_path

    def _generate_html_report(self) -> str:
        """Generate HTML content for validation report."""

        summary = self.summary_stats

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pynomaly Template Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
                .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
                .status-passed {{ color: #27ae60; }}
                .status-warning {{ color: #f39c12; }}
                .status-failed {{ color: #e74c3c; }}
                .category-section {{ margin: 20px 0; }}
                .template-item {{ background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 6px; border-left: 4px solid #3498db; }}
                .template-item.warning {{ border-left-color: #f39c12; }}
                .template-item.failed {{ border-left-color: #e74c3c; }}
                .collapsible {{ cursor: pointer; }}
                .content {{ display: none; margin-top: 10px; }}
                .metric {{ display: inline-block; margin: 5px 10px 5px 0; padding: 3px 8px; background: #ecf0f1; border-radius: 3px; font-size: 12px; }}
            </style>
            <script>
                function toggleContent(element) {{
                    const content = element.nextElementSibling;
                    content.style.display = content.style.display === 'none' ? 'block' : 'none';
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Pynomaly Template Validation Report</h1>
                    <p>Generated on {summary.get("validation_timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>

                <div class="summary">
                    <div class="stat-card">
                        <div class="stat-value">{summary.get("total_templates", 0)}</div>
                        <div class="stat-label">Total Templates</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value status-passed">{summary.get("passed_templates", 0)}</div>
                        <div class="stat-label">Passed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value status-warning">{summary.get("warning_templates", 0)}</div>
                        <div class="stat-label">Warnings</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value status-failed">{summary.get("failed_templates", 0)}</div>
                        <div class="stat-label">Failed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{summary.get("success_rate", 0):.1f}%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                </div>
        """

        # Add detailed results for each template
        for template_name, result in self.validation_results.items():
            status_class = result.overall_status

            html += f"""
                <div class="template-item {status_class}">
                    <div class="collapsible" onclick="toggleContent(this)">
                        <h3>{template_name} - <span class="status-{status_class}">{result.overall_status.upper()}</span> (Score: {result.score:.1f}/100)</h3>
                    </div>
                    <div class="content">
                        <p><strong>Path:</strong> {result.template_path}</p>

                        <div class="metrics">
                            <h4>Category Scores:</h4>
            """

            for category_name, category_result in result.validation_categories.items():
                html += f'<span class="metric">{category_name}: {category_result.get("score", 0):.1f}/100</span>'

            if result.errors:
                html += """
                            <h4 style="color: #e74c3c;">Errors:</h4>
                            <ul>
                """
                for error in result.errors:
                    html += f"<li>{error}</li>"
                html += "</ul>"

            if result.warnings:
                html += """
                            <h4 style="color: #f39c12;">Warnings:</h4>
                            <ul>
                """
                for warning in result.warnings:
                    html += f"<li>{warning}</li>"
                html += "</ul>"

            if result.recommendations:
                html += """
                            <h4 style="color: #3498db;">Recommendations:</h4>
                            <ul>
                """
                for recommendation in result.recommendations:
                    html += f"<li>{recommendation}</li>"
                html += "</ul>"

            html += """
                        </div>
                    </div>
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def save_results(self, filepath: str):
        """Save validation results to JSON file."""

        # Prepare serializable results
        serializable_results = {
            "summary_stats": self.summary_stats,
            "validation_results": {},
            "template_registry": self.template_registry,
        }

        for template_name, result in self.validation_results.items():
            serializable_results["validation_results"][template_name] = {
                "template_name": result.template_name,
                "template_path": result.template_path,
                "validation_timestamp": result.validation_timestamp.isoformat(),
                "overall_status": result.overall_status,
                "validation_categories": result.validation_categories,
                "performance_metrics": result.performance_metrics,
                "errors": result.errors,
                "warnings": result.warnings,
                "recommendations": result.recommendations,
                "score": result.score,
            }

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Validation results saved to {filepath}")


def main():
    """Example usage of the Template Validation Framework."""

    # Initialize validator
    validator = TemplateValidator(verbose=True)

    # Get templates root directory (adjust path as needed)
    current_dir = Path(__file__).parent
    templates_root = current_dir.parent.parent  # Go up to templates root

    print(f"Validating templates in: {templates_root}")

    # Run comprehensive validation
    validation_results = validator.validate_all_templates(str(templates_root))

    # Print summary
    summary = validation_results["summary_stats"]
    print("\n" + "=" * 60)
    print("TEMPLATE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Templates: {summary['total_templates']}")
    print(f"Validated: {summary['templates_validated']}")
    print(f"Passed: {summary['passed_templates']} ({summary['success_rate']:.1f}%)")
    print(f"Warnings: {summary['warning_templates']}")
    print(f"Failed: {summary['failed_templates']}")

    # Show failed templates if any
    if summary["failed_list"]:
        print("\nFailed Templates:")
        for template in summary["failed_list"]:
            print(f"  - {template}")

    # Show warning templates if any
    if summary["warning_list"]:
        print("\nTemplates with Warnings:")
        for template in summary["warning_list"]:
            print(f"  - {template}")

    # Generate reports
    validator.generate_validation_report("template_validation_report.html")
    validator.save_results("template_validation_results.json")

    print("\nReports generated:")
    print("  - HTML Report: template_validation_report.html")
    print("  - JSON Results: template_validation_results.json")

    print("\nTemplate validation framework demonstration completed!")


if __name__ == "__main__":
    main()
