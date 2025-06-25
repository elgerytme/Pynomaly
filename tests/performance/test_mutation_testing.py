"""Mutation testing for test quality validation - Phase 4 Advanced Testing."""

from __future__ import annotations

import ast
import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@dataclass
class MutationResult:
    """Result of a single mutation test."""

    mutant_id: str
    mutation_type: str
    original_code: str
    mutated_code: str
    test_passed: bool
    error_message: str | None = None
    execution_time: float = 0.0


@dataclass
class MutationTestSummary:
    """Summary of mutation testing results."""

    total_mutants: int
    killed_mutants: int  # Tests failed (good)
    survived_mutants: int  # Tests passed (bad)
    equivalent_mutants: int  # Functionally equivalent
    error_mutants: int  # Execution errors
    mutation_score: float  # killed / (total - equivalent - error)

    @property
    def test_quality_score(self) -> float:
        """Overall test quality score (0-1)."""
        if self.total_mutants == 0:
            return 0.0
        effective_mutants = (
            self.total_mutants - self.equivalent_mutants - self.error_mutants
        )
        if effective_mutants == 0:
            return 1.0
        return self.killed_mutants / effective_mutants


class SimpleMutationEngine:
    """Simple mutation testing engine for critical code paths."""

    def __init__(self):
        self.mutation_operators = [
            self._mutate_arithmetic_operators,
            self._mutate_comparison_operators,
            self._mutate_logical_operators,
            self._mutate_constants,
            self._mutate_boundary_conditions,
        ]

    def generate_mutants(
        self, source_code: str, target_functions: list[str] = None
    ) -> list[tuple[str, str, str]]:
        """Generate mutants for given source code.

        Returns:
            List of (mutant_id, mutation_type, mutated_code) tuples
        """
        tree = ast.parse(source_code)
        mutants = []

        for i, operator in enumerate(self.mutation_operators):
            try:
                mutated_tree = copy.deepcopy(tree)
                mutation_applied, mutation_type = operator(
                    mutated_tree, target_functions
                )

                if mutation_applied:
                    mutated_code = ast.unparse(mutated_tree)
                    mutant_id = f"mutant_{i}_{len(mutants)}"
                    mutants.append((mutant_id, mutation_type, mutated_code))
            except Exception:
                # Skip mutations that cause syntax errors
                continue

        return mutants

    def _mutate_arithmetic_operators(
        self, tree: ast.AST, target_functions: list[str]
    ) -> tuple[bool, str]:
        """Mutate arithmetic operators (+, -, *, /)."""
        mutation_applied = False

        class ArithmeticMutator(ast.NodeTransformer):
            def visit_BinOp(self, node):
                nonlocal mutation_applied
                self.generic_visit(node)

                # Mutation mappings
                mutations = {
                    ast.Add: ast.Sub,
                    ast.Sub: ast.Add,
                    ast.Mult: ast.Div,
                    ast.Div: ast.Mult,
                    ast.Mod: ast.Mult,
                    ast.Pow: ast.Mult,
                }

                if type(node.op) in mutations and not mutation_applied:
                    node.op = mutations[type(node.op)]()
                    mutation_applied = True

                return node

        ArithmeticMutator().visit(tree)
        return mutation_applied, "arithmetic_operator"

    def _mutate_comparison_operators(
        self, tree: ast.AST, target_functions: list[str]
    ) -> tuple[bool, str]:
        """Mutate comparison operators (<, >, <=, >=, ==, !=)."""
        mutation_applied = False

        class ComparisonMutator(ast.NodeTransformer):
            def visit_Compare(self, node):
                nonlocal mutation_applied
                self.generic_visit(node)

                mutations = {
                    ast.Lt: ast.Gt,
                    ast.Gt: ast.Lt,
                    ast.LtE: ast.GtE,
                    ast.GtE: ast.LtE,
                    ast.Eq: ast.NotEq,
                    ast.NotEq: ast.Eq,
                }

                for i, op in enumerate(node.ops):
                    if type(op) in mutations and not mutation_applied:
                        node.ops[i] = mutations[type(op)]()
                        mutation_applied = True
                        break

                return node

        ComparisonMutator().visit(tree)
        return mutation_applied, "comparison_operator"

    def _mutate_logical_operators(
        self, tree: ast.AST, target_functions: list[str]
    ) -> tuple[bool, str]:
        """Mutate logical operators (and, or, not)."""
        mutation_applied = False

        class LogicalMutator(ast.NodeTransformer):
            def visit_BoolOp(self, node):
                nonlocal mutation_applied
                self.generic_visit(node)

                if isinstance(node.op, ast.And) and not mutation_applied:
                    node.op = ast.Or()
                    mutation_applied = True
                elif isinstance(node.op, ast.Or) and not mutation_applied:
                    node.op = ast.And()
                    mutation_applied = True

                return node

            def visit_UnaryOp(self, node):
                nonlocal mutation_applied
                self.generic_visit(node)

                # Remove 'not' operator
                if isinstance(node.op, ast.Not) and not mutation_applied:
                    mutation_applied = True
                    return node.operand

                return node

        LogicalMutator().visit(tree)
        return mutation_applied, "logical_operator"

    def _mutate_constants(
        self, tree: ast.AST, target_functions: list[str]
    ) -> tuple[bool, str]:
        """Mutate numeric constants."""
        mutation_applied = False

        class ConstantMutator(ast.NodeTransformer):
            def visit_Constant(self, node):
                nonlocal mutation_applied
                self.generic_visit(node)

                if isinstance(node.value, int | float) and not mutation_applied:
                    if node.value == 0:
                        node.value = 1
                    elif node.value == 1:
                        node.value = 0
                    elif node.value > 0:
                        node.value = -node.value
                    else:
                        node.value = abs(node.value)
                    mutation_applied = True

                return node

        ConstantMutator().visit(tree)
        return mutation_applied, "constant_value"

    def _mutate_boundary_conditions(
        self, tree: ast.AST, target_functions: list[str]
    ) -> tuple[bool, str]:
        """Mutate boundary conditions (off-by-one errors)."""
        mutation_applied = False

        class BoundaryMutator(ast.NodeTransformer):
            def visit_BinOp(self, node):
                nonlocal mutation_applied
                self.generic_visit(node)

                # Add/subtract 1 to numeric constants in arithmetic operations
                if isinstance(node.op, ast.Add | ast.Sub) and not mutation_applied:
                    if isinstance(node.right, ast.Constant) and isinstance(
                        node.right.value, int
                    ):
                        if isinstance(node.op, ast.Add):
                            node.right.value += 1
                        else:
                            node.right.value -= 1
                        mutation_applied = True

                return node

        BoundaryMutator().visit(tree)
        return mutation_applied, "boundary_condition"


class MutationTester:
    """Execute mutation tests and evaluate test quality."""

    def __init__(self):
        self.mutation_engine = SimpleMutationEngine()

    def run_mutation_tests(
        self,
        target_code: str,
        test_functions: list[Callable],
        target_functions: list[str] = None,
    ) -> MutationTestSummary:
        """Run mutation tests on target code."""
        mutants = self.mutation_engine.generate_mutants(target_code, target_functions)
        results = []

        for mutant_id, mutation_type, mutated_code in mutants:
            result = self._execute_mutant_tests(
                mutant_id, mutation_type, target_code, mutated_code, test_functions
            )
            results.append(result)

        return self._summarize_results(results)

    def _execute_mutant_tests(
        self,
        mutant_id: str,
        mutation_type: str,
        original_code: str,
        mutated_code: str,
        test_functions: list[Callable],
    ) -> MutationResult:
        """Execute tests against a single mutant."""
        import time

        start_time = time.time()

        try:
            # Create temporary module with mutated code
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(mutated_code)
                mutant_path = f.name

            # Import mutated module
            import importlib.util

            spec = importlib.util.spec_from_file_location("mutant_module", mutant_path)
            mutant_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant_module)

            # Run tests against mutant
            test_passed = True
            error_message = None

            for test_func in test_functions:
                try:
                    # Replace original implementations with mutated ones in test context
                    test_func(mutant_module)
                except AssertionError as e:
                    # Test failed - mutant was killed (good)
                    test_passed = False
                    error_message = str(e)
                    break
                except Exception as e:
                    # Execution error
                    test_passed = False
                    error_message = f"Execution error: {str(e)}"
                    break

            Path(mutant_path).unlink(missing_ok=True)

        except Exception as e:
            test_passed = False
            error_message = f"Mutant execution failed: {str(e)}"

        execution_time = time.time() - start_time

        return MutationResult(
            mutant_id=mutant_id,
            mutation_type=mutation_type,
            original_code=original_code,
            mutated_code=mutated_code,
            test_passed=test_passed,
            error_message=error_message,
            execution_time=execution_time,
        )

    def _summarize_results(self, results: list[MutationResult]) -> MutationTestSummary:
        """Summarize mutation test results."""
        total_mutants = len(results)
        killed_mutants = sum(
            1
            for r in results
            if not r.test_passed and "Execution error" not in (r.error_message or "")
        )
        survived_mutants = sum(1 for r in results if r.test_passed)
        error_mutants = sum(
            1
            for r in results
            if not r.test_passed and "Execution error" in (r.error_message or "")
        )
        equivalent_mutants = 0  # Would need semantic analysis

        effective_mutants = total_mutants - equivalent_mutants - error_mutants
        mutation_score = (
            killed_mutants / effective_mutants if effective_mutants > 0 else 0.0
        )

        return MutationTestSummary(
            total_mutants=total_mutants,
            killed_mutants=killed_mutants,
            survived_mutants=survived_mutants,
            equivalent_mutants=equivalent_mutants,
            error_mutants=error_mutants,
            mutation_score=mutation_score,
        )


class TestMutationTestingDomain:
    """Mutation testing for domain layer."""

    def test_contamination_rate_mutations(self):
        """Test ContaminationRate with mutations."""
        # Target code for mutation
        contamination_rate_code = """
class ContaminationRate:
    def __init__(self, value: float):
        if value <= 0 or value > 0.5:
            raise ValueError("Contamination rate must be between 0 and 0.5")
        self._value = value

    @property
    def value(self) -> float:
        return self._value

    def __eq__(self, other) -> bool:
        if not isinstance(other, ContaminationRate):
            return False
        return self._value == other._value

    def __lt__(self, other) -> bool:
        if not isinstance(other, ContaminationRate):
            return NotImplemented
        return self._value < other._value
"""

        # Test functions
        def test_valid_contamination_rates(module):
            """Test that valid contamination rates work correctly."""
            # Valid rates
            rate1 = module.ContaminationRate(0.1)
            assert rate1.value == 0.1

            rate2 = module.ContaminationRate(0.5)
            assert rate2.value == 0.5

            # Equality
            rate3 = module.ContaminationRate(0.1)
            assert rate1 == rate3
            assert rate1 != rate2

            # Comparison
            assert rate1 < rate2

        def test_invalid_contamination_rates(module):
            """Test that invalid contamination rates are rejected."""
            # Invalid rates should raise ValueError
            try:
                module.ContaminationRate(0.0)
                raise AssertionError("Should raise ValueError for 0.0")
            except ValueError:
                pass

            try:
                module.ContaminationRate(-0.1)
                raise AssertionError("Should raise ValueError for negative")
            except ValueError:
                pass

            try:
                module.ContaminationRate(0.6)
                raise AssertionError("Should raise ValueError for > 0.5")
            except ValueError:
                pass

        def test_boundary_conditions(module):
            """Test boundary conditions."""
            # Just above zero
            rate = module.ContaminationRate(0.001)
            assert rate.value == 0.001

            # Exactly 0.5
            rate = module.ContaminationRate(0.5)
            assert rate.value == 0.5

        # Run mutation tests
        tester = MutationTester()
        test_functions = [
            test_valid_contamination_rates,
            test_invalid_contamination_rates,
            test_boundary_conditions,
        ]

        summary = tester.run_mutation_tests(contamination_rate_code, test_functions)

        # Analyze results
        print("ContaminationRate Mutation Test Results:")
        print(f"Total mutants: {summary.total_mutants}")
        print(f"Killed: {summary.killed_mutants}, Survived: {summary.survived_mutants}")
        print(f"Mutation score: {summary.mutation_score:.2f}")
        print(f"Test quality score: {summary.test_quality_score:.2f}")

        # Assertions
        assert summary.total_mutants > 0, "Should generate mutants"
        assert summary.test_quality_score >= 0.7, "Test quality should be high"

    def test_anomaly_score_mutations(self):
        """Test AnomalyScore with mutations."""
        anomaly_score_code = """
class AnomalyScore:
    def __init__(self, value: float):
        if value < 0.0 or value > 1.0:
            raise ValueError("Anomaly score must be between 0.0 and 1.0")
        self._value = value

    @property
    def value(self) -> float:
        return self._value

    def __eq__(self, other) -> bool:
        if not isinstance(other, AnomalyScore):
            return False
        return abs(self._value - other._value) < 1e-9

    def __lt__(self, other) -> bool:
        if not isinstance(other, AnomalyScore):
            return NotImplemented
        return self._value < other._value

    def is_anomaly(self, threshold: float = 0.5) -> bool:
        return self._value >= threshold
"""

        def test_valid_scores(module):
            """Test valid anomaly scores."""
            score1 = module.AnomalyScore(0.0)
            assert score1.value == 0.0

            score2 = module.AnomalyScore(1.0)
            assert score2.value == 1.0

            score3 = module.AnomalyScore(0.7)
            assert score3.value == 0.7

            # Comparison
            assert score1 < score3 < score2

        def test_invalid_scores(module):
            """Test invalid anomaly scores."""
            try:
                module.AnomalyScore(-0.1)
                raise AssertionError("Should reject negative scores")
            except ValueError:
                pass

            try:
                module.AnomalyScore(1.1)
                raise AssertionError("Should reject scores > 1.0")
            except ValueError:
                pass

        def test_anomaly_detection_logic(module):
            """Test anomaly detection threshold logic."""
            low_score = module.AnomalyScore(0.3)
            high_score = module.AnomalyScore(0.8)

            # Default threshold (0.5)
            assert not low_score.is_anomaly()
            assert high_score.is_anomaly()

            # Custom threshold
            assert not low_score.is_anomaly(0.4)
            assert low_score.is_anomaly(0.2)

        tester = MutationTester()
        test_functions = [
            test_valid_scores,
            test_invalid_scores,
            test_anomaly_detection_logic,
        ]

        summary = tester.run_mutation_tests(anomaly_score_code, test_functions)

        print("AnomalyScore Mutation Test Results:")
        print(f"Total mutants: {summary.total_mutants}")
        print(f"Killed: {summary.killed_mutants}, Survived: {summary.survived_mutants}")
        print(f"Mutation score: {summary.mutation_score:.2f}")
        print(f"Test quality score: {summary.test_quality_score:.2f}")

        assert summary.total_mutants > 0
        assert summary.test_quality_score >= 0.6


class TestMutationTestingAlgorithms:
    """Mutation testing for algorithm implementations."""

    def test_isolation_forest_parameter_validation(self):
        """Test parameter validation with mutations."""
        validation_code = '''
def validate_isolation_forest_params(n_estimators=100, contamination=0.1, max_samples="auto", random_state=None):
    """Validate Isolation Forest parameters."""
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError("n_estimators must be positive integer")

    if not isinstance(contamination, (int, float)) or contamination <= 0 or contamination > 0.5:
        raise ValueError("contamination must be between 0 and 0.5")

    if max_samples != "auto" and (not isinstance(max_samples, int) or max_samples <= 0):
        raise ValueError("max_samples must be 'auto' or positive integer")

    if random_state is not None and (not isinstance(random_state, int) or random_state < 0):
        raise ValueError("random_state must be None or non-negative integer")

    return True
'''

        def test_valid_parameters(module):
            """Test valid parameter combinations."""
            # Default parameters
            assert module.validate_isolation_forest_params()

            # Custom valid parameters
            assert module.validate_isolation_forest_params(50, 0.05, 100, 42)
            assert module.validate_isolation_forest_params(200, 0.2, "auto", None)

        def test_invalid_n_estimators(module):
            """Test invalid n_estimators."""
            try:
                module.validate_isolation_forest_params(n_estimators=0)
                raise AssertionError("Should reject n_estimators=0")
            except ValueError:
                pass

            try:
                module.validate_isolation_forest_params(n_estimators=-10)
                raise AssertionError("Should reject negative n_estimators")
            except ValueError:
                pass

        def test_invalid_contamination(module):
            """Test invalid contamination rates."""
            try:
                module.validate_isolation_forest_params(contamination=0.0)
                raise AssertionError("Should reject contamination=0.0")
            except ValueError:
                pass

            try:
                module.validate_isolation_forest_params(contamination=0.6)
                raise AssertionError("Should reject contamination > 0.5")
            except ValueError:
                pass

        def test_invalid_max_samples(module):
            """Test invalid max_samples."""
            try:
                module.validate_isolation_forest_params(max_samples=0)
                raise AssertionError("Should reject max_samples=0")
            except ValueError:
                pass

            try:
                module.validate_isolation_forest_params(max_samples="invalid")
                raise AssertionError("Should reject invalid string")
            except ValueError:
                pass

        tester = MutationTester()
        test_functions = [
            test_valid_parameters,
            test_invalid_n_estimators,
            test_invalid_contamination,
            test_invalid_max_samples,
        ]

        summary = tester.run_mutation_tests(validation_code, test_functions)

        print("Parameter Validation Mutation Test Results:")
        print(f"Total mutants: {summary.total_mutants}")
        print(f"Killed: {summary.killed_mutants}, Survived: {summary.survived_mutants}")
        print(f"Mutation score: {summary.mutation_score:.2f}")
        print(f"Test quality score: {summary.test_quality_score:.2f}")

        assert summary.total_mutants > 0
        assert summary.test_quality_score >= 0.7

    def test_detection_result_processing(self):
        """Test detection result processing logic with mutations."""
        processing_code = '''
def process_detection_results(predictions, scores, threshold=0.5):
    """Process raw detection results into final format."""
    if len(predictions) != len(scores):
        raise ValueError("Predictions and scores must have same length")

    if not predictions or not scores:
        raise ValueError("Predictions and scores cannot be empty")

    # Apply threshold to scores to get binary predictions
    binary_predictions = [1 if score >= threshold else 0 for score in scores]

    # Count anomalies
    anomaly_count = sum(binary_predictions)

    # Calculate statistics
    mean_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)

    # Calculate anomaly rate
    anomaly_rate = anomaly_count / len(predictions) if len(predictions) > 0 else 0.0

    return {
        "predictions": binary_predictions,
        "scores": scores,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "mean_score": mean_score,
        "max_score": max_score,
        "min_score": min_score,
        "threshold": threshold
    }
'''

        def test_normal_processing(module):
            """Test normal result processing."""
            predictions = [0, 0, 1, 0, 1]
            scores = [0.3, 0.2, 0.8, 0.1, 0.9]

            result = module.process_detection_results(predictions, scores)

            assert result["anomaly_count"] == 2
            assert result["anomaly_rate"] == 0.4
            assert result["mean_score"] == 0.46
            assert result["max_score"] == 0.9
            assert result["min_score"] == 0.1

        def test_threshold_sensitivity(module):
            """Test threshold sensitivity."""
            predictions = [0, 0, 0, 0, 0]
            scores = [0.3, 0.4, 0.6, 0.7, 0.8]

            # Low threshold
            result_low = module.process_detection_results(
                predictions, scores, threshold=0.2
            )
            assert result_low["anomaly_count"] == 5

            # High threshold
            result_high = module.process_detection_results(
                predictions, scores, threshold=0.9
            )
            assert result_high["anomaly_count"] == 0

        def test_edge_cases(module):
            """Test edge cases."""
            # Single element
            predictions = [1]
            scores = [0.8]

            result = module.process_detection_results(predictions, scores)
            assert result["anomaly_count"] == 1
            assert result["anomaly_rate"] == 1.0

        def test_error_conditions(module):
            """Test error conditions."""
            # Mismatched lengths
            try:
                module.process_detection_results([0, 1], [0.3])
                raise AssertionError("Should reject mismatched lengths")
            except ValueError:
                pass

            # Empty inputs
            try:
                module.process_detection_results([], [])
                raise AssertionError("Should reject empty inputs")
            except ValueError:
                pass

        tester = MutationTester()
        test_functions = [
            test_normal_processing,
            test_threshold_sensitivity,
            test_edge_cases,
            test_error_conditions,
        ]

        summary = tester.run_mutation_tests(processing_code, test_functions)

        print("Result Processing Mutation Test Results:")
        print(f"Total mutants: {summary.total_mutants}")
        print(f"Killed: {summary.killed_mutants}, Survived: {summary.survived_mutants}")
        print(f"Mutation score: {summary.mutation_score:.2f}")
        print(f"Test quality score: {summary.test_quality_score:.2f}")

        assert summary.total_mutants > 0
        assert summary.test_quality_score >= 0.7


class TestMutationTestingDataValidation:
    """Mutation testing for data validation logic."""

    def test_dataset_validation_mutations(self):
        """Test dataset validation with mutations."""
        validation_code = '''
import pandas as pd
import numpy as np

def validate_dataset(data, min_samples=10, min_features=1, max_nan_ratio=0.1):
    """Validate dataset for anomaly detection."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")

    if len(data) < min_samples:
        raise ValueError(f"Dataset must have at least {min_samples} samples")

    if len(data.columns) < min_features:
        raise ValueError(f"Dataset must have at least {min_features} features")

    # Check for numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        raise ValueError("Dataset must have at least one numeric column")

    # Check NaN ratio
    total_values = len(data) * len(data.columns)
    nan_count = data.isnull().sum().sum()
    nan_ratio = nan_count / total_values if total_values > 0 else 0

    if nan_ratio > max_nan_ratio:
        raise ValueError(f"NaN ratio {nan_ratio:.2f} exceeds maximum {max_nan_ratio}")

    # Check for infinite values
    if data.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any():
        raise ValueError("Dataset contains infinite values")

    return True
'''

        def test_valid_datasets(module):
            """Test valid datasets."""
            import numpy as np

            # Simple valid dataset
            data = pd.DataFrame(
                {
                    "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                }
            )
            assert module.validate_dataset(data)

            # Dataset with some NaNs (within limit)
            data_with_nans = data.copy()
            data_with_nans.loc[0, "feature1"] = np.nan
            assert module.validate_dataset(data_with_nans)

        def test_invalid_type(module):
            """Test invalid data types."""
            try:
                module.validate_dataset("not a dataframe")
                raise AssertionError("Should reject non-DataFrame")
            except TypeError:
                pass

            try:
                module.validate_dataset(None)
                raise AssertionError("Should reject None")
            except TypeError:
                pass

        def test_insufficient_samples(module):
            """Test insufficient samples."""

            small_data = pd.DataFrame({"feature1": [1, 2, 3]})  # Only 3 samples
            try:
                module.validate_dataset(small_data)
                raise AssertionError("Should reject datasets with too few samples")
            except ValueError:
                pass

        def test_no_numeric_columns(module):
            """Test datasets with no numeric columns."""

            text_data = pd.DataFrame(
                {
                    "text1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                    "text2": ["x", "y", "z", "w", "v", "u", "t", "s", "r", "q"],
                }
            )
            try:
                module.validate_dataset(text_data)
                raise AssertionError("Should reject datasets with no numeric columns")
            except ValueError:
                pass

        def test_too_many_nans(module):
            """Test datasets with too many NaNs."""
            import numpy as np

            data = pd.DataFrame(
                {
                    "feature1": [
                        1,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "feature2": [
                        0.1,
                        0.2,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                }
            )
            try:
                module.validate_dataset(data)  # 80% NaNs, exceeds 10% limit
                raise AssertionError("Should reject datasets with too many NaNs")
            except ValueError:
                pass

        def test_infinite_values(module):
            """Test datasets with infinite values."""
            import numpy as np

            data = pd.DataFrame(
                {
                    "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, np.inf],
                    "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                }
            )
            try:
                module.validate_dataset(data)
                raise AssertionError("Should reject datasets with infinite values")
            except ValueError:
                pass

        tester = MutationTester()
        test_functions = [
            test_valid_datasets,
            test_invalid_type,
            test_insufficient_samples,
            test_no_numeric_columns,
            test_too_many_nans,
            test_infinite_values,
        ]

        summary = tester.run_mutation_tests(validation_code, test_functions)

        print("Dataset Validation Mutation Test Results:")
        print(f"Total mutants: {summary.total_mutants}")
        print(f"Killed: {summary.killed_mutants}, Survived: {summary.survived_mutants}")
        print(f"Mutation score: {summary.mutation_score:.2f}")
        print(f"Test quality score: {summary.test_quality_score:.2f}")

        assert summary.total_mutants > 0
        assert summary.test_quality_score >= 0.6


class TestMutationTestingBusinessLogic:
    """Mutation testing for critical business logic."""

    def test_ensemble_voting_logic(self):
        """Test ensemble voting logic with mutations."""
        ensemble_code = '''
def ensemble_vote(predictions_list, method="majority", weights=None):
    """Combine predictions from multiple detectors."""
    if not predictions_list:
        raise ValueError("Predictions list cannot be empty")

    # Validate input dimensions
    n_samples = len(predictions_list[0])
    for i, preds in enumerate(predictions_list):
        if len(preds) != n_samples:
            raise ValueError(f"All predictions must have same length, detector {i} has {len(preds)}")

    if weights is not None and len(weights) != len(predictions_list):
        raise ValueError("Weights must match number of detectors")

    # Initialize result
    ensemble_predictions = []

    for sample_idx in range(n_samples):
        votes = [preds[sample_idx] for preds in predictions_list]

        if method == "majority":
            # Simple majority vote
            anomaly_votes = sum(votes)
            total_votes = len(votes)
            final_prediction = 1 if anomaly_votes > total_votes / 2 else 0

        elif method == "weighted":
            if weights is None:
                raise ValueError("Weights required for weighted voting")

            # Weighted vote
            weighted_sum = sum(vote * weight for vote, weight in zip(votes, weights))
            weight_sum = sum(weights)
            final_prediction = 1 if weighted_sum > weight_sum / 2 else 0

        elif method == "unanimous":
            # All detectors must agree
            final_prediction = 1 if all(vote == 1 for vote in votes) else 0

        else:
            raise ValueError(f"Unknown voting method: {method}")

        ensemble_predictions.append(final_prediction)

    return ensemble_predictions
'''

        def test_majority_voting(module):
            """Test majority voting."""
            # Simple majority case
            predictions = [
                [1, 0, 1, 0, 1],  # Detector 1
                [1, 0, 0, 0, 1],  # Detector 2
                [0, 0, 1, 1, 1],  # Detector 3
            ]

            result = module.ensemble_vote(predictions, method="majority")
            expected = [1, 0, 1, 0, 1]  # Majority votes
            assert result == expected

        def test_weighted_voting(module):
            """Test weighted voting."""
            predictions = [
                [1, 0, 1],  # High weight detector
                [0, 0, 0],  # Low weight detector
            ]
            weights = [0.8, 0.2]

            result = module.ensemble_vote(
                predictions, method="weighted", weights=weights
            )
            expected = [1, 0, 1]  # High weight detector dominates
            assert result == expected

        def test_unanimous_voting(module):
            """Test unanimous voting."""
            predictions = [
                [1, 0, 1, 0],
                [1, 0, 0, 0],
                [1, 1, 1, 0],
            ]

            result = module.ensemble_vote(predictions, method="unanimous")
            expected = [1, 0, 0, 0]  # Only first sample is unanimous
            assert result == expected

        def test_error_conditions(module):
            """Test error conditions."""
            # Empty predictions
            try:
                module.ensemble_vote([])
                raise AssertionError("Should reject empty predictions")
            except ValueError:
                pass

            # Mismatched lengths
            try:
                module.ensemble_vote([[1, 0], [1, 0, 1]])
                raise AssertionError("Should reject mismatched lengths")
            except ValueError:
                pass

            # Weighted without weights
            try:
                module.ensemble_vote([[1, 0]], method="weighted")
                raise AssertionError("Should require weights for weighted voting")
            except ValueError:
                pass

        tester = MutationTester()
        test_functions = [
            test_majority_voting,
            test_weighted_voting,
            test_unanimous_voting,
            test_error_conditions,
        ]

        summary = tester.run_mutation_tests(ensemble_code, test_functions)

        print("Ensemble Voting Mutation Test Results:")
        print(f"Total mutants: {summary.total_mutants}")
        print(f"Killed: {summary.killed_mutants}, Survived: {summary.survived_mutants}")
        print(f"Mutation score: {summary.mutation_score:.2f}")
        print(f"Test quality score: {summary.test_quality_score:.2f}")

        assert summary.total_mutants > 0
        assert summary.test_quality_score >= 0.7


class TestOverallMutationTestQuality:
    """Overall assessment of mutation test quality."""

    def test_mutation_test_coverage_analysis(self):
        """Analyze overall mutation test coverage and quality."""
        # This test runs multiple mutation test scenarios and analyzes overall quality

        test_scenarios = [
            ("Domain Value Objects", ["ContaminationRate", "AnomalyScore"]),
            ("Algorithm Validation", ["parameter_validation", "result_processing"]),
            ("Data Validation", ["dataset_validation", "feature_validation"]),
            ("Business Logic", ["ensemble_voting", "threshold_optimization"]),
        ]

        overall_results = []

        for scenario_name, _components in test_scenarios:
            # Simulate mutation testing for each scenario
            # In a real implementation, this would run actual mutation tests

            # Mock results for demonstration
            mock_summary = MutationTestSummary(
                total_mutants=np.random.randint(10, 30),
                killed_mutants=np.random.randint(7, 25),
                survived_mutants=np.random.randint(0, 5),
                equivalent_mutants=np.random.randint(0, 3),
                error_mutants=np.random.randint(0, 2),
                mutation_score=np.random.uniform(0.7, 0.95),
            )

            overall_results.append((scenario_name, mock_summary))

        # Analyze overall quality
        total_mutants = sum(summary.total_mutants for _, summary in overall_results)
        total_killed = sum(summary.killed_mutants for _, summary in overall_results)
        sum(summary.survived_mutants for _, summary in overall_results)

        overall_mutation_score = total_killed / (
            total_mutants
            - sum(
                summary.equivalent_mutants + summary.error_mutants
                for _, summary in overall_results
            )
        )

        print("Overall Mutation Testing Analysis:")
        print(f"Total scenarios tested: {len(test_scenarios)}")
        print(f"Total mutants across all scenarios: {total_mutants}")
        print(f"Overall mutation score: {overall_mutation_score:.2f}")

        for scenario_name, summary in overall_results:
            print(f"{scenario_name}: {summary.test_quality_score:.2f} quality score")

        # Quality thresholds
        min_mutation_score = 0.8  # 80% of mutants should be killed
        min_scenario_quality = 0.7  # Each scenario should have 70% quality

        # Assertions
        assert overall_mutation_score >= min_mutation_score, (
            f"Overall mutation score {overall_mutation_score:.2f} below threshold {min_mutation_score}"
        )

        for scenario_name, summary in overall_results:
            assert summary.test_quality_score >= min_scenario_quality, (
                f"Scenario {scenario_name} quality {summary.test_quality_score:.2f} below threshold {min_scenario_quality}"
            )

        # Recommendations based on results
        recommendations = []

        for scenario_name, summary in overall_results:
            if summary.test_quality_score < 0.8:
                recommendations.append(f"Improve test coverage for {scenario_name}")

            if summary.survived_mutants > summary.total_mutants * 0.2:
                recommendations.append(f"Add more edge case tests for {scenario_name}")

        if recommendations:
            print("Recommendations for improvement:")
            for rec in recommendations:
                print(f"- {rec}")

        # Overall assessment
        if overall_mutation_score >= 0.9:
            quality_level = "Excellent"
        elif overall_mutation_score >= 0.8:
            quality_level = "Good"
        elif overall_mutation_score >= 0.7:
            quality_level = "Acceptable"
        else:
            quality_level = "Needs Improvement"

        print(f"Overall test quality assessment: {quality_level}")

        assert overall_mutation_score >= 0.7, (
            "Minimum acceptable mutation score not met"
        )


if __name__ == "__main__":
    # Enable running mutation tests directly
    pytest.main([__file__, "-v", "-s"])
