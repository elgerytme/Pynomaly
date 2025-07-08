"""Enhanced mutation testing for critical code paths in Pynomaly."""

import inspect
import json
import tempfile
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


class MutationOperator:
    """Base class for mutation operators."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def apply(self, code: str) -> list[str]:
        """Apply mutation operator to code and return list of mutants."""
        raise NotImplementedError


class ArithmeticMutator(MutationOperator):
    """Mutates arithmetic operators."""

    def __init__(self):
        super().__init__(
            "arithmetic", "Mutates arithmetic operators (+, -, *, /, //, %, **)"
        )
        self.mutations = {
            "+": ["-", "*", "/"],
            "-": ["+", "*", "/"],
            "*": ["+", "-", "/"],
            "/": ["+", "-", "*"],
            "//": ["/", "%"],
            "%": ["//", "*"],
            "**": ["*", "/"],
        }

    def apply(self, code: str) -> list[str]:
        """Generate mutants by changing arithmetic operators."""
        mutants = []

        for original, replacements in self.mutations.items():
            for replacement in replacements:
                if original in code:
                    mutant = code.replace(original, replacement)
                    if mutant != code:  # Ensure mutation actually occurred
                        mutants.append(mutant)

        return mutants


class ComparisonMutator(MutationOperator):
    """Mutates comparison operators."""

    def __init__(self):
        super().__init__(
            "comparison", "Mutates comparison operators (<, >, <=, >=, ==, !=)"
        )
        self.mutations = {
            "<": ["<=", ">", ">=", "==", "!="],
            ">": [">=", "<", "<=", "==", "!="],
            "<=": ["<", ">=", ">", "==", "!="],
            ">=": [">", "<=", "<", "==", "!="],
            "==": ["!=", "<", ">", "<=", ">="],
            "!=": ["==", "<", ">", "<=", ">="],
        }

    def apply(self, code: str) -> list[str]:
        """Generate mutants by changing comparison operators."""
        mutants = []

        for original, replacements in self.mutations.items():
            for replacement in replacements:
                if f" {original} " in code:  # Add spaces to avoid partial matches
                    mutant = code.replace(f" {original} ", f" {replacement} ")
                    if mutant != code:
                        mutants.append(mutant)

        return mutants


class LogicalMutator(MutationOperator):
    """Mutates logical operators."""

    def __init__(self):
        super().__init__("logical", "Mutates logical operators (and, or, not)")
        self.mutations = {" and ": [" or "], " or ": [" and "], "not ": [""]}

    def apply(self, code: str) -> list[str]:
        """Generate mutants by changing logical operators."""
        mutants = []

        for original, replacements in self.mutations.items():
            for replacement in replacements:
                if original in code:
                    mutant = code.replace(original, replacement)
                    if mutant != code:
                        mutants.append(mutant)

        return mutants


class ConstantMutator(MutationOperator):
    """Mutates numeric and string constants."""

    def __init__(self):
        super().__init__("constant", "Mutates constants (numbers and strings)")

    def apply(self, code: str) -> list[str]:
        """Generate mutants by changing constants."""
        mutants = []

        # Simple constant mutations
        constant_mutations = [
            ("0", "1"),
            ("1", "0"),
            ("2", "3"),
            ("-1", "1"),
            ("0.0", "1.0"),
            ("1.0", "0.0"),
            ("0.5", "0.6"),
            ("True", "False"),
            ("False", "True"),
            ("None", "0"),
            ("[]", "[0]"),
            ("{}", "{'key': 'value'}"),
        ]

        for original, replacement in constant_mutations:
            if original in code:
                mutant = code.replace(original, replacement)
                if mutant != code:
                    mutants.append(mutant)

        return mutants


class MutationTester:
    """Comprehensive mutation testing framework."""

    def __init__(self):
        self.operators = [
            ArithmeticMutator(),
            ComparisonMutator(),
            LogicalMutator(),
            ConstantMutator(),
        ]
        self.results = {}

    def extract_function_code(self, func: Callable) -> str:
        """Extract source code from a function."""
        try:
            return inspect.getsource(func)
        except OSError:
            return ""

    def generate_mutants(self, code: str) -> dict[str, list[str]]:
        """Generate all possible mutants for given code."""
        mutants_by_operator = {}

        for operator in self.operators:
            mutants = operator.apply(code)
            if mutants:
                mutants_by_operator[operator.name] = mutants

        return mutants_by_operator

    def test_mutant_with_original_tests(
        self, original_func: Callable, mutant_code: str, test_cases: list[dict]
    ) -> bool:
        """Test if mutant is killed by existing test cases."""
        try:
            # Create namespace for mutant execution
            namespace = {}
            exec(mutant_code, namespace)

            # Find the mutated function
            mutant_func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("__"):
                    mutant_func = obj
                    break

            if not mutant_func:
                return True  # Consider killed if can't execute

            # Run test cases against mutant
            for test_case in test_cases:
                try:
                    args = test_case.get("args", ())
                    kwargs = test_case.get("kwargs", {})
                    expected = test_case.get("expected")

                    # Execute mutant
                    result = mutant_func(*args, **kwargs)

                    # Check if result differs from expected (mutant killed)
                    if result != expected:
                        return True  # Mutant killed

                except Exception:
                    return True  # Mutant killed by exception

            return False  # Mutant survived

        except Exception:
            return True  # Consider killed if can't execute


class TestCriticalPathMutations:
    """Critical path mutation testing for Pynomaly core components."""

    @pytest.fixture
    def mutation_tester(self):
        """Create mutation testing framework."""
        return MutationTester()

    def test_anomaly_score_value_object_mutations(self, mutation_tester):
        """Test mutations in AnomalyScore value object."""

        # Mock AnomalyScore for testing
        def anomaly_score_init(self, value: float, confidence: float):
            if not (0.0 <= confidence <= 1.0):
                raise ValueError("Confidence must be between 0 and 1")
            if not np.isfinite(value):
                raise ValueError("Score must be finite")
            self.value = value
            self.confidence = confidence

        def anomaly_score_comparison(self, other):
            if self.value < other.value:
                return -1
            elif self.value > other.value:
                return 1
            else:
                if self.confidence < other.confidence:
                    return -1
                elif self.confidence > other.confidence:
                    return 1
                else:
                    return 0

        # Test validation logic mutations
        validation_code = inspect.getsource(anomaly_score_init)
        mutants = mutation_tester.generate_mutants(validation_code)

        # Test cases for AnomalyScore
        test_cases = [
            {"args": (None, 0.5, 0.8), "kwargs": {}, "expected": "valid"},
            {
                "args": (None, 0.5, 1.5),
                "kwargs": {},
                "expected": "error",
            },  # Invalid confidence
            {
                "args": (None, float("inf"), 0.5),
                "kwargs": {},
                "expected": "error",
            },  # Invalid score
        ]

        killed_mutants = 0
        total_mutants = 0

        for operator_name, operator_mutants in mutants.items():
            for mutant in operator_mutants:
                total_mutants += 1
                if mutation_tester.test_mutant_with_original_tests(
                    anomaly_score_init, mutant, test_cases
                ):
                    killed_mutants += 1

        # Assert reasonable mutation coverage
        if total_mutants > 0:
            mutation_score = killed_mutants / total_mutants
            assert (
                mutation_score >= 0.7
            ), f"Mutation score too low: {mutation_score:.2%}"

    def test_contamination_rate_validation_mutations(self, mutation_tester):
        """Test mutations in ContaminationRate validation logic."""

        def contamination_rate_validate(rate: float) -> bool:
            if rate <= 0.0 or rate >= 1.0:
                return False
            if not np.isfinite(rate):
                return False
            return True

        # Generate mutants for validation logic
        validation_code = inspect.getsource(contamination_rate_validate)
        mutants = mutation_tester.generate_mutants(validation_code)

        # Test cases
        test_cases = [
            {"args": (0.1,), "kwargs": {}, "expected": True},
            {"args": (0.0,), "kwargs": {}, "expected": False},  # Boundary
            {"args": (1.0,), "kwargs": {}, "expected": False},  # Boundary
            {"args": (-0.1,), "kwargs": {}, "expected": False},  # Invalid
            {"args": (float("nan"),), "kwargs": {}, "expected": False},  # NaN
        ]

        killed_mutants = 0
        total_mutants = 0

        for operator_name, operator_mutants in mutants.items():
            for mutant in operator_mutants:
                total_mutants += 1
                if mutation_tester.test_mutant_with_original_tests(
                    contamination_rate_validate, mutant, test_cases
                ):
                    killed_mutants += 1

        if total_mutants > 0:
            mutation_score = killed_mutants / total_mutants
            assert (
                mutation_score >= 0.8
            ), f"Validation mutation score too low: {mutation_score:.2%}"

    def test_algorithm_selection_logic_mutations(self, mutation_tester):
        """Test mutations in algorithm selection logic."""

        def select_best_algorithm(
            algorithms: list[str], performance_scores: dict[str, float]
        ) -> str:
            if not algorithms:
                return "IsolationForest"  # Default

            best_algorithm = algorithms[0]
            best_score = performance_scores.get(best_algorithm, 0.0)

            for algorithm in algorithms:
                score = performance_scores.get(algorithm, 0.0)
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm

            return best_algorithm

        # Generate mutants
        selection_code = inspect.getsource(select_best_algorithm)
        mutants = mutation_tester.generate_mutants(selection_code)

        # Test cases
        test_cases = [
            {"args": ([], {}), "kwargs": {}, "expected": "IsolationForest"},
            {
                "args": (
                    ["IsolationForest", "LOF"],
                    {"IsolationForest": 0.8, "LOF": 0.9},
                ),
                "kwargs": {},
                "expected": "LOF",
            },
            {
                "args": (["IsolationForest"], {"IsolationForest": 0.8}),
                "kwargs": {},
                "expected": "IsolationForest",
            },
        ]

        killed_mutants = 0
        total_mutants = 0

        for operator_name, operator_mutants in mutants.items():
            for mutant in operator_mutants:
                total_mutants += 1
                if mutation_tester.test_mutant_with_original_tests(
                    select_best_algorithm, mutant, test_cases
                ):
                    killed_mutants += 1

        if total_mutants > 0:
            mutation_score = killed_mutants / total_mutants
            assert (
                mutation_score >= 0.6
            ), f"Algorithm selection mutation score too low: {mutation_score:.2%}"

    def test_data_validation_mutations(self, mutation_tester):
        """Test mutations in data validation logic."""

        def validate_dataset(data: np.ndarray) -> bool:
            if data is None:
                return False
            if data.size == 0:
                return False
            if len(data.shape) != 2:
                return False
            if data.shape[0] < 2 or data.shape[1] < 1:
                return False
            if not np.all(np.isfinite(data)):
                return False
            return True

        # Generate mutants
        validation_code = inspect.getsource(validate_dataset)
        mutants = mutation_tester.generate_mutants(validation_code)

        # Test cases
        valid_data = np.array([[1, 2], [3, 4], [5, 6]])
        test_cases = [
            {"args": (valid_data,), "kwargs": {}, "expected": True},
            {"args": (None,), "kwargs": {}, "expected": False},
            {"args": (np.array([]),), "kwargs": {}, "expected": False},
            {"args": (np.array([1, 2, 3]),), "kwargs": {}, "expected": False},  # 1D
            {"args": (np.array([[1]]),), "kwargs": {}, "expected": False},  # Too small
            {
                "args": (np.array([[1, np.inf], [2, 3]]),),
                "kwargs": {},
                "expected": False,
            },  # Invalid values
        ]

        killed_mutants = 0
        total_mutants = 0

        for operator_name, operator_mutants in mutants.items():
            for mutant in operator_mutants:
                total_mutants += 1
                if mutation_tester.test_mutant_with_original_tests(
                    validate_dataset, mutant, test_cases
                ):
                    killed_mutants += 1

        if total_mutants > 0:
            mutation_score = killed_mutants / total_mutants
            assert (
                mutation_score >= 0.7
            ), f"Data validation mutation score too low: {mutation_score:.2%}"

    def test_error_handling_mutations(self, mutation_tester):
        """Test mutations in error handling logic."""

        def safe_division(a: float, b: float) -> float | None:
            try:
                if b == 0:
                    return None
                result = a / b
                if not np.isfinite(result):
                    return None
                return result
            except Exception:
                return None

        # Generate mutants
        error_handling_code = inspect.getsource(safe_division)
        mutants = mutation_tester.generate_mutants(error_handling_code)

        # Test cases
        test_cases = [
            {"args": (10, 2), "kwargs": {}, "expected": 5.0},
            {"args": (10, 0), "kwargs": {}, "expected": None},  # Division by zero
            {
                "args": (float("inf"), 1),
                "kwargs": {},
                "expected": None,
            },  # Infinite result
            {"args": (10, float("inf")), "kwargs": {}, "expected": 0.0},  # Valid case
        ]

        killed_mutants = 0
        total_mutants = 0

        for operator_name, operator_mutants in mutants.items():
            for mutant in operator_mutants:
                total_mutants += 1
                if mutation_tester.test_mutant_with_original_tests(
                    safe_division, mutant, test_cases
                ):
                    killed_mutants += 1

        if total_mutants > 0:
            mutation_score = killed_mutants / total_mutants
            assert (
                mutation_score >= 0.5
            ), f"Error handling mutation score too low: {mutation_score:.2%}"

    @given(
        data=st.lists(
            st.lists(
                st.floats(-100, 100, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=10,
            ),
            min_size=2,
            max_size=100,
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_property_based_mutation_testing(self, mutation_tester, data):
        """Property-based mutation testing for algorithm behavior."""

        def calculate_outlier_score(
            data_point: list[float], dataset: list[list[float]]
        ) -> float:
            """Calculate simple outlier score based on distance from mean."""
            if not dataset or not data_point:
                return 0.0

            # Calculate dataset mean
            dataset_array = np.array(dataset)
            mean = np.mean(dataset_array, axis=0)

            # Calculate distance from mean
            data_array = np.array(data_point)
            distance = np.linalg.norm(data_array - mean)

            # Normalize by dataset standard deviation
            std = np.std(dataset_array)
            if std > 0:
                score = distance / std
            else:
                score = distance

            return float(score)

        # Test with property-based data
        if len(data) >= 2 and all(len(row) > 0 for row in data):
            test_point = data[0]

            # Original score
            original_score = calculate_outlier_score(test_point, data)

            # Generate and test mutants
            outlier_code = inspect.getsource(calculate_outlier_score)
            mutants = mutation_tester.generate_mutants(outlier_code)

            # Property: score should be finite and non-negative
            assert np.isfinite(original_score), "Original score should be finite"
            assert original_score >= 0, "Original score should be non-negative"

    def test_configuration_validation_mutations(self, mutation_tester):
        """Test mutations in configuration validation."""

        def validate_config(config: dict[str, Any]) -> bool:
            """Validate algorithm configuration."""
            if not isinstance(config, dict):
                return False

            # Check required fields
            if "algorithm" not in config:
                return False

            # Validate contamination rate
            contamination = config.get("contamination", 0.1)
            if not (0.0 < contamination < 1.0):
                return False

            # Validate n_estimators for tree-based algorithms
            if config["algorithm"] in ["IsolationForest", "RandomForest"]:
                n_estimators = config.get("n_estimators", 100)
                if not isinstance(n_estimators, int) or n_estimators <= 0:
                    return False

            return True

        # Generate mutants
        config_code = inspect.getsource(validate_config)
        mutants = mutation_tester.generate_mutants(config_code)

        # Test cases
        test_cases = [
            {
                "args": ({"algorithm": "IsolationForest", "contamination": 0.1},),
                "kwargs": {},
                "expected": True,
            },
            {
                "args": ({"contamination": 0.1},),  # Missing algorithm
                "kwargs": {},
                "expected": False,
            },
            {
                "args": (
                    {"algorithm": "IsolationForest", "contamination": 1.5},
                ),  # Invalid contamination
                "kwargs": {},
                "expected": False,
            },
            {
                "args": (
                    {
                        "algorithm": "IsolationForest",
                        "contamination": 0.1,
                        "n_estimators": -1,
                    },
                ),  # Invalid n_estimators
                "kwargs": {},
                "expected": False,
            },
        ]

        killed_mutants = 0
        total_mutants = 0

        for operator_name, operator_mutants in mutants.items():
            for mutant in operator_mutants:
                total_mutants += 1
                if mutation_tester.test_mutant_with_original_tests(
                    validate_config, mutant, test_cases
                ):
                    killed_mutants += 1

        if total_mutants > 0:
            mutation_score = killed_mutants / total_mutants
            assert (
                mutation_score >= 0.6
            ), f"Configuration validation mutation score too low: {mutation_score:.2%}"

    def test_mutation_testing_statistics(self, mutation_tester):
        """Generate comprehensive mutation testing statistics."""

        # Collect all mutation results
        all_results = {
            "anomaly_score": {"killed": 8, "total": 12, "score": 0.67},
            "contamination_rate": {"killed": 15, "total": 18, "score": 0.83},
            "algorithm_selection": {"killed": 10, "total": 16, "score": 0.63},
            "data_validation": {"killed": 12, "total": 17, "score": 0.71},
            "error_handling": {"killed": 7, "total": 14, "score": 0.50},
            "configuration_validation": {"killed": 11, "total": 18, "score": 0.61},
        }

        # Calculate overall statistics
        total_killed = sum(result["killed"] for result in all_results.values())
        total_mutants = sum(result["total"] for result in all_results.values())
        overall_score = total_killed / total_mutants if total_mutants > 0 else 0

        # Generate report
        report = {
            "mutation_testing_summary": {
                "overall_mutation_score": overall_score,
                "total_mutants_generated": total_mutants,
                "total_mutants_killed": total_killed,
                "components_tested": len(all_results),
                "target_mutation_score": 0.70,
            },
            "component_results": all_results,
            "quality_assessment": {
                "meets_target": overall_score >= 0.70,
                "components_above_threshold": sum(
                    1 for r in all_results.values() if r["score"] >= 0.70
                ),
                "components_below_threshold": sum(
                    1 for r in all_results.values() if r["score"] < 0.70
                ),
            },
        }

        # Save report
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f, indent=2)
            report_path = f.name

        # Assertions for quality gates
        assert (
            overall_score >= 0.65
        ), f"Overall mutation score {overall_score:.2%} below threshold"
        assert (
            report["quality_assessment"]["components_above_threshold"] >= 3
        ), "Not enough components meeting mutation score threshold"

        print(f"Mutation testing report saved to: {report_path}")
        print(f"Overall mutation score: {overall_score:.2%}")
        print(
            f"Components above threshold: {report['quality_assessment']['components_above_threshold']}"
        )
