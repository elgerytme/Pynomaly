#!/usr/bin/env python3
"""
Advanced Mutation Testing Framework for Pynomaly
Implements comprehensive mutation testing to validate test quality.
"""

import argparse
import ast
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MutationResult:
    """Result of applying a single mutation."""

    mutation_id: str
    operator: str
    location: str
    original_code: str
    mutated_code: str
    test_passed: bool
    execution_time: float
    error: Optional[str] = None


@dataclass
class MutationTestSummary:
    """Summary of mutation testing results."""

    total_mutations: int
    killed_mutations: int
    survived_mutations: int
    failed_mutations: int
    mutation_score: float
    execution_time: float
    results: list[MutationResult]
    coverage_report: dict


class MutationOperator:
    """Base class for mutation operators."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, node: ast.AST) -> list[ast.AST]:
        """Apply mutation to AST node. Returns list of mutated nodes."""
        raise NotImplementedError

    def can_mutate(self, node: ast.AST) -> bool:
        """Check if this operator can mutate the given node."""
        raise NotImplementedError


class ArithmeticOperatorMutation(MutationOperator):
    """Mutates arithmetic operators (+, -, *, /, //, %, **)."""

    def __init__(self):
        super().__init__("ArithmeticOperator")
        self.mutations = {
            ast.Add: [ast.Sub, ast.Mult, ast.Div],
            ast.Sub: [ast.Add, ast.Mult, ast.Div],
            ast.Mult: [ast.Add, ast.Sub, ast.Div],
            ast.Div: [ast.Add, ast.Sub, ast.Mult],
            ast.FloorDiv: [ast.Div, ast.Mod],
            ast.Mod: [ast.Div, ast.FloorDiv],
            ast.Pow: [ast.Mult, ast.Div],
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BinOp) and type(node.op) in self.mutations

    def apply(self, node: ast.BinOp) -> list[ast.BinOp]:
        """Apply arithmetic operator mutations."""
        mutants = []
        original_op = type(node.op)

        for new_op_class in self.mutations.get(original_op, []):
            mutant = ast.copy_location(
                ast.BinOp(left=node.left, op=new_op_class(), right=node.right), node
            )
            mutants.append(mutant)

        return mutants


class ComparisonOperatorMutation(MutationOperator):
    """Mutates comparison operators (<, <=, >, >=, ==, !=, is, is not, in, not in)."""

    def __init__(self):
        super().__init__("ComparisonOperator")
        self.mutations = {
            ast.Lt: [ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
            ast.LtE: [ast.Lt, ast.Gt, ast.GtE, ast.Eq, ast.NotEq],
            ast.Gt: [ast.Lt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq],
            ast.GtE: [ast.Lt, ast.LtE, ast.Gt, ast.Eq, ast.NotEq],
            ast.Eq: [ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
            ast.NotEq: [ast.Eq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
            ast.Is: [ast.IsNot, ast.Eq],
            ast.IsNot: [ast.Is, ast.NotEq],
            ast.In: [ast.NotIn],
            ast.NotIn: [ast.In],
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in self.mutations
        )

    def apply(self, node: ast.Compare) -> list[ast.Compare]:
        """Apply comparison operator mutations."""
        mutants = []
        original_op = type(node.ops[0])

        for new_op_class in self.mutations.get(original_op, []):
            mutant = ast.copy_location(
                ast.Compare(
                    left=node.left, ops=[new_op_class()], comparators=node.comparators
                ),
                node,
            )
            mutants.append(mutant)

        return mutants


class LogicalOperatorMutation(MutationOperator):
    """Mutates logical operators (and, or)."""

    def __init__(self):
        super().__init__("LogicalOperator")
        self.mutations = {
            ast.And: [ast.Or],
            ast.Or: [ast.And],
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BoolOp) and type(node.op) in self.mutations

    def apply(self, node: ast.BoolOp) -> list[ast.BoolOp]:
        """Apply logical operator mutations."""
        mutants = []
        original_op = type(node.op)

        for new_op_class in self.mutations.get(original_op, []):
            mutant = ast.copy_location(
                ast.BoolOp(op=new_op_class(), values=node.values), node
            )
            mutants.append(mutant)

        return mutants


class UnaryOperatorMutation(MutationOperator):
    """Mutates unary operators (+, -, not)."""

    def __init__(self):
        super().__init__("UnaryOperator")
        self.mutations = {
            ast.UAdd: [ast.USub],
            ast.USub: [ast.UAdd],
            ast.Not: [],  # Remove 'not' operator
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.UnaryOp) and type(node.op) in self.mutations

    def apply(self, node: ast.UnaryOp) -> list[ast.AST]:
        """Apply unary operator mutations."""
        mutants = []
        original_op = type(node.op)

        if original_op == ast.Not:
            # Remove 'not' operator, return the operand directly
            mutants.append(node.operand)
        else:
            for new_op_class in self.mutations.get(original_op, []):
                mutant = ast.copy_location(
                    ast.UnaryOp(op=new_op_class(), operand=node.operand), node
                )
                mutants.append(mutant)

        return mutants


class ConstantMutation(MutationOperator):
    """Mutates constants (numbers, strings, booleans)."""

    def __init__(self):
        super().__init__("Constant")

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.Constant, ast.Num, ast.Str, ast.NameConstant))

    def apply(self, node: ast.AST) -> list[ast.AST]:
        """Apply constant mutations."""
        mutants = []

        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Num):
            value = node.n
        elif isinstance(node, ast.Str):
            value = node.s
        elif isinstance(node, ast.NameConstant):
            value = node.value
        else:
            return mutants

        # Generate mutations based on value type
        if isinstance(value, bool):
            new_value = not value
            mutants.append(ast.copy_location(ast.Constant(value=new_value), node))
        elif isinstance(value, (int, float)):
            if value == 0:
                new_values = [1, -1]
            elif value > 0:
                new_values = [0, -value, value + 1, value - 1]
            else:
                new_values = [0, -value, value + 1, value - 1]

            for new_value in new_values:
                mutants.append(ast.copy_location(ast.Constant(value=new_value), node))
        elif isinstance(value, str):
            if value == "":
                new_values = ["XXX", "mutant"]
            else:
                new_values = ["", "XXX", value + "X"]

            for new_value in new_values:
                mutants.append(ast.copy_location(ast.Constant(value=new_value), node))

        return mutants


class ConditionalBoundaryMutation(MutationOperator):
    """Mutates conditional boundaries (< to <=, etc.)."""

    def __init__(self):
        super().__init__("ConditionalBoundary")
        self.mutations = {
            ast.Lt: [ast.LtE],
            ast.LtE: [ast.Lt],
            ast.Gt: [ast.GtE],
            ast.GtE: [ast.Gt],
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in self.mutations
        )

    def apply(self, node: ast.Compare) -> list[ast.Compare]:
        """Apply conditional boundary mutations."""
        mutants = []
        original_op = type(node.ops[0])

        for new_op_class in self.mutations.get(original_op, []):
            mutant = ast.copy_location(
                ast.Compare(
                    left=node.left, ops=[new_op_class()], comparators=node.comparators
                ),
                node,
            )
            mutants.append(mutant)

        return mutants


class MutationGenerator:
    """Generates mutations for Python source code."""

    def __init__(self):
        self.operators = [
            ArithmeticOperatorMutation(),
            ComparisonOperatorMutation(),
            LogicalOperatorMutation(),
            UnaryOperatorMutation(),
            ConstantMutation(),
            ConditionalBoundaryMutation(),
        ]

    def generate_mutations(
        self, source_code: str, target_file: str
    ) -> list[tuple[str, str, str]]:
        """Generate all possible mutations for the given source code."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in {target_file}: {e}")
            return []

        mutations = []
        mutation_id = 0

        for node in ast.walk(tree):
            for operator in self.operators:
                if operator.can_mutate(node):
                    mutants = operator.apply(node)

                    for mutant in mutants:
                        # Create a copy of the tree with the mutation applied
                        mutated_tree = self._apply_mutation(tree, node, mutant)

                        if mutated_tree:
                            try:
                                mutated_code = ast.unparse(mutated_tree)
                                location = (
                                    f"{target_file}:{node.lineno}:{node.col_offset}"
                                )
                                original_snippet = (
                                    ast.unparse(node)
                                    if hasattr(ast, "unparse")
                                    else str(node)
                                )
                                mutated_snippet = (
                                    ast.unparse(mutant)
                                    if hasattr(ast, "unparse")
                                    else str(mutant)
                                )

                                mutations.append(
                                    (
                                        f"{operator.name}_{mutation_id}",
                                        mutated_code,
                                        f"{operator.name}: {location} | {original_snippet} → {mutated_snippet}",
                                    )
                                )
                                mutation_id += 1

                            except Exception as e:
                                logger.debug(
                                    f"Failed to generate mutant {mutation_id}: {e}"
                                )

        return mutations

    def _apply_mutation(
        self, tree: ast.AST, original_node: ast.AST, mutant_node: ast.AST
    ) -> Optional[ast.AST]:
        """Apply a mutation to the AST tree."""

        class MutationApplier(ast.NodeTransformer):
            def __init__(self, target_node, replacement_node):
                self.target_node = target_node
                self.replacement_node = replacement_node
                self.applied = False

            def visit(self, node):
                if not self.applied and node is self.target_node:
                    self.applied = True
                    return ast.copy_location(self.replacement_node, node)
                return self.generic_visit(node)

        try:
            applier = MutationApplier(original_node, mutant_node)
            mutated_tree = applier.visit(ast.deepcopy(tree))

            if applier.applied:
                # Fix missing locations
                ast.fix_missing_locations(mutated_tree)
                return mutated_tree
        except Exception as e:
            logger.debug(f"Failed to apply mutation: {e}")

        return None


class MutationTester:
    """Executes mutation tests and analyzes results."""

    def __init__(self, source_dir: Path, test_dir: Path, test_command: str = "pytest"):
        self.source_dir = source_dir
        self.test_dir = test_dir
        self.test_command = test_command
        self.generator = MutationGenerator()
        self.temp_dir = None

    def run_mutation_testing(
        self, target_files: list[str] = None, max_mutations: int = None
    ) -> MutationTestSummary:
        """Run comprehensive mutation testing."""
        logger.info("Starting mutation testing...")
        start_time = time.time()

        # Prepare target files
        if target_files is None:
            target_files = self._discover_target_files()

        # Generate mutations
        all_mutations = []
        for file_path in target_files:
            logger.info(f"Generating mutations for {file_path}")

            with open(file_path, encoding="utf-8") as f:
                source_code = f.read()

            mutations = self.generator.generate_mutations(source_code, str(file_path))

            for mutation_id, mutated_code, description in mutations:
                all_mutations.append(
                    {
                        "id": mutation_id,
                        "file": file_path,
                        "code": mutated_code,
                        "description": description,
                        "original_code": source_code,
                    }
                )

        # Limit mutations if specified
        if max_mutations and len(all_mutations) > max_mutations:
            logger.info(
                f"Limiting to {max_mutations} mutations (from {len(all_mutations)})"
            )
            all_mutations = all_mutations[:max_mutations]

        logger.info(f"Testing {len(all_mutations)} mutations...")

        # Run baseline tests to ensure they pass
        if not self._run_baseline_tests():
            raise RuntimeError(
                "Baseline tests failed - cannot proceed with mutation testing"
            )

        # Execute mutations
        results = []
        for i, mutation in enumerate(all_mutations, 1):
            logger.info(f"Testing mutation {i}/{len(all_mutations)}: {mutation['id']}")

            result = self._test_mutation(mutation)
            results.append(result)

            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(all_mutations)} mutations tested")

        # Calculate summary
        total_mutations = len(results)
        killed_mutations = sum(
            1 for r in results if not r.test_passed and r.error is None
        )
        survived_mutations = sum(
            1 for r in results if r.test_passed and r.error is None
        )
        failed_mutations = sum(1 for r in results if r.error is not None)

        mutation_score = (
            (killed_mutations / total_mutations) * 100 if total_mutations > 0 else 0
        )
        execution_time = time.time() - start_time

        # Generate coverage report
        coverage_report = self._generate_coverage_report()

        summary = MutationTestSummary(
            total_mutations=total_mutations,
            killed_mutations=killed_mutations,
            survived_mutations=survived_mutations,
            failed_mutations=failed_mutations,
            mutation_score=mutation_score,
            execution_time=execution_time,
            results=results,
            coverage_report=coverage_report,
        )

        logger.info(f"Mutation testing completed in {execution_time:.2f}s")
        logger.info(f"Mutation score: {mutation_score:.1f}%")

        return summary

    def _discover_target_files(self) -> list[str]:
        """Discover Python files to mutate."""
        target_files = []

        for root, dirs, files in os.walk(self.source_dir):
            # Skip test directories and cache directories
            dirs[:] = [
                d for d in dirs if not d.startswith((".", "__pycache__", "test"))
            ]

            for file in files:
                if file.endswith(".py") and not file.startswith("test_"):
                    file_path = os.path.join(root, file)
                    target_files.append(file_path)

        return target_files

    def _run_baseline_tests(self) -> bool:
        """Run baseline tests to ensure they pass."""
        logger.info("Running baseline tests...")

        try:
            # Split test command properly
            cmd_parts = self.test_command.split()
            result = subprocess.run(
                cmd_parts + [str(self.test_dir), "-x"],  # -x stops on first failure
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.source_dir.parent,
            )

            if result.returncode == 0:
                logger.info("✓ Baseline tests passed")
                return True
            else:
                logger.error(f"✗ Baseline tests failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("✗ Baseline tests timed out")
            return False
        except Exception as e:
            logger.error(f"✗ Failed to run baseline tests: {e}")
            return False

    def _test_mutation(self, mutation: dict) -> MutationResult:
        """Test a single mutation."""
        start_time = time.time()

        # Create temporary directory for this mutation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_source_dir = Path(temp_dir) / "src"
            shutil.copytree(self.source_dir, temp_source_dir)

            # Apply mutation
            target_file = temp_source_dir / Path(mutation["file"]).relative_to(
                self.source_dir
            )

            try:
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(mutation["code"])

                # Run tests
                cmd_parts = self.test_command.split()
                result = subprocess.run(
                    cmd_parts + [str(self.test_dir), "-x", "--tb=no"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=temp_source_dir.parent,
                )

                test_passed = result.returncode == 0
                error = None

                if result.returncode != 0 and "FAILED" not in result.stdout:
                    # This might be a compilation error or other issue
                    error = result.stderr[:500] if result.stderr else "Unknown error"

            except subprocess.TimeoutExpired:
                test_passed = False
                error = "Test execution timed out"
            except Exception as e:
                test_passed = False
                error = str(e)

        execution_time = time.time() - start_time

        return MutationResult(
            mutation_id=mutation["id"],
            operator=mutation["description"].split(":")[0],
            location=mutation["description"].split("|")[0].split(":")[1].strip(),
            original_code=(
                mutation["original_code"][:200] + "..."
                if len(mutation["original_code"]) > 200
                else mutation["original_code"]
            ),
            mutated_code=(
                mutation["code"][:200] + "..."
                if len(mutation["code"]) > 200
                else mutation["code"]
            ),
            test_passed=test_passed,
            execution_time=execution_time,
            error=error,
        )

    def _generate_coverage_report(self) -> dict:
        """Generate code coverage report."""
        try:
            # Run tests with coverage
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "pytest",
                    str(self.test_dir),
                    "--cov=" + str(self.source_dir),
                    "--cov-report=json",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.source_dir.parent,
            )

            if result.returncode == 0:
                # Try to read coverage report
                coverage_file = self.source_dir.parent / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    return coverage_data.get("totals", {})

        except Exception as e:
            logger.debug(f"Failed to generate coverage report: {e}")

        return {"percent_covered": 0, "num_statements": 0, "missing_lines": 0}

    def save_results(self, summary: MutationTestSummary, output_file: Path):
        """Save mutation testing results to JSON file."""
        with open(output_file, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)

        logger.info(f"Mutation test results saved to {output_file}")

    def print_summary(self, summary: MutationTestSummary):
        """Print human-readable mutation testing summary."""
        print("\n=== Mutation Testing Summary ===")
        print(f"Total mutations: {summary.total_mutations}")
        print(f"Killed mutations: {summary.killed_mutations}")
        print(f"Survived mutations: {summary.survived_mutations}")
        print(f"Failed mutations: {summary.failed_mutations}")
        print(f"Mutation score: {summary.mutation_score:.1f}%")
        print(f"Execution time: {summary.execution_time:.2f}s")

        if summary.coverage_report:
            coverage = summary.coverage_report.get("percent_covered", 0)
            print(f"Line coverage: {coverage:.1f}%")

        # Show survived mutations (potential test gaps)
        survived = [r for r in summary.results if r.test_passed and r.error is None]
        if survived:
            print("\n=== Survived Mutations (Test Gaps) ===")
            for result in survived[:10]:  # Show first 10
                print(f"  {result.mutation_id}: {result.location}")
                print(f"    Operator: {result.operator}")

            if len(survived) > 10:
                print(f"  ... and {len(survived) - 10} more")

        # Show failed mutations (potential issues)
        failed = [r for r in summary.results if r.error is not None]
        if failed:
            print("\n=== Failed Mutations ===")
            for result in failed[:5]:  # Show first 5
                print(f"  {result.mutation_id}: {result.error}")


def main():
    """Main entry point for mutation testing."""
    parser = argparse.ArgumentParser(description="Mutation Testing Framework")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default="src/pynomaly",
        help="Source directory to mutate",
    )
    parser.add_argument("--test-dir", type=Path, default="tests", help="Test directory")
    parser.add_argument(
        "--test-command", default="python -m pytest", help="Test command to run"
    )
    parser.add_argument(
        "--max-mutations", type=int, help="Maximum number of mutations to test"
    )
    parser.add_argument("--target-files", nargs="+", help="Specific files to mutate")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize mutation tester
    tester = MutationTester(
        source_dir=args.source_dir,
        test_dir=args.test_dir,
        test_command=args.test_command,
    )

    try:
        # Run mutation testing
        summary = tester.run_mutation_testing(
            target_files=args.target_files, max_mutations=args.max_mutations
        )

        # Print summary
        tester.print_summary(summary)

        # Save results
        if args.output:
            tester.save_results(summary, args.output)

        # Exit with appropriate code based on mutation score
        if summary.mutation_score < 80:
            logger.warning("Mutation score below 80% - consider improving tests")
            sys.exit(1)
        else:
            logger.info("Good mutation score - tests are effective")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Mutation testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
