#!/usr/bin/env python3
"""
Detailed Test Mapping Analysis
Creates a detailed mapping between source modules and their corresponding test files
"""

from pathlib import Path


def analyze_specific_modules():
    """Analyze specific modules and their test coverage"""

    base_path = Path("/mnt/c/Users/andre/Pynomaly")
    src_path = base_path / "src" / "monorepo"
    test_path = base_path / "tests"

    # Key source modules to analyze
    key_modules = [
        "application/dto",
        "application/services",
        "application/use_cases",
        "domain/entities",
        "domain/services",
        "domain/value_objects",
        "infrastructure/adapters",
        "infrastructure/repositories",
        "infrastructure/persistence",
        "presentation/api",
        "presentation/cli",
    ]

    module_analysis = {}

    for module_path in key_modules:
        full_module_path = src_path / module_path
        if full_module_path.exists():
            # Get all Python files in this module
            module_files = []
            for file in full_module_path.glob("*.py"):
                if file.name != "__init__.py":
                    module_files.append(file.name)

            # Look for corresponding test files
            test_coverage = {}
            for file in module_files:
                base_name = file.replace(".py", "")
                corresponding_tests = []

                # Search for test files that might test this module
                for test_file in test_path.rglob("*.py"):
                    if test_file.name.startswith("test_"):
                        # Check if this test file tests our module
                        if (
                            base_name in test_file.name
                            or module_path.replace("/", "_") in test_file.name
                            or any(
                                part in test_file.name
                                for part in module_path.split("/")
                            )
                        ):
                            corresponding_tests.append(
                                str(test_file.relative_to(test_path))
                            )

                test_coverage[file] = corresponding_tests

            module_analysis[module_path] = {
                "files": module_files,
                "test_coverage": test_coverage,
                "total_files": len(module_files),
                "tested_files": len([f for f, tests in test_coverage.items() if tests]),
            }

    return module_analysis


def print_detailed_analysis():
    """Print detailed analysis"""

    analysis = analyze_specific_modules()

    print("# Detailed Test Coverage Mapping")
    print("=" * 50)

    for module_path, data in analysis.items():
        coverage_pct = (
            (data["tested_files"] / data["total_files"] * 100)
            if data["total_files"] > 0
            else 0
        )

        print(f"\n## {module_path}")
        print(f"Total files: {data['total_files']}")
        print(f"Tested files: {data['tested_files']}")
        print(f"Coverage: {coverage_pct:.1f}%")
        print()

        # Show files with their test coverage
        for file, tests in data["test_coverage"].items():
            if tests:
                print(f"✅ {file}")
                for test in tests[:3]:  # Show first 3 tests
                    print(f"   → {test}")
                if len(tests) > 3:
                    print(f"   → ... and {len(tests) - 3} more tests")
            else:
                print(f"❌ {file} (NO TESTS)")

        print()


def identify_missing_tests():
    """Identify specific files that need tests"""

    analysis = analyze_specific_modules()

    print("\n# Files Missing Tests")
    print("=" * 30)

    for module_path, data in analysis.items():
        missing_tests = [
            file for file, tests in data["test_coverage"].items() if not tests
        ]

        if missing_tests:
            print(f"\n## {module_path}")
            for file in missing_tests:
                print(f"- {file}")


def main():
    print_detailed_analysis()
    identify_missing_tests()


if __name__ == "__main__":
    main()
