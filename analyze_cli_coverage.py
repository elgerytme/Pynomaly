#!/usr/bin/env python3
"""
Comprehensive CLI Test Coverage Analysis for Pynomaly

This script analyzes the CLI test coverage by examining:
1. All CLI source files and their commands
2. Existing CLI test files and their test functions
3. Coverage gaps and missing tests
4. Test utilities and helpers
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CLICommand:
    """Represents a CLI command."""
    module: str
    name: str
    function: str
    help_text: str = ""
    params: list[str] = None

    def __post_init__(self):
        if self.params is None:
            self.params = []

@dataclass
class CLITest:
    """Represents a CLI test."""
    module: str
    name: str
    function: str
    target_command: str = ""
    test_type: str = ""

@dataclass
class CLIModule:
    """Represents a CLI module."""
    name: str
    file_path: str
    commands: list[CLICommand]
    tests: list[CLITest]

    def __post_init__(self):
        if not self.commands:
            self.commands = []
        if not self.tests:
            self.tests = []

class CLICoverageAnalyzer:
    """Analyzes CLI test coverage."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cli_src_path = self.project_root / "src" / "pynomaly" / "presentation" / "cli"
        self.cli_test_path = self.project_root / "tests" / "cli"

        self.cli_modules: dict[str, CLIModule] = {}
        self.cli_commands: list[CLICommand] = []
        self.cli_tests: list[CLITest] = []
        self.coverage_gaps: list[str] = []

    def analyze_cli_modules(self) -> None:
        """Analyze CLI source modules and extract commands."""
        print("🔍 Analyzing CLI source modules...")

        if not self.cli_src_path.exists():
            print(f"❌ CLI source path not found: {self.cli_src_path}")
            return

        for py_file in self.cli_src_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            module_name = py_file.stem
            self.cli_modules[module_name] = CLIModule(
                name=module_name,
                file_path=str(py_file),
                commands=[],
                tests=[]
            )

            # Extract commands from this module
            commands = self._extract_commands_from_file(py_file, module_name)
            self.cli_modules[module_name].commands = commands
            self.cli_commands.extend(commands)

        print(f"✅ Found {len(self.cli_modules)} CLI modules")
        print(f"✅ Found {len(self.cli_commands)} CLI commands")

    def analyze_cli_tests(self) -> None:
        """Analyze CLI test files and extract test functions."""
        print("🔍 Analyzing CLI test files...")

        if not self.cli_test_path.exists():
            print(f"❌ CLI test path not found: {self.cli_test_path}")
            return

        # Analyze test files recursively
        test_files = []
        for py_file in self.cli_test_path.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
            test_files.append(py_file)

        for test_file in test_files:
            tests = self._extract_tests_from_file(test_file)
            self.cli_tests.extend(tests)

            # Associate tests with modules
            for test in tests:
                if test.module in self.cli_modules:
                    self.cli_modules[test.module].tests.append(test)

        print(f"✅ Found {len(test_files)} CLI test files")
        print(f"✅ Found {len(self.cli_tests)} CLI test functions")

    def _extract_commands_from_file(self, file_path: Path, module_name: str) -> list[CLICommand]:
        """Extract CLI commands from a Python file."""
        commands = []

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Find @app.command() decorators and their functions
            pattern = r'@app\.command\((?:"([^"]*)"|\))\s*\ndef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            matches = re.finditer(pattern, content)

            for match in matches:
                command_name = match.group(1) if match.group(1) else match.group(2)
                function_name = match.group(2)

                # Extract help text if available
                help_text = self._extract_help_text(content, match.end())

                # Extract parameters
                params = self._extract_function_params(content, match.end())

                commands.append(CLICommand(
                    module=module_name,
                    name=command_name,
                    function=function_name,
                    help_text=help_text,
                    params=params
                ))

        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")

        return commands

    def _extract_tests_from_file(self, file_path: Path) -> list[CLITest]:
        """Extract test functions from a Python file."""
        tests = []

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Find test functions
            pattern = r'def\s+(test_[a-zA-Z0-9_]+)\s*\('
            matches = re.finditer(pattern, content)

            for match in matches:
                test_name = match.group(1)

                # Try to determine which CLI module this test targets
                target_module = self._determine_target_module(file_path, content, test_name)

                # Determine test type
                test_type = self._determine_test_type(test_name, content)

                tests.append(CLITest(
                    module=target_module,
                    name=test_name,
                    function=test_name,
                    target_command=self._extract_target_command(content, test_name),
                    test_type=test_type
                ))

        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")

        return tests

    def _extract_help_text(self, content: str, start_pos: int) -> str:
        """Extract help text from function docstring."""
        # Look for docstring after function definition
        function_start = content.find('"""', start_pos)
        if function_start == -1:
            return ""

        function_end = content.find('"""', function_start + 3)
        if function_end == -1:
            return ""

        return content[function_start + 3:function_end].strip()

    def _extract_function_params(self, content: str, start_pos: int) -> list[str]:
        """Extract function parameters from function definition."""
        params = []

        # Find the function definition
        func_start = content.find('def ', start_pos - 100)
        if func_start == -1:
            return params

        # Find the end of the function definition
        paren_count = 0
        i = func_start
        while i < len(content):
            if content[i] == '(':
                paren_count += 1
            elif content[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    break
            i += 1

        if i < len(content):
            func_def = content[func_start:i+1]
            # Extract parameter names using regex
            param_pattern = r'(\w+)\s*:'
            param_matches = re.finditer(param_pattern, func_def)
            for match in param_matches:
                param_name = match.group(1)
                if param_name not in ['self', 'cls']:
                    params.append(param_name)

        return params

    def _determine_target_module(self, file_path: Path, content: str, test_name: str) -> str:
        """Determine which CLI module a test targets."""
        # Extract from file path
        relative_path = file_path.relative_to(self.cli_test_path)
        path_parts = relative_path.parts

        # Look for imports in the test file
        import_pattern = r'from\s+pynomaly\.presentation\.cli\.(\w+)\s+import'
        import_matches = re.finditer(import_pattern, content)

        for match in import_matches:
            module_name = match.group(1)
            if module_name in self.cli_modules:
                return module_name

        # Try to infer from test name
        for module_name in self.cli_modules:
            if module_name in test_name or module_name in str(file_path):
                return module_name

        return "unknown"

    def _determine_test_type(self, test_name: str, content: str) -> str:
        """Determine the type of test."""
        if "integration" in test_name.lower():
            return "integration"
        elif "unit" in test_name.lower():
            return "unit"
        elif "workflow" in test_name.lower():
            return "workflow"
        elif "error" in test_name.lower():
            return "error_handling"
        elif "help" in test_name.lower():
            return "help"
        else:
            return "functional"

    def _extract_target_command(self, content: str, test_name: str) -> str:
        """Extract the target command from test content."""
        # Look for runner.invoke calls
        invoke_pattern = r'runner\.invoke\([^,]+,\s*\[(?:"([^"]*)")?'
        invoke_matches = re.finditer(invoke_pattern, content)

        for match in invoke_matches:
            if match.group(1):
                return match.group(1)

        return ""

    def analyze_coverage_gaps(self) -> None:
        """Analyze coverage gaps between CLI commands and tests."""
        print("🔍 Analyzing coverage gaps...")

        # Create mapping of tested commands
        tested_commands = set()
        for test in self.cli_tests:
            if test.target_command:
                tested_commands.add(f"{test.module}.{test.target_command}")

        # Check each command for test coverage
        for command in self.cli_commands:
            command_key = f"{command.module}.{command.name}"
            if command_key not in tested_commands:
                self.coverage_gaps.append(command_key)

        print(f"⚠️  Found {len(self.coverage_gaps)} coverage gaps")

    def generate_report(self) -> dict:
        """Generate comprehensive coverage report."""
        report = {
            "summary": {
                "total_cli_modules": len(self.cli_modules),
                "total_cli_commands": len(self.cli_commands),
                "total_cli_tests": len(self.cli_tests),
                "coverage_gaps": len(self.coverage_gaps),
                "test_coverage_percentage": (
                    (len(self.cli_commands) - len(self.coverage_gaps)) / len(self.cli_commands) * 100
                ) if self.cli_commands else 0
            },
            "cli_modules": {},
            "cli_commands": [],
            "cli_tests": [],
            "coverage_gaps": self.coverage_gaps,
            "test_utilities": self._analyze_test_utilities()
        }

        # Add module details
        for module_name, module in self.cli_modules.items():
            report["cli_modules"][module_name] = {
                "file_path": module.file_path,
                "commands": len(module.commands),
                "tests": len(module.tests),
                "command_details": [
                    {
                        "name": cmd.name,
                        "function": cmd.function,
                        "help_text": cmd.help_text,
                        "params": cmd.params
                    }
                    for cmd in module.commands
                ],
                "test_details": [
                    {
                        "name": test.name,
                        "function": test.function,
                        "target_command": test.target_command,
                        "test_type": test.test_type
                    }
                    for test in module.tests
                ]
            }

        # Add command details
        for command in self.cli_commands:
            report["cli_commands"].append({
                "module": command.module,
                "name": command.name,
                "function": command.function,
                "help_text": command.help_text,
                "params": command.params
            })

        # Add test details
        for test in self.cli_tests:
            report["cli_tests"].append({
                "module": test.module,
                "name": test.name,
                "function": test.function,
                "target_command": test.target_command,
                "test_type": test.test_type
            })

        return report

    def _analyze_test_utilities(self) -> dict:
        """Analyze test utilities and helpers."""
        utilities = {
            "fixtures": [],
            "helpers": [],
            "mocks": [],
            "test_data": []
        }

        # Check for common test utilities
        test_files = list(self.cli_test_path.rglob("*.py"))

        for test_file in test_files:
            try:
                with open(test_file, encoding='utf-8') as f:
                    content = f.read()

                # Look for fixtures
                if "@pytest.fixture" in content:
                    fixture_pattern = r'@pytest\.fixture[^\n]*\ndef\s+([a-zA-Z_][a-zA-Z0-9_]*)'
                    fixtures = re.findall(fixture_pattern, content)
                    utilities["fixtures"].extend(fixtures)

                # Look for helper functions
                if "def test_" not in content and "def " in content:
                    helper_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                    helpers = re.findall(helper_pattern, content)
                    utilities["helpers"].extend(helpers)

                # Look for mocks
                if "Mock" in content or "patch" in content:
                    utilities["mocks"].append(str(test_file.relative_to(self.cli_test_path)))

            except Exception as e:
                print(f"⚠️  Error analyzing test utilities in {test_file}: {e}")

        return utilities

    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        print("\n" + "="*60)
        print("📊 PYNOMALY CLI TEST COVERAGE ANALYSIS")
        print("="*60)

        print("\n🏗️  CLI Structure:")
        print(f"   • CLI Modules: {len(self.cli_modules)}")
        print(f"   • CLI Commands: {len(self.cli_commands)}")
        print(f"   • CLI Tests: {len(self.cli_tests)}")

        coverage_pct = (
            (len(self.cli_commands) - len(self.coverage_gaps)) / len(self.cli_commands) * 100
        ) if self.cli_commands else 0

        print("\n📈 Test Coverage:")
        print(f"   • Coverage Gaps: {len(self.coverage_gaps)}")
        print(f"   • Test Coverage: {coverage_pct:.1f}%")

        print("\n📋 Top CLI Modules by Commands:")
        module_stats = [
            (name, len(module.commands), len(module.tests))
            for name, module in self.cli_modules.items()
        ]
        module_stats.sort(key=lambda x: x[1], reverse=True)

        for name, commands, tests in module_stats[:10]:
            print(f"   • {name}: {commands} commands, {tests} tests")

        print("\n⚠️  Coverage Gaps:")
        for gap in self.coverage_gaps[:10]:
            print(f"   • {gap}")

        if len(self.coverage_gaps) > 10:
            print(f"   ... and {len(self.coverage_gaps) - 10} more")

    def run_analysis(self) -> dict:
        """Run the complete CLI coverage analysis."""
        print("🚀 Starting CLI Test Coverage Analysis...")

        self.analyze_cli_modules()
        self.analyze_cli_tests()
        self.analyze_coverage_gaps()

        report = self.generate_report()
        self.print_summary()

        return report

def main():
    """Main function."""
    project_root = "/mnt/c/Users/andre/Pynomaly"

    analyzer = CLICoverageAnalyzer(project_root)
    report = analyzer.run_analysis()

    # Save report to file
    report_file = Path(project_root) / "cli_coverage_analysis.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n💾 Report saved to: {report_file}")

    return report

if __name__ == "__main__":
    main()
