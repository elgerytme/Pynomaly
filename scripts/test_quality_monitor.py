#!/usr/bin/env python3
"""Test quality monitoring and metrics for Pynomaly."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class TestQualityMonitor:
    """Monitors and analyzes test quality metrics."""
    
    def __init__(self, root_dir: Path = None):
        """Initialize test quality monitor."""
        self.root_dir = root_dir or Path.cwd()
        self.reports_dir = self.root_dir / "reports" / "test_quality"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def find_test_files(self) -> List[Path]:
        """Find all test files in the project."""
        test_files = []
        
        # Common test directories
        test_dirs = [
            "tests/",
            "src/pynomaly/tests/",
            "src/integration_tests/",
        ]
        
        for test_dir in test_dirs:
            test_path = self.root_dir / test_dir
            if test_path.exists():
                # Find Python test files
                test_files.extend(test_path.rglob("test_*.py"))
                test_files.extend(test_path.rglob("*_test.py"))
        
        return test_files
    
    def analyze_test_file(self, file_path: Path) -> Dict:
        """Analyze a single test file for quality metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Initialize metrics
            metrics = {
                "file_path": str(file_path),
                "lines_of_code": len(content.splitlines()),
                "test_functions": 0,
                "test_classes": 0,
                "assertions": 0,
                "fixtures": 0,
                "mocks": 0,
                "parametrized_tests": 0,
                "async_tests": 0,
                "docstrings": 0,
                "complexity_issues": [],
                "long_functions": [],
                "missing_docstrings": [],
            }
            
            for node in ast.walk(tree):
                # Count test functions and classes
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        metrics["test_functions"] += 1
                        
                        # Check for async tests
                        if isinstance(node, ast.AsyncFunctionDef):
                            metrics["async_tests"] += 1
                        
                        # Check for docstrings
                        if (node.body and isinstance(node.body[0], ast.Expr) 
                            and isinstance(node.body[0].value, ast.Constant)):
                            metrics["docstrings"] += 1
                        else:
                            metrics["missing_docstrings"].append(node.name)
                        
                        # Check function length
                        func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        if func_lines > 50:  # Long function threshold
                            metrics["long_functions"].append({
                                "name": node.name,
                                "lines": func_lines
                            })
                
                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith('Test'):
                        metrics["test_classes"] += 1
                
                # Count assertions
                elif isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) 
                        and node.func.attr.startswith('assert')):
                        metrics["assertions"] += 1
                    elif (isinstance(node.func, ast.Name) 
                          and node.func.id.startswith('assert')):
                        metrics["assertions"] += 1
            
            # Check for pytest markers and fixtures in content
            if '@pytest.fixture' in content:
                metrics["fixtures"] = len(re.findall(r'@pytest\.fixture', content))
            
            if '@pytest.mark.parametrize' in content:
                metrics["parametrized_tests"] = len(re.findall(r'@pytest\.mark\.parametrize', content))
            
            if 'mock' in content.lower():
                metrics["mocks"] = len(re.findall(r'mock\w*', content, re.IGNORECASE))
            
            return metrics
            
        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": str(e),
                "lines_of_code": 0,
                "test_functions": 0,
            }
    
    def calculate_test_coverage_per_module(self) -> Dict:
        """Calculate test coverage per module."""
        try:
            # Run coverage with JSON output
            subprocess.run([
                "coverage", "run", "-m", "pytest", "--tb=no", "-q"
            ], check=True, capture_output=True)
            
            subprocess.run([
                "coverage", "json", "-o", str(self.reports_dir / "coverage_per_module.json")
            ], check=True, capture_output=True)
            
            with open(self.reports_dir / "coverage_per_module.json") as f:
                coverage_data = json.load(f)
            
            return coverage_data.get("files", {})
            
        except Exception as e:
            print(f"Error calculating coverage per module: {e}")
            return {}
    
    def run_quality_analysis(self) -> Dict:
        """Run comprehensive test quality analysis."""
        print("🔍 Analyzing test quality...")
        
        test_files = self.find_test_files()
        print(f"Found {len(test_files)} test files")
        
        # Analyze each test file
        file_metrics = []
        for test_file in test_files:
            metrics = self.analyze_test_file(test_file)
            file_metrics.append(metrics)
        
        # Calculate aggregate metrics
        total_metrics = {
            "total_files": len(test_files),
            "total_test_functions": sum(m["test_functions"] for m in file_metrics),
            "total_test_classes": sum(m["test_classes"] for m in file_metrics),
            "total_assertions": sum(m["assertions"] for m in file_metrics),
            "total_fixtures": sum(m["fixtures"] for m in file_metrics),
            "total_mocks": sum(m["mocks"] for m in file_metrics),
            "total_parametrized": sum(m["parametrized_tests"] for m in file_metrics),
            "total_async_tests": sum(m["async_tests"] for m in file_metrics),
            "total_docstrings": sum(m["docstrings"] for m in file_metrics),
            "total_loc": sum(m["lines_of_code"] for m in file_metrics),
            "files_missing_docstrings": len([m for m in file_metrics if m["missing_docstrings"]]),
            "long_functions_count": sum(len(m["long_functions"]) for m in file_metrics),
        }
        
        # Calculate quality scores
        if total_metrics["total_test_functions"] > 0:
            total_metrics["assertions_per_test"] = (
                total_metrics["total_assertions"] / total_metrics["total_test_functions"]
            )
            total_metrics["docstring_coverage"] = (
                total_metrics["total_docstrings"] / total_metrics["total_test_functions"] * 100
            )
        else:
            total_metrics["assertions_per_test"] = 0
            total_metrics["docstring_coverage"] = 0
        
        # Get coverage per module
        coverage_data = self.calculate_test_coverage_per_module()
        
        return {
            "summary": total_metrics,
            "file_details": file_metrics,
            "coverage_per_module": coverage_data,
            "quality_issues": self._identify_quality_issues(file_metrics, total_metrics)
        }
    
    def _identify_quality_issues(self, file_metrics: List[Dict], total_metrics: Dict) -> List[str]:
        """Identify test quality issues."""
        issues = []
        
        # Check for low assertion density
        if total_metrics["assertions_per_test"] < 2:
            issues.append(f"Low assertion density: {total_metrics['assertions_per_test']:.1f} assertions per test")
        
        # Check for missing docstrings
        if total_metrics["docstring_coverage"] < 50:
            issues.append(f"Low docstring coverage: {total_metrics['docstring_coverage']:.1f}%")
        
        # Check for long test functions
        if total_metrics["long_functions_count"] > 0:
            issues.append(f"{total_metrics['long_functions_count']} test functions are too long (>50 lines)")
        
        # Check for files with no tests
        empty_files = [m for m in file_metrics if m["test_functions"] == 0]
        if empty_files:
            issues.append(f"{len(empty_files)} test files contain no test functions")
        
        return issues
    
    def generate_quality_report(self, analysis: Dict) -> None:
        """Generate test quality report."""
        print("📊 Generating test quality report...")
        
        report_file = self.reports_dir / "test_quality_report.md"
        
        with open(report_file, "w") as f:
            f.write("# 🧪 Test Quality Report\n\n")
            
            # Summary
            summary = analysis["summary"]
            f.write("## 📈 Summary Metrics\n\n")
            f.write(f"- **Total Test Files**: {summary['total_files']}\n")
            f.write(f"- **Total Test Functions**: {summary['total_test_functions']}\n")
            f.write(f"- **Total Test Classes**: {summary['total_test_classes']}\n")
            f.write(f"- **Total Assertions**: {summary['total_assertions']}\n")
            f.write(f"- **Assertions per Test**: {summary['assertions_per_test']:.2f}\n")
            f.write(f"- **Docstring Coverage**: {summary['docstring_coverage']:.1f}%\n")
            f.write(f"- **Total Lines of Test Code**: {summary['total_loc']}\n")
            f.write(f"- **Async Tests**: {summary['total_async_tests']}\n")
            f.write(f"- **Parametrized Tests**: {summary['total_parametrized']}\n")
            f.write(f"- **Mock Usage**: {summary['total_mocks']} instances\n")
            f.write(f"- **Fixtures**: {summary['total_fixtures']}\n\n")
            
            # Quality issues
            if analysis["quality_issues"]:
                f.write("## ⚠️  Quality Issues\n\n")
                for issue in analysis["quality_issues"]:
                    f.write(f"- {issue}\n")
                f.write("\n")
            else:
                f.write("## ✅ No Major Quality Issues Found\n\n")
            
            # Coverage per module
            if analysis["coverage_per_module"]:
                f.write("## 📊 Coverage per Module\n\n")
                f.write("| Module | Coverage | Lines | Missing |\n")
                f.write("|--------|----------|-------|----------|\n")
                
                for module, data in analysis["coverage_per_module"].items():
                    coverage = data.get("summary", {}).get("percent_covered", 0)
                    lines = data.get("summary", {}).get("num_statements", 0)
                    missing = len(data.get("missing_lines", []))
                    f.write(f"| {module} | {coverage:.1f}% | {lines} | {missing} |\n")
                f.write("\n")
            
            # Recommendations
            f.write("## 🎯 Recommendations\n\n")
            
            if summary["assertions_per_test"] < 2:
                f.write("- **Increase assertion density**: Add more assertions to test functions\n")
            
            if summary["docstring_coverage"] < 80:
                f.write("- **Add test docstrings**: Document test purposes and expected outcomes\n")
            
            if summary["total_parametrized"] < summary["total_test_functions"] * 0.1:
                f.write("- **Use parametrized tests**: Reduce code duplication with pytest.mark.parametrize\n")
            
            if summary["total_async_tests"] == 0:
                f.write("- **Consider async testing**: Add async tests for async code paths\n")
            
            f.write("- **Regular code review**: Maintain test quality through peer review\n")
            f.write("- **Refactor long functions**: Break down complex test functions\n")
            f.write("- **Mock external dependencies**: Improve test isolation and speed\n")
        
        print(f"✅ Test quality report generated: {report_file}")
        
        # Save detailed data as JSON
        with open(self.reports_dir / "test_quality_data.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test quality monitoring")
    parser.add_argument("--report-only", action="store_true", 
                       help="Only generate report from existing data")
    
    args = parser.parse_args()
    
    monitor = TestQualityMonitor()
    
    if not args.report_only:
        analysis = monitor.run_quality_analysis()
    else:
        # Load existing data
        data_file = monitor.reports_dir / "test_quality_data.json"
        if data_file.exists():
            with open(data_file) as f:
                analysis = json.load(f)
        else:
            print("❌ No existing quality data found. Run without --report-only first.")
            sys.exit(1)
    
    monitor.generate_quality_report(analysis)
    print("🎉 Test quality analysis completed!")


if __name__ == "__main__":
    main()