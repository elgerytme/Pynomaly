import json
import os
import subprocess
from pathlib import Path
import argparse

class TestOrchestrator:
    def __init__(self, mode: str):
        self.mode = mode
        self.reports_dir = Path("reports/advanced")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_pytest(self):
        print("Running pytest...")
        result = subprocess.run(["pytest", "--cov", "--cov-report=json:reports/advanced/coverage.json", "tests"],
                                capture_output=True, text=True)
        self._log_result(result)

    def run_property_tests(self):
        print("Running property-based tests...")
        result = subprocess.run(["pytest", "--hypothesis-show-statistics", "tests"],
                                capture_output=True, text=True)
        self._log_result(result)
        
    def run_mutation_tests(self):
        print("Running mutation tests...")
        result = subprocess.run(["mutmut", "run"], capture_output=True, text=True)
        self._log_result(result)

    def _log_result(self, result):
        if result.returncode == 0:
            print("Success")
        else:
            print("Failure")
            print(result.stdout)
            print(result.stderr)

    def run_all(self):
        self.run_pytest()
        self.run_property_tests()
        self.run_mutation_tests()

    def generate_report(self):
        print("Generating report...")
        report_data = {
            "mode": self.mode,
            "summary": "Detailed test results",
            # Placeholder for example
            "pass_fail_status": "TBD",
            "coverage": self._load_coverage_data(),
            "mutation_score": "TBD"
        }
        report_file = self.reports_dir / "aggregate_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"Report generated at: {report_file}")

    def _load_coverage_data(self):
        try:
            with open("reports/advanced/coverage.json") as f:
                coverage_data = json.load(f)
            print("Coverage data loaded.")
            return coverage_data
        except Exception as e:
            print(f"Failed to load coverage data: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(description='Advanced Test Orchestrator')
    parser.add_argument('--mode', choices=['all', 'mutation', 'property'],
                        default='all', help='Test mode to run')
    args = parser.parse_args()

    orchestrator = TestOrchestrator(mode=args.mode)
    if orchestrator.mode == "all":
        orchestrator.run_all()
    elif orchestrator.mode == "mutation":
        orchestrator.run_mutation_tests()
    elif orchestrator.mode == "property":
        orchestrator.run_property_tests()
    
    orchestrator.generate_report()

if __name__ == "__main__":
    main()


