#!/usr/bin/env python3
"""Mutation testing automation for Pynomaly."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class MutationTester:
    """Manages mutation testing for code quality validation."""
    
    def __init__(self, root_dir: Path = None):
        """Initialize mutation tester."""
        self.root_dir = root_dir or Path.cwd()
        self.reports_dir = self.root_dir / "reports" / "mutation"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def run_mutmut_testing(self, paths: List[str] = None) -> bool:
        """Run mutation testing with mutmut."""
        print("🧬 Running mutation testing with mutmut...")
        
        if not paths:
            paths = ["src/pynomaly/"]
        
        try:
            # Initialize mutmut
            subprocess.run(["mutmut", "run", "--paths-to-mutate"] + paths, 
                         check=True, cwd=self.root_dir)
            
            # Generate HTML report
            subprocess.run(["mutmut", "html"], check=True, cwd=self.root_dir)
            
            # Generate JSON report
            result = subprocess.run(
                ["mutmut", "results"],
                capture_output=True, text=True, check=True, cwd=self.root_dir
            )
            
            with open(self.reports_dir / "mutmut_results.json", "w") as f:
                json.dump({"results": result.stdout}, f, indent=2)
            
            print("✅ Mutation testing completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Mutation testing failed: {e}")
            return False
    
    def run_cosmic_ray_testing(self, module_path: str = "src/pynomaly") -> bool:
        """Run mutation testing with cosmic-ray."""
        print("🌌 Running mutation testing with cosmic-ray...")
        
        try:
            # Initialize cosmic-ray session
            session_file = self.reports_dir / "cosmic_ray_session.toml"
            
            subprocess.run([
                "cosmic-ray", "init", str(session_file),
                module_path, "--test-command", "python -m pytest tests/"
            ], check=True, cwd=self.root_dir)
            
            # Run mutations
            subprocess.run([
                "cosmic-ray", "exec", str(session_file)
            ], check=True, cwd=self.root_dir)
            
            # Generate report
            subprocess.run([
                "cosmic-ray", "dump", str(session_file), 
                str(self.reports_dir / "cosmic_ray_results.json")
            ], check=True, cwd=self.root_dir)
            
            print("✅ Cosmic-ray testing completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Cosmic-ray testing failed: {e}")
            return False
    
    def analyze_mutation_results(self) -> Dict:
        """Analyze mutation testing results."""
        print("🔍 Analyzing mutation testing results...")
        
        results = {}
        
        # Analyze mutmut results
        mutmut_file = self.reports_dir / "mutmut_results.json"
        if mutmut_file.exists():
            with open(mutmut_file) as f:
                mutmut_data = json.load(f)
            results["mutmut"] = mutmut_data
        
        # Analyze cosmic-ray results
        cosmic_ray_file = self.reports_dir / "cosmic_ray_results.json"
        if cosmic_ray_file.exists():
            with open(cosmic_ray_file) as f:
                cosmic_ray_data = json.load(f)
            results["cosmic_ray"] = cosmic_ray_data
        
        return results
    
    def generate_mutation_report(self, results: Dict) -> None:
        """Generate comprehensive mutation testing report."""
        print("📊 Generating mutation testing report...")
        
        report_file = self.reports_dir / "mutation_report.md"
        
        with open(report_file, "w") as f:
            f.write("# 🧬 Mutation Testing Report\n\n")
            f.write(f"Generated on: {Path.cwd()}\n\n")
            
            if "mutmut" in results:
                f.write("## Mutmut Results\n\n")
                f.write("```\n")
                f.write(results["mutmut"].get("results", "No results available"))
                f.write("\n```\n\n")
            
            if "cosmic_ray" in results:
                f.write("## Cosmic-Ray Results\n\n")
                f.write("```json\n")
                f.write(json.dumps(results["cosmic_ray"], indent=2))
                f.write("\n```\n\n")
            
            f.write("## Summary\n\n")
            f.write("Mutation testing helps identify gaps in test coverage by ")
            f.write("introducing small changes (mutations) to the code and ")
            f.write("verifying that tests catch these changes.\n\n")
            f.write("- **Killed mutations**: Tests successfully caught the change ✅\n")
            f.write("- **Survived mutations**: Tests did not catch the change ❌\n")
            f.write("- **Timeout/Error**: Mutation caused timeout or error ⚠️\n")
        
        print(f"✅ Mutation report generated: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mutation testing automation")
    parser.add_argument("--tool", choices=["mutmut", "cosmic-ray", "both"], 
                       default="mutmut", help="Mutation testing tool to use")
    parser.add_argument("--paths", nargs="*", help="Paths to mutate")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze existing results")
    
    args = parser.parse_args()
    
    tester = MutationTester()
    
    if not args.analyze_only:
        success = True
        
        if args.tool in ["mutmut", "both"]:
            if not tester.run_mutmut_testing(args.paths):
                success = False
        
        if args.tool in ["cosmic-ray", "both"]:
            module_path = args.paths[0] if args.paths else "src/pynomaly"
            if not tester.run_cosmic_ray_testing(module_path):
                success = False
        
        if not success:
            sys.exit(1)
    
    # Analyze results
    results = tester.analyze_mutation_results()
    tester.generate_mutation_report(results)
    
    print("🎉 Mutation testing analysis completed!")


if __name__ == "__main__":
    main()