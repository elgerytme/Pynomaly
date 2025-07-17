#!/usr/bin/env python3
"""
Automated Repository Governance System
Main entry point for all repository quality checks and automated fixes.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .checks import (
    ArchitectureChecker,
    DomainLeakageChecker,
    LayoutChecker,
    TidinessChecker,
)
from .fixes import (
    AutoFixer,
    BackupFileFixer,
    DomainLeakageFixer,
    StructureFixer,
)
from .reporting import (
    ConsoleReporter,
    GitHubIssueReporter,
    JSONReporter,
    MarkdownReporter,
)
from .rules import RuleEngine


class RepositoryGovernanceRunner:
    """Main governance runner that orchestrates all checks and fixes."""
    
    def __init__(self, root_path: Path, config_path: Optional[Path] = None):
        """Initialize the governance runner."""
        self.root_path = root_path
        self.config_path = config_path or root_path / "scripts" / "repository_governance" / "config.json"
        self.config = self._load_config()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.checkers = self._init_checkers()
        self.fixers = self._init_fixers()
        self.reporters = self._init_reporters()
        self.rule_engine = RuleEngine(self.config.get('rules', {}))
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "log_level": "INFO",
            "checks": {
                "tidiness": {"enabled": True, "auto_fix": True},
                "layout": {"enabled": True, "auto_fix": False},
                "domain_leakage": {"enabled": True, "auto_fix": True},
                "architecture": {"enabled": True, "auto_fix": False}
            },
            "rules": {
                "max_backup_files": 0,
                "max_monorepo_imports": 50,
                "required_package_structure": [
                    "domain", "application", "infrastructure", "interfaces"
                ],
                "forbidden_directories": [
                    "src/archive", "src/temporary", "temp", "backup"
                ],
                "max_file_size_mb": 10,
                "max_directory_depth": 8
            },
            "reporting": {
                "console": {"enabled": True},
                "markdown": {"enabled": True, "output_file": "governance_report.md"},
                "json": {"enabled": True, "output_file": "governance_report.json"},
                "github_issues": {"enabled": False, "create_issues": False}
            }
        }
    
    def _init_checkers(self) -> Dict[str, object]:
        """Initialize all checkers."""
        return {
            "tidiness": TidinessChecker(self.root_path),
            "layout": LayoutChecker(self.root_path),
            "domain_leakage": DomainLeakageChecker(self.root_path),
            "architecture": ArchitectureChecker(self.root_path)
        }
    
    def _init_fixers(self) -> Dict[str, object]:
        """Initialize all fixers."""
        return {
            "backup_files": BackupFileFixer(self.root_path),
            "domain_leakage": DomainLeakageFixer(self.root_path),
            "structure": StructureFixer(self.root_path),
            "auto": AutoFixer(self.root_path)
        }
    
    def _init_reporters(self) -> Dict[str, object]:
        """Initialize all reporters."""
        return {
            "console": ConsoleReporter(),
            "markdown": MarkdownReporter(self.root_path / "governance_report.md"),
            "json": JSONReporter(self.root_path / "governance_report.json"),
            "github": GitHubIssueReporter()
        }
    
    def run_checks(self, check_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Run specified checks or all enabled checks."""
        if check_types is None:
            check_types = [
                name for name, config in self.config["checks"].items() 
                if config.get("enabled", True)
            ]
        
        results = {}
        
        for check_type in check_types:
            if check_type not in self.checkers:
                self.logger.warning(f"Unknown check type: {check_type}")
                continue
            
            self.logger.info(f"Running {check_type} check...")
            try:
                checker = self.checkers[check_type]
                result = checker.check()
                results[check_type] = result
                
                # Apply rules
                rule_violations = self.rule_engine.evaluate_rules(check_type, result)
                if rule_violations:
                    results[check_type]["rule_violations"] = rule_violations
                
            except Exception as e:
                self.logger.error(f"Error running {check_type} check: {e}")
                results[check_type] = {"error": str(e)}
        
        return results
    
    def run_fixes(self, fix_types: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, Dict]:
        """Run specified fixes or all auto-fixes."""
        if fix_types is None:
            fix_types = [
                name for name, config in self.config["checks"].items() 
                if config.get("auto_fix", False)
            ]
        
        results = {}
        
        for fix_type in fix_types:
            if fix_type not in self.fixers:
                self.logger.warning(f"Unknown fix type: {fix_type}")
                continue
            
            self.logger.info(f"Running {fix_type} fix (dry_run={dry_run})...")
            try:
                fixer = self.fixers[fix_type]
                result = fixer.fix(dry_run=dry_run)
                results[fix_type] = result
                
            except Exception as e:
                self.logger.error(f"Error running {fix_type} fix: {e}")
                results[fix_type] = {"error": str(e)}
        
        return results
    
    def generate_reports(self, check_results: Dict, fix_results: Dict = None) -> None:
        """Generate all enabled reports."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "check_results": check_results,
            "fix_results": fix_results or {},
            "config": self.config
        }
        
        for reporter_name, reporter in self.reporters.items():
            if self.config["reporting"].get(reporter_name, {}).get("enabled", False):
                try:
                    reporter.generate_report(report_data)
                    self.logger.info(f"Generated {reporter_name} report")
                except Exception as e:
                    self.logger.error(f"Error generating {reporter_name} report: {e}")
    
    def run_full_governance(self, auto_fix: bool = True, dry_run: bool = False) -> Dict:
        """Run complete governance cycle."""
        self.logger.info("Starting full repository governance cycle...")
        
        # Run all checks
        check_results = self.run_checks()
        
        # Run fixes if enabled
        fix_results = {}
        if auto_fix:
            fix_results = self.run_fixes(dry_run=dry_run)
        
        # Generate reports
        self.generate_reports(check_results, fix_results)
        
        # Calculate summary
        total_violations = sum(
            len(result.get("violations", []))
            for result in check_results.values()
            if isinstance(result, dict)
        )
        
        total_fixes = sum(
            len(result.get("fixes_applied", []))
            for result in fix_results.values()
            if isinstance(result, dict)
        )
        
        summary = {
            "total_violations": total_violations,
            "total_fixes": total_fixes,
            "checks_run": len(check_results),
            "fixes_run": len(fix_results),
            "status": "PASS" if total_violations == 0 else "FAIL"
        }
        
        self.logger.info(f"Governance cycle complete: {summary}")
        return {
            "summary": summary,
            "check_results": check_results,
            "fix_results": fix_results
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Repository Governance System")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root path")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--checks", nargs="*", help="Specific checks to run")
    parser.add_argument("--fixes", nargs="*", help="Specific fixes to run")
    parser.add_argument("--auto-fix", action="store_true", help="Run auto-fixes")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no changes)")
    parser.add_argument("--full", action="store_true", help="Run full governance cycle")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = RepositoryGovernanceRunner(args.root, args.config)
    
    if args.full:
        # Run full governance cycle
        result = runner.run_full_governance(auto_fix=args.auto_fix, dry_run=args.dry_run)
        sys.exit(0 if result["summary"]["status"] == "PASS" else 1)
    
    # Run specific checks or fixes
    if args.checks:
        results = runner.run_checks(args.checks)
        runner.generate_reports(results)
    
    if args.fixes:
        results = runner.run_fixes(args.fixes, dry_run=args.dry_run)
        runner.generate_reports({}, results)


if __name__ == "__main__":
    main()