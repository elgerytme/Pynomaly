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
    HTMLReporter,
)
from .config import GovernanceConfig, GovernanceConfigManager, RulesEngine


class RepositoryGovernanceRunner:
    """Main governance runner that orchestrates all checks and fixes."""
    
    def __init__(self, root_path: Path, config_path: Optional[Path] = None):
        """Initialize the governance runner."""
        self.root_path = root_path
        self.config_path = config_path or root_path / "governance.toml"
        
        # Load configuration
        config_manager = GovernanceConfigManager(self.config_path)
        self.config = config_manager.load_config()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.checkers = self._init_checkers()
        self.fixers = self._init_fixers()
        self.reporters = self._init_reporters()
        self.rules_engine = RulesEngine()
        self.rules_engine.load_default_rules()
        
        self.logger.info("Repository Governance Runner initialized")
    
    def _init_checkers(self) -> Dict[str, object]:
        """Initialize all checkers."""
        checkers = {}
        
        # Initialize each checker if enabled
        if self.config.checkers.get("TidinessChecker", {}).enabled:
            checkers["TidinessChecker"] = TidinessChecker(self.root_path)
        
        if self.config.checkers.get("DomainLeakageChecker", {}).enabled:
            checkers["DomainLeakageChecker"] = DomainLeakageChecker(self.root_path)
        
        if self.config.checkers.get("ArchitectureChecker", {}).enabled:
            checkers["ArchitectureChecker"] = ArchitectureChecker(self.root_path)
        
        return checkers
    
    def _init_fixers(self) -> Dict[str, object]:
        """Initialize all fixers."""
        fixers = {}
        
        # Initialize each fixer if enabled
        if self.config.fixers.get("BackupFileFixer", {}).enabled:
            dry_run = self.config.fixers.get("BackupFileFixer", {}).dry_run or self.config.dry_run
            fixers["BackupFileFixer"] = BackupFileFixer(self.root_path, dry_run=dry_run)
        
        if self.config.fixers.get("DomainLeakageFixer", {}).enabled:
            dry_run = self.config.fixers.get("DomainLeakageFixer", {}).dry_run or self.config.dry_run
            fixers["DomainLeakageFixer"] = DomainLeakageFixer(self.root_path, dry_run=dry_run)
        
        if self.config.fixers.get("StructureFixer", {}).enabled:
            dry_run = self.config.fixers.get("StructureFixer", {}).dry_run or self.config.dry_run
            fixers["StructureFixer"] = StructureFixer(self.root_path, dry_run=dry_run)
        
        return fixers
    
    def _init_reporters(self) -> Dict[str, object]:
        """Initialize all reporters."""
        reporters = {}
        output_dir = self.config.reporting.output_directory
        
        for report_format in self.config.reporting.formats:
            if report_format == "console":
                reporters["console"] = ConsoleReporter(output_dir)
            elif report_format == "json":
                reporters["json"] = JSONReporter(output_dir)
            elif report_format == "markdown":
                reporters["markdown"] = MarkdownReporter(output_dir)
            elif report_format == "html":
                reporters["html"] = HTMLReporter(output_dir, include_charts=self.config.reporting.include_charts)
            elif report_format == "github":
                reporters["github"] = GitHubIssueReporter(output_dir)
        
        return reporters
    
    def run_checks(self, check_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Run specified checks or all enabled checks."""
        if check_types is None:
            check_types = list(self.checkers.keys())
        
        results = {}
        
        for check_type in check_types:
            if check_type not in self.checkers:
                self.logger.warning(f"Unknown check type: {check_type}")
                continue
            
            checker_config = self.config.checkers.get(check_type)
            if not checker_config or not checker_config.enabled:
                self.logger.info(f"Skipping disabled checker: {check_type}")
                continue
            
            self.logger.info(f"Running {check_type} check...")
            try:
                checker = self.checkers[check_type]
                result = checker.check()
                results[check_type] = result
                
                # Apply custom rules if configured
                if checker_config.custom_rules:
                    self._apply_custom_rules(check_type, result, checker_config.custom_rules)
                
                # Check violation limits
                if checker_config.max_violations:
                    total_violations = result.get("total_violations", 0)
                    if total_violations > checker_config.max_violations:
                        self.logger.warning(f"{check_type} exceeded max violations: {total_violations} > {checker_config.max_violations}")
                
            except Exception as e:
                self.logger.error(f"Error running {check_type} check: {e}")
                results[check_type] = {"error": str(e), "violations": [], "total_violations": 0}
        
        return results
    
    def run_fixes(self, fix_types: Optional[List[str]] = None, dry_run: bool = None) -> Dict[str, List]:
        """Run specified fixes or all enabled fixes."""
        if fix_types is None:
            fix_types = list(self.fixers.keys())
        
        if dry_run is None:
            dry_run = self.config.dry_run
        
        results = {}
        
        for fix_type in fix_types:
            if fix_type not in self.fixers:
                self.logger.warning(f"Unknown fix type: {fix_type}")
                continue
            
            fixer_config = self.config.fixers.get(fix_type)
            if not fixer_config or not fixer_config.enabled:
                self.logger.info(f"Skipping disabled fixer: {fix_type}")
                continue
            
            # Check if auto-fix is enabled
            if not fixer_config.auto_fix:
                self.logger.info(f"Skipping non-auto fixer: {fix_type}")
                continue
            
            self.logger.info(f"Running {fix_type} fix (dry_run={dry_run})...")
            try:
                fixer = self.fixers[fix_type]
                
                # Get violations from previous check results
                violations = self._get_violations_for_fixer(fix_type)
                
                if violations:
                    fix_results = fixer.apply_fixes(violations)
                    results[fix_type] = fix_results
                    
                    # Check fix limits
                    if fixer_config.max_fixes:
                        if len(fix_results) > fixer_config.max_fixes:
                            self.logger.warning(f"{fix_type} exceeded max fixes: {len(fix_results)} > {fixer_config.max_fixes}")
                            results[fix_type] = fix_results[:fixer_config.max_fixes]
                else:
                    results[fix_type] = []
                    self.logger.info(f"No violations found for {fix_type}")
                
            except Exception as e:
                self.logger.error(f"Error running {fix_type} fix: {e}")
                results[fix_type] = [{"success": False, "message": f"Error: {str(e)}"}]
        
        return results
    
    def generate_reports(self, check_results: Dict, fix_results: Dict = None) -> None:
        """Generate all enabled reports."""
        for reporter_name, reporter in self.reporters.items():
            try:
                self.logger.info(f"Generating {reporter_name} report...")
                report_content = reporter.generate_report(check_results, fix_results)
                
                # Save report
                success = reporter.save_report(report_content)
                if success:
                    self.logger.info(f"Generated {reporter_name} report successfully")
                else:
                    self.logger.error(f"Failed to save {reporter_name} report")
                    
            except Exception as e:
                self.logger.error(f"Error generating {reporter_name} report: {e}")
    
    def run_full_governance(self, auto_fix: bool = True, dry_run: bool = False) -> Dict:
        """Run complete governance process with checks and fixes."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self._serialize_config(),
            "check_results": {},
            "fix_results": {},
            "summary": {}
        }
        
        try:
            # Run checks
            self.logger.info("Running governance checks...")
            check_results = self.run_checks()
            results["check_results"] = check_results
            
            # Run fixes if enabled
            if auto_fix:
                self.logger.info("Running governance fixes...")
                fix_results = self.run_fixes(dry_run=dry_run)
                results["fix_results"] = fix_results
            
            # Generate reports
            self.logger.info("Generating reports...")
            self.generate_reports(check_results, results.get("fix_results"))
            
            # Calculate summary
            results["summary"] = self._calculate_summary(check_results, results.get("fix_results"))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in governance runner: {e}")
            results["error"] = str(e)
            return results
    
    def _serialize_config(self) -> Dict:
        """Serialize config for JSON output."""
        return {
            "enabled": self.config.enabled,
            "dry_run": self.config.dry_run,
            "fail_on_violations": self.config.fail_on_violations,
            "max_violations": self.config.max_violations,
            "include_patterns": self.config.include_patterns,
            "exclude_patterns": self.config.exclude_patterns,
            "checkers": {
                name: {
                    "enabled": checker.enabled,
                    "severity_override": checker.severity_override,
                    "fail_on_violation": checker.fail_on_violation,
                    "max_violations": checker.max_violations
                }
                for name, checker in self.config.checkers.items()
            },
            "fixers": {
                name: {
                    "enabled": fixer.enabled,
                    "auto_fix": fixer.auto_fix,
                    "dry_run": fixer.dry_run,
                    "max_fixes": fixer.max_fixes
                }
                for name, fixer in self.config.fixers.items()
            },
            "reporting": {
                "formats": self.config.reporting.formats,
                "output_directory": str(self.config.reporting.output_directory) if self.config.reporting.output_directory else None,
                "include_charts": self.config.reporting.include_charts,
                "create_github_issues": self.config.reporting.create_github_issues
            }
        }
    
    def _calculate_summary(self, check_results: Dict, fix_results: Dict = None) -> Dict:
        """Calculate summary statistics."""
        total_violations = 0
        total_fixes = 0
        checkers_passed = 0
        checkers_failed = 0
        
        # Count check results
        for checker_name, result in check_results.items():
            if isinstance(result, dict):
                violations = result.get("violations", [])
                total_violations += len(violations)
                
                if violations:
                    checkers_failed += 1
                else:
                    checkers_passed += 1
        
        # Count fix results
        if fix_results:
            for fixer_name, results in fix_results.items():
                if isinstance(results, list):
                    successful_fixes = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
                    total_fixes += successful_fixes
        
        # Determine overall status
        status = "PASS" if total_violations == 0 else "FAIL"
        
        # Check if we should fail on violations
        should_fail = (
            self.config.fail_on_violations and total_violations > 0
        ) or (
            self.config.max_violations and total_violations > self.config.max_violations
        )
        
        return {
            "total_violations": total_violations,
            "total_fixes": total_fixes,
            "checkers_run": len(check_results),
            "checkers_passed": checkers_passed,
            "checkers_failed": checkers_failed,
            "fixers_run": len(fix_results) if fix_results else 0,
            "status": status,
            "should_fail": should_fail,
            "governance_score": self._calculate_governance_score(check_results),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_governance_score(self, check_results: Dict) -> float:
        """Calculate overall governance score (0-100)."""
        if not check_results:
            return 100.0
        
        total_score = 0
        checker_count = 0
        
        for checker_name, result in check_results.items():
            if isinstance(result, dict) and "score" in result:
                total_score += result["score"]
                checker_count += 1
        
        return total_score / checker_count if checker_count > 0 else 100.0
    
    def _apply_custom_rules(self, checker_name: str, result: Dict, custom_rules: Dict) -> None:
        """Apply custom rules to check results."""
        # This is a placeholder for custom rule application
        # In practice, you'd implement specific custom rule logic here
        pass
    
    def _get_violations_for_fixer(self, fixer_name: str) -> List[Dict]:
        """Get violations that a specific fixer can handle."""
        # This would typically be called after checks have run
        # For now, return empty list as this is a placeholder
        return []


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Repository Governance System")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root path")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--checks", nargs="*", help="Specific checks to run")
    parser.add_argument("--fixes", nargs="*", help="Specific fixes to run")
    parser.add_argument("--auto-fix", action="store_true", help="Run auto-fixes")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no changes)")
    parser.add_argument("--full", action="store_true", default=True, help="Run full governance cycle")
    parser.add_argument("--reports", help="Report formats (comma-separated)")
    parser.add_argument("--output-dir", type=Path, help="Output directory for reports")
    parser.add_argument("--fail-on-violations", action="store_true", help="Fail on any violations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration file")
    
    args = parser.parse_args()
    
    # Set up debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize runner
    runner = RepositoryGovernanceRunner(args.root, args.config)
    
    # Override config with CLI arguments
    if args.dry_run:
        runner.config.dry_run = True
    
    if args.fail_on_violations:
        runner.config.fail_on_violations = True
    
    if args.reports:
        runner.config.reporting.formats = args.reports.split(",")
    
    if args.output_dir:
        runner.config.reporting.output_directory = args.output_dir
    
    # Validate configuration if requested
    if args.validate_config:
        config_manager = GovernanceConfigManager(args.config)
        errors = config_manager.validate_config(runner.config)
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("Configuration is valid")
            sys.exit(0)
    
    # Run governance
    if args.full or (not args.checks and not args.fixes):
        # Run full governance cycle
        result = runner.run_full_governance(auto_fix=args.auto_fix, dry_run=args.dry_run)
        
        # Print summary
        summary = result.get("summary", {})
        print(f"\\nGovernance Summary:")
        print(f"  Status: {summary.get('status', 'UNKNOWN')}")
        print(f"  Total Violations: {summary.get('total_violations', 0)}")
        print(f"  Total Fixes: {summary.get('total_fixes', 0)}")
        print(f"  Governance Score: {summary.get('governance_score', 0):.1f}/100")
        
        # Exit with appropriate code
        sys.exit(0 if not summary.get("should_fail", False) else 1)
    
    # Run specific checks or fixes
    if args.checks:
        results = runner.run_checks(args.checks)
        runner.generate_reports(results)
    
    if args.fixes:
        results = runner.run_fixes(args.fixes, dry_run=args.dry_run)
        runner.generate_reports({}, results)


if __name__ == "__main__":
    main()