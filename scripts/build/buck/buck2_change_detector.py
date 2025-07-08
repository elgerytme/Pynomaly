#!/usr/bin/env python3
"""
Buck2 Change Detection System for Pynomaly
Identifies affected files and their dependencies for incremental testing.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChangeAnalysis:
    """Analysis of changes and their impact on Buck2 targets."""
    changed_files: List[str]
    affected_targets: Set[str]
    test_targets: Set[str]
    build_targets: Set[str]
    commit_range: str
    analysis_metadata: Dict

class Buck2ChangeDetector:
    """Detects changes and maps them to Buck2 targets for incremental testing."""

    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path.cwd()
        self.buck_file = self.repo_root / "BUCK"
        self.target_map = self._build_target_map()

    def _build_target_map(self) -> Dict[str, Set[str]]:
        """Build mapping from file patterns to Buck2 targets."""
        target_map = {
            # Domain layer
            "src/pynomaly/domain/": {"domain", "test-domain"},
            # Application layer
            "src/pynomaly/application/": {"application", "test-application"},
            # Infrastructure layer
            "src/pynomaly/infrastructure/": {"infrastructure", "test-infrastructure"},
            # Presentation layer
            "src/pynomaly/presentation/": {"presentation", "test-presentation"},
            # Shared utilities
            "src/pynomaly/shared/": {"shared"},
            # Tests
            "tests/domain/": {"test-domain"},
            "tests/application/": {"test-application"},
            "tests/infrastructure/": {"test-infrastructure"},
            "tests/presentation/": {"test-presentation"},
            "tests/integration/": {"test-integration"},
            "tests/e2e/": {"test-integration"},
            "tests/performance/": {"benchmarks"},
            "tests/benchmarks/": {"benchmarks"},
            "tests/property/": {"property-tests"},
            "tests/mutation/": {"mutation-tests"},
            "tests/security/": {"security-tests"},
            # Scripts and utilities
            "scripts/": {"pynomaly-cli", "pynomaly-api", "pynomaly-web"},
            # Web assets
            "src/pynomaly/presentation/web/": {"tailwind-build", "pynomaly-js", "web-assets"},
            "config/web/": {"tailwind-build", "web-assets"},
            # Documentation
            "docs/": {"docs"},
            # Configuration files
            "pyproject.toml": {"pynomaly-lib", "wheel", "sdist"},
            "requirements*.txt": {"pynomaly-lib"},
            "BUCK": {"build-all"},
            ".buckconfig": {"build-all"},
        }
        return target_map

    def get_changed_files(self, base_commit: str = "HEAD~1", target_commit: str = "HEAD") -> List[str]:
        """Get list of changed files between commits."""
        try:
            cmd = ["git", "diff", "--name-only", f"{base_commit}..{target_commit}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            changed_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            logger.info(f"Found {len(changed_files)} changed files between {base_commit} and {target_commit}")
            return changed_files
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get changed files: {e}")
            return []

    def map_files_to_targets(self, changed_files: List[str]) -> Set[str]:
        """Map changed files to affected Buck2 targets."""
        affected_targets = set()

        for file_path in changed_files:
            # Check direct file matches
            if file_path in self.target_map:
                affected_targets.update(self.target_map[file_path])
                continue

            # Check directory pattern matches
            for pattern, targets in self.target_map.items():
                if file_path.startswith(pattern):
                    affected_targets.update(targets)
                    break
            else:
                # If no specific mapping, trigger broader tests
                if file_path.startswith("src/pynomaly/"):
                    affected_targets.update({"pynomaly-lib", "test-integration"})
                elif file_path.startswith("tests/"):
                    affected_targets.add("test-integration")
                elif file_path.startswith("scripts/"):
                    affected_targets.update({"pynomaly-cli", "pynomaly-api", "pynomaly-web"})

        logger.info(f"Mapped {len(changed_files)} files to {len(affected_targets)} targets")
        return affected_targets

    def get_dependent_targets(self, targets: Set[str]) -> Set[str]:
        """Get targets that depend on the given targets."""
        # Define dependency relationships
        dependencies = {
            "domain": {"application", "infrastructure", "presentation", "pynomaly-lib"},
            "application": {"infrastructure", "presentation", "pynomaly-lib"},
            "infrastructure": {"presentation", "pynomaly-lib"},
            "shared": {"domain", "application", "infrastructure", "presentation", "pynomaly-lib"},
            "pynomaly-lib": {"pynomaly-cli", "pynomaly-api", "pynomaly-web", "wheel", "sdist"},
        }

        dependent_targets = set(targets)

        for target in targets:
            if target in dependencies:
                dependent_targets.update(dependencies[target])

        return dependent_targets

    def categorize_targets(self, targets: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Categorize targets into test and build targets."""
        test_targets = {t for t in targets if t.startswith("test-") or t in ["benchmarks", "property-tests", "mutation-tests", "security-tests"]}
        build_targets = targets - test_targets

        return test_targets, build_targets

    def analyze_changes(self, base_commit: str = "HEAD~1", target_commit: str = "HEAD") -> ChangeAnalysis:
        """Perform comprehensive change analysis."""
        logger.info(f"Analyzing changes between {base_commit} and {target_commit}")

        # Get changed files
        changed_files = self.get_changed_files(base_commit, target_commit)
        if not changed_files:
            logger.info("No changes detected")
            return ChangeAnalysis(
                changed_files=[],
                affected_targets=set(),
                test_targets=set(),
                build_targets=set(),
                commit_range=f"{base_commit}..{target_commit}",
                analysis_metadata={"status": "no_changes"}
            )

        # Map to targets
        affected_targets = self.map_files_to_targets(changed_files)

        # Include dependent targets
        all_affected_targets = self.get_dependent_targets(affected_targets)

        # Categorize targets
        test_targets, build_targets = self.categorize_targets(all_affected_targets)

        # Generate metadata
        analysis_metadata = {
            "total_changed_files": len(changed_files),
            "direct_targets": len(affected_targets),
            "total_affected_targets": len(all_affected_targets),
            "test_targets_count": len(test_targets),
            "build_targets_count": len(build_targets),
            "analysis_timestamp": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
        }

        return ChangeAnalysis(
            changed_files=changed_files,
            affected_targets=all_affected_targets,
            test_targets=test_targets,
            build_targets=build_targets,
            commit_range=f"{base_commit}..{target_commit}",
            analysis_metadata=analysis_metadata
        )

    def save_analysis(self, analysis: ChangeAnalysis, output_file: Path = None) -> Path:
        """Save change analysis to JSON file."""
        if output_file is None:
            output_file = self.repo_root / "buck2_change_analysis.json"

        # Convert sets to lists for JSON serialization
        analysis_dict = asdict(analysis)
        analysis_dict["affected_targets"] = list(analysis.affected_targets)
        analysis_dict["test_targets"] = list(analysis.test_targets)
        analysis_dict["build_targets"] = list(analysis.build_targets)

        with open(output_file, 'w') as f:
            json.dump(analysis_dict, f, indent=2)

        logger.info(f"Analysis saved to {output_file}")
        return output_file

    def load_analysis(self, input_file: Path) -> ChangeAnalysis:
        """Load change analysis from JSON file."""
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Convert lists back to sets
        data["affected_targets"] = set(data["affected_targets"])
        data["test_targets"] = set(data["test_targets"])
        data["build_targets"] = set(data["build_targets"])

        return ChangeAnalysis(**data)

def main():
    """Main entry point for the change detector."""
    parser = argparse.ArgumentParser(description="Buck2 Change Detection System")
    parser.add_argument("--base", default="HEAD~1", help="Base commit for comparison")
    parser.add_argument("--target", default="HEAD", help="Target commit for comparison")
    parser.add_argument("--output", type=Path, help="Output file for analysis results")
    parser.add_argument("--format", choices=["json", "summary"], default="summary", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize detector
    detector = Buck2ChangeDetector()

    # Analyze changes
    analysis = detector.analyze_changes(args.base, args.target)

    # Output results
    if args.format == "json":
        output_file = args.output or Path("buck2_change_analysis.json")
        detector.save_analysis(analysis, output_file)
        print(f"Analysis saved to: {output_file}")
    else:
        # Print summary
        print(f"\n=== Buck2 Change Analysis ({analysis.commit_range}) ===")
        print(f"Changed files: {len(analysis.changed_files)}")
        print(f"Affected targets: {len(analysis.affected_targets)}")
        print(f"Test targets: {len(analysis.test_targets)}")
        print(f"Build targets: {len(analysis.build_targets)}")

        if analysis.changed_files:
            print(f"\nChanged files:")
            for file in analysis.changed_files[:10]:  # Show first 10
                print(f"  - {file}")
            if len(analysis.changed_files) > 10:
                print(f"  ... and {len(analysis.changed_files) - 10} more")

        if analysis.test_targets:
            print(f"\nTest targets to run:")
            for target in sorted(analysis.test_targets):
                print(f"  - {target}")

        if analysis.build_targets:
            print(f"\nBuild targets to run:")
            for target in sorted(analysis.build_targets):
                print(f"  - {target}")

if __name__ == "__main__":
    main()
