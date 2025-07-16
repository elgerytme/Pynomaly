#!/usr/bin/env python3
"""
Automated Repository Organization Tool

This script automatically organizes files in the repository according to
the organization rules defined in .project-rules/REPOSITORY_ORGANIZATION_RULES.md
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse
import json


@dataclass
class OrganizationAction:
    """Represents an organization action to be taken."""
    action_type: str  # 'move', 'delete', 'create_dir'
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0  # 0.0 to 1.0


class RepositoryOrganizer:
    """Automatically organizes repository files according to rules."""
    
    def __init__(self, repo_root: Path, dry_run: bool = True):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.actions: List[OrganizationAction] = []
        
        # Organization rules
        self.config_patterns = {
            r'^\.buckconfig.*': 'scripts/config/buck/',
            r'^\.dockerignore$': 'scripts/config/docker/',
            r'^\.env.*': 'scripts/config/env/',
            r'^\.eslintrc.*': 'scripts/config/linting/',
            r'^\.stylelintrc.*': 'scripts/config/linting/',
            r'^\.prettierrc.*': 'scripts/config/linting/',
            r'^\.mutmut\.toml$': 'scripts/config/testing/',
            r'^\.percyrc\.yml$': 'scripts/config/testing/',
            r'^\.pre-commit.*': 'scripts/config/git/',
            r'^\.gitmessage$': 'scripts/config/git/',
            r'^\.python-version-dev$': 'scripts/config/python/',
            r'^\.secrets\.baseline$': 'scripts/config/security/',
            r'^\.pyno-org\.yaml$': 'scripts/config/',
        }
        
        self.doc_patterns = {
            r'.*_GUIDE\.md$': 'docs/guides/',
            r'.*_PLAN\.md$': 'docs/plans/',
            r'.*_SUMMARY\.md$': 'docs/summaries/',
            r'.*_REPORT\.md$': 'docs/reports/',
            r'.*_ANALYSIS.*\.md$': 'docs/analysis/',
            r'^DEPLOYMENT_.*\.md$': 'docs/deployment/',
            r'^IMPLEMENTATION_.*\.md$': 'docs/implementation/',
            r'^TESTING_.*\.md$': 'docs/testing/',
            r'MIGRATION.*\.md$': 'docs/migration/',
            r'REORGANIZATION.*\.md$': 'docs/architecture/',
            r'DOMAIN.*\.md$': 'docs/architecture/',
        }
        
        self.script_patterns = {
            r'^validate_.*\.py$': 'scripts/validation/',
            r'^test_.*\.py$': 'scripts/testing/',
            r'^fix_.*\.py$': 'scripts/maintenance/',
            r'^setup_.*\.py$': 'scripts/setup/',
            r'^deploy_.*\.py$': 'scripts/deployment/',
            r'^run_.*\.py$': 'scripts/development/',
            r'^BUCK$': 'scripts/build/',
        }

    def analyze_repository(self) -> List[OrganizationAction]:
        """Analyze repository and generate organization actions."""
        self.actions = []
        
        self._analyze_root_directory()
        self._analyze_misplaced_files()
        self._analyze_temporary_files()
        self._create_missing_directories()
        
        return self.actions

    def _analyze_root_directory(self):
        """Analyze files in root directory for organization opportunities."""
        allowed_root_files = {
            'README.md', 'CHANGELOG.md', 'LICENSE', 'pyproject.toml',
            '.gitignore', '.python-version'
        }
        
        for item in os.listdir(self.repo_root):
            item_path = self.repo_root / item
            
            if item_path.is_file() and item not in allowed_root_files:
                self._check_file_organization(item, str(item_path))

    def _check_file_organization(self, filename: str, file_path: str):
        """Check if a file needs to be organized."""
        relative_path = str(Path(file_path).relative_to(self.repo_root))
        
        # Check configuration files
        for pattern, target_dir in self.config_patterns.items():
            if re.match(pattern, filename):
                target_path = self.repo_root / target_dir / filename
                self.actions.append(OrganizationAction(
                    action_type='move',
                    source_path=relative_path,
                    target_path=str(target_path.relative_to(self.repo_root)),
                    reason=f"Configuration file should be in {target_dir}",
                    confidence=0.9
                ))
                return
        
        # Check documentation files
        for pattern, target_dir in self.doc_patterns.items():
            if re.match(pattern, filename):
                target_path = self.repo_root / target_dir / filename
                self.actions.append(OrganizationAction(
                    action_type='move',
                    source_path=relative_path,
                    target_path=str(target_path.relative_to(self.repo_root)),
                    reason=f"Documentation file should be in {target_dir}",
                    confidence=0.8
                ))
                return
        
        # Check script files
        for pattern, target_dir in self.script_patterns.items():
            if re.match(pattern, filename):
                target_path = self.repo_root / target_dir / filename
                self.actions.append(OrganizationAction(
                    action_type='move',
                    source_path=relative_path,
                    target_path=str(target_path.relative_to(self.repo_root)),
                    reason=f"Script file should be in {target_dir}",
                    confidence=0.8
                ))
                return
        
        # Check for Python files that should be in src/
        if filename.endswith('.py') and filename not in ['setup.py']:
            target_path = self.repo_root / 'src' / filename
            self.actions.append(OrganizationAction(
                action_type='move',
                source_path=relative_path,
                target_path=str(target_path.relative_to(self.repo_root)),
                reason="Python files should be in src/",
                confidence=0.7
            ))
            return
        
        # Check for prohibited temporary files
        temp_patterns = [r'.*\.tmp$', r'.*\.temp$', r'.*\.bak$', r'.*\.backup$', 
                        r'^temp_.*', r'^tmp_.*', r'^debug_.*', r'^scratch_.*']
        
        for pattern in temp_patterns:
            if re.match(pattern, filename):
                self.actions.append(OrganizationAction(
                    action_type='delete',
                    source_path=relative_path,
                    reason="Temporary file should be removed",
                    confidence=0.6
                ))
                return

    def _analyze_misplaced_files(self):
        """Find files that are in wrong directories."""
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.repo_root)
            
            # Skip certain directories
            if any(part in ['.git', '.github', '.vscode', '.claude', '.hypothesis', 
                           '.project-rules', '.ruff_cache', '.storybook'] 
                   for part in relative_root.parts):
                continue
            
            for filename in files:
                file_path = root_path / filename
                relative_file_path = file_path.relative_to(self.repo_root)
                
                # Check for configuration files outside scripts/config/
                if not str(relative_file_path).startswith('scripts/config/'):
                    for pattern, target_dir in self.config_patterns.items():
                        if re.match(pattern, filename):
                            target_path = self.repo_root / target_dir / filename
                            self.actions.append(OrganizationAction(
                                action_type='move',
                                source_path=str(relative_file_path),
                                target_path=str(target_path.relative_to(self.repo_root)),
                                reason=f"Configuration file misplaced, should be in {target_dir}",
                                confidence=0.8
                            ))

    def _analyze_temporary_files(self):
        """Find and flag temporary files for cleanup."""
        temp_patterns = [r'.*\.tmp$', r'.*\.temp$', r'.*\.bak$', r'.*\.backup$', 
                        r'.*~$', r'.*\.swp$', r'.*\.swo$']
        
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.repo_root)
            
            # Skip temp directories (they're allowed to have temp files)
            if any(part in ['temp', 'tmp', 'cache', '.cache'] for part in relative_root.parts):
                continue
            
            # Skip git and other system directories
            if any(part.startswith('.') for part in relative_root.parts):
                continue
            
            for filename in files:
                for pattern in temp_patterns:
                    if re.match(pattern, filename):
                        file_path = root_path / filename
                        relative_file_path = file_path.relative_to(self.repo_root)
                        
                        self.actions.append(OrganizationAction(
                            action_type='delete',
                            source_path=str(relative_file_path),
                            reason="Temporary file cleanup",
                            confidence=0.7
                        ))

    def _create_missing_directories(self):
        """Create missing directory structure."""
        required_dirs = [
            'docs/architecture',
            'docs/migration', 
            'docs/guides',
            'docs/plans',
            'docs/summaries',
            'docs/reports',
            'docs/analysis',
            'docs/deployment',
            'docs/implementation',
            'docs/testing',
            'scripts/config/buck',
            'scripts/config/docker',
            'scripts/config/env',
            'scripts/config/linting',
            'scripts/config/testing',
            'scripts/config/git',
            'scripts/config/python',
            'scripts/config/security',
            'scripts/build',
            'scripts/validation',
            'scripts/cleanup',
            'scripts/development',
            'scripts/deployment',
            'scripts/maintenance',
            'scripts/setup',
            'scripts/testing',
        ]
        
        for dir_path in required_dirs:
            full_path = self.repo_root / dir_path
            if not full_path.exists():
                self.actions.append(OrganizationAction(
                    action_type='create_dir',
                    target_path=dir_path,
                    reason="Create missing directory structure",
                    confidence=1.0
                ))

    def execute_actions(self, force: bool = False) -> Tuple[int, int]:
        """Execute the organization actions."""
        if not self.actions:
            print("No organization actions needed.")
            return 0, 0
        
        executed = 0
        skipped = 0
        
        # Sort actions by confidence (highest first)
        sorted_actions = sorted(self.actions, key=lambda x: x.confidence, reverse=True)
        
        for action in sorted_actions:
            if not force and action.confidence < 0.8:
                print(f"⚠️  Skipping low-confidence action: {action.reason}")
                skipped += 1
                continue
            
            if self.dry_run:
                print(f"🔍 DRY RUN: Would {action.action_type}", end="")
                if action.source_path:
                    print(f" {action.source_path}", end="")
                if action.target_path:
                    print(f" → {action.target_path}", end="")
                print(f" ({action.reason})")
                executed += 1
                continue
            
            try:
                if action.action_type == 'move':
                    source = self.repo_root / action.source_path
                    target = self.repo_root / action.target_path
                    
                    # Create target directory if needed
                    target.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move file
                    shutil.move(str(source), str(target))
                    print(f"✅ Moved {action.source_path} → {action.target_path}")
                    
                elif action.action_type == 'delete':
                    source = self.repo_root / action.source_path
                    if source.exists():
                        source.unlink()
                        print(f"🗑️  Deleted {action.source_path}")
                    
                elif action.action_type == 'create_dir':
                    target = self.repo_root / action.target_path
                    target.mkdir(parents=True, exist_ok=True)
                    print(f"📁 Created directory {action.target_path}")
                
                executed += 1
                
            except Exception as e:
                print(f"❌ Failed to {action.action_type} {action.source_path}: {e}")
                skipped += 1
        
        return executed, skipped

    def generate_report(self) -> str:
        """Generate a report of organization actions."""
        if not self.actions:
            return "# Organization Report\n\nNo organization actions needed. Repository is well organized! ✅"
        
        report_lines = [
            "# Repository Organization Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Mode**: {'DRY RUN' if self.dry_run else 'EXECUTION'}",
            "",
            f"## Summary",
            f"- **Total Actions**: {len(self.actions)}",
            f"- **Move Operations**: {sum(1 for a in self.actions if a.action_type == 'move')}",
            f"- **Delete Operations**: {sum(1 for a in self.actions if a.action_type == 'delete')}",
            f"- **Directory Creations**: {sum(1 for a in self.actions if a.action_type == 'create_dir')}",
            "",
        ]
        
        # Group by action type
        by_type = {}
        for action in self.actions:
            if action.action_type not in by_type:
                by_type[action.action_type] = []
            by_type[action.action_type].append(action)
        
        for action_type, actions in by_type.items():
            report_lines.extend([
                f"## {action_type.title()} Operations",
                ""
            ])
            
            for action in sorted(actions, key=lambda x: x.confidence, reverse=True):
                confidence_icon = "🔴" if action.confidence < 0.6 else "🟡" if action.confidence < 0.8 else "🟢"
                report_lines.append(f"{confidence_icon} **{action.reason}** (confidence: {action.confidence:.1f})")
                
                if action.source_path:
                    report_lines.append(f"   - Source: `{action.source_path}`")
                
                if action.target_path:
                    report_lines.append(f"   - Target: `{action.target_path}`")
                
                report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated repository organization tool")
    parser.add_argument('--execute', action='store_true', help='Execute actions (default is dry run)')
    parser.add_argument('--force', action='store_true', help='Execute all actions regardless of confidence')
    parser.add_argument('--report', type=str, help='Write report to file')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.parent
    organizer = RepositoryOrganizer(repo_root, dry_run=not args.execute)
    
    if not args.quiet:
        mode = "EXECUTION" if args.execute else "DRY RUN"
        print(f"🔧 Repository organization tool ({mode})")
        print(f"📂 Repository: {repo_root}")
        print()
    
    # Analyze repository
    actions = organizer.analyze_repository()
    
    if not actions:
        if not args.quiet:
            print("✅ Repository is well organized! No actions needed.")
        return
    
    # Generate and optionally save report
    report = organizer.generate_report()
    
    if args.report:
        report_file = Path(args.report)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report)
        if not args.quiet:
            print(f"📄 Report written to: {report_file}")
    
    # Execute actions
    executed, skipped = organizer.execute_actions(force=args.force)
    
    if not args.quiet:
        print()
        print(f"📊 Results: {executed} executed, {skipped} skipped")
        
        if organizer.dry_run:
            print("💡 Run with --execute to apply changes")
        else:
            print("✅ Organization complete!")


if __name__ == "__main__":
    main()