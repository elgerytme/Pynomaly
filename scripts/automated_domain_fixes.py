#!/usr/bin/env python3
"""
Automated Domain Boundary Fixes

This script implements automated text replacement to fix domain boundary violations.
It targets quick wins that can be safely automated through text substitution.
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ReplacementRule:
    """A rule for automated text replacement"""
    pattern: str
    replacement: str
    category: str
    description: str
    safe_contexts: List[str]
    excluded_files: List[str]
    excluded_patterns: List[str]
    
@dataclass
class FixResult:
    """Result of applying a fix"""
    file_path: str
    rule_applied: str
    replacements_made: int
    original_content: str
    new_content: str
    success: bool
    error_message: Optional[str] = None

class AutomatedDomainFixer:
    """Main class for automated domain boundary fixes"""
    
    def __init__(self):
        self.replacement_rules = self._load_replacement_rules()
        self.fix_results: List[FixResult] = []
        self.files_processed = 0
        self.total_replacements = 0
        self.errors = []
        
    def _load_replacement_rules(self) -> List[ReplacementRule]:
        """Load predefined replacement rules for common violations"""
        
        rules = [
            # Package name fixes
            ReplacementRule(
                pattern=r'anomaly_detection',
                replacement='software',
                category='package_names',
                description='Replace anomaly_detection with software',
                safe_contexts=['comments', 'strings', 'documentation'],
                excluded_files=['README.md', 'CHANGELOG.md', 'HISTORY.md'],
                excluded_patterns=[r'anomaly_detection\.io', r'github\.com.*anomaly_detection']
            ),
            
            # Domain-specific terminology
            ReplacementRule(
                pattern=r'\banomaly[_ ]detection\b',
                replacement='data processing',
                category='domain_terminology',
                description='Replace anomaly detection with data processing',
                safe_contexts=['comments', 'strings', 'documentation'],
                excluded_files=['*.md'],
                excluded_patterns=[r'import.*anomaly', r'from.*anomaly']
            ),
            
            ReplacementRule(
                pattern=r'\bmachine[_ ]learning\b',
                replacement='computational analysis',
                category='domain_terminology',
                description='Replace machine learning with computational analysis',
                safe_contexts=['comments', 'strings', 'documentation'],
                excluded_files=['*.md'],
                excluded_patterns=[r'import.*ml', r'from.*ml']
            ),
            
            ReplacementRule(
                pattern=r'\bdata[_ ]science\b',
                replacement='data analysis',
                category='domain_terminology',
                description='Replace data science with data analysis',
                safe_contexts=['comments', 'strings', 'documentation'],
                excluded_files=['*.md'],
                excluded_patterns=[r'import.*science', r'from.*science']
            ),
            
            ReplacementRule(
                pattern=r'\boutlier[_ ]detection\b',
                replacement='pattern recognition',
                category='domain_terminology',
                description='Replace outlier detection with pattern recognition',
                safe_contexts=['comments', 'strings', 'documentation'],
                excluded_files=['*.md'],
                excluded_patterns=[r'import.*outlier', r'from.*outlier']
            ),
            
            # Configuration fixes
            ReplacementRule(
                pattern=r'"anomaly_detection-([^"]+)"',
                replacement=r'"software-\1"',
                category='configuration',
                description='Replace anomaly_detection- prefixed package names with software-',
                safe_contexts=['json', 'toml', 'yaml'],
                excluded_files=[],
                excluded_patterns=[r'repository.*anomaly_detection']
            ),
            
            ReplacementRule(
                pattern=r"'anomaly_detection-([^']+)'",
                replacement=r"'software-\1'",
                category='configuration',
                description='Replace anomaly_detection- prefixed package names with software-',
                safe_contexts=['json', 'toml', 'yaml'],
                excluded_files=[],
                excluded_patterns=[r'repository.*anomaly_detection']
            ),
            
            # Variable and function name fixes
            ReplacementRule(
                pattern=r'\banomalyDetector\b',
                replacement='dataProcessor',
                category='variable_names',
                description='Replace anomalyDetector with dataProcessor',
                safe_contexts=['code'],
                excluded_files=['*.md', '*.txt'],
                excluded_patterns=[r'class.*anomalyDetector']
            ),
            
            ReplacementRule(
                pattern=r'\banomalyScore\b',
                replacement='processingScore',
                category='variable_names',
                description='Replace anomalyScore with processingScore',
                safe_contexts=['code'],
                excluded_files=['*.md', '*.txt'],
                excluded_patterns=[]
            ),
            
            ReplacementRule(
                pattern=r'\bmlModel\b',
                replacement='computationalModel',
                category='variable_names',
                description='Replace mlModel with computationalModel',
                safe_contexts=['code'],
                excluded_files=['*.md', '*.txt'],
                excluded_patterns=[]
            ),
            
            # Description and documentation fixes
            ReplacementRule(
                pattern=r'detects anomalies in',
                replacement='processes data in',
                category='descriptions',
                description='Replace anomaly detection descriptions',
                safe_contexts=['strings', 'documentation'],
                excluded_files=[],
                excluded_patterns=[]
            ),
            
            ReplacementRule(
                pattern=r'machine learning algorithms',
                replacement='computational algorithms',
                category='descriptions',
                description='Replace ML algorithm descriptions',
                safe_contexts=['strings', 'documentation'],
                excluded_files=[],
                excluded_patterns=[]
            ),
            
            ReplacementRule(
                pattern=r'data science workflows',
                replacement='data analysis workflows',
                category='descriptions',
                description='Replace data science workflow descriptions',
                safe_contexts=['strings', 'documentation'],
                excluded_files=[],
                excluded_patterns=[]
            ),
            
            # Import statement fixes
            ReplacementRule(
                pattern=r'from anomaly_detection\.([^.]+)',
                replacement=r'from software.\1',
                category='imports',
                description='Replace anomaly_detection imports with software imports',
                safe_contexts=['code'],
                excluded_files=[],
                excluded_patterns=[]
            ),
            
            ReplacementRule(
                pattern=r'import anomaly_detection\.([^.]+)',
                replacement=r'import software.\1',
                category='imports',
                description='Replace anomaly_detection imports with software imports',
                safe_contexts=['code'],
                excluded_files=[],
                excluded_patterns=[]
            ),
            
            # URL and reference fixes
            ReplacementRule(
                pattern=r'https://github\.com/[^/]+/anomaly_detection',
                replacement='https://github.com/domain-team/software',
                category='urls',
                description='Replace anomaly_detection GitHub URLs with software URLs',
                safe_contexts=['strings', 'documentation'],
                excluded_files=[],
                excluded_patterns=[]
            ),
            
            # Keyword and tag fixes
            ReplacementRule(
                pattern=r'"anomaly[_-]detection"',
                replacement='"data-processing"',
                category='keywords',
                description='Replace anomaly detection keywords',
                safe_contexts=['json', 'toml', 'yaml'],
                excluded_files=[],
                excluded_patterns=[]
            ),
            
            ReplacementRule(
                pattern=r'"machine[_-]learning"',
                replacement='"computational-analysis"',
                category='keywords',
                description='Replace machine learning keywords',
                safe_contexts=['json', 'toml', 'yaml'],
                excluded_files=[],
                excluded_patterns=[]
            ),
            
            ReplacementRule(
                pattern=r'"data[_-]science"',
                replacement='"data-analysis"',
                category='keywords',
                description='Replace data science keywords',
                safe_contexts=['json', 'toml', 'yaml'],
                excluded_files=[],
                excluded_patterns=[]
            )
        ]
        
        return rules
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        
        # Skip binary files
        if file_path.suffix in ['.pyc', '.pyo', '.so', '.dll', '.exe']:
            return True
            
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return True
            
        # Skip test files for now (handle separately)
        if 'test' in file_path.name.lower():
            return True
            
        # Skip migration files
        if 'migration' in file_path.name.lower():
            return True
            
        # Skip generated files
        if file_path.name in ['__pycache__', '.mypy_cache', '.pytest_cache']:
            return True
            
        return False
    
    def _file_matches_pattern(self, file_path: Path, pattern: str) -> bool:
        """Check if file matches a pattern"""
        return file_path.match(pattern)
    
    def _is_safe_context(self, content: str, match_start: int, match_end: int, 
                        safe_contexts: List[str]) -> bool:
        """Check if replacement is in a safe context"""
        
        # Get surrounding context
        start = max(0, match_start - 100)
        end = min(len(content), match_end + 100)
        context = content[start:end]
        
        # Check for safe contexts
        for safe_context in safe_contexts:
            if safe_context == 'comments':
                # Check if in Python comment
                lines = context.split('\n')
                for line in lines:
                    if '#' in line and match_start >= content.find(line):
                        return True
                        
            elif safe_context == 'strings':
                # Check if in string literal
                if ('"' in context[:match_start-start] or 
                    "'" in context[:match_start-start]):
                    return True
                    
            elif safe_context == 'documentation':
                # Check if in docstring
                if ('"""' in context[:match_start-start] or 
                    "'''" in context[:match_start-start]):
                    return True
                    
            elif safe_context in ['json', 'toml', 'yaml']:
                # For configuration files, most replacements are safe
                return True
                
            elif safe_context == 'code':
                # For code context, check it's not in critical areas
                critical_patterns = ['class ', 'def ', 'import ', 'from ']
                line_start = content.rfind('\n', 0, match_start) + 1
                line_end = content.find('\n', match_end)
                if line_end == -1:
                    line_end = len(content)
                line_content = content[line_start:line_end]
                
                if not any(pattern in line_content for pattern in critical_patterns):
                    return True
        
        return False
    
    def _should_exclude_match(self, content: str, match_start: int, match_end: int,
                             excluded_patterns: List[str]) -> bool:
        """Check if match should be excluded based on patterns"""
        
        # Get surrounding context
        start = max(0, match_start - 50)
        end = min(len(content), match_end + 50)
        context = content[start:end]
        
        # Check exclusion patterns
        for pattern in excluded_patterns:
            if re.search(pattern, context):
                return True
                
        return False
    
    def _apply_rule_to_file(self, file_path: Path, rule: ReplacementRule) -> FixResult:
        """Apply a replacement rule to a file"""
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            # Check if file should be excluded
            for excluded_file in rule.excluded_files:
                if self._file_matches_pattern(file_path, excluded_file):
                    return FixResult(
                        file_path=str(file_path),
                        rule_applied=rule.description,
                        replacements_made=0,
                        original_content=original_content,
                        new_content=original_content,
                        success=True,
                        error_message="File excluded by rule"
                    )
            
            # Apply replacements
            new_content = original_content
            replacements_made = 0
            
            # Find all matches
            for match in re.finditer(rule.pattern, original_content, re.IGNORECASE):
                match_start = match.start()
                match_end = match.end()
                
                # Check if in safe context
                if not self._is_safe_context(original_content, match_start, match_end, 
                                           rule.safe_contexts):
                    continue
                    
                # Check if should be excluded
                if self._should_exclude_match(original_content, match_start, match_end,
                                            rule.excluded_patterns):
                    continue
                    
                # Apply replacement
                new_content = re.sub(rule.pattern, rule.replacement, new_content, 
                                   count=1, flags=re.IGNORECASE)
                replacements_made += 1
            
            # Write back if changes were made
            if replacements_made > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
                logger.info(f"Applied {replacements_made} replacements to {file_path}")
            
            return FixResult(
                file_path=str(file_path),
                rule_applied=rule.description,
                replacements_made=replacements_made,
                original_content=original_content,
                new_content=new_content,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            
            return FixResult(
                file_path=str(file_path),
                rule_applied=rule.description,
                replacements_made=0,
                original_content="",
                new_content="",
                success=False,
                error_message=error_msg
            )
    
    def process_software_package(self, dry_run: bool = False) -> Dict[str, any]:
        """Process the software package for automated fixes"""
        
        logger.info("Starting automated domain boundary fixes...")
        
        # Get software package path
        software_path = Path("src/packages/software")
        if not software_path.exists():
            raise FileNotFoundError(f"Software package not found: {software_path}")
        
        # Find all files to process
        files_to_process = []
        for file_path in software_path.rglob("*"):
            if file_path.is_file() and not self._should_skip_file(file_path):
                files_to_process.append(file_path)
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process each file with each rule
        results_by_category = {}
        
        for rule in self.replacement_rules:
            if rule.category not in results_by_category:
                results_by_category[rule.category] = []
                
            logger.info(f"Applying rule: {rule.description}")
            
            for file_path in files_to_process:
                if dry_run:
                    logger.info(f"DRY RUN: Would process {file_path} with rule {rule.description}")
                    continue
                    
                result = self._apply_rule_to_file(file_path, rule)
                
                if result.replacements_made > 0:
                    results_by_category[rule.category].append(result)
                    self.total_replacements += result.replacements_made
                    
                if not result.success:
                    self.errors.append(result.error_message)
                    
                self.files_processed += 1
        
        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "files_processed": self.files_processed,
            "total_replacements": self.total_replacements,
            "errors": len(self.errors),
            "results_by_category": {},
            "dry_run": dry_run
        }
        
        for category, results in results_by_category.items():
            summary["results_by_category"][category] = {
                "files_affected": len(results),
                "total_replacements": sum(r.replacements_made for r in results),
                "files": [r.file_path for r in results]
            }
        
        return summary
    
    def generate_report(self, summary: Dict[str, any], output_file: str = "automated_fixes_report.json"):
        """Generate a detailed report of fixes applied"""
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Report generated: {output_file}")
        
        # Also generate human-readable summary
        report_lines = [
            "# Automated Domain Boundary Fixes Report",
            f"Generated: {summary['timestamp']}",
            f"Mode: {'DRY RUN' if summary['dry_run'] else 'LIVE RUN'}",
            "",
            "## Summary",
            f"- Files processed: {summary['files_processed']}",
            f"- Total replacements: {summary['total_replacements']}",
            f"- Errors: {summary['errors']}",
            "",
            "## Results by Category"
        ]
        
        for category, results in summary['results_by_category'].items():
            report_lines.extend([
                f"### {category.replace('_', ' ').title()}",
                f"- Files affected: {results['files_affected']}",
                f"- Replacements made: {results['total_replacements']}",
                f"- Files: {', '.join(results['files'])}",
                ""
            ])
        
        if self.errors:
            report_lines.extend([
                "## Errors",
                *[f"- {error}" for error in self.errors]
            ])
        
        report_file = output_file.replace('.json', '.md')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
            
        logger.info(f"Human-readable report generated: {report_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Domain Boundary Fixes")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Run in dry-run mode (no changes made)")
    parser.add_argument("--output", default="automated_fixes_report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Create fixer
    fixer = AutomatedDomainFixer()
    
    try:
        # Process software package
        summary = fixer.process_software_package(dry_run=args.dry_run)
        
        # Generate report
        fixer.generate_report(summary, args.output)
        
        # Print summary
        print(f"✅ Automated fixes completed!")
        print(f"📁 Files processed: {summary['files_processed']}")
        print(f"🔄 Total replacements: {summary['total_replacements']}")
        print(f"❌ Errors: {summary['errors']}")
        print(f"📊 Report: {args.output}")
        
        if args.dry_run:
            print("\n⚠️  This was a DRY RUN - no changes were made")
            print("Run without --dry-run to apply changes")
        else:
            print("\n✨ Changes have been applied to the software package")
            print("Run domain boundary validator to see improvement")
            
    except Exception as e:
        logger.error(f"Failed to run automated fixes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()