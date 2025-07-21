#!/usr/bin/env python3
"""
Pynomaly Terminology Replacement Script
======================================
Systematically replace pynomaly references with appropriate contextual terms.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


class TerminologyReplacer:
    """Replaces pynomaly terminology with contextually appropriate terms"""
    
    def __init__(self, repository_root: Path):
        self.repository_root = repository_root
        
        # Define replacement mappings based on context
        self.replacements = {
            # Project/Repository references
            "project": "project",
            "project": "project",
            "repository": "repository", 
            "repository": "repository",
            "codebase": "codebase",
            "codebase": "codebase",
            "repository": "repository",
            "repository": "repository",
            
            # Platform/System references  
            "Anomaly Detection Platform": "Anomaly Detection Platform",
            "anomaly detection platform": "anomaly detection platform",
            "anomaly detection system": "anomaly detection system",
            "anomaly detection system": "anomaly detection system",
            "anomaly detection framework": "anomaly detection framework",
            "anomaly detection framework": "anomaly detection framework",
            
            # Package/Module references
            "anomaly_detection package": "anomaly_detection package", 
            "anomaly_detection package": "anomaly_detection package",
            "anomaly_detection module": "anomaly_detection module",
            "anomaly_detection module": "anomaly_detection module",
            "anomaly_detection library": "anomaly_detection library",
            "anomaly_detection library": "anomaly_detection library",
            
            # CLI/Tool references
            "anomaly detection CLI": "anomaly detection CLI",
            "anomaly detection CLI": "anomaly detection CLI", 
            "anomaly-detector command": "anomaly-detector command",
            "anomaly-detector command": "anomaly-detector command",
            "anomaly detection tool": "anomaly detection tool",
            "anomaly detection tool": "anomaly detection tool",
            
            # Team/Organization references
            "Anomaly Detection Team": "Anomaly Detection Team",
            "anomaly detection team": "anomaly detection team", 
            "project developers": "project developers",
            "project developers": "project developers",
            
            # Application references
            "anomaly detection app": "anomaly detection app",
            "anomaly detection app": "anomaly detection app",
            "anomaly detection application": "anomaly detection application", 
            "anomaly detection application": "anomaly detection application",
            
            # Service references
            "anomaly detection service": "anomaly detection service",
            "anomaly detection service": "anomaly detection service",
            "anomaly detection API": "anomaly detection API",
            "anomaly detection API": "anomaly detection API",
            
            # Standalone references (context-dependent)
            "\\bPynomaly\\b": "Anomaly Detection Platform",  # Regex for word boundary
            "\\bpynomaly\\b": "anomaly_detection",          # For code identifiers
        }
        
        # File extensions to process
        self.text_extensions = {
            '.py', '.md', '.rst', '.txt', '.yml', '.yaml', '.json',
            '.toml', '.cfg', '.ini', '.sh', '.bat', '.ps1', '.dockerfile'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.venv', 'venv', 'env', '.env', 'environments'
        }
    
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed"""
        # Skip if in excluded directory
        for part in file_path.parts:
            if part in self.skip_dirs:
                return False
        
        # Skip if not a text file
        if file_path.suffix.lower() not in self.text_extensions:
            return False
            
        # Skip if it's a binary file or too large
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
                return False
        except OSError:
            return False
            
        return True
    
    def process_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Process a single file and return (changed, changes_made)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except (UnicodeDecodeError, OSError) as e:
            return False, [f"Could not read file: {e}"]
        
        modified_content = original_content
        changes_made = []
        
        # Apply replacements
        for old_term, new_term in self.replacements.items():
            if old_term.startswith('\\\\b') and old_term.endswith('\\\\b'):
                # Regex replacement for word boundaries
                pattern = old_term
                matches = re.findall(pattern, modified_content, re.IGNORECASE)
                if matches:
                    modified_content = re.sub(pattern, new_term, modified_content, flags=re.IGNORECASE)
                    changes_made.append(f"Replaced {len(matches)} occurrences of '{old_term.strip('\\\\b')}' with '{new_term}'")
            else:
                # Simple string replacement
                if old_term in modified_content:
                    count = modified_content.count(old_term)
                    modified_content = modified_content.replace(old_term, new_term)
                    changes_made.append(f"Replaced {count} occurrences of '{old_term}' with '{new_term}'")
        
        # Write back if changed
        if modified_content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                return True, changes_made
            except OSError as e:
                return False, [f"Could not write file: {e}"]
        
        return False, []
    
    def process_repository(self) -> Dict[str, List[str]]:
        """Process all files in the repository"""
        results = {
            'processed': [],
            'changed': [],
            'errors': []
        }
        
        print("🔍 Scanning repository for pynomaly references...")
        
        # Walk through all files
        for file_path in self.repository_root.rglob('*'):
            if not file_path.is_file():
                continue
                
            if not self.should_process_file(file_path):
                continue
            
            relative_path = file_path.relative_to(self.repository_root)
            results['processed'].append(str(relative_path))
            
            try:
                changed, changes = self.process_file(file_path)
                if changed:
                    results['changed'].append({
                        'file': str(relative_path),
                        'changes': changes
                    })
                    print(f"✅ {relative_path}")
                    for change in changes:
                        print(f"   • {change}")
                
            except Exception as e:
                error_msg = f"Error processing {relative_path}: {e}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        return results
    
    def generate_report(self, results: Dict[str, List[str]]) -> str:
        """Generate a summary report"""
        changed_files = len(results['changed'])
        total_processed = len(results['processed'])
        errors = len(results['errors'])
        
        report = f"""
Pynomaly Terminology Replacement Report
=======================================

## Summary
- Files processed: {total_processed}
- Files changed: {changed_files}
- Errors: {errors}

## Changed Files
"""
        
        for change in results['changed']:
            report += f"\n### {change['file']}\n"
            for change_detail in change['changes']:
                report += f"- {change_detail}\n"
        
        if results['errors']:
            report += "\n## Errors\n"
            for error in results['errors']:
                report += f"- {error}\n"
        
        return report


def main():
    """Main entry point"""
    repository_root = Path(__file__).parent.parent
    
    print("🔄 Starting pynomaly terminology replacement...")
    print(f"📁 Repository root: {repository_root}")
    
    replacer = TerminologyReplacer(repository_root)
    results = replacer.process_repository()
    
    # Generate report
    report = replacer.generate_report(results)
    
    # Save report
    report_path = repository_root / "docs" / "terminology_replacement_report.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Summary:")
    print(f"   Files processed: {len(results['processed'])}")
    print(f"   Files changed: {len(results['changed'])}")
    print(f"   Errors: {len(results['errors'])}")
    print(f"   Report saved: {report_path}")
    
    if results['errors']:
        print(f"\n⚠️  {len(results['errors'])} errors occurred during processing")
        return 1
    
    print("\n✅ Pynomaly terminology replacement completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())