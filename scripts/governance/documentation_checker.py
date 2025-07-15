#!/usr/bin/env python3
"""
Documentation Checker
Validates documentation completeness and quality across the repository.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set
import re


class DocumentationChecker:
    """Checks for documentation completeness and quality."""
    
    REQUIRED_REPO_DOCS = {
        "README.md": "Main repository documentation",
        "CONTRIBUTING.md": "Contribution guidelines", 
        "CHANGELOG.md": "Version history and changes",
        "LICENSE": "License information"
    }
    
    REQUIRED_PACKAGE_DOCS = {
        "README.md": "Package overview and usage",
        "CHANGELOG.md": "Package version history",
        "__init__.py": "Package initialization"
    }
    
    README_REQUIRED_SECTIONS = {
        "# ": "Package title",
        "## Overview": "Package overview section",
        "## Installation": "Installation instructions", 
        "## Usage": "Usage examples",
        "## Contributing": "Contribution guidelines",
        "## License": "License information"
    }
    
    DOCUMENTATION_PATTERNS = {
        r'```python': "Python code examples",
        r'```bash': "Shell command examples", 
        r'\[.*\]\(.*\)': "Internal/external links",
        r'!\[.*\]\(.*\)': "Images and diagrams"
    }
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.packages_dir = repo_root / "src" / "packages"
        self.violations: List[Dict] = []
        
    def check_repository_documentation(self) -> List[Dict]:
        """Check main repository documentation."""
        violations = []
        
        for doc_file, description in self.REQUIRED_REPO_DOCS.items():
            doc_path = self.repo_root / doc_file
            if not doc_path.exists():
                violations.append({
                    "type": "missing_repo_documentation",
                    "file": doc_file,
                    "description": description,
                    "severity": "error" if doc_file == "README.md" else "warning"
                })
        
        return violations
    
    def check_package_documentation(self) -> List[Dict]:
        """Check package-level documentation."""
        violations = []
        
        if not self.packages_dir.exists():
            return violations
            
        for package_path in self.packages_dir.glob("*"):
            if not package_path.is_dir():
                continue
                
            package_name = package_path.name
            
            # Check required package files
            for doc_file, description in self.REQUIRED_PACKAGE_DOCS.items():
                doc_path = package_path / doc_file
                if not doc_path.exists():
                    violations.append({
                        "type": "missing_package_documentation",
                        "package": package_name,
                        "file": doc_file,
                        "description": description,
                        "severity": "error" if doc_file == "README.md" else "warning"
                    })
            
            # Check README quality if it exists
            readme_path = package_path / "README.md"
            if readme_path.exists():
                readme_violations = self._check_readme_quality(readme_path, package_name)
                violations.extend(readme_violations)
        
        return violations
    
    def _check_readme_quality(self, readme_path: Path, package_name: str) -> List[Dict]:
        """Check README content quality."""
        violations = []
        
        try:
            content = readme_path.read_text(encoding='utf-8')
        except Exception as e:
            violations.append({
                "type": "readme_read_error",
                "package": package_name,
                "error": str(e),
                "severity": "error"
            })
            return violations
        
        # Check for required sections
        for section, description in self.README_REQUIRED_SECTIONS.items():
            if section not in content:
                violations.append({
                    "type": "missing_readme_section",
                    "package": package_name,
                    "section": section,
                    "description": description,
                    "severity": "warning"
                })
        
        # Check content quality
        violations.extend(self._check_content_quality(content, package_name))
        
        return violations
    
    def _check_content_quality(self, content: str, package_name: str) -> List[Dict]:
        """Check documentation content quality.""" 
        violations = []
        
        # Check minimum length
        if len(content) < 500:
            violations.append({
                "type": "readme_too_short",
                "package": package_name,
                "length": len(content),
                "severity": "warning",
                "description": "README is very short, consider adding more details"
            })
        
        # Check for code examples
        has_code_examples = any(
            re.search(pattern, content, re.MULTILINE) 
            for pattern in ['```python', '```bash', '```']
        )
        
        if not has_code_examples:
            violations.append({
                "type": "missing_code_examples",
                "package": package_name,
                "severity": "info",
                "description": "Consider adding code examples to improve usability"
            })
        
        # Check for broken internal links
        internal_links = re.findall(r'\[.*?\]\(((?!https?://).*?)\)', content)
        for link in internal_links:
            # Skip anchor links and external links
            if link.startswith('#') or link.startswith('http'):
                continue
                
            link_path = (Path(package_name).parent / link).resolve()
            if not link_path.exists():
                violations.append({
                    "type": "broken_internal_link",
                    "package": package_name,
                    "link": link,
                    "severity": "warning"
                })
        
        # Check for placeholder text
        placeholders = [
            "{{", "}}", "TODO", "FIXME", "XXX", 
            "Coming soon", "To be implemented"
        ]
        
        for placeholder in placeholders:
            if placeholder in content:
                violations.append({
                    "type": "placeholder_text",
                    "package": package_name,
                    "placeholder": placeholder,
                    "severity": "info",
                    "description": "Consider replacing placeholder text with actual content"
                })
        
        return violations
    
    def check_documentation_consistency(self) -> List[Dict]:
        """Check consistency across documentation."""
        violations = []
        
        # Get all README files
        readme_files = []
        
        # Main README
        main_readme = self.repo_root / "README.md"
        if main_readme.exists():
            readme_files.append(("main", main_readme))
        
        # Package READMEs
        for package_path in self.packages_dir.glob("*/README.md"):
            package_name = package_path.parent.name
            readme_files.append((package_name, package_path))
        
        # Check for consistent structure
        section_patterns = {}
        for name, readme_path in readme_files:
            try:
                content = readme_path.read_text(encoding='utf-8')
                sections = re.findall(r'^## (.+)$', content, re.MULTILINE)
                section_patterns[name] = set(sections)
            except Exception:
                continue
        
        # Check for common sections across packages
        if len(section_patterns) > 1:
            package_sections = [
                sections for name, sections in section_patterns.items() 
                if name != "main"
            ]
            
            if package_sections:
                common_sections = set.intersection(*package_sections)
                expected_sections = {"Overview", "Installation", "Usage"}
                
                missing_common = expected_sections - common_sections
                if missing_common:
                    violations.append({
                        "type": "inconsistent_documentation_structure",
                        "missing_sections": list(missing_common),
                        "severity": "info",
                        "description": "Some packages missing common documentation sections"
                    })
        
        return violations
    
    def check_changelog_format(self) -> List[Dict]:
        """Check changelog format compliance with Keep a Changelog."""
        violations = []
        
        # Check main repository changelog
        main_changelog = self.repo_root / "CHANGELOG.md"
        if main_changelog.exists():
            changelog_violations = self._validate_changelog_format(main_changelog, "main")
            violations.extend(changelog_violations)
        
        # Check package changelogs
        if self.packages_dir.exists():
            for package_path in self.packages_dir.glob("*"):
                if not package_path.is_dir():
                    continue
                    
                changelog_path = package_path / "CHANGELOG.md"
                if changelog_path.exists():
                    package_name = package_path.name
                    changelog_violations = self._validate_changelog_format(changelog_path, package_name)
                    violations.extend(changelog_violations)
        
        return violations
    
    def _validate_changelog_format(self, changelog_path: Path, context: str) -> List[Dict]:
        """Validate individual changelog format."""
        violations = []
        
        try:
            content = changelog_path.read_text(encoding='utf-8')
        except Exception as e:
            violations.append({
                "type": "changelog_read_error",
                "context": context,
                "error": str(e),
                "severity": "error"
            })
            return violations
        
        # Check for Keep a Changelog format requirements
        required_patterns = {
            r'# Changelog': "Main title should be '# Changelog'",
            r'## \[Unreleased\]': "Should have Unreleased section",
            r'### Added|### Changed|### Deprecated|### Removed|### Fixed|### Security': "Should use standard subsections",
            r'https://keepachangelog\.com': "Should reference Keep a Changelog"
        }
        
        for pattern, description in required_patterns.items():
            if not re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                violations.append({
                    "type": "changelog_format_violation",
                    "context": context,
                    "missing_pattern": pattern,
                    "description": description,
                    "severity": "warning"
                })
        
        # Check for semantic versioning patterns
        version_pattern = r'## \[(\d+\.\d+\.\d+)\]'
        versions = re.findall(version_pattern, content)
        
        if not versions and context != "main":
            violations.append({
                "type": "missing_version_entries",
                "context": context,
                "description": "Changelog should have versioned releases",
                "severity": "info"
            })
        
        return violations

    def check_api_documentation(self) -> List[Dict]:
        """Check for API documentation completeness.""" 
        violations = []
        
        # Look for Python files that might need documentation
        python_files = list(self.packages_dir.rglob("*.py"))
        
        # Count files with docstrings
        files_with_docs = 0
        total_files = 0
        
        for py_file in python_files:
            # Skip test files and __pycache__
            if any(skip in str(py_file) for skip in ['test_', '__pycache__', '.pyc']):
                continue
                
            total_files += 1
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check for module docstring
                if '"""' in content[:500] or "'''" in content[:500]:
                    files_with_docs += 1
                elif content.strip().startswith('#'):
                    # Consider files starting with comments as documented
                    files_with_docs += 1
                    
            except Exception:
                continue
        
        if total_files > 0:
            documentation_ratio = files_with_docs / total_files
            
            if documentation_ratio < 0.5:
                violations.append({
                    "type": "low_api_documentation_coverage",
                    "documented_files": files_with_docs,
                    "total_files": total_files,
                    "coverage": f"{documentation_ratio:.1%}",
                    "severity": "warning",
                    "description": "Consider adding docstrings to more Python files"
                })
        
        return violations
    
    def run_check(self) -> bool:
        """Run complete documentation check."""
        print("📚 Checking documentation completeness and quality...")
        
        violations = []
        violations.extend(self.check_repository_documentation())
        violations.extend(self.check_package_documentation())
        violations.extend(self.check_documentation_consistency())
        violations.extend(self.check_changelog_format())
        violations.extend(self.check_api_documentation())
        
        if violations:
            errors = [v for v in violations if v.get("severity") == "error"]
            warnings = [v for v in violations if v.get("severity") == "warning"]
            info = [v for v in violations if v.get("severity") == "info"]
            
            if errors:
                print(f"❌ Found {len(errors)} documentation error(s):")
                for error in errors:
                    self._print_violation(error)
            
            if warnings:
                print(f"⚠️  Found {len(warnings)} documentation warning(s):")
                for warning in warnings:
                    self._print_violation(warning)
            
            if info:
                print(f"💡 Found {len(info)} documentation suggestion(s):")
                for suggestion in info:
                    self._print_violation(suggestion)
            
            print("\\n💡 Run package structure enforcer to check for missing READMEs")
            
            return len(errors) == 0
        
        print("✅ Documentation is complete and well-structured!")
        return True
    
    def _print_violation(self, violation: Dict) -> None:
        """Print a formatted violation."""
        violation_type = violation.get("type", "unknown")
        
        if violation_type == "missing_repo_documentation":
            print(f"  • Missing {violation['file']}: {violation['description']}")
        elif violation_type == "missing_package_documentation":
            print(f"  • Package {violation['package']}: Missing {violation['file']}")
        elif violation_type == "missing_readme_section":
            print(f"  • {violation['package']}: Missing section '{violation['section']}'")
        elif violation_type == "readme_too_short":
            print(f"  • {violation['package']}: README too short ({violation['length']} chars)")
        elif violation_type == "broken_internal_link":
            print(f"  • {violation['package']}: Broken link '{violation['link']}'")
        elif violation_type == "placeholder_text":
            print(f"  • {violation['package']}: Contains placeholder '{violation['placeholder']}'")
        elif violation_type == "low_api_documentation_coverage":
            print(f"  • API documentation coverage: {violation['coverage']} ({violation['documented_files']}/{violation['total_files']})")
        elif violation_type == "changelog_format_violation":
            print(f"  • {violation['context']}: {violation['description']}")
        elif violation_type == "missing_version_entries":
            print(f"  • {violation['context']}: {violation['description']}")
        elif violation_type == "changelog_read_error":
            print(f"  • {violation['context']}: Error reading changelog - {violation['error']}")
        else:
            print(f"  • {violation}")


def main():
    """Main entry point for documentation checker."""
    repo_root = Path(__file__).parent.parent.parent
    checker = DocumentationChecker(repo_root)
    
    success = checker.run_check()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()