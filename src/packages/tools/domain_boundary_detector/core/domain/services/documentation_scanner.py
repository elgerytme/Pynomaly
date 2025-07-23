"""Documentation scanner for detecting domain boundary violations in documentation files."""

import re
import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Pattern
from dataclasses import dataclass, field
import yaml
from fnmatch import fnmatch


@dataclass
class DocumentationViolation:
    """Represents a documentation domain boundary violation."""
    file_path: str
    line_number: int
    line_content: str
    violation_type: str
    message: str
    severity: str
    rule_name: str
    pattern: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.severity.upper()}: {self.message} in {self.file_path}:{self.line_number}"


@dataclass
class DocumentationRule:
    """Represents a documentation scanning rule."""
    name: str
    description: str
    severity: str
    scope: str
    patterns: List[Dict[str, Any]]
    exceptions: List[str] = field(default_factory=list)

    def matches_scope(self, file_path: str) -> bool:
        """Check if the file path matches this rule's scope."""
        if self.scope == "**/*.md":
            return file_path.endswith('.md') or file_path.endswith('.rst')
        
        # Convert glob pattern to regex for more complex matching
        scope_parts = self.scope.split('/')
        path_parts = file_path.replace('\\', '/').split('/')
        
        return self._match_glob_pattern(scope_parts, path_parts)
    
    def _match_glob_pattern(self, pattern_parts: List[str], path_parts: List[str]) -> bool:
        """Match glob pattern against path parts."""
        if not pattern_parts:
            return not path_parts
        
        if pattern_parts[0] == '**':
            # Recursive wildcard
            if len(pattern_parts) == 1:
                return True
            
            # Try matching remaining pattern at each position
            for i in range(len(path_parts) + 1):
                if self._match_glob_pattern(pattern_parts[1:], path_parts[i:]):
                    return True
            return False
        
        if not path_parts:
            return False
        
        if pattern_parts[0] == '*':
            # Single wildcard
            return self._match_glob_pattern(pattern_parts[1:], path_parts[1:])
        
        if fnmatch(path_parts[0], pattern_parts[0]):
            return self._match_glob_pattern(pattern_parts[1:], path_parts[1:])
        
        return False


class DocumentationScanner:
    """Scanner for detecting domain boundary violations in documentation files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the documentation scanner."""
        self.config_path = config_path or '.domain-boundaries.yaml'
        self.rules: List[DocumentationRule] = []
        self.violations: List[DocumentationViolation] = []
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            documentation_config = config.get('documentation', {})
            rules_config = documentation_config.get('rules', [])
            exceptions_config = documentation_config.get('exceptions', [])
            
            # Create exception file set for quick lookup
            exception_files = {exc.get('file', '') for exc in exceptions_config}
            
            for rule_config in rules_config:
                rule = DocumentationRule(
                    name=rule_config['name'],
                    description=rule_config['description'],
                    severity=rule_config['severity'],
                    scope=rule_config['scope'],
                    patterns=rule_config['patterns'],
                    exceptions=list(exception_files)
                )
                self.rules.append(rule)
                
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            self._load_default_rules()
    
    def _load_default_rules(self) -> None:
        """Load default rules if configuration file is not available."""
        default_rules = [
            DocumentationRule(
                name="no_cross_package_references_in_package_docs",
                description="Package documentation must not reference other packages",
                severity="critical",
                scope="src/packages/*/docs/**/*.md",
                patterns=[
                    {
                        "pattern": r"from\s+(?!\.{1,2})([a-zA-Z_][\w_]*)\.*import",
                        "message": "Package documentation must not reference other packages - use relative imports"
                    },
                    {
                        "pattern": r"(?<!\.)\\b(anomaly_detection|mlops|data_science|enterprise_\w+|neuro_symbolic|machine_learning)\\b(?!\w)",
                        "message": "Package documentation must not reference other package names"
                    }
                ]
            ),
            DocumentationRule(
                name="no_package_specific_refs_in_repo_docs",
                description="Repository documentation must not reference specific packages",
                severity="critical",
                scope="docs/**/*.md",
                patterns=[
                    {
                        "pattern": r"(?<!\.)\\b(anomaly_detection|mlops|data_science|enterprise_\w+|neuro_symbolic|machine_learning)\\b(?!\w)",
                        "message": "Repository documentation must not reference specific packages - keep it generic"
                    },
                    {
                        "pattern": r"src/packages/[^\s/]+/[^\s]*",
                        "message": "Repository documentation must not reference specific package paths"
                    }
                ]
            )
        ]
        self.rules = default_rules
    
    def scan_file(self, file_path: str) -> List[DocumentationViolation]:
        """Scan a single documentation file for violations."""
        violations = []
        
        # Skip if file is in exceptions
        normalized_path = file_path.replace('\\', '/')
        for rule in self.rules:
            if normalized_path in rule.exceptions:
                continue
                
            if not rule.matches_scope(normalized_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                violations.extend(self._scan_file_content(file_path, lines, rule))
            except Exception as e:
                print(f"Warning: Could not scan file {file_path}: {e}")
        
        return violations
    
    def _scan_file_content(self, file_path: str, lines: List[str], rule: DocumentationRule) -> List[DocumentationViolation]:
        """Scan file content for violations according to a specific rule."""
        violations = []
        in_code_block = False
        code_block_language = ""
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Track code blocks
            if line_stripped.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_block_language = line_stripped[3:].strip().lower()
                else:
                    in_code_block = False
                    code_block_language = ""
                continue
            
            # Check for violations in line
            for pattern_config in rule.patterns:
                pattern_str = pattern_config['pattern']
                message = pattern_config.get('message', f"Violation of rule: {rule.name}")
                exceptions = pattern_config.get('exceptions', [])
                
                # Skip if line matches any exception
                if any(exc in line for exc in exceptions):
                    continue
                
                # Special handling for import patterns - only check code blocks
                if 'import' in pattern_str.lower() and not in_code_block:
                    continue
                
                # Special handling for package name references in package docs
                if rule.name == "no_cross_package_references_in_package_docs":
                    # Extract package name from file path for exclude_self logic
                    package_name = self._extract_package_name_from_path(file_path)
                    if package_name and pattern_config.get('exclude_self', False):
                        # Skip if the reference is to the same package
                        if package_name in line.lower():
                            # Check if it's actually a self-reference
                            if self._is_self_reference(line, package_name):
                                continue
                
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                    matches = pattern.finditer(line)
                    
                    for match in matches:
                        suggestion = self._generate_suggestion(pattern_str, match.group(0), file_path, rule.name)
                        
                        violation = DocumentationViolation(
                            file_path=file_path,
                            line_number=line_num,
                            line_content=line.rstrip(),
                            violation_type=rule.name,
                            message=message,
                            severity=rule.severity,
                            rule_name=rule.name,
                            pattern=pattern_str,
                            suggestion=suggestion
                        )
                        violations.append(violation)
                        
                except re.error as e:
                    print(f"Warning: Invalid regex pattern '{pattern_str}': {e}")
        
        return violations
    
    def _extract_package_name_from_path(self, file_path: str) -> Optional[str]:
        """Extract package name from file path."""
        normalized_path = file_path.replace('\\', '/')
        if 'src/packages/' in normalized_path:
            parts = normalized_path.split('src/packages/')[1].split('/')
            if len(parts) >= 2:
                # Return the last part of the package path (e.g., 'mlops' from 'ai/mlops')
                return parts[1]
        return None
    
    def _is_self_reference(self, line: str, package_name: str) -> bool:
        """Check if a package reference in a line is actually a self-reference."""
        # Simple heuristic: if the line contains relative import indicators or
        # is clearly referring to the same package in context
        return (
            'from .' in line or 
            'from ..' in line or
            f'from {package_name}' in line.lower() or
            f'import {package_name}' in line.lower()
        )
    
    def _generate_suggestion(self, pattern: str, match: str, file_path: str, rule_name: str) -> Optional[str]:
        """Generate a suggestion for fixing the violation."""
        suggestions = {
            "no_cross_package_references_in_package_docs": {
                "import": "Use relative imports like 'from .module import Class' instead",
                "package_name": "Remove references to other packages or make them generic"
            },
            "no_package_specific_refs_in_repo_docs": {
                "package_name": "Use generic terms instead of specific package names",
                "path": "Use generic path patterns instead of specific package paths"
            },
            "no_monorepo_imports": {
                "import": "Use relative imports like 'from .module import Class' instead"
            },
            "no_absolute_package_imports_in_package_docs": {
                "import": "Use relative imports like 'from .module import Class' instead"
            }
        }
        
        rule_suggestions = suggestions.get(rule_name, {})
        
        if 'import' in pattern.lower():
            return rule_suggestions.get('import')
        elif 'src/packages' in pattern:
            return rule_suggestions.get('path')
        else:
            return rule_suggestions.get('package_name')
    
    def scan_directory(self, directory_path: str) -> List[DocumentationViolation]:
        """Scan all documentation files in a directory recursively."""
        violations = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.md', '.rst')):
                    file_path = os.path.join(root, file)
                    violations.extend(self.scan_file(file_path))
        
        return violations
    
    def scan_repository(self, repository_path: str = ".") -> List[DocumentationViolation]:
        """Scan the entire repository for documentation violations."""
        self.violations = []
        
        # Scan repository-level documentation
        docs_path = os.path.join(repository_path, "docs")
        if os.path.exists(docs_path):
            self.violations.extend(self.scan_directory(docs_path))
        
        # Scan package-level documentation
        packages_path = os.path.join(repository_path, "src", "packages")
        if os.path.exists(packages_path):
            for domain in os.listdir(packages_path):
                domain_path = os.path.join(packages_path, domain)
                if os.path.isdir(domain_path):
                    for package in os.listdir(domain_path):
                        package_path = os.path.join(domain_path, package)
                        if os.path.isdir(package_path):
                            # Scan package docs directory
                            docs_path = os.path.join(package_path, "docs")
                            if os.path.exists(docs_path):
                                self.violations.extend(self.scan_directory(docs_path))
                            
                            # Scan README files
                            readme_path = os.path.join(package_path, "README.md")
                            if os.path.exists(readme_path):
                                self.violations.extend(self.scan_file(readme_path))
        
        return self.violations
    
    def get_violations_by_severity(self) -> Dict[str, List[DocumentationViolation]]:
        """Group violations by severity."""
        violations_by_severity = {}
        for violation in self.violations:
            if violation.severity not in violations_by_severity:
                violations_by_severity[violation.severity] = []
            violations_by_severity[violation.severity].append(violation)
        return violations_by_severity
    
    def get_violations_by_rule(self) -> Dict[str, List[DocumentationViolation]]:
        """Group violations by rule name."""
        violations_by_rule = {}
        for violation in self.violations:
            if violation.rule_name not in violations_by_rule:
                violations_by_rule[violation.rule_name] = []
            violations_by_rule[violation.rule_name].append(violation)
        return violations_by_rule
    
    def generate_report(self, format_type: str = "console") -> str:
        """Generate a report of documentation violations."""
        if format_type == "console":
            return self._generate_console_report()
        elif format_type == "json":
            return self._generate_json_report()
        elif format_type == "markdown":
            return self._generate_markdown_report()
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _generate_console_report(self) -> str:
        """Generate a console-formatted report."""
        if not self.violations:
            return "✅ No documentation domain boundary violations found!"
        
        violations_by_severity = self.get_violations_by_severity()
        
        report = ["Documentation Domain Boundary Scan Report", "=" * 45, ""]
        
        # Summary
        critical_count = len(violations_by_severity.get('critical', []))
        warning_count = len(violations_by_severity.get('warning', []))
        info_count = len(violations_by_severity.get('info', []))
        
        report.append(f"❌ VIOLATIONS FOUND: {len(self.violations)}")
        report.append(f"  ● Critical: {critical_count}")
        report.append(f"  ● Warning: {warning_count}")
        report.append(f"  ● Info: {info_count}")
        report.append("")
        
        # Detailed violations by severity
        for severity in ['critical', 'warning', 'info']:
            violations = violations_by_severity.get(severity, [])
            if not violations:
                continue
            
            report.append(f"{severity.upper()} VIOLATIONS:")
            report.append("-" * 40)
            
            for i, violation in enumerate(violations, 1):
                report.append(f"{i}. {violation.message}")
                report.append(f"   File: {violation.file_path}:{violation.line_number}")
                report.append(f"   Content: {violation.line_content[:100]}...")
                if violation.suggestion:
                    report.append(f"   Suggestion: {violation.suggestion}")
                report.append("")
        
        return "\n".join(report)
    
    def _generate_json_report(self) -> str:
        """Generate a JSON-formatted report."""
        import json
        
        violations_data = []
        for violation in self.violations:
            violations_data.append({
                'file_path': violation.file_path,
                'line_number': violation.line_number,
                'line_content': violation.line_content,
                'violation_type': violation.violation_type,
                'message': violation.message,
                'severity': violation.severity,
                'rule_name': violation.rule_name,
                'pattern': violation.pattern,
                'suggestion': violation.suggestion
            })
        
        report_data = {
            'summary': {
                'total_violations': len(self.violations),
                'critical': len([v for v in self.violations if v.severity == 'critical']),
                'warning': len([v for v in self.violations if v.severity == 'warning']),
                'info': len([v for v in self.violations if v.severity == 'info'])
            },
            'violations': violations_data
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_markdown_report(self) -> str:
        """Generate a Markdown-formatted report."""
        if not self.violations:
            return "# Documentation Domain Boundary Report\n\n✅ No violations found!"
        
        violations_by_severity = self.get_violations_by_severity()
        
        report = ["# Documentation Domain Boundary Report", ""]
        
        # Summary
        critical_count = len(violations_by_severity.get('critical', []))
        warning_count = len(violations_by_severity.get('warning', []))
        info_count = len(violations_by_severity.get('info', []))
        
        report.append(f"**Total Violations:** {len(self.violations)}")
        report.append(f"- Critical: {critical_count}")
        report.append(f"- Warning: {warning_count}")
        report.append(f"- Info: {info_count}")
        report.append("")
        
        # Detailed violations
        for severity in ['critical', 'warning', 'info']:
            violations = violations_by_severity.get(severity, [])
            if not violations:
                continue
            
            report.append(f"## {severity.title()} Violations")
            report.append("")
            
            for i, violation in enumerate(violations, 1):
                report.append(f"### {i}. {violation.message}")
                report.append("")
                report.append(f"**File:** `{violation.file_path}:{violation.line_number}`")
                report.append("")
                report.append("**Content:**")
                report.append("```")
                report.append(violation.line_content)
                report.append("```")
                report.append("")
                
                if violation.suggestion:
                    report.append(f"**Suggestion:** {violation.suggestion}")
                    report.append("")
        
        return "\n".join(report)