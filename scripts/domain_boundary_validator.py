#!/usr/bin/env python3
"""
Domain Boundary Validator

Validates that packages maintain proper domain boundaries and don't contain
references to other domains. This enforces clean architecture principles.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class DomainViolation:
    """Represents a domain boundary violation"""
    file_path: str
    line_number: int
    line_content: str
    violation_type: str
    prohibited_term: str
    domain_package: str
    severity: str = "error"  # error, warning, info

@dataclass
class DomainRule:
    """Represents a domain boundary rule"""
    domain_package: str
    prohibited_terms: List[str]
    allowed_exceptions: List[str]
    file_patterns: List[str]
    description: str

class DomainBoundaryValidator:
    """Validates domain boundaries across packages"""
    
    def __init__(self):
        self.violations: List[DomainViolation] = []
        self.rules = self._load_domain_rules()
    
    def _load_domain_rules(self) -> Dict[str, DomainRule]:
        """Load domain boundary rules"""
        return {
            "software": DomainRule(
                domain_package="software",
                prohibited_terms=[
                    # Core domain violations
                    "pynomaly", "anomaly", "anomaly_detection", "outlier", "detection",
                    "automl", "ensemble", "explainability", "explainable_ai",
                    "drift", "model", "training", "dataset", "preprocessing",
                    "contamination", "threshold", "severity", "confidence_interval",
                    "hyperparameters", "model_metrics", "performance_metrics",
                    "active_learning", "continuous_learning", "streaming_session",
                    
                    # ML/AI violations
                    "machine_learning", "artificial_intelligence", "deep_learning",
                    "neural_network", "sklearn", "scikit_learn", "tensorflow",
                    "pytorch", "keras", "xgboost", "lightgbm", "catboost",
                    "random_forest", "isolation_forest", "one_class_svm",
                    "local_outlier_factor", "dbscan", "kmeans", "pca",
                    "feature_selection", "feature_engineering", "cross_validation",
                    "grid_search", "random_search", "bayesian_optimization",
                    
                    # Data science violations
                    "data_science", "analytics", "statistics", "regression",
                    "classification", "clustering", "dimensionality_reduction",
                    "time_series", "forecasting", "prediction", "inference",
                    "bias", "variance", "overfitting", "underfitting",
                    "gradient_descent", "backpropagation", "epoch", "batch",
                    
                    # Domain-specific violations
                    "fraud", "intrusion", "cybersecurity", "financial",
                    "banking", "healthcare", "iot", "network_security",
                    "malware", "spam", "phishing", "credit_card",
                    "transaction", "payment", "risk", "compliance",
                    
                    # Business logic violations
                    "business_intelligence", "dashboard", "reporting",
                    "visualization", "charts", "graphs", "metrics",
                    "kpi", "alert", "notification", "monitoring",
                    "observability", "telemetry", "logging"
                ],
                allowed_exceptions=[
                    # Generic terms that may appear in software contexts
                    "service", "repository", "entity", "value_object",
                    "exception", "protocol", "interface", "adapter",
                    "infrastructure", "application", "domain", "shared",
                    "config", "settings", "environment", "migration",
                    "database", "cache", "storage", "network",
                    "authentication", "authorization", "security",
                    "validation", "serialization", "deserialization",
                    "logging", "monitoring", "health", "performance",
                    "error", "exception", "handler", "middleware",
                    "router", "endpoint", "request", "response",
                    "client", "server", "api", "rest", "graphql",
                    "websocket", "stream", "queue", "event",
                    "message", "command", "query", "dto",
                    "use_case", "interactor", "presenter", "view",
                    "controller", "facade", "factory", "builder",
                    "observer", "strategy", "template", "decorator",
                    "singleton", "dependency", "injection", "container"
                ],
                file_patterns=[
                    "*.py", "*.toml", "*.yaml", "*.yml", "*.json",
                    "*.md", "*.rst", "*.txt", "*.cfg", "*.ini"
                ],
                description="Software package should contain only generic software infrastructure"
            ),
            "domain_library": DomainRule(
                domain_package="domain_library", 
                prohibited_terms=[
                    "pynomaly", "anomaly", "detection", "automl", "model",
                    "training", "dataset", "fraud", "intrusion", "cybersecurity"
                ],
                allowed_exceptions=[
                    "domain", "entity", "value_object", "repository",
                    "service", "business_logic", "template", "catalog"
                ],
                file_patterns=["*.py", "*.toml", "*.md"],
                description="Domain library should contain only generic domain patterns"
            ),
            "enterprise": DomainRule(
                domain_package="enterprise",
                prohibited_terms=[
                    "pynomaly", "anomaly", "detection", "automl", "model",
                    "training", "dataset", "fraud", "intrusion"
                ],
                allowed_exceptions=[
                    "enterprise", "tenant", "organization", "subscription",
                    "billing", "user", "role", "permission", "security"
                ],
                file_patterns=["*.py", "*.toml", "*.md"],
                description="Enterprise package should contain only enterprise infrastructure"
            )
        }
    
    def validate_package(self, package_path: str) -> List[DomainViolation]:
        """Validate a package against domain boundary rules"""
        package_name = Path(package_path).name
        
        if package_name not in self.rules:
            return []
        
        rule = self.rules[package_name]
        violations = []
        
        # Walk through all files in the package
        for root, dirs, files in os.walk(package_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if any(self._matches_pattern(file, pattern) for pattern in rule.file_patterns):
                    file_path = Path(root) / file
                    file_violations = self._validate_file(file_path, rule)
                    violations.extend(file_violations)
        
        return violations
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches glob pattern"""
        if pattern.startswith("*."):
            extension = pattern[2:]
            return filename.endswith(f".{extension}")
        return filename == pattern
    
    def _validate_file(self, file_path: Path, rule: DomainRule) -> List[DomainViolation]:
        """Validate a single file against domain rules"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                
                # Check for prohibited terms
                for term in rule.prohibited_terms:
                    if self._contains_prohibited_term(line_lower, term, rule.allowed_exceptions):
                        violations.append(DomainViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            line_content=line.strip(),
                            violation_type="prohibited_term",
                            prohibited_term=term,
                            domain_package=rule.domain_package,
                            severity="error"
                        ))
        
        except Exception as e:
            # Skip files that can't be read
            pass
        
        return violations
    
    def _contains_prohibited_term(self, line: str, term: str, allowed_exceptions: List[str]) -> bool:
        """Check if line contains prohibited term, considering exceptions"""
        # Check if term is in line
        if term not in line:
            return False
        
        # Check if it's part of an allowed exception
        for exception in allowed_exceptions:
            if exception in line:
                return False
        
        # Use word boundaries to avoid false positives
        pattern = r'\b' + re.escape(term) + r'\b'
        return bool(re.search(pattern, line))
    
    def validate_all_packages(self, root_path: str = "src/packages") -> Dict[str, List[DomainViolation]]:
        """Validate all packages in the repository"""
        results = {}
        root = Path(root_path)
        
        if not root.exists():
            return results
        
        # Find all package directories
        for package_dir in root.iterdir():
            if package_dir.is_dir() and not package_dir.name.startswith('.'):
                violations = self.validate_package(str(package_dir))
                if violations:
                    results[package_dir.name] = violations
        
        return results
    
    def generate_report(self, results: Dict[str, List[DomainViolation]], output_file: str = "domain_violations_report.json"):
        """Generate a comprehensive violation report"""
        report = {
            "summary": {
                "total_packages_checked": len(self.rules),
                "packages_with_violations": len(results),
                "total_violations": sum(len(violations) for violations in results.values())
            },
            "violations_by_package": {},
            "violations_by_type": {},
            "most_common_violations": {}
        }
        
        # Group violations by package
        for package_name, violations in results.items():
            report["violations_by_package"][package_name] = {
                "total_violations": len(violations),
                "violations": [
                    {
                        "file": v.file_path,
                        "line": v.line_number,
                        "content": v.line_content,
                        "term": v.prohibited_term,
                        "severity": v.severity
                    } for v in violations
                ]
            }
        
        # Count violations by type
        violation_counts = {}
        for violations in results.values():
            for violation in violations:
                term = violation.prohibited_term
                violation_counts[term] = violation_counts.get(term, 0) + 1
        
        report["most_common_violations"] = dict(sorted(violation_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self, results: Dict[str, List[DomainViolation]]):
        """Print a summary of violations"""
        total_violations = sum(len(violations) for violations in results.values())
        
        print(f"\n🔍 Domain Boundary Validation Results")
        print(f"=" * 50)
        print(f"Packages checked: {len(self.rules)}")
        print(f"Packages with violations: {len(results)}")
        print(f"Total violations: {total_violations}")
        
        if results:
            print(f"\n📊 Violations by Package:")
            for package_name, violations in results.items():
                print(f"  {package_name}: {len(violations)} violations")
            
            print(f"\n🚨 Most Common Violations:")
            violation_counts = {}
            for violations in results.values():
                for violation in violations:
                    term = violation.prohibited_term
                    violation_counts[term] = violation_counts.get(term, 0) + 1
            
            for term, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {term}: {count} occurrences")
        
        if total_violations == 0:
            print(f"\n✅ No domain boundary violations found!")
        else:
            print(f"\n❌ Domain boundary violations detected. Please review and fix.")

def main():
    """Main entry point for domain boundary validation"""
    validator = DomainBoundaryValidator()
    
    # Validate all packages
    results = validator.validate_all_packages()
    
    # Generate report
    report = validator.generate_report(results)
    
    # Print summary
    validator.print_summary(results)
    
    # Exit with error code if violations found
    total_violations = sum(len(violations) for violations in results.values())
    sys.exit(1 if total_violations > 0 else 0)

if __name__ == "__main__":
    main()