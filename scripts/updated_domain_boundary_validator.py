#!/usr/bin/env python3
"""
Updated Domain Boundary Validator for Feature Architecture

Validates that features maintain proper domain boundaries and don't contain
references to other domains. This enforces clean architecture principles
in the new domain → package → feature → layer structure.
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
    feature_name: str
    severity: str = "error"  # error, warning, info

@dataclass
class DomainRule:
    """Represents a domain boundary rule"""
    domain_package: str
    prohibited_terms: List[str]
    allowed_exceptions: List[str]
    file_patterns: List[str]
    description: str

class FeatureDomainBoundaryValidator:
    """Validates domain boundaries across features"""
    
    def __init__(self, packages_root: str = "src/packages_new"):
        self.packages_root = Path(packages_root)
        self.violations: List[DomainViolation] = []
        self.rules = self._load_domain_rules()
    
    def _load_domain_rules(self) -> Dict[str, DomainRule]:
        """Load domain boundary rules for feature architecture"""
        return {
            "software": DomainRule(
                domain_package="software",
                prohibited_terms=[
                    # Core domain violations - business logic terms
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
                    
                    # Domain-specific business violations
                    "fraud", "intrusion", "cybersecurity", "financial",
                    "banking", "healthcare", "iot", "network_security",
                    "malware", "spam", "phishing", "credit_card",
                    "transaction", "payment", "risk", "compliance",
                    
                    # Business intelligence violations
                    "business_intelligence", "dashboard", "reporting",
                    "visualization", "charts", "graphs", "metrics",
                    "kpi", "alert", "notification", "monitoring",
                    "observability", "telemetry", "logging"
                ],
                allowed_exceptions=[
                    # Generic software terms
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
                description="Software domain should contain only generic software infrastructure"
            ),
            
            "business": DomainRule(
                domain_package="business",
                prohibited_terms=[
                    # Technical implementation violations
                    "sklearn", "tensorflow", "pytorch", "keras", "xgboost",
                    "sqlalchemy", "psycopg2", "pymongo", "redis",
                    "fastapi", "flask", "django", "celery", "rabbitmq",
                    "docker", "kubernetes", "terraform", "ansible",
                    
                    # Low-level technical terms
                    "algorithm", "neural_network", "gradient_descent",
                    "backpropagation", "clustering", "classification",
                    "regression", "dimensionality_reduction", "pca",
                    "feature_engineering", "cross_validation", "grid_search"
                ],
                allowed_exceptions=[
                    # Business domain terms
                    "business", "analytics", "intelligence", "reporting",
                    "dashboard", "kpi", "metrics", "administration",
                    "governance", "compliance", "audit", "risk",
                    "cost", "optimization", "budget", "forecast",
                    "onboarding", "workflow", "process", "policy",
                    "user", "customer", "client", "stakeholder",
                    "organization", "department", "team", "role",
                    "permission", "access", "authorization", "security"
                ],
                file_patterns=["*.py", "*.toml", "*.yaml", "*.yml", "*.json", "*.md"],
                description="Business domain should contain only business logic and rules"
            ),
            
            "data": DomainRule(
                domain_package="data",
                prohibited_terms=[
                    # Business logic violations
                    "business_intelligence", "governance", "compliance",
                    "administration", "onboarding", "cost_optimization",
                    "budget", "forecast", "audit", "risk_assessment",
                    "policy", "workflow", "process", "user_management",
                    "role", "permission", "authorization", "authentication",
                    
                    # Infrastructure violations
                    "fastapi", "flask", "django", "click", "typer",
                    "kubernetes", "docker", "terraform", "ansible",
                    "nginx", "gunicorn", "uwsgi", "celery", "rabbitmq"
                ],
                allowed_exceptions=[
                    # Data domain terms
                    "data", "dataset", "dataframe", "pipeline", "etl",
                    "transformation", "processing", "ingestion", "validation",
                    "quality", "profiling", "lineage", "catalog", "schema",
                    "anomaly", "detection", "threshold", "alert", "monitoring",
                    "observability", "metric", "measurement", "analysis",
                    "statistical", "aggregation", "filtering", "sorting",
                    "storage", "warehouse", "lake", "mart", "repository"
                ],
                file_patterns=["*.py", "*.toml", "*.yaml", "*.yml", "*.json", "*.md"],
                description="Data domain should contain only data-related functionality"
            ),
            
            "ai": DomainRule(
                domain_package="ai",
                prohibited_terms=[
                    # Business logic violations
                    "business_intelligence", "governance", "compliance",
                    "administration", "onboarding", "cost_optimization",
                    "budget", "forecast", "audit", "risk_assessment",
                    "policy", "workflow", "process", "user_management",
                    "role", "permission", "authorization", "authentication",
                    
                    # Infrastructure violations
                    "fastapi", "flask", "django", "click", "typer",
                    "kubernetes", "docker", "terraform", "ansible",
                    "nginx", "gunicorn", "uwsgi", "celery", "rabbitmq"
                ],
                allowed_exceptions=[
                    # AI/ML domain terms
                    "artificial_intelligence", "machine_learning", "deep_learning",
                    "neural_network", "model", "training", "inference",
                    "prediction", "classification", "regression", "clustering",
                    "automl", "mlops", "experiment", "hyperparameter",
                    "optimization", "feature", "engineering", "selection",
                    "algorithm", "ensemble", "cross_validation", "metric",
                    "evaluation", "performance", "accuracy", "precision",
                    "recall", "f1_score", "roc_auc", "confusion_matrix",
                    "overfitting", "underfitting", "bias", "variance",
                    "gradient", "descent", "backpropagation", "epoch",
                    "batch", "learning_rate", "optimizer", "loss_function"
                ],
                file_patterns=["*.py", "*.toml", "*.yaml", "*.yml", "*.json", "*.md"],
                description="AI domain should contain only AI/ML related functionality"
            )
        }
    
    def validate_feature(self, feature_path: Path) -> List[DomainViolation]:
        """Validate a single feature against domain boundary rules"""
        domain_name = feature_path.parent.parent.name
        package_name = feature_path.parent.name
        feature_name = feature_path.name
        
        # Get domain rule (check both domain and package level)
        rule = self.rules.get(domain_name)
        if not rule:
            rule = self.rules.get(package_name)
        
        if not rule:
            return []
        
        violations = []
        
        # Walk through all files in the feature
        for root, dirs, files in os.walk(feature_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if any(self._matches_pattern(file, pattern) for pattern in rule.file_patterns):
                    file_path = Path(root) / file
                    file_violations = self._validate_file(file_path, rule, feature_name)
                    violations.extend(file_violations)
        
        return violations
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches glob pattern"""
        if pattern.startswith("*."):
            extension = pattern[2:]
            return filename.endswith(f".{extension}")
        return filename == pattern
    
    def _validate_file(self, file_path: Path, rule: DomainRule, feature_name: str) -> List[DomainViolation]:
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
                            feature_name=feature_name,
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
        pattern = r'\\b' + re.escape(term) + r'\\b'
        return bool(re.search(pattern, line))
    
    def validate_all_features(self) -> Dict[str, List[DomainViolation]]:
        """Validate all features in the repository"""
        results = {}
        
        if not self.packages_root.exists():
            return results
        
        # Walk through domain/package/feature structure
        for domain_dir in self.packages_root.iterdir():
            if not domain_dir.is_dir() or domain_dir.name.startswith('.'):
                continue
                
            for package_dir in domain_dir.iterdir():
                if not package_dir.is_dir() or package_dir.name.startswith('.'):
                    continue
                    
                for feature_dir in package_dir.iterdir():
                    if not feature_dir.is_dir() or feature_dir.name in ['shared', 'docs'] or feature_dir.name.startswith('.'):
                        continue
                    
                    feature_key = f"{domain_dir.name}/{package_dir.name}/{feature_dir.name}"
                    violations = self.validate_feature(feature_dir)
                    if violations:
                        results[feature_key] = violations
        
        return results
    
    def generate_report(self, results: Dict[str, List[DomainViolation]], output_file: str = "feature_domain_violations_report.json"):
        """Generate a comprehensive violation report"""
        report = {
            "summary": {
                "total_features_checked": len(results) if results else 0,
                "features_with_violations": len(results),
                "total_violations": sum(len(violations) for violations in results.values())
            },
            "violations_by_feature": {},
            "violations_by_domain": {},
            "most_common_violations": {}
        }
        
        # Group violations by feature
        for feature_name, violations in results.items():
            report["violations_by_feature"][feature_name] = {
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
        
        # Group violations by domain
        domain_violations = {}
        for violations in results.values():
            for violation in violations:
                domain = violation.domain_package
                if domain not in domain_violations:
                    domain_violations[domain] = []
                domain_violations[domain].append(violation)
        
        for domain, violations in domain_violations.items():
            report["violations_by_domain"][domain] = {
                "total_violations": len(violations),
                "features_affected": len(set(v.feature_name for v in violations))
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
        
        print(f"\\n🔍 Feature Domain Boundary Validation Results")
        print(f"=" * 60)
        print(f"Features checked: {len(results) if results else 0}")
        print(f"Features with violations: {len(results)}")
        print(f"Total violations: {total_violations}")
        
        if results:
            print(f"\\n📊 Violations by Feature:")
            for feature_name, violations in results.items():
                print(f"  {feature_name}: {len(violations)} violations")
            
            print(f"\\n🚨 Most Common Violations:")
            violation_counts = {}
            for violations in results.values():
                for violation in violations:
                    term = violation.prohibited_term
                    violation_counts[term] = violation_counts.get(term, 0) + 1
            
            for term, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {term}: {count} occurrences")
        
        if total_violations == 0:
            print(f"\\n✅ No domain boundary violations found!")
        else:
            print(f"\\n❌ Domain boundary violations detected. Please review and fix.")

def main():
    """Main entry point for feature domain boundary validation"""
    validator = FeatureDomainBoundaryValidator()
    
    # Validate all features
    results = validator.validate_all_features()
    
    # Generate report
    report = validator.generate_report(results)
    
    # Print summary
    validator.print_summary(results)
    
    # Exit with error code if violations found
    total_violations = sum(len(violations) for violations in results.values())
    sys.exit(1 if total_violations > 0 else 0)

if __name__ == "__main__":
    main()