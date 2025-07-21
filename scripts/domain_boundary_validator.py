#!/usr/bin/env python3
"""
Domain Boundary Validator

Validates that packages maintain proper domain boundaries and don't contain
references to other domains. This enforces clean architecture principles.
Enhanced with semantic analysis and new domain detection capabilities.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import ast
from collections import Counter, defaultdict
import subprocess
from datetime import datetime

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
    confidence: float = 1.0  # Confidence level for semantic violations
    suggested_domain: Optional[str] = None  # Suggested domain for the concept
    context: List[str] = field(default_factory=list)  # Surrounding context

@dataclass
class NewDomainDetection:
    """Represents a newly detected domain concept"""
    domain_name: str
    concepts: List[str]
    files: List[str]
    confidence: float
    suggested_package_structure: Dict[str, Any]
    reasoning: str

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
        self.new_domains: List[NewDomainDetection] = []
        self.rules = self._load_domain_rules()
        self.semantic_patterns = self._load_semantic_patterns()
        self.existing_domains = self._discover_existing_domains()
    
    def _discover_existing_domains(self) -> Set[str]:
        """Discover existing domain packages in the repository"""
        domains = set()
        packages_dir = Path("src/packages")
        
        if packages_dir.exists():
            for package_path in packages_dir.iterdir():
                if package_path.is_dir() and not package_path.name.startswith('.'):
                    # Handle nested domain structure like ai/machine_learning
                    if any(sub.is_dir() for sub in package_path.iterdir() if not sub.name.startswith('.')):
                        for subdir in package_path.iterdir():
                            if subdir.is_dir() and not subdir.name.startswith('.'):
                                domains.add(f"{package_path.name}/{subdir.name}")
                    else:
                        domains.add(package_path.name)
        
        return domains
    
    def _load_semantic_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load semantic patterns for domain detection"""
        return {
            # Business domains
            "financial_services": {
                "keywords": ["payment", "transaction", "billing", "invoice", "banking", "finance", "credit", "debit", "wallet", "money", "currency"],
                "patterns": [r"\b(payment|transaction|billing|finance)_\w+", r"\bfinancial\w*", r"\bmoney\w*"],
                "context_indicators": ["banking", "fintech", "payments", "wallet"],
                "confidence_threshold": 0.7
            },
            "healthcare": {
                "keywords": ["patient", "medical", "diagnosis", "treatment", "clinic", "hospital", "health", "medicine", "doctor", "nurse"],
                "patterns": [r"\b(patient|medical|health)_\w+", r"\bhealthcare\w*", r"\bmedical\w*"],
                "context_indicators": ["healthcare", "medical", "clinical"],
                "confidence_threshold": 0.7
            },
            "e_commerce": {
                "keywords": ["product", "catalog", "inventory", "order", "cart", "checkout", "shipping", "customer", "purchase", "sale"],
                "patterns": [r"\b(product|order|cart|customer)_\w+", r"\becommerce\w*", r"\bcommerce\w*"],
                "context_indicators": ["ecommerce", "retail", "marketplace"],
                "confidence_threshold": 0.7
            },
            "iot_devices": {
                "keywords": ["sensor", "device", "telemetry", "measurement", "actuator", "gateway", "edge", "firmware", "embedded"],
                "patterns": [r"\b(sensor|device|telemetry)_\w+", r"\biot\w*", r"\bedge\w*"],
                "context_indicators": ["iot", "devices", "sensors", "embedded"],
                "confidence_threshold": 0.8
            },
            "cybersecurity": {
                "keywords": ["security", "threat", "vulnerability", "attack", "malware", "intrusion", "firewall", "encryption", "authentication"],
                "patterns": [r"\b(security|threat|vulnerability)_\w+", r"\bcyber\w*", r"\bsecurity\w*"],
                "context_indicators": ["cybersecurity", "infosec", "security"],
                "confidence_threshold": 0.8
            },
            "logistics": {
                "keywords": ["shipment", "delivery", "warehouse", "inventory", "tracking", "logistics", "freight", "cargo", "route", "fleet"],
                "patterns": [r"\b(shipment|delivery|warehouse)_\w+", r"\blogistics\w*", r"\bfreight\w*"],
                "context_indicators": ["logistics", "shipping", "supply_chain"],
                "confidence_threshold": 0.7
            },
            "content_management": {
                "keywords": ["content", "article", "media", "document", "publishing", "cms", "editor", "author", "publication"],
                "patterns": [r"\b(content|article|media)_\w+", r"\bcms\w*", r"\bpublishing\w*"],
                "context_indicators": ["cms", "publishing", "content"],
                "confidence_threshold": 0.7
            },
            "social_media": {
                "keywords": ["post", "comment", "like", "share", "follow", "feed", "timeline", "social", "community", "user_profile"],
                "patterns": [r"\b(post|comment|social)_\w+", r"\bsocial\w*", r"\bcommunity\w*"],
                "context_indicators": ["social", "community", "networking"],
                "confidence_threshold": 0.8
            },
            "gaming": {
                "keywords": ["game", "player", "score", "level", "achievement", "leaderboard", "match", "tournament", "gameplay"],
                "patterns": [r"\b(game|player|score)_\w+", r"\bgaming\w*", r"\bgame\w*"],
                "context_indicators": ["gaming", "games", "entertainment"],
                "confidence_threshold": 0.8
            },
            "education": {
                "keywords": ["course", "lesson", "student", "teacher", "assignment", "grade", "curriculum", "learning", "education"],
                "patterns": [r"\b(course|student|lesson)_\w+", r"\beducation\w*", r"\blearning\w*"],
                "context_indicators": ["education", "learning", "academic"],
                "confidence_threshold": 0.7
            }
        }
    
    def _load_domain_rules(self) -> Dict[str, DomainRule]:
        """Load domain boundary rules"""
        return {
            "software": DomainRule(
                domain_package="software",
                prohibited_terms=[
                    # Core domain violations
                    "anomaly_detection", "anomaly", "anomaly_detection", "outlier", "detection",
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
                    "anomaly_detection", "anomaly", "detection", "automl", "model",
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
                    "anomaly_detection", "anomaly", "detection", "automl", "model",
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
    
    def detect_new_domains(self, root_path: str = "src/packages") -> List[NewDomainDetection]:
        """Detect potential new domains in the codebase using semantic analysis"""
        domain_candidates = defaultdict(lambda: {
            'concepts': set(), 
            'files': set(), 
            'confidence_scores': [],
            'context': set()
        })
        
        # Analyze all files in the repository
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.json')):
                    file_path = Path(root) / file
                    domain_matches = self._analyze_file_for_domains(file_path)
                    
                    for domain_name, matches in domain_matches.items():
                        domain_candidates[domain_name]['concepts'].update(matches['concepts'])
                        domain_candidates[domain_name]['files'].add(str(file_path))
                        domain_candidates[domain_name]['confidence_scores'].extend(matches['confidence_scores'])
                        domain_candidates[domain_name]['context'].update(matches['context'])
        
        # Convert candidates to NewDomainDetection objects
        new_domains = []
        for domain_name, data in domain_candidates.items():
            if domain_name not in self.existing_domains and len(data['concepts']) >= 3:
                avg_confidence = sum(data['confidence_scores']) / len(data['confidence_scores']) if data['confidence_scores'] else 0
                
                if avg_confidence >= self.semantic_patterns[domain_name]['confidence_threshold']:
                    new_domain = NewDomainDetection(
                        domain_name=domain_name,
                        concepts=list(data['concepts']),
                        files=list(data['files']),
                        confidence=avg_confidence,
                        suggested_package_structure=self._generate_package_structure(domain_name, data['concepts']),
                        reasoning=self._generate_reasoning(domain_name, data)
                    )
                    new_domains.append(new_domain)
        
        self.new_domains = new_domains
        return new_domains
    
    def _analyze_file_for_domains(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Analyze a single file for domain-specific concepts"""
        matches = defaultdict(lambda: {'concepts': set(), 'confidence_scores': [], 'context': set()})
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Analyze each line for domain concepts
            for line_num, line in enumerate(lines):
                line_lower = line.lower()
                
                # Check against semantic patterns
                for domain_name, pattern_data in self.semantic_patterns.items():
                    confidence = 0.0
                    detected_concepts = set()
                    context_indicators = set()
                    
                    # Check keywords
                    for keyword in pattern_data['keywords']:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', line_lower):
                            detected_concepts.add(keyword)
                            confidence += 0.3
                    
                    # Check regex patterns
                    for pattern in pattern_data['patterns']:
                        pattern_matches = re.findall(pattern, line_lower)
                        if pattern_matches:
                            detected_concepts.update(pattern_matches)
                            confidence += 0.5
                    
                    # Check context indicators
                    for indicator in pattern_data['context_indicators']:
                        if indicator in line_lower:
                            context_indicators.add(indicator)
                            confidence += 0.2
                    
                    # If we found domain concepts in this line
                    if detected_concepts:
                        matches[domain_name]['concepts'].update(detected_concepts)
                        matches[domain_name]['confidence_scores'].append(min(confidence, 1.0))
                        matches[domain_name]['context'].update(context_indicators)
                        
                        # Add surrounding context
                        start_line = max(0, line_num - 2)
                        end_line = min(len(lines), line_num + 3)
                        context_lines = lines[start_line:end_line]
                        matches[domain_name]['context'].update([l.strip() for l in context_lines if l.strip()])
        
        except Exception:
            pass  # Skip files that can't be read
        
        return dict(matches)
    
    def _generate_package_structure(self, domain_name: str, concepts: Set[str]) -> Dict[str, Any]:
        """Generate suggested package structure for a new domain"""
        structure = {
            "package_name": domain_name,
            "suggested_path": f"src/packages/{domain_name}",
            "core_entities": [],
            "services": [],
            "value_objects": [],
            "repositories": []
        }
        
        # Analyze concepts to suggest structure
        for concept in concepts:
            if any(word in concept for word in ['service', 'manager', 'handler', 'processor']):
                structure['services'].append(f"{concept}_service")
            elif any(word in concept for word in ['entity', 'model', 'record']):
                structure['core_entities'].append(f"{concept}_entity")
            elif any(word in concept for word in ['value', 'dto', 'data']):
                structure['value_objects'].append(f"{concept}_value_object")
            elif any(word in concept for word in ['repository', 'dao', 'storage']):
                structure['repositories'].append(f"{concept}_repository")
        
        return structure
    
    def _generate_reasoning(self, domain_name: str, data: Dict[str, Any]) -> str:
        """Generate reasoning for why this domain was detected"""
        concepts_count = len(data['concepts'])
        files_count = len(data['files'])
        avg_confidence = sum(data['confidence_scores']) / len(data['confidence_scores']) if data['confidence_scores'] else 0
        
        reasoning = (
            f"Detected {domain_name} domain based on {concepts_count} domain-specific concepts "
            f"found across {files_count} files with average confidence {avg_confidence:.2f}. "
            f"Key concepts include: {', '.join(list(data['concepts'])[:5])}."
        )
        
        if data['context']:
            reasoning += f" Context indicators: {', '.join(list(data['context'])[:3])}."
        
        return reasoning
    
    def analyze_changed_files(self, changed_files: List[str]) -> Dict[str, Any]:
        """Analyze changed files for new domain concepts and violations"""
        analysis_result = {
            'new_domain_concepts': [],
            'cross_domain_violations': [],
            'suggested_package_moves': [],
            'auto_fix_suggestions': []
        }
        
        # Get git diff to analyze actual changes
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                changed_files = result.stdout.strip().split('\n')
            else:
                # Fallback to provided files list
                pass
        except Exception:
            pass
        
        # Analyze each changed file
        for file_path in changed_files:
            if not file_path or not Path(file_path).exists():
                continue
                
            # Check if file introduces new domain concepts
            domain_matches = self._analyze_file_for_domains(Path(file_path))
            for domain_name, matches in domain_matches.items():
                if domain_name not in self.existing_domains and matches['concepts']:
                    analysis_result['new_domain_concepts'].append({
                        'file': file_path,
                        'domain': domain_name,
                        'concepts': list(matches['concepts']),
                        'confidence': sum(matches['confidence_scores']) / len(matches['confidence_scores']) if matches['confidence_scores'] else 0
                    })
            
            # Check for cross-domain violations
            current_package = self._get_package_from_path(file_path)
            if current_package:
                violations = self._validate_file(Path(file_path), self.rules.get(current_package))
                if violations:
                    analysis_result['cross_domain_violations'].extend(violations)
                    
                    # Generate auto-fix suggestions
                    for violation in violations:
                        suggestion = self._generate_auto_fix_suggestion(violation, file_path)
                        if suggestion:
                            analysis_result['auto_fix_suggestions'].append(suggestion)
        
        return analysis_result
    
    def _get_package_from_path(self, file_path: str) -> Optional[str]:
        """Extract package name from file path"""
        path_parts = Path(file_path).parts
        if 'src' in path_parts and 'packages' in path_parts:
            try:
                src_idx = path_parts.index('src')
                packages_idx = path_parts.index('packages')
                if packages_idx == src_idx + 1 and len(path_parts) > packages_idx + 1:
                    return path_parts[packages_idx + 1]
            except (ValueError, IndexError):
                pass
        return None
    
    def _generate_auto_fix_suggestion(self, violation: DomainViolation, file_path: str) -> Optional[Dict[str, Any]]:
        """Generate automatic fix suggestions for domain violations"""
        # Analyze the violation and suggest appropriate domain
        for domain_name, pattern_data in self.semantic_patterns.items():
            if violation.prohibited_term in pattern_data['keywords']:
                return {
                    'violation_file': file_path,
                    'violation_line': violation.line_number,
                    'violation_term': violation.prohibited_term,
                    'suggested_action': 'move_to_domain',
                    'target_domain': domain_name,
                    'reasoning': f"Term '{violation.prohibited_term}' belongs to {domain_name} domain",
                    'suggested_path': f"src/packages/{domain_name}",
                    'confidence': 0.8
                }
        
        return None
    
    def generate_report(self, results: Dict[str, List[DomainViolation]], output_file: str = "domain_violations_report.json"):
        """Generate a comprehensive violation report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_packages_checked": len(self.rules),
                "packages_with_violations": len(results),
                "total_violations": sum(len(violations) for violations in results.values()),
                "new_domains_detected": len(self.new_domains),
                "existing_domains": list(self.existing_domains)
            },
            "violations_by_package": {},
            "violations_by_type": {},
            "most_common_violations": {},
            "new_domain_detections": [],
            "recommendations": []
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
        
        # Add new domain detections
        for new_domain in self.new_domains:
            report["new_domain_detections"].append({
                "domain_name": new_domain.domain_name,
                "concepts": new_domain.concepts,
                "files": new_domain.files,
                "confidence": new_domain.confidence,
                "suggested_package_structure": new_domain.suggested_package_structure,
                "reasoning": new_domain.reasoning
            })
        
        # Generate recommendations
        recommendations = []
        
        # Recommend creating new packages for detected domains
        for new_domain in self.new_domains:
            recommendations.append({
                "type": "create_package",
                "priority": "high" if new_domain.confidence > 0.8 else "medium",
                "action": f"Create new package for {new_domain.domain_name} domain",
                "command": f"python scripts/create_domain_package.py {new_domain.domain_name}",
                "reasoning": new_domain.reasoning
            })
        
        # Recommend fixing violations
        if results:
            recommendations.append({
                "type": "fix_violations",
                "priority": "high",
                "action": "Fix domain boundary violations",
                "command": "python scripts/fix_domain_leakage.py",
                "reasoning": f"Found {sum(len(v) for v in results.values())} domain boundary violations"
            })
        
        report["recommendations"] = recommendations
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self, results: Dict[str, List[DomainViolation]]):
        """Print a summary of violations and new domain detections"""
        total_violations = sum(len(violations) for violations in results.values())
        
        print(f"\n🔍 Domain Boundary Validation Results")
        print(f"=" * 50)
        print(f"Packages checked: {len(self.rules)}")
        print(f"Existing domains: {len(self.existing_domains)}")
        print(f"Packages with violations: {len(results)}")
        print(f"Total violations: {total_violations}")
        print(f"New domains detected: {len(self.new_domains)}")
        
        # Show existing domains
        if self.existing_domains:
            print(f"\n📦 Existing Domains:")
            for domain in sorted(self.existing_domains):
                print(f"  ✓ {domain}")
        
        # Show new domain detections
        if self.new_domains:
            print(f"\n🆕 New Domain Detections:")
            for domain in self.new_domains:
                confidence_emoji = "🟢" if domain.confidence > 0.8 else "🟡" if domain.confidence > 0.6 else "🔴"
                print(f"  {confidence_emoji} {domain.domain_name} (confidence: {domain.confidence:.2f})")
                print(f"    Concepts: {', '.join(domain.concepts[:5])}{'...' if len(domain.concepts) > 5 else ''}")
                print(f"    Files: {len(domain.files)} files")
                print(f"    Suggested path: {domain.suggested_package_structure['suggested_path']}")
                print()
        
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
        
        # Print recommendations
        print(f"\n💡 Recommendations:")
        if self.new_domains:
            for domain in self.new_domains:
                if domain.confidence > 0.7:
                    print(f"  • Create package: python scripts/create_domain_package.py {domain.domain_name}")
        
        if results:
            print(f"  • Fix violations: python scripts/fix_domain_leakage.py")
            print(f"  • Review report: cat domain_violations_report.json")
        
        if total_violations == 0 and not self.new_domains:
            print(f"\n✅ No domain boundary violations found and no new domains detected!")
        elif self.new_domains:
            print(f"\n🎯 Found {len(self.new_domains)} potential new domains. Consider creating packages for high-confidence detections.")
        
        if total_violations > 0:
            print(f"\n❌ Domain boundary violations detected. Please review and fix.")

def main():
    """Main entry point for domain boundary validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain Boundary Validator with semantic analysis")
    parser.add_argument("--detect-new-domains", action="store_true", 
                       help="Detect new domains using semantic analysis")
    parser.add_argument("--analyze-changes", action="store_true",
                       help="Analyze git changes for domain violations")
    parser.add_argument("--root-path", default="src/packages", 
                       help="Root path to analyze (default: src/packages)")
    parser.add_argument("--output", default="domain_violations_report.json",
                       help="Output report file (default: domain_violations_report.json)")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                       help="Minimum confidence for new domain detection (default: 0.7)")
    
    args = parser.parse_args()
    
    validator = DomainBoundaryValidator()
    
    # Detect new domains if requested
    if args.detect_new_domains:
        print("🔍 Detecting new domains using semantic analysis...")
        validator.detect_new_domains(args.root_path)
    
    # Analyze git changes if requested
    if args.analyze_changes:
        print("🔍 Analyzing git changes for domain violations...")
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                changed_files = [f for f in result.stdout.strip().split('\n') if f]
                change_analysis = validator.analyze_changed_files(changed_files)
                
                if change_analysis['new_domain_concepts']:
                    print("\n🆕 New domain concepts detected in changes:")
                    for concept in change_analysis['new_domain_concepts']:
                        print(f"  • {concept['domain']} in {concept['file']} (confidence: {concept['confidence']:.2f})")
        except Exception as e:
            print(f"Warning: Could not analyze git changes: {e}")
    
    # Validate all packages
    results = validator.validate_all_packages()
    
    # Generate report
    report = validator.generate_report(results, args.output)
    
    # Print summary
    validator.print_summary(results)
    
    # Provide specific recommendations
    if validator.new_domains:
        high_confidence_domains = [d for d in validator.new_domains if d.confidence >= args.confidence_threshold]
        if high_confidence_domains:
            print(f"\n🚀 Quick Start Commands:")
            for domain in high_confidence_domains:
                print(f"  python scripts/create_domain_package.py {domain.domain_name} --description \"{domain.domain_name.replace('_', ' ').title()} domain package\"")
    
    # Exit with error code if violations found or high-confidence new domains detected
    total_violations = sum(len(violations) for violations in results.values())
    high_confidence_new_domains = len([d for d in validator.new_domains if d.confidence >= 0.8])
    
    if total_violations > 0:
        print(f"\n❌ Exiting with error code 1 due to {total_violations} domain violations")
        sys.exit(1)
    elif high_confidence_new_domains > 0:
        print(f"\n⚠️  Exiting with error code 2 due to {high_confidence_new_domains} high-confidence new domains detected")
        print("Consider creating packages for these domains before proceeding")
        sys.exit(2)
    else:
        print(f"\n✅ Domain boundaries are compliant!")
        sys.exit(0)

if __name__ == "__main__":
    main()