"""
Rules engine for repository governance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
import logging
import re

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of governance rules."""
    FILE_PATTERN = "file_pattern"
    CONTENT_PATTERN = "content_pattern"
    STRUCTURE_RULE = "structure_rule"
    METRIC_THRESHOLD = "metric_threshold"
    DEPENDENCY_RULE = "dependency_rule"
    CUSTOM_FUNCTION = "custom_function"


class RuleAction(Enum):
    """Actions that can be taken when a rule is violated."""
    WARN = "warn"
    ERROR = "error"
    FIX = "fix"
    IGNORE = "ignore"


@dataclass
class RuleResult:
    """Result of a rule evaluation."""
    rule_id: str
    passed: bool
    message: str
    severity: str = "info"
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class Rule(ABC):
    """Abstract base class for governance rules."""
    
    def __init__(self, rule_id: str, description: str, severity: str = "medium", 
                 action: RuleAction = RuleAction.WARN, enabled: bool = True):
        self.rule_id = rule_id
        self.description = description
        self.severity = severity
        self.action = action
        self.enabled = enabled
        self.logger = logging.getLogger(f"Rule.{rule_id}")
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate the rule against the given context."""
        pass
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if the rule is applicable to the given context."""
        return self.enabled


class FilePatternRule(Rule):
    """Rule that checks file patterns."""
    
    def __init__(self, rule_id: str, description: str, pattern: str, 
                 should_exist: bool = True, **kwargs):
        super().__init__(rule_id, description, **kwargs)
        self.pattern = pattern
        self.should_exist = should_exist
        self.compiled_pattern = re.compile(pattern)
    
    def evaluate(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate file pattern rule."""
        results = []
        root_path = context.get("root_path", Path.cwd())
        
        if not isinstance(root_path, Path):
            root_path = Path(root_path)
        
        # Find matching files
        matching_files = []
        for file_path in root_path.rglob("*"):
            if file_path.is_file() and self.compiled_pattern.search(str(file_path)):
                matching_files.append(file_path)
        
        if self.should_exist:
            # Files should exist
            if not matching_files:
                results.append(RuleResult(
                    rule_id=self.rule_id,
                    passed=False,
                    message=f"No files matching pattern '{self.pattern}' found",
                    severity=self.severity
                ))
            else:
                results.append(RuleResult(
                    rule_id=self.rule_id,
                    passed=True,
                    message=f"Found {len(matching_files)} files matching pattern '{self.pattern}'",
                    severity=self.severity,
                    details={"matching_files": [str(f) for f in matching_files]}
                ))
        else:
            # Files should not exist
            if matching_files:
                for file_path in matching_files:
                    results.append(RuleResult(
                        rule_id=self.rule_id,
                        passed=False,
                        message=f"File should not exist: {file_path.name}",
                        severity=self.severity,
                        file_path=file_path
                    ))
            else:
                results.append(RuleResult(
                    rule_id=self.rule_id,
                    passed=True,
                    message=f"No forbidden files matching pattern '{self.pattern}' found",
                    severity=self.severity
                ))
        
        return results


class ContentPatternRule(Rule):
    """Rule that checks content patterns in files."""
    
    def __init__(self, rule_id: str, description: str, content_pattern: str,
                 file_pattern: str = "**/*.py", should_match: bool = False, **kwargs):
        super().__init__(rule_id, description, **kwargs)
        self.content_pattern = content_pattern
        self.file_pattern = file_pattern
        self.should_match = should_match
        self.compiled_content_pattern = re.compile(content_pattern)
    
    def evaluate(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate content pattern rule."""
        results = []
        root_path = context.get("root_path", Path.cwd())
        
        if not isinstance(root_path, Path):
            root_path = Path(root_path)
        
        # Find files matching file pattern
        matching_files = list(root_path.glob(self.file_pattern))
        
        for file_path in matching_files:
            if not file_path.is_file():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                matches = list(self.compiled_content_pattern.finditer(content))
                
                if self.should_match:
                    # Content should match
                    if not matches:
                        results.append(RuleResult(
                            rule_id=self.rule_id,
                            passed=False,
                            message=f"Required pattern not found in {file_path.name}",
                            severity=self.severity,
                            file_path=file_path
                        ))
                else:
                    # Content should not match
                    if matches:
                        for match in matches:
                            line_number = content[:match.start()].count('\\n') + 1
                            results.append(RuleResult(
                                rule_id=self.rule_id,
                                passed=False,
                                message=f"Forbidden pattern found in {file_path.name}",
                                severity=self.severity,
                                file_path=file_path,
                                line_number=line_number,
                                details={"matched_text": match.group()}
                            ))
                            
            except Exception as e:
                self.logger.warning(f"Failed to read {file_path}: {e}")
                continue
        
        return results


class StructureRule(Rule):
    """Rule that checks repository structure."""
    
    def __init__(self, rule_id: str, description: str, required_dirs: List[str] = None,
                 required_files: List[str] = None, forbidden_dirs: List[str] = None,
                 forbidden_files: List[str] = None, **kwargs):
        super().__init__(rule_id, description, **kwargs)
        self.required_dirs = required_dirs or []
        self.required_files = required_files or []
        self.forbidden_dirs = forbidden_dirs or []
        self.forbidden_files = forbidden_files or []
    
    def evaluate(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate structure rule."""
        results = []
        root_path = context.get("root_path", Path.cwd())
        
        if not isinstance(root_path, Path):
            root_path = Path(root_path)
        
        # Check required directories
        for required_dir in self.required_dirs:
            dir_path = root_path / required_dir
            if not dir_path.exists():
                results.append(RuleResult(
                    rule_id=self.rule_id,
                    passed=False,
                    message=f"Required directory not found: {required_dir}",
                    severity=self.severity,
                    details={"path": str(dir_path)}
                ))
        
        # Check required files
        for required_file in self.required_files:
            file_path = root_path / required_file
            if not file_path.exists():
                results.append(RuleResult(
                    rule_id=self.rule_id,
                    passed=False,
                    message=f"Required file not found: {required_file}",
                    severity=self.severity,
                    details={"path": str(file_path)}
                ))
        
        # Check forbidden directories
        for forbidden_dir in self.forbidden_dirs:
            dir_path = root_path / forbidden_dir
            if dir_path.exists():
                results.append(RuleResult(
                    rule_id=self.rule_id,
                    passed=False,
                    message=f"Forbidden directory found: {forbidden_dir}",
                    severity=self.severity,
                    file_path=dir_path
                ))
        
        # Check forbidden files
        for forbidden_file in self.forbidden_files:
            file_path = root_path / forbidden_file
            if file_path.exists():
                results.append(RuleResult(
                    rule_id=self.rule_id,
                    passed=False,
                    message=f"Forbidden file found: {forbidden_file}",
                    severity=self.severity,
                    file_path=file_path
                ))
        
        # If no violations found, rule passes
        if not results:
            results.append(RuleResult(
                rule_id=self.rule_id,
                passed=True,
                message="Repository structure is compliant",
                severity=self.severity
            ))
        
        return results


class MetricThresholdRule(Rule):
    """Rule that checks metrics against thresholds."""
    
    def __init__(self, rule_id: str, description: str, metric_name: str,
                 threshold: Union[int, float], operator: str = "<=", **kwargs):
        super().__init__(rule_id, description, **kwargs)
        self.metric_name = metric_name
        self.threshold = threshold
        self.operator = operator
        
        # Define operator functions
        self.operators = {
            "<=": lambda x, y: x <= y,
            "<": lambda x, y: x < y,
            ">=": lambda x, y: x >= y,
            ">": lambda x, y: x > y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y
        }
    
    def evaluate(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate metric threshold rule."""
        results = []
        metrics = context.get("metrics", {})
        
        if self.metric_name not in metrics:
            results.append(RuleResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Metric '{self.metric_name}' not found in context",
                severity=self.severity
            ))
            return results
        
        metric_value = metrics[self.metric_name]
        operator_func = self.operators.get(self.operator)
        
        if not operator_func:
            results.append(RuleResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Unknown operator: {self.operator}",
                severity=self.severity
            ))
            return results
        
        passed = operator_func(metric_value, self.threshold)
        
        results.append(RuleResult(
            rule_id=self.rule_id,
            passed=passed,
            message=f"Metric '{self.metric_name}' = {metric_value} {self.operator} {self.threshold}: {'PASS' if passed else 'FAIL'}",
            severity=self.severity,
            details={
                "metric_name": self.metric_name,
                "metric_value": metric_value,
                "threshold": self.threshold,
                "operator": self.operator
            }
        ))
        
        return results


class CustomFunctionRule(Rule):
    """Rule that uses a custom function for evaluation."""
    
    def __init__(self, rule_id: str, description: str, 
                 evaluation_function: Callable[[Dict[str, Any]], List[RuleResult]], **kwargs):
        super().__init__(rule_id, description, **kwargs)
        self.evaluation_function = evaluation_function
    
    def evaluate(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate custom function rule."""
        try:
            return self.evaluation_function(context)
        except Exception as e:
            self.logger.error(f"Error in custom function rule {self.rule_id}: {e}")
            return [RuleResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Error evaluating custom rule: {str(e)}",
                severity="high"
            )]


class RulesEngine:
    """Main rules engine for repository governance."""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self.rules[rule.rule_id] = rule
        self.logger.debug(f"Added rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule from the engine."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.debug(f"Removed rule: {rule_id}")
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        return self.rules.get(rule_id)
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.logger.debug(f"Enabled rule: {rule_id}")
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.logger.debug(f"Disabled rule: {rule_id}")
    
    def evaluate_all(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate all enabled rules."""
        all_results = []
        
        for rule_id, rule in self.rules.items():
            if not rule.is_applicable(context):
                continue
            
            try:
                results = rule.evaluate(context)
                all_results.extend(results)
                self.logger.debug(f"Rule {rule_id} evaluated: {len(results)} results")
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule_id}: {e}")
                all_results.append(RuleResult(
                    rule_id=rule_id,
                    passed=False,
                    message=f"Error evaluating rule: {str(e)}",
                    severity="high"
                ))
        
        return all_results
    
    def evaluate_rule(self, rule_id: str, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate a specific rule."""
        rule = self.rules.get(rule_id)
        if not rule:
            return [RuleResult(
                rule_id=rule_id,
                passed=False,
                message=f"Rule not found: {rule_id}",
                severity="high"
            )]
        
        if not rule.is_applicable(context):
            return [RuleResult(
                rule_id=rule_id,
                passed=True,
                message=f"Rule not applicable: {rule_id}",
                severity="info"
            )]
        
        try:
            return rule.evaluate(context)
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule_id}: {e}")
            return [RuleResult(
                rule_id=rule_id,
                passed=False,
                message=f"Error evaluating rule: {str(e)}",
                severity="high"
            )]
    
    def get_rules_by_severity(self, severity: str) -> List[Rule]:
        """Get all rules with a specific severity."""
        return [rule for rule in self.rules.values() if rule.severity == severity]
    
    def get_enabled_rules(self) -> List[Rule]:
        """Get all enabled rules."""
        return [rule for rule in self.rules.values() if rule.enabled]
    
    def get_disabled_rules(self) -> List[Rule]:
        """Get all disabled rules."""
        return [rule for rule in self.rules.values() if not rule.enabled]
    
    def load_default_rules(self) -> None:
        """Load default governance rules."""
        # File pattern rules
        self.add_rule(FilePatternRule(
            rule_id="no_backup_files",
            description="Repository should not contain backup files",
            pattern=r"\\.(bak|backup|orig|old|~)$",
            should_exist=False,
            severity="medium"
        ))
        
        self.add_rule(FilePatternRule(
            rule_id="has_readme",
            description="Repository should have a README file",
            pattern=r"README\\.(md|rst|txt)$",
            should_exist=True,
            severity="low"
        ))
        
        self.add_rule(FilePatternRule(
            rule_id="has_gitignore",
            description="Repository should have a .gitignore file",
            pattern=r"^\\.gitignore$",
            should_exist=True,
            severity="low"
        ))
        
        # Content pattern rules
        self.add_rule(ContentPatternRule(
            rule_id="no_todo_comments",
            description="Code should not contain TODO comments",
            content_pattern=r"#\\s*TODO|#\\s*FIXME|#\\s*XXX",
            file_pattern="**/*.py",
            should_match=False,
            severity="low"
        ))
        
        self.add_rule(ContentPatternRule(
            rule_id="no_hardcoded_secrets",
            description="Code should not contain hardcoded secrets",
            content_pattern=r"(password|secret|key)\\s*=\\s*['\\\"][^'\\\"]+['\\\"]",
            file_pattern="**/*.py",
            should_match=False,
            severity="high"
        ))
        
        # Structure rules
        self.add_rule(StructureRule(
            rule_id="python_package_structure",
            description="Python packages should have proper structure",
            required_files=["__init__.py"],
            forbidden_files=["Thumbs.db", ".DS_Store"],
            severity="medium"
        ))
        
        self.logger.info("Loaded default governance rules")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the rules engine."""
        enabled_rules = self.get_enabled_rules()
        disabled_rules = self.get_disabled_rules()
        
        severity_counts = {}
        for rule in enabled_rules:
            severity_counts[rule.severity] = severity_counts.get(rule.severity, 0) + 1
        
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len(enabled_rules),
            "disabled_rules": len(disabled_rules),
            "severity_counts": severity_counts,
            "rule_types": {
                "file_pattern": len([r for r in enabled_rules if isinstance(r, FilePatternRule)]),
                "content_pattern": len([r for r in enabled_rules if isinstance(r, ContentPatternRule)]),
                "structure": len([r for r in enabled_rules if isinstance(r, StructureRule)]),
                "metric_threshold": len([r for r in enabled_rules if isinstance(r, MetricThresholdRule)]),
                "custom_function": len([r for r in enabled_rules if isinstance(r, CustomFunctionRule)])
            }
        }