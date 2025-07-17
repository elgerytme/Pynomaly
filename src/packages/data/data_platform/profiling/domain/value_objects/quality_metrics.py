"""Quality metrics value objects for data profiling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import UUID


class QualityDimension(str, Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    RELEVANCE = "relevance"
    INTEGRITY = "integrity"


class SeverityLevel(str, Enum):
    """Quality issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class QualityRuleType(str, Enum):
    """Types of quality rules."""
    COMPLETENESS_RULE = "completeness_rule"
    PATTERN_RULE = "pattern_rule"
    RANGE_RULE = "range_rule"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    UNIQUENESS_RULE = "uniqueness_rule"
    FRESHNESS_RULE = "freshness_rule"
    CONSISTENCY_RULE = "consistency_rule"
    CUSTOM_RULE = "custom_rule"


@dataclass(frozen=True)
class QualityScore:
    """Represents a quality score for a specific dimension."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    weight: float = 1.0
    confidence: float = 1.0
    calculation_method: str = "automatic"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Quality score must be between 0.0 and 1.0, got {self.score}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted score."""
        return self.score * self.weight
    
    @property
    def is_passing(self) -> bool:
        """Check if score meets minimum threshold."""
        return self.score >= 0.7  # Default threshold
    
    @classmethod
    def create_completeness_score(
        cls, 
        complete_values: int, 
        total_values: int, 
        **kwargs
    ) -> QualityScore:
        """Create completeness score from counts."""
        if total_values == 0:
            score = 1.0
        else:
            score = complete_values / total_values
        
        return cls(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            calculation_method="count_based",
            metadata={
                "complete_values": complete_values,
                "total_values": total_values,
                "missing_values": total_values - complete_values
            },
            **kwargs
        )
    
    @classmethod
    def create_uniqueness_score(
        cls, 
        unique_values: int, 
        total_values: int, 
        **kwargs
    ) -> QualityScore:
        """Create uniqueness score from counts."""
        if total_values == 0:
            score = 1.0
        else:
            score = unique_values / total_values
        
        return cls(
            dimension=QualityDimension.UNIQUENESS,
            score=score,
            calculation_method="uniqueness_ratio",
            metadata={
                "unique_values": unique_values,
                "total_values": total_values,
                "duplicate_count": total_values - unique_values
            },
            **kwargs
        )


@dataclass(frozen=True)
class QualityRule:
    """Represents a data quality rule."""
    rule_id: str
    rule_type: QualityRuleType
    name: str
    description: str
    dimension: QualityDimension
    severity: SeverityLevel
    target_columns: List[str] = field(default_factory=list)
    rule_expression: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_column_specific(self) -> bool:
        """Check if rule applies to specific columns."""
        return len(self.target_columns) > 0
    
    @property
    def weight_factor(self) -> float:
        """Get weight factor based on severity."""
        weights = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.LOW: 0.4,
            SeverityLevel.INFO: 0.2
        }
        return weights.get(self.severity, 0.5)
    
    @classmethod
    def create_completeness_rule(
        cls,
        rule_id: str,
        column_name: str,
        min_completeness: float = 0.95,
        **kwargs
    ) -> QualityRule:
        """Create a completeness rule for a column."""
        return cls(
            rule_id=rule_id,
            rule_type=QualityRuleType.COMPLETENESS_RULE,
            name=f"Completeness check for {column_name}",
            description=f"Column {column_name} must have at least {min_completeness*100}% non-null values",
            dimension=QualityDimension.COMPLETENESS,
            severity=SeverityLevel.HIGH,
            target_columns=[column_name],
            rule_expression=f"completeness >= {min_completeness}",
            parameters={"min_completeness": min_completeness},
            **kwargs
        )
    
    @classmethod
    def create_pattern_rule(
        cls,
        rule_id: str,
        column_name: str,
        pattern: str,
        pattern_name: str = "custom",
        **kwargs
    ) -> QualityRule:
        """Create a pattern validation rule."""
        return cls(
            rule_id=rule_id,
            rule_type=QualityRuleType.PATTERN_RULE,
            name=f"Pattern validation for {column_name}",
            description=f"Column {column_name} must match pattern: {pattern_name}",
            dimension=QualityDimension.VALIDITY,
            severity=SeverityLevel.MEDIUM,
            target_columns=[column_name],
            rule_expression=f"matches_pattern('{pattern}')",
            parameters={"pattern": pattern, "pattern_name": pattern_name},
            **kwargs
        )


@dataclass(frozen=True)
class QualityViolation:
    """Represents a quality rule violation."""
    violation_id: str
    rule_id: str
    column_name: Optional[str]
    row_identifier: Optional[str]
    violation_type: str
    severity: SeverityLevel
    description: str
    detected_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    detection_timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    confidence: float = 1.0
    
    @property
    def is_critical(self) -> bool:
        """Check if violation is critical."""
        return self.severity == SeverityLevel.CRITICAL
    
    @property
    def impact_score(self) -> float:
        """Calculate impact score based on severity."""
        impact_weights = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.LOW: 0.4,
            SeverityLevel.INFO: 0.2
        }
        return impact_weights.get(self.severity, 0.5) * self.confidence


@dataclass(frozen=True)
class QualityReport:
    """Comprehensive quality assessment report."""
    report_id: str
    dataset_id: str
    assessment_timestamp: datetime
    overall_score: float
    dimension_scores: Dict[QualityDimension, QualityScore] = field(default_factory=dict)
    violations: List[QualityViolation] = field(default_factory=list)
    rules_applied: List[str] = field(default_factory=list)
    column_scores: Dict[str, float] = field(default_factory=dict)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError(f"Overall score must be between 0.0 and 1.0, got {self.overall_score}")
    
    @property
    def critical_violations(self) -> List[QualityViolation]:
        """Get critical violations."""
        return [v for v in self.violations if v.is_critical]
    
    @property
    def violation_count_by_severity(self) -> Dict[SeverityLevel, int]:
        """Count violations by severity."""
        counts = {severity: 0 for severity in SeverityLevel}
        for violation in self.violations:
            counts[violation.severity] += 1
        return counts
    
    @property
    def is_passing(self) -> bool:
        """Check if overall quality is acceptable."""
        return self.overall_score >= 0.7 and len(self.critical_violations) == 0
    
    def get_dimension_score(self, dimension: QualityDimension) -> Optional[float]:
        """Get score for specific dimension."""
        score_obj = self.dimension_scores.get(dimension)
        return score_obj.score if score_obj else None
    
    def get_violations_for_column(self, column_name: str) -> List[QualityViolation]:
        """Get violations for specific column."""
        return [v for v in self.violations if v.column_name == column_name]
    
    @classmethod
    def create_from_scores(
        cls,
        report_id: str,
        dataset_id: str,
        dimension_scores: Dict[QualityDimension, QualityScore],
        violations: List[QualityViolation] = None,
        **kwargs
    ) -> QualityReport:
        """Create report from dimension scores."""
        # Calculate overall score as weighted average
        total_weight = sum(score.weight for score in dimension_scores.values())
        if total_weight > 0:
            overall_score = sum(
                score.weighted_score for score in dimension_scores.values()
            ) / total_weight
        else:
            overall_score = 0.0
        
        return cls(
            report_id=report_id,
            dataset_id=dataset_id,
            assessment_timestamp=datetime.now(),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            violations=violations or [],
            **kwargs
        )


@dataclass(frozen=True)
class QualityThreshold:
    """Quality threshold configuration."""
    dimension: QualityDimension
    minimum_score: float
    target_score: float
    weight: float = 1.0
    is_mandatory: bool = False
    description: str = ""
    
    def __post_init__(self):
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError(f"Minimum score must be between 0.0 and 1.0, got {self.minimum_score}")
        if not 0.0 <= self.target_score <= 1.0:
            raise ValueError(f"Target score must be between 0.0 and 1.0, got {self.target_score}")
        if self.minimum_score > self.target_score:
            raise ValueError("Minimum score cannot be greater than target score")
    
    def evaluate_score(self, score: float) -> tuple[bool, str]:
        """Evaluate if score meets threshold."""
        if score >= self.target_score:
            return True, "Exceeds target"
        elif score >= self.minimum_score:
            return True, "Meets minimum requirement"
        else:
            return False, "Below minimum threshold"


@dataclass(frozen=True)
class QualityConfiguration:
    """Quality assessment configuration."""
    config_id: str
    name: str
    description: str
    thresholds: Dict[QualityDimension, QualityThreshold] = field(default_factory=dict)
    rules: List[QualityRule] = field(default_factory=list)
    global_settings: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    @property
    def mandatory_dimensions(self) -> List[QualityDimension]:
        """Get mandatory quality dimensions."""
        return [
            dim for dim, threshold in self.thresholds.items() 
            if threshold.is_mandatory
        ]
    
    @property
    def active_rules(self) -> List[QualityRule]:
        """Get active quality rules."""
        return [rule for rule in self.rules if rule.is_active]
    
    def get_threshold(self, dimension: QualityDimension) -> Optional[QualityThreshold]:
        """Get threshold for specific dimension."""
        return self.thresholds.get(dimension)
    
    def add_rule(self, rule: QualityRule) -> QualityConfiguration:
        """Add a quality rule to configuration."""
        new_rules = list(self.rules) + [rule]
        return dataclass.replace(self, rules=new_rules)
    
    @classmethod
    def create_default_config(cls, config_id: str, name: str) -> QualityConfiguration:
        """Create default quality configuration."""
        default_thresholds = {
            QualityDimension.COMPLETENESS: QualityThreshold(
                dimension=QualityDimension.COMPLETENESS,
                minimum_score=0.95,
                target_score=0.99,
                weight=1.0,
                is_mandatory=True,
                description="Data completeness threshold"
            ),
            QualityDimension.UNIQUENESS: QualityThreshold(
                dimension=QualityDimension.UNIQUENESS,
                minimum_score=0.90,
                target_score=0.95,
                weight=0.8,
                description="Data uniqueness threshold"
            ),
            QualityDimension.VALIDITY: QualityThreshold(
                dimension=QualityDimension.VALIDITY,
                minimum_score=0.85,
                target_score=0.95,
                weight=0.9,
                description="Data validity threshold"
            )
        }
        
        return cls(
            config_id=config_id,
            name=name,
            description="Default quality configuration with standard thresholds",
            thresholds=default_thresholds,
            global_settings={
                "enable_automatic_fixes": False,
                "quality_monitoring_enabled": True,
                "alert_on_critical_violations": True
            }
        )