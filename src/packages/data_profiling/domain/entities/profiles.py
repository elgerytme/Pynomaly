from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class Pattern:
    pattern_type: str
    regex: str
    frequency: int
    percentage: float
    examples: List[str]
    confidence: float

@dataclass(frozen=True)
class ValueDistribution:
    value_counts: Dict[Any, int]
    top_values: List[Any]
    histogram: Optional[Any]
    statistical_summary: Dict[str, float]
    distribution_type: str

@dataclass(frozen=True)
class ColumnProfile:
    column_name: str
    data_type: str
    inferred_type: Optional[str]
    nullable: bool
    unique_count: int
    null_count: int
    completeness_ratio: float
    cardinality: str
    distribution: ValueDistribution
    patterns: List[Pattern]
    quality_score: Optional[float]
    semantic_type: Optional[str]

@dataclass(frozen=True)
class SchemaProfile:
    table_count: int
    column_count: int
    columns: List[ColumnProfile]
    relationships: List[Any]
    constraints: List[Any]
    indexes: List[Any]
    size_metrics: Dict[str, Any]
    schema_evolution: Optional[Any]

@dataclass(frozen=True)
class StatisticalProfile:
    numeric_stats: Dict[str, Dict[str, float]]

@dataclass(frozen=True)
class DataProfile:
    schema_profile: SchemaProfile
    statistical_profile: StatisticalProfile
    content_profile: Any
    quality_assessment: Any
    profiling_metadata: Any
    created_at: datetime
    last_updated: datetime
    version: str