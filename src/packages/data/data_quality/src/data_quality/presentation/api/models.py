
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel


class DataProfileCreate(BaseModel):
    dataset_name: str
    file_path: str


class ColumnProfileResponse(BaseModel):
    id: UUID
    column_name: str
    data_type: str
    inferred_type: Optional[str]
    position: int
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    foreign_key_table: Optional[str]
    foreign_key_column: Optional[str]
    statistics: Dict[str, Any]
    common_patterns: List[str]
    format_patterns: List[str]
    regex_patterns: List[str]
    top_values: List[Dict[str, Any]]
    sample_values: List[Any]
    invalid_values: List[Any]
    quality_score: float
    anomaly_count: int
    outlier_count: int
    created_at: datetime
    updated_at: datetime
    is_numeric: bool
    is_categorical: bool
    is_high_cardinality: bool
    has_quality_issues: bool


class DataProfileResponse(BaseModel):
    id: UUID
    dataset_name: str
    table_name: Optional[str]
    schema_name: Optional[str]
    status: str
    version: str
    total_rows: int
    total_columns: int
    file_size_bytes: Optional[int]
    column_profiles: List[ColumnProfileResponse]
    completeness_score: float
    uniqueness_score: float
    validity_score: float
    overall_quality_score: float
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    relationships: List[Dict[str, Any]]
    profiling_started_at: Optional[datetime]
    profiling_completed_at: Optional[datetime]
    profiling_duration_ms: float
    profiling_duration_seconds: float
    sample_size: Optional[int]
    sampling_method: Optional[str]
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str
    config: Dict[str, Any]
    tags: List[str]
    is_completed: bool
    is_running: bool
    has_quality_issues: bool
    target_name: str


class RuleConditionCreate(BaseModel):
    column_name: str
    operator: str
    value: Optional[Any]
    case_sensitive: bool = True
    min_value: Optional[Any]
    max_value: Optional[Any]
    pattern: Optional[str]
    reference_table: Optional[str]
    reference_column: Optional[str]
    description: str = ""


class DataQualityRuleCreate(BaseModel):
    name: str
    description: str
    rule_type: str
    severity: str
    dataset_name: str
    table_name: Optional[str]
    schema_name: Optional[str]
    conditions: List[RuleConditionCreate]
    logical_operator: str = "AND"
    expression: Optional[str]
    is_active: bool = True
    is_blocking: bool = False
    auto_fix: bool = False
    fix_action: Optional[str]
    violation_threshold: float = 0.0
    sample_size: Optional[int]
    created_by: str = ""
    updated_by: str = ""
    config: Dict[str, Any] = {}
    tags: List[str] = []
    depends_on: List[UUID] = []


class DataQualityRuleResponse(BaseModel):
    id: UUID
    name: str
    description: str
    rule_type: str
    severity: str
    dataset_name: str
    table_name: Optional[str]
    schema_name: Optional[str]
    conditions: List[Dict[str, Any]]
    logical_operator: str
    expression: Optional[str]
    is_active: bool
    is_blocking: bool
    auto_fix: bool
    fix_action: Optional[str]
    violation_threshold: float
    sample_size: Optional[int]
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str
    last_evaluated_at: Optional[datetime]
    evaluation_count: int
    violation_count: int
    last_violation_at: Optional[datetime]
    config: Dict[str, Any]
    tags: List[str]
    depends_on: List[UUID]
    is_simple_rule: bool
    is_complex_rule: bool
    violation_rate: float
    is_healthy: bool
    has_recent_violations: bool
    target_name: str


class DataQualityCheckCreate(BaseModel):
    name: str
    description: str
    check_type: str
    rule_id: UUID
    dataset_name: str
    column_name: Optional[str]
    schema_name: Optional[str]
    table_name: Optional[str]
    query: Optional[str]
    expression: Optional[str]
    expected_value: Optional[Any]
    threshold: float = 0.95
    tolerance: float = 0.0
    is_active: bool = True
    schedule_cron: Optional[str]
    timeout_seconds: int = 300
    retry_attempts: int = 3
    created_by: str = ""
    updated_by: str = ""
    config: Dict[str, Any] = {}
    environment_vars: Dict[str, str] = {}
    tags: List[str] = []
    depends_on: List[UUID] = []
    blocks: List[UUID] = []


class CheckResultResponse(BaseModel):
    id: UUID
    check_id: UUID
    dataset_name: str
    column_name: Optional[str]
    passed: bool
    score: float
    total_records: int
    passed_records: int
    failed_records: int
    pass_rate: float
    fail_rate: float
    executed_at: datetime
    execution_time_ms: float
    severity: str
    message: str
    details: Dict[str, Any]
    failed_values: List[Any]
    sample_failures: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    tags: List[str]
    is_passed: bool
    is_critical: bool


class DataQualityCheckResponse(BaseModel):
    id: UUID
    name: str
    description: str
    check_type: str
    rule_id: UUID
    dataset_name: str
    column_name: Optional[str]
    schema_name: Optional[str]
    table_name: Optional[str]
    query: Optional[str]
    expression: Optional[str]
    expected_value: Optional[Any]
    threshold: float
    tolerance: float
    is_active: bool
    schedule_cron: Optional[str]
    timeout_seconds: int
    retry_attempts: int
    status: str
    last_executed_at: Optional[datetime]
    next_execution_at: Optional[datetime]
    execution_count: int
    last_result: Optional[CheckResultResponse]
    consecutive_failures: int
    success_rate: float
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str
    config: Dict[str, Any]
    environment_vars: Dict[str, str]
    tags: List[str]
    depends_on: List[UUID]
    blocks: List[UUID]
    is_column_level: bool
    is_table_level: bool
    is_overdue: bool
    has_recent_failure: bool
    is_healthy: bool
    full_target_name: str


class RunCheckRequest(BaseModel):
    file_path: str
