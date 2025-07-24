
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text
from sqlalchemy import JSON, UUID as SA_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class DataProfileModel(Base):
    __tablename__ = "data_profiles"

    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_name = Column(String, nullable=False)
    table_name = Column(String)
    schema_name = Column(String)
    status = Column(String, default="pending")
    version = Column(String, default="1.0.0")
    total_rows = Column(Integer, default=0)
    total_columns = Column(Integer, default=0)
    file_size_bytes = Column(Integer)
    column_profiles = Column(JSON, default=list)  # Store as JSON for flexibility
    completeness_score = Column(Float, default=0.0)
    uniqueness_score = Column(Float, default=0.0)
    validity_score = Column(Float, default=0.0)
    overall_quality_score = Column(Float, default=0.0)
    primary_keys = Column(JSON, default=list)
    foreign_keys = Column(JSON, default=list)
    relationships = Column(JSON, default=list)
    profiling_started_at = Column(DateTime(timezone=True))
    profiling_completed_at = Column(DateTime(timezone=True))
    profiling_duration_ms = Column(Float, default=0.0)
    sample_size = Column(Integer)
    sampling_method = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    updated_by = Column(String)
    config = Column(JSON, default=dict)
    tags = Column(JSON, default=list)

class DataQualityCheckModel(Base):
    __tablename__ = "data_quality_checks"

    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(Text)
    check_type = Column(String, nullable=False)
    rule_id = Column(SA_UUID(as_uuid=True), nullable=True)
    dataset_name = Column(String, nullable=False)
    column_name = Column(String)
    schema_name = Column(String)
    table_name = Column(String)
    query = Column(Text)
    expression = Column(Text)
    expected_value = Column(Text)  # Store as text, convert in application
    threshold = Column(Float, default=0.95)
    tolerance = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    schedule_cron = Column(String)
    timeout_seconds = Column(Integer, default=300)
    retry_attempts = Column(Integer, default=3)
    status = Column(String, default="pending")
    last_executed_at = Column(DateTime(timezone=True))
    next_execution_at = Column(DateTime(timezone=True))
    execution_count = Column(Integer, default=0)
    last_result = Column(JSON)  # Store last result as JSON
    consecutive_failures = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    updated_by = Column(String)
    config = Column(JSON, default=dict)
    environment_vars = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    depends_on = Column(JSON, default=list)
    blocks = Column(JSON, default=list)

class DataQualityRuleModel(Base):
    __tablename__ = "data_quality_rules"

    id = Column(SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(Text)
    rule_type = Column(String, nullable=False)
    severity = Column(String, default="error")
    dataset_name = Column(String, nullable=False)
    table_name = Column(String)
    schema_name = Column(String)
    conditions = Column(JSON, default=list)  # Store conditions as JSON
    logical_operator = Column(String, default="AND")
    expression = Column(Text)
    is_active = Column(Boolean, default=True)
    is_blocking = Column(Boolean, default=False)
    auto_fix = Column(Boolean, default=False)
    fix_action = Column(Text)
    violation_threshold = Column(Float, default=0.0)
    sample_size = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    updated_by = Column(String)
    last_evaluated_at = Column(DateTime(timezone=True))
    evaluation_count = Column(Integer, default=0)
    violation_count = Column(Integer, default=0)
    last_violation_at = Column(DateTime(timezone=True))
    config = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    depends_on = Column(JSON, default=list)
